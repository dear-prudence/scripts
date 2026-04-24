import sys
import numpy as np
from archive.hestia import append_particles, filter_particles, cosmo_transform
from archive.hestia import center_halo
from archive.hestia import calc_temperature
from archive.hestia import transform_haloFrame


def stream_filtering(particles, ids):
    rho_cutoff = 28

    if ids is None:
        for key in particles.keys():
            particles[key] = np.array(particles[key])  # Convert to numpy array for easier indexing

        radii = np.zeros(particles['Halo_Coordinates'].shape[0])
        rho = np.zeros(particles['Halo_Coordinates'].shape[0])
        for i in range(particles['Halo_Coordinates'].shape[0]):
            radii[i] = np.linalg.norm(particles['Halo_Coordinates'][i])
            rho[i] = np.linalg.norm(
                np.array([particles['Halo_Coordinates'][i, 0], particles['Halo_Coordinates'][i, 1]]))
        particles['Radii'] = radii
        particles['rho'] = rho

        metals = np.log10(particles['GFM_Metallicity'] / particles['GFM_Metals'][:, 0])
        particles['Z'] = metals

        indices_to_keep = np.where(
            # (1e3 <= particles['Masses']) & (particles['Masses'] <= 1e5) &
            # (-3 <= particles['Z']) & (-1 > particles['Z']) &
            (rho_cutoff <= particles['rho']) &
            (-150 < particles['Halo_Coordinates'][:, 0]) &
            (150 > particles['Halo_Coordinates'][:, 0]) &
            (-150 < particles['Halo_Coordinates'][:, 1]) &
            (150 > particles['Halo_Coordinates'][:, 1]) &
            (-150 < particles['Halo_Coordinates'][:, 2]) &
            (150 > particles['Halo_Coordinates'][:, 2]) &
            # (1 < particles['Angular_Momenta'][:, 0]) &
            # (1e8 > particles['Angular_Momenta'][:, 0]) &
            (10 ** 7.2 < particles['Angular_Momenta'][:, 1]) &
            (10 ** 8.6 > particles['Angular_Momenta'][:, 1]) &
            (10 ** 7.8 < particles['Angular_Momenta'][:, 2]) &
            (10 ** 8.8 > particles['Angular_Momenta'][:, 2]) &
            (-3 < particles['Z']) &
            (-2 > particles['Z'])
        )[0]

        print('Performing initial selection...')
        for key in particles.keys():
            particles[key] = particles[key][indices_to_keep]

        particles_fini = particles

    else:
        # filters particles to only those listed in the particle_ids array
        idx_ids = np.in1d(particles['ParticleIDs'].astype(np.int64), np.array(ids, dtype=np.int64),
                          assume_unique=True)
        # idx_ids = np.where(particles['ParticleIDs'].astype(np.int64) == ids.astype(np.int64))[0]
        print('Before rho filtering...')
        print('len(ids) = ' + str(len(ids)))
        print('len(idx_ids: True) = ' + str(sum(idx_ids)))
        for key in particles.keys():
            particles[key] = particles[key][idx_ids]

        rho = np.zeros(particles['Halo_Coordinates'].shape[0])
        for i in range(particles['Halo_Coordinates'].shape[0]):
            rho[i] = np.linalg.norm(
                np.array([particles['Halo_Coordinates'][i, 0], particles['Halo_Coordinates'][i, 1]]))
        particles['rho'] = rho

        # filters out those particles which accrete onto the neutral disk
        disk_mask = particles['rho'] > rho_cutoff
        particles_fini = {key: val[disk_mask] for key, val in particles.items()}

        print('After rho filtering...')
        print('len(particle[IDs]) = ' + str(len(particles_fini['ParticleIDs'])))

    return particles_fini


def create_weighted_histogram(x, y, weights, bins, mode, part_type, bounds=None):
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=bins, range=bounds)
    # Compute the sum of densities in each bin
    sum_hist, _, _ = np.histogram2d(x, y, bins=(x_e, y_e), range=bounds, weights=weights)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_hist = np.divide(sum_hist, hist, where=(hist != 0))
    if mode == 'temperature' or 'massDen':
        threshold = 1
        avg_hist[hist < threshold] = 0
    elif mode == 'metallicity':
        threshold = 1
        avg_hist[hist < threshold] = -10  # metal-poor primordial gas
    if part_type == 'PartType0':
        return avg_hist, x_e, y_e
    # returns the total (mass) per bin, instead of average mass per particle
    elif part_type == 'PartType4':
        vol_per_bin = float(bounds[0, 1] - bounds[0, 0] / bins[0])
        return sum_hist / vol_per_bin, x_e, y_e


def filter_unphysical_metallicities(data):
    for key in data.keys():
        data[key] = np.array(data[key])  # Convert to numpy array for easier indexing
    # Get the indices where metallicities are negative
    original_length = len(data['GFM_Metallicity'])
    indices_to_keep = np.where(data['GFM_Metallicity'][:] > 0)[0]
    print('Done with unphysical metallicity filtering, number of thrown-out particles: '
          + str(original_length - len(indices_to_keep)))
    for key in data.keys():
        data[key] = data[key][indices_to_keep]
    return data


def param_processing(param, particles, run, snap, part_type, ids=None):
    h = 0.677

    particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                e_abundance=np.array(particles['ElectronAbundance']),
                                                x_h=np.array(particles['GFM_Metals'][:, 0]))

    particles_z = filter_unphysical_metallicities(particles)
    particles_fini = stream_filtering(particles_z, ids)
    weights = particles_fini['Density']
    print('mass(stream_particles) = ' + str(np.sum(particles_fini['Masses'])))
    sum_mass = np.sum(particles_fini['Masses'] * particles_fini['GFM_Metals'][:, 0])
    print('nH0(stream_particles) = ' + str(np.average(particles_fini['NeutralHydrogenAbundance'])))
    avg_nH0 = np.average(particles_fini['NeutralHydrogenAbundance'])
    print('avg(T) = ' + str(np.average(particles_fini['Temperature'])))
    avg_temp = np.average(particles_fini['Temperature'])
    print('sum(AGN_Radiation) = ' + str(np.sum(particles_fini['GFM_AGNRadiation'])))
    sum_agn = np.sum(particles_fini['GFM_AGNRadiation'])
    print('len(stream_particles) = ' + str(len(particles_fini['ParticleIDs'])))
    print(particles_fini['ParticleIDs'])

    return weights, particles_fini, sum_mass, avg_nH0, avg_temp, sum_agn


def retrieve_particles(snap_i, z, sim_run, part_type, size, particle_halo, reference_frame, bool_isolated_halo=False):
    from archive.hestia import halo_dictionary
    h = 0.677
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    halo_id = halo_dictionary(sim_run, reference_frame)

    l_b, u_b = center_halo(run=sim_run, halo_id=halo_id, snap=snap, size=size) * 1e-3  # these are in _h units!
    # l_b and u_b carve a cube with side length 2x specified in "size", to account for the gaps in spatial
    # distribution of particles after the halo coordinate transformation

    if part_type == 'PartType0':
        key_names = ['ParticleIDs', 'Coordinates', 'Density', 'ElectronAbundance', 'GFM_AGNRadiation',
                     'GFM_Metallicity', 'GFM_Metals',
                     'InternalEnergy', 'Masses', 'NeutralHydrogenAbundance', 'StarFormationRate', 'Velocities']
    elif part_type == 'PartType4':
        key_names = ['Coordinates', 'GFM_Metallicity', 'GFM_Metals', 'Masses', 'Velocities', 'GFM_StellarFormationTime']
    else:
        print('Error: \"' + part_type + '\" is in invalid particle type at this time!')
        exit(1)

    if bool_isolated_halo is True:
        base_path = ('/z/rschisholm/storage/snapshots_stream/snapshot_' + snap + '.hdf5')
        file_paths = [base_path]
    else:
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim_run + '/output_2x2.5Mpc/snapdir_'
                     + snap + '/snapshot_' + snap + '.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
    all_particles = {name: None for name in key_names}
    print('Processing Snapshot ' + snap + '...')
    for file_path in file_paths:
        all_particles = append_particles(part_type, file_path, key_names=key_names,
                                         existing_arrays=all_particles)

    print('len all particles: ' + str(len(all_particles['Masses'])))
    # Filter particles based on the bounding box
    if bool_isolated_halo is True:
        filtered_particles = cosmo_transform(all_particles, 'ckpc/h', 'ckpc', part_type=part_type)
    else:
        filtered_particles = cosmo_transform(filter_particles(all_particles, l_b, u_b),
                                             'ckpc/h', 'ckpc', z, part_type=part_type)

    rotated_particles = transform_haloFrame(sim_run, halo_id, snap, filtered_particles)
    return rotated_particles, l_b / h, u_b / h


def make_snap(particles, weights, part_type, param, bins, axis, size):
    h = 0.677

    cartesian = np.array([0, 1, 2])
    # Get the indices not equal to the specified axis
    axs = cartesian[cartesian != axis]

    bounds = np.array([[-1 * size[axs[0]] / 2, size[axs[0]] / 2], [-1 * size[axs[1]] / 2, size[axs[1]] / 2]]) / h

    return create_weighted_histogram(particles['Halo_Coordinates'][:, axs[0]],
                                     particles['Halo_Coordinates'][:, axs[1]],
                                     weights=weights, bins=bins, bounds=bounds, part_type=part_type, mode=param)


def package_data(sim_run, param, dims, n_bins, snaps, particle, particle_halo, reference_halo, bool_isolated_halo):
    from archive.hestia import time_edges

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4'}
    part_type = part_to_type[particle]

    bins = [n_bins, n_bins]  # for the 2-dim histograms

    time_e = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1))

    reference_frame = particle_halo if reference_halo == 'halo' else reference_halo

    # Dictionary to hold results for each axis/dimension
    all_image = {'y-z': None,
                 'x-z': None,
                 'x-y': None,
                 'y_e': None, 'z_e': None, 'x_e': None}

    # ---------------------------------------------------------
    # handles snap 118 first to get particle_ids
    primary_snap = 112
    # creates the cube to restrict all particles used in this script by
    S = {'Coordinates': np.array([dims[1], dims[1], dims[1]])}

    # Transform S coordinates
    z_value = float(time_e[snaps[1] - primary_snap][0])
    S_h = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=z_value)
    print('-------- $z = {}$ --------'.format(z_value))

    # l_b and u_b are in ckpc (converted from _h units)
    particles, l_b, u_b = retrieve_particles(primary_snap, z_value, sim_run, part_type, np.array(S_h['Coordinates']),
                                             particle_halo, reference_frame=reference_frame,
                                             bool_isolated_halo=bool_isolated_halo)

    _, particles_fini, _, _, _, _ = param_processing(param, particles, sim_run, primary_snap, part_type,
                                                  ids=None)
    particle_ids = particles_fini['ParticleIDs']
    print('len(particle_ids) = ' + str(len(particle_ids)))
    # ---------------------------------------------------------
    masses = np.array([])
    nH0s = np.array([])
    temps = np.array([])
    agns = np.array([])

    for i in range(snaps[1], snaps[0], -1):

        # creates the cube to restrict all particles used in this script by
        S = {'Coordinates': np.array([dims[1], dims[1], dims[1]])}

        # Transform S coordinates
        z_value = float(time_e[snaps[1] - i][0])
        S_h = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=z_value)
        print('-------- $z = {}$ --------'.format(z_value))

        # l_b and u_b are in ckpc (converted from _h units)
        particles, l_b, u_b = retrieve_particles(i, z_value, sim_run, part_type, np.array(S_h['Coordinates']),
                                                 particle_halo, reference_frame=reference_frame,
                                                 bool_isolated_halo=bool_isolated_halo)

        weights, particles_fini, mass, nH0, agn, temp = param_processing(param, particles, sim_run, i, part_type,
                                                                         ids=particle_ids)

        masses = np.append(masses, mass)
        nH0s = np.append(nH0s, nH0)
        temps = np.append(temps, temp)
        agns = np.append(agns, agn)
        # Loop through the axes
        for axis, name_i, edge_i in zip(range(3), ['y-z', 'x-z', 'x-y'], ['y_e', 'z_e', 'x_e']):

            # creates the prism to restrict particles being plotted by
            S = {'Coordinates': np.array([dims[1], dims[1], dims[1]])}
            S['Coordinates'][axis] = dims[0]
            S_h = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=z_value)

            current_image, i_edge, j_edge = make_snap(particles_fini, weights, part_type, param, bins, axis,
                                                      np.array(S_h['Coordinates']))
            print('current_image.shape = ' + str(current_image.shape))

            # Initialize arrays if this is the first snapshot
            if i == snaps[1]:
                particle_ids = particles_fini['ParticleIDs']
                all_image[name_i] = current_image
                # store the edges, for the x-z case, store the z_edges (second axis) instead
                all_image[edge_i] = j_edge if axis == 1 else i_edge
                # print('all_image[' + name_i + '].shape = ' + str(all_image[name_i].shape))
            else:
                # Append new snapshot data
                # print('all_image[' + name_i + '].shape = ' + str(all_image[name_i].shape))
                all_image[name_i] = np.dstack((all_image[name_i], current_image))
                all_image[edge_i] = j_edge if axis == 1 else i_edge  # edges only need to be stored once

        # Now all_image['x-y'], all_image['y-z'], image['x-z'], etc., hold the stacked data
        print('Snapshot ' + str(i) + ': check')

    # Combine all dictionaries into one
    data_to_save = all_image.copy()  # Start with all_planes
    data_to_save['time'] = time_e  # timestamps
    data_to_save['column_width'] = round(dims[1] / float(n_bins), 3)  # in ckpc
    data_to_save['column_depth'] = dims[0]  # in ckpc
    data_to_save['image_size'] = dims[1]  # in ckpc

    # Save data
    np.savez('/z/rschisholm/storage/isolateStream/' + parameter + '/'
             + simulation_run + '_' + particle_halo[0:4] + particle_halo[5:]
             + '_' + particle + '_' + parameter + '_'
             + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc'
             + ('_stream' if bool_isolated_halo is True else '') + '.npz', **data_to_save)

    # Indicate that the script has completed its task
    print('Done!')
    # Terminate the script
    sys.exit(0)


def plotting(sim_run, param, dims, particle, particle_halo, bool_isolated_halo, planes=None):
    from scripts.util.archive.plots_old import plot_imageMap_frames

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/isolateStream/' + param + '/'
                  + sim_run + '_'
                  + (particle_halo[0:4] + particle_halo[5:] if len(particle_halo) != 3 else particle_halo)
                  + '_' + particle + '_' + param + '_'
                  + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc'
                  + ('_stream' if bool_isolated_halo is True else '') + '.npz')

    output_path = ('/Users/dear-prudence/Desktop/smorgasbord/isolateStream/' + param + '/'
                   + str(dims[0]) + 'x' + str(dims[1]) + '_frames/')

    # optional argument "axis" where default is face-on, edge-on
    plot_imageMap_frames(param, input_path, output_path, scale=dims, planes=planes)


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
parameter = 'massDen'
dims_ = [400, 400]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
bins_ = 800
snaps_ = [101, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
halo_ = 'halo_08'  # chosen halo frame of reference
particle_type = 'gas'
frame_ = 'halo'  # will transform to halo coords if "frame_" is set to 'halo', else will transform to specified halo
isolated_halo = False
type_plot = 'frames'
manual = False
planes_ = ['x-y', 'x-z']
# ------------------------------------

if machine == 'geras':
    package_data(simulation_run, parameter, dims_, bins_, snaps_, particle_type, halo_, frame_, isolated_halo)

elif machine == 'dear-prudence':
    plotting(simulation_run, parameter, dims_, particle_type, halo_, isolated_halo, planes=planes_)

