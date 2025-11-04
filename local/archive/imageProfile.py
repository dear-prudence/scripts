import sys
import numpy as np
from hestia import append_particles, filter_particles
from hestia import get_halo_params
from hestia import calc_temperature
from hestia import transform_haloFrame, rid_h_units


def create_weighted_histogram(part_type, x, weights, log, bins, densities, bounds=None):
    # Create a 2D histogram
    hist, x_e = np.histogram(x, bins=bins, range=bounds, weights=densities)
    # Compute the sum of densities in each bin
    sum_hist, _ = np.histogram(x, bins=bins, range=bounds, weights=weights * densities)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_hist = np.divide(sum_hist, hist, where=(hist != 0))

    # threshold = 1
    # avg_hist[hist < threshold] = background
    if log is True:
        avg_hist = np.log10(avg_hist)
    else:
        pass

    if part_type == 'PartType0':
        return avg_hist, x_e
    # returns the total (mass) per bin, instead of average mass per particle
    elif part_type == 'PartType4':
        vol_per_bin = float(bounds[0, 1] - bounds[0, 0] / bins[0])
        return sum_hist / vol_per_bin, x_e


def filter_unphysical_Z(data):
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


def filter_stars(particles):
    # stars have SFT > 0, wind particles have SFT < 0
    stellar_mask = particles['GFM_StellarFormationTime'] > 0
    return {key: val[stellar_mask] for key, val in particles.items()}


def add_temperature(particles):
    temp_column = calc_temperature(u=np.array(particles['InternalEnergy']), e_abundance=np.array(particles['ElectronAbundance']),
                                   x_h=np.array(particles['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    particles['Temperature'] = temp_column
    return particles


def add_velocity(particles):
    v_xyz = np.zeros(particles['Halo_Velocities'].shape[0])
    for j in range(particles['Halo_Velocities'].shape[0]):
        v_xyz[j] = np.sqrt(particles['Halo_Velocities'][j, 0] ** 2
                           + particles['Halo_Velocities'][j, 1] ** 2
                           + particles['Halo_Velocities'][j, 2] ** 2)
    particles['Vel_Mag'] = v_xyz
    return particles


def rhoH_to_nH(particles, mu=0.59):
    rhoH = particles['Density'] * particles['GFM_Metals'][:, 0]
    # 1.00784 for mostly H0, 0.59 for mostly H1
    M_solar_to_g = 1.989e33  # Solar mass in grams
    kpc_to_cm = 3.086e21  # 1 kpc in cm
    amu_to_g = 1.66053906660e-24  # Atomic mass unit in grams
    # Safeguard against extremely large inputs
    rhoH = np.asarray(rhoH, dtype=np.float64)  # Ensure it's a float array or scalar
    if np.any(rhoH <= 0):
        raise ValueError("Hydrogen mass density must be positive and non-zero.")
    # Convert hydrogen mass density to g/cm^3
    rhoH_cgs = (rhoH * M_solar_to_g) / (kpc_to_cm ** 3)
    # Convert mass density to number density
    m_h = mu * amu_to_g  # Mass of a hydrogen atom/molecule in grams
    nH = rhoH_cgs / m_h
    particles['n_H'] = nH
    return particles


def param_processing(part_type, param, particles, snap):
    if part_type == 'PartType0':
        if param == 'massDen':
            fini = particles
            weights = particles['Density']
            background = 1  # mass density of IGM
        elif param == 'temperature':
            fini = particles
            weights = calc_temperature(u=np.array(particles['InternalEnergy']),
                                       e_abundance=np.array(particles['ElectronAbundance']),
                                       x_h=np.array(particles['GFM_Metals'][:, 0]))
            background = 1e5  # temperature of IGM
        elif param == 'velMag':
            fini = add_velocity(particles)
            weights = fini['Vel_Mag']
            background = 0
        elif param == 'metallicity':
            fini = filter_unphysical_Z(particles)
            weights = np.log10(fini['GFM_Metallicity'] / fini['GFM_Metals'][:, 0])
            background = -6,  # very metal poor primordial gas
        elif param == 'num_H0':
            fini = rhoH_to_nH(particles, mu=1.00784)
            weights = fini['n_H'] * fini['NeutralHydrogenAbundance']
            log = True
        elif param == 'num_H1':
            fini = rhoH_to_nH(particles, mu=0.59)
            weights = fini['n_H'] * (1 - fini['NeutralHydrogenAbundance'])
            background = 1e-7,  # number density of H1 in IGM
        elif param == 'vx':
            fini = particles
            _, _, _, halo_vel, _, _ = get_halo_params(simulation_run, halo_, snap)
            weights = fini['Halo_Velocities'][:, 0]
            background = 0
        elif param == 'vy':
            fini = particles
            _, _, _, halo_vel, _, _ = get_halo_params(simulation_run, halo_, snap)
            weights = fini['Halo_Velocities'][:, 1]
            background = 0
        elif param == 'vz':
            fini = particles
            _, _, _, halo_vel, _, _ = get_halo_params(simulation_run, halo_, snap)
            weights = fini['Halo_Velocities'][:, 2]
            background = 0
        else:
            print('Error: \"' + param + '\" is an invalid parameter!')
            exit(0)

    elif part_type == 'PartType4':
        if param == 'massDen':
            fini = filter_stars(particles)
            weights = fini['Masses']
            background = 1e-10  # arbitrarily low stellar density for IGM
        elif param == 'velocity':
            fini = add_velocity(filter_stars(particles))
            weights = fini['Vel_Mag']
            background = 0
        else:
            print('Error: \"' + param + '\" is an invalid parameter!')
            exit(0)

    else:
        print('Error: \"' + part_type + '\" is an invalid particle type!')
        exit(0)

    return fini, weights, log


def retrieve_particles(run, halo, snap, z, part_type, using_isolated_halo=False):
    import h5py

    h = 0.677
    alpha = 1 / h if halo == 'stream' else 1e-3
    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    halo_id, _, pos_h, vel_h, l_h, r_vir_h = get_halo_params(run, halo, snap)

    lb, ub = (pos_h - 4 * r_vir_h) * alpha, (pos_h + 4 * r_vir_h) * alpha

    if using_isolated_halo is True:
        file_path = '/z/rschisholm/storage/snapshots_' + halo + '/snapshot_' + snap_ + '.hdf5'
        with h5py.File(file_path, 'r') as file:
            keys = file[part_type].keys()
        all_particles = {name: None for name in keys}
        all_particles = append_particles(part_type, file_path, key_names=keys, existing_arrays=all_particles)

    else:
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run + '/output_2x2.5Mpc/snapdir_'
                     + snap_ + '/snapshot_' + snap_ + '.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        with h5py.File(base_path + '0' + file_extension, 'r') as file:
            keys = file[part_type].keys()
        all_particles = {name: None for name in keys}
        for file_path in file_paths:
            all_particles = append_particles(part_type, file_path, key_names=keys,
                                             existing_arrays=all_particles)

    processed_particles = transform_haloFrame(run, halo_id, snap,
                                              rid_h_units(
                                         filter_particles(all_particles, lb, ub), z, part_type)) \
        if halo != 'stream' else transform_haloFrame(run, halo_id, snap, filter_particles(all_particles, lb, ub))
    return processed_particles


def make_snap(part_type, particles_full_depth, weights_full_depth, column_depth, width, log, bins, axis):
    bounds = np.array([-1 * width / 2, width / 2])

    column_mask = np.abs(particles_full_depth['Halo_Coordinates'][:, axis]) < (column_depth / 2)
    particles = {key: val[column_mask] for key, val in particles_full_depth.items()}
    weights = weights_full_depth[column_mask]

    return create_weighted_histogram(part_type, particles['Halo_Coordinates'][:, axis],
                                     weights=weights, log=log, bins=bins, densities=particles['n_H'], bounds=bounds)


def package_data(run, param, width, column_depth, bins, particle_type, halo, bool_isolated_halo):
    from hestia import get_lookbackTimes

    print('---------------------------------------------------\n'
          + 'Creating image-map...\n'
          + 'sim_run -- ' + run + '\n'
          + 'halo -- ' + halo + '\n'
          + 'part_type -- ' + particle_type + '\n'
          + 'parameter -- ' + param + '\n'
          + 'width, column_depth, bins -- ' + str(width) + ', ' + str(column_depth) + ', ' + str(bins) + '\n'
          + '---------------------------------------------------')

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4'}
    part_type = part_to_type[particle_type]

    redshifts, lookback_times = get_lookbackTimes(snaps)

    # Dictionary to hold results for each axis/dimension
    all_profiles = {'x': None, 'y': None, 'z': None,
                    'x_e': None, 'y_e': None, 'z_e': None}

    for snap in range(snaps[1], snaps[0], -1):
        z = float(redshifts[snaps[1] - snap])
        print('-------- $z = {}$ --------'.format(z))

        # l_b and u_b are in ckpc (converted from _h units)
        particles = retrieve_particles(run, halo, snap, z, part_type, using_isolated_halo=bool_isolated_halo)
        particles_fini, weights, log = param_processing(part_type, param, particles, snap)

        # Loop through the axes
        for i, r_i, edge_i in zip(range(3), ['x', 'y', 'z'], ['x_e', 'y_e', 'z_e']):

            current_profile, i_edge = make_snap(part_type, particles_fini, weights,
                                                column_depth, width, log, bins, i)
            print('current_image.shape = ' + str(current_profile.shape))

            # Initialize arrays if this is the first snapshot
            # np.dstack is the reason you have to take the transpose in the plotting routine, consider np.vstack

            if snap == snaps[1]:
                all_profiles[r_i] = current_profile
                # store the edges, for the x-z case, store the z_edges (second axis) instead
                all_profiles[edge_i] = i_edge
                print('all_image[' + r_i + '].shape = ' + str(all_profiles[r_i].shape))
            else:
                # Append new snapshot data
                all_profiles[r_i] = np.vstack((all_profiles[r_i], current_profile))
                print('all_image[' + r_i + '].shape = ' + str(all_profiles[r_i].shape))

        print('Snapshot ' + str(snap) + ': check')

    # Combine all dictionaries into one
    data_to_save = all_profiles.copy()  # Start with all_planes
    data_to_save['redshifts'], data_to_save['lookback_times'] = redshifts, lookback_times
    data_to_save['column_depth'] = column_depth  # in ckpc
    data_to_save['image_width'] = width  # in ckpc

    # Save data
    output_path = '/z/rschisholm/storage/images/' + param + '/'
    output_name = run + '_' + halo + '_' + particle_type + '_' + param + '_profiles' + str(width) + 'ckpc.npz'
    np.savez(output_path + output_name, **data_to_save)

    print('Done!\n----------------------------------------------')
    print('scp -P 2222 rschisholm@geras.aip.de:' + output_path + output_name +
          ' /Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/')
    print('----------------------------------------------')
    sys.exit(0)


def plotting(run, param, width, particle_type, halo, axis=None):
    from scripts.local.archive.plots_old import plot_imageProfile

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
                  + run + '_' + halo + '_' + particle_type + '_' + param + '_profiles' + str(width) + 'ckpc.npz')

    output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
                   + particle_type + '_' + param + '_profile' + str(width) + 'kpc_' + axis + '.png')

    plot_imageProfile(param, input_path, output_path, axis=axis)

    print('Done!')


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
parameter = 'num_H0'
width_ = 300  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
column_depth_ = width_
bins_ = 100
snaps = [121, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
halo_ = 'stream'  # chosen halo frame of reference, or 'stream' for MS-analog
particle_type_ = 'gas'
isolated_halo = True
# ------------------------------------
manual = False
axis_ = 'y'
# ------------------------------------

if machine == 'geras':
    package_data(simulation_run, parameter, width_, column_depth_, bins_, particle_type_, halo_, isolated_halo)

elif machine == 'dear-prudence':
    plotting(simulation_run, parameter, width_, particle_type_, halo_, axis=axis_)
