import sys
import numpy as np
from scripts.hestia import append_particles, filter_particles, cosmo_transform
from scripts.hestia import time_edges, center_halo
from scripts.hestia import get_halo_params
from scripts.hestia import calc_temperature
from scripts.hestia import transform_haloFrame


def create_weighted_histogram(x, y, weights, bins, mode, part_type, bounds=None):
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=bins, range=bounds)
    # Compute the sum of densities in each bin
    sum_hist, _, _ = np.histogram2d(x, y, bins=(x_e, y_e), range=bounds, weights=weights)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_hist = np.divide(sum_hist, hist, where=(hist != 0))
    if mode == 'massDen' or 'temperature':
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


def param_processing(param, particles, run, snap, part_type):
    h = 0.677
    if param == 'numDen':
        particles_fini = particles
        weights = None
    elif param == 'massDen':
        if part_type == 'PartType0':
            particles_fini = particles
            weights = particles['Density']
        elif part_type == 'PartType4':
            # stars have SFT > 0, wind particles have SFT < 0
            stellar_mask = particles['GFM_StellarFormationTime'] > 0
            particles_fini = {key: val[stellar_mask] for key, val in particles.items()}
            weights = particles_fini['Masses'] * 1e10 / h
        else:
            exit(1)
    elif param == 'temperature':
        particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                    e_abundance=np.array(particles['ElectronAbundance']),
                                                    x_h=np.array(particles['GFM_Metals'][:, 0]))
        particles_fini = particles
        weights = particles_fini['Temperature']
    elif param == 'velocity':
        _, _, halo_vel, _ = get_halo_params(run, halo, snap)
        vel_mags = np.array([])
        for j in range(len(particles['Velocities'])):
            vel_mags = np.append(vel_mags, np.linalg.norm(particles['Velocities'][j] - halo_vel))
        particles_fini = particles
        weights = particles_fini['Velocity']
    elif param == 'metallicity':
        particles_fini = filter_unphysical_metallicities(particles)
        weights = np.log10(particles_fini['GFM_Metallicity'] / particles_fini['GFM_Metals'][:, 0])
    elif param == 'AGNradiation':
        particles_fini = particles
        weights = particles_fini['GFM_AGNRadiation']
    elif param == 'nH0':
        particles_fini = particles
        weights = particles_fini['NeutralHydrogenAbundance']
    else:
        print('Error: invalid post-processing parameter!')
        exit(1)

    return weights, particles_fini


def retrieve_particles(snap_i, z, run, part_type, size, follow_gal=None, isolate_halo=False, corona=False, frame=None):
    h = 0.677
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    halo_to_id = {'lmc': '010', 'mw': '003', 'halo_08': '008', 'halo_11': '011',
                  'halo_15': '015', 'halo_16': '016', 'halo_21': '021', 'halo_23': '023', 'halo_27': '027',
                  'halo_28': '028', 'halo_54': '054', 'halo_94': '094', 'halo_130': '130'}
    halo_id = halo_to_id[follow_gal]

    l_b, u_b = center_halo(run=run, halo_id=halo_id, snap=snap, size=size) * 1e-3  # these are in _h units!

    if part_type == 'PartType0':
        key_names = ['Coordinates', 'Density', 'ElectronAbundance', 'GFM_AGNRadiation', 'GFM_Metallicity', 'GFM_Metals',
                     'InternalEnergy', 'Masses', 'NeutralHydrogenAbundance', 'StarFormationRate', 'Velocities']
    elif part_type == 'PartType4':
        key_names = ['Coordinates', 'GFM_Metallicity', 'GFM_Metals', 'Masses', 'Velocities', 'GFM_StellarFormationTime']
    else:
        print('Error: Invalid particle type!')
        exit(1)

    if isolate_halo is True:
        base_path = ('/z/rschisholm/storage/snapshots_' + halo + ('/cgm' if corona is True else '')
                     + '/snapshot_' + snap + '.hdf5')
        file_paths = [base_path]
    else:
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run + '/output_2x2.5Mpc/snapdir_'
                     + snap + '/snapshot_' + snap + '.')
        file_extension = '.hdf5'
        # Generate file paths using a loop
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_particles = {name: None for name in key_names}
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + snap + '...')
    for file_path in file_paths:
        all_particles = append_particles(part_type, file_path, key_names=key_names,
                                         existing_arrays=all_particles)
    # Filter particles based on the bounding box
    filtered_particles = cosmo_transform(filter_particles(all_particles, l_b, u_b),
                                         'ckpc/h', 'ckpc', z, part_type=part_type)
    if frame is not None:
        rotated_particles = transform_haloFrame(run, halo_id, snap, filtered_particles)
    else:
        rotated_particles = filtered_particles

    return rotated_particles, l_b / h, u_b / h


def make_snap(snap_i, z, run, part_type, param, bins, size, axs, follow_gal=None, isolate_halo=False,
              corona=False, frame=None):
    h = 0.677
    # bins object must be a two element list
    # axes object follows the formal [x_axis, y_axis] (i.e. if plotting z vs y, axes = [1, 2])

    # l_b and u_b are in ckpc (converted from _h units)
    particles, l_b, u_b = retrieve_particles(snap_i, z, run, part_type, size, follow_gal=follow_gal,
                                             isolate_halo=isolate_halo, corona=corona, frame=frame)
    weights, particles_fini = param_processing(param, particles, run, snap_i, part_type)
    print(len(particles_fini['Coordinates'][:, 0]))
    print(particles_fini['Coordinates'][:, 0][0])
    if frame is not None:
        # size is in _h units
        bounds = np.array([[-1 * size[axs[0]] / 2, size[axs[0]] / 2], [-1 * size[axs[1]] / 2, size[axs[1]] / 2]]) / h
    else:
        bounds = np.array([[l_b[axs[0]], u_b[axs[0]]], [l_b[axs[1]], u_b[axs[1]]]])

    return create_weighted_histogram(particles_fini['Coordinates' if frame is None else 'Halo_Coordinates'][:, axs[0]],
                                     particles_fini['Coordinates' if frame is None else 'Halo_Coordinates'][:, axs[1]],
                                     weights=weights, bins=bins,
                                     bounds=bounds, part_type=part_type,
                                     mode=param)


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
parameter = 'temperature'
dims = [100, 800]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
axes = [0, 1]
bins_ = 400
snaps = [67, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
halo = 'halo_15'
Particle = 'gas'
frame_ = ''
isolated_halo = False
Corona = False
type_plot = 'frames'
manual = False
# ------------------------------------

n_bins = [bins_, bins_]
if Particle == 'stars':
    particle = 'PartType4'
else:
    particle = 'PartType0'

if machine == 'geras':
    if __name__ == "__main__":

        spatial_size = [dims[0] for i in range(3)]
        for ax in axes:
            spatial_size[ax] = dims[1]
        S = {'Coordinates': np.array(spatial_size)}

        time_edges = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1))
        for i in range(snaps[1], snaps[0], -1):
            S_h = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=time_edges[127 - i][0])
            print('time: $z = $' + str(time_edges[127 - i][0]))
            current_snapshot, x_edges, y_edges = make_snap(i, float(time_edges[127 - i][0]), simulation_run,
                                                           part_type=particle, param=parameter, bins=n_bins,
                                                           size=np.array(S_h['Coordinates']),
                                                           axs=axes, follow_gal=halo, isolate_halo=isolated_halo,
                                                           corona=Corona, frame=frame_)
            print(np.max(current_snapshot))
            print(current_snapshot)
            if i == snaps[1]:
                all_snapshots = current_snapshot
            else:
                # Append the new snapshot
                all_snapshots = np.dstack((all_snapshots, current_snapshot))
            print('Snapshot ' + str(i) + ': check')

            # Save data
            np.savez('/z/rschisholm/storage/images/' + parameter + '/' + simulation_run + '_'
                     + ('gas' if particle == 'PartType0' else 'stars') + '_' + parameter + '_' + halo + '_'
                     + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc_'
                     + 'bin' + str(round(spatial_size[axes[0]] / float(n_bins[0]), 2)) + 'ckpc_'
                     + str(axes[0]) + '_' + str(axes[1])
                     + ('isolated' if isolated_halo is True else '')
                     + ('corona' if Corona is True else '')
                     + ('haloFrame' if frame_ == 'halo' else '')
                     + '.npz',
                     data=all_snapshots, x_edges=x_edges, y_edges=y_edges, time=time_edges)

        # Indicate that the script has completed its task
        print('Done!')
        # Terminate the script
        sys.exit(0)

    # ------------------------------------
elif machine == 'dear-prudence':
    from scripts.archive import ursa
    from scripts.local.archive.plots_old import plot_image, plot_frames

    bin_size = round(dims[1] / float(n_bins[0]), 2)

    if type_plot == 'image':
        input_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + halo + '/'
                      + ('isolated/' if isolated_halo is True else '')
                      + ('corona/' if Corona is True else '')
                      + parameter + '/'
                      + str(dims[0]) + 'x' + str(dims[1]) + '/09_18_gas_' + parameter + '_' + halo
                      + '_' + str(dims[0]) + 'x' + str(dims[1])
                      + 'ckpc_bin' + str(bin_size) + 'ckpc_0_1'
                      + ('isolated' if isolated_halo is True else '')
                      + ('isolatedcorona' if Corona is True else '')
                      + '.npz')
        output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + halo + '/'
                       + ('isolated/' if isolated_halo is True else '')
                       + ('corona/' if Corona is True else '')
                       + parameter + '/'
                       + str(dims[0]) + 'x' + str(dims[1]) + '/'
                       '09_18_gas_' + parameter + '_' + halo + '_' + str(dims[0]) + 'x' + str(dims[1])
                       + 'ckpc_0_1.png')
        snaps = [83, 89, 110, 116, 121, 127]
        # snap 95 (z = 0.506): "primordial LMC"
        # snap 108 (z = 0.258): fully-formed, near-peak mass LMC
        # snap 114 (z = 0.165): LMC peak mass, M_vir ~ 3.52e+11
        # snap 119 (z = 0.099): after major ejection of gas
        # snap 122 (z = 0.060): LMC passes R_vir of the MW
        # snap 127 (z = 0.0): present-day (M_vir ~ 1.85e+11, M_gas ~ 1.26e+10)

        plot_image(parameter, input_path, output_path, snaps=snaps, scale=[50, 400])

    elif type_plot == 'frames':
        if manual is True:
            input_paths = ['/Users/dear-prudence/Desktop/smorgasbord/images/halo_08/stars/massDen/200x400_rot/'
                           '09_18_stars_massDen_halo_08_200x400ckpc_bin0.5ckpc_0_1haloFrame.npz',
                           '/Users/dear-prudence/Desktop/smorgasbord/images/halo_08/massDen/400x400_rot/'
                           '09_18_gas_massDen_halo_08_400x400ckpc_bin0.5ckpc_0_1haloFrame.npz']
            output_path = '/Users/ursa/Desktop/smorgasbord/images/halo_08/starsVgas/'
        else:
            input_paths = ['/Users/dear-prudence/Desktop/smorgasbord/images/' + halo + '/'
                           + ('isolated/' if isolated_halo is True else '')
                           + ('corona/' if Corona is True else '')
                           + ('stars/' if Particle == 'stars' else '')
                           + parameter + '/'
                           + str(dims[0]) + 'x' + str(dims[1]) + '/09_18_' + Particle + '_' + parameter + '_' + halo
                           + '_' + str(dims[0]) + 'x' + str(dims[1])
                           + 'ckpc_bin' + str(bin_size) + 'ckpc_0_1'
                           + ('isolated' if isolated_halo is True else '')
                           + ('isolatedcorona' if Corona is True else '')
                           + ('haloFrame' if frame_ == 'halo' else '')
                           + '.npz',
                           '/Users/dear-prudence/Desktop/smorgasbord/images/' + halo + '/'
                           + ('isolated/' if isolated_halo is True else '')
                           + ('corona/' if Corona is True else '')
                           + ('stars/' if Particle == 'stars' else '')
                           + parameter + '/'
                           + str(dims[0]) + 'x' + str(dims[1]) + '/09_18_' + Particle + '_' + parameter + '_' + halo
                           + '_' + str(dims[0]) + 'x' + str(dims[1])
                           + 'ckpc_bin' + str(bin_size) + 'ckpc_2_1'
                           + ('isolated' if isolated_halo is True else '')
                           + ('isolatedcorona' if Corona is True else '')
                           + ('haloFrame' if frame_ == 'halo' else '')
                           + '.npz']
            output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + halo + '/'
                           + ('isolated/' if isolated_halo is True else '')
                           + ('corona/' if Corona is True else '')
                           + ('stars/' if Particle == 'stars' else '')
                           + parameter + '/'
                           + str(dims[0]) + 'x' + str(dims[1]) + '/')

        plot_frames(parameter, input_paths, output_path, scale=dims)

    elif type_plot == 'combo_frames':
        input_paths = ['/Users/dear-prudence/Desktop/smorgasbord/images/temperature/50x400ckpc/'
                       '09_18_gas_temperature_LMC_50x400x400ckpc_bin1.0ckpc_2_1.npz',
                       '/Users/dear-prudence/Desktop/smorgasbord/images/velMag/'
                       '09_18_gas_velocityMag_lmc_50x400ckpc_bin1.0ckpc_2_1.npz']
        output_path = '/Users/ursa/Desktop/smorgasbord/images/combination_temperature_velocity_50x400/'

        ursa.plot_combo_frames(['temperature', 'velocity'], input_paths, output_path, scale=[100, 800])

    print('Done!')
