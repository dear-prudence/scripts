# This script will bin gas at all temps and return average column temperature over a specified number of snapshots;
# it will return two numpy data files, which can be used to create a two-panel side-by-side movie
import sys
import numpy as np
from scripts.hestia import append_particles
from scripts.hestia import filter_particles
from scripts.hestia import cosmo_transform
from scripts.hestia import time_edges
from scripts.hestia import center_lmc
from scripts.hestia import calc_temperature


def create_weighted_histogram(x, y, temperature, bins):
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=bins)

    # Compute the sum of temperatures in each bin
    sum_temps, _, _ = np.histogram2d(x, y, bins=(x_e, y_e), weights=temperature)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_temps = np.divide(sum_temps, hist, where=(hist != 0))
    return avg_temps, x_e, y_e


def add_temperature(data):
    temp_column = calc_temperature(u=np.array(data['InternalEnergy']), e_abundance=np.array(data['ElectronAbundance']),
                                   x_h=np.array(data['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    data['Temperature'] = temp_column
    return data


def make_snap(snap_i, z, run, bins, size, axs, follow_gal=None):
    # bins object must be a two element list
    # axes object follows the formal [x_axis, y_axis] (i.e. if plotting z vs y, axes = [1, 2])
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)
    key_names = ['Coordinates', 'InternalEnergy', 'ElectronAbundance', 'GFM_Metals']
    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_'
                 + str(snap) + '/snapshot_' + str(snap) + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_particles = {name: None for name in key_names}
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + str(snap) + '...')
    for file_path in file_paths:
        all_particles = append_particles('PartType0', file_path, key_names=key_names, existing_arrays=all_particles)
    if follow_gal == 'lmc':
        l_b, u_b = center_lmc(run=run, lmc_halo_id='10', snap=snap, size=size) * 1e-3  # convert from kpc to Mpc
    else:
        print('Error: Invalid reference galaxy! routine is WIP.')
        exit()
    # Filter particles based on the bounding box
    filtered_proper_particles = cosmo_transform(filter_particles(all_particles, l_b, u_b),
                                                'ckpc/h', 'ckpc', z)
    particles_w_t = add_temperature(filtered_proper_particles)
    return create_weighted_histogram(particles_w_t['Coordinates'][:, axs[0]],
                                     particles_w_t['Coordinates'][:, axs[1]],
                                     particles_w_t['Temperature'], bins=bins)


# ------------------------------------
mode = 'dear-prudence'
# ------------------------------------

if mode == 'geras':
    if __name__ == "__main__":
        # -------------------
        simulation_run = '09_18'
        # for this script, box needs to be a cube
        dims = [50, 400]  # in c-kpc
        axes = [0, 1]
        n_bins = [800, 800]
        snaps = [67, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
        snap_stepSize = 1
        halo = 'lmc'
        # -------------------
        spatial_size = [dims[0] for i in range(3)]
        for ax in axes:
            spatial_size[ax] = dims[1]
        S = {'Coordinates': np.array(spatial_size)}
        # module that writes the panel
        time_edges = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1 * snap_stepSize))

        for i in range(snaps[1], snaps[0], -1 * snap_stepSize):
            S_co = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=time_edges[127 - i][0])
            print('time: z = ' + str(time_edges[127 - i][0]))
            print('S_co coordinates: ' + str(S_co['Coordinates']))
            current_snapshot, x_edges, y_edges = make_snap(i, float(time_edges[127 - i][0]), simulation_run, bins=n_bins,
                                                           size=np.array(S_co['Coordinates']),
                                                           axs=axes, follow_gal='lmc')
            if i == snaps[1]:
                all_snapshots = current_snapshot
            else:
                # Append the new snapshot
                all_snapshots = np.dstack((all_snapshots, current_snapshot))
            print('Snapshot ' + str(i) + ': check')
        # Save data
        np.savez('/z/rschisholm/storage/images/' + simulation_run + '_gas_temperature_' + halo + '_'
                 + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc_'
                 + 'bin' + str(round(spatial_size[axes[0]] / float(n_bins[0]), 2)) + 'ckpc_'
                 + str(axes[0]) + '_' + str(axes[1]) + '.npz',
                 data=all_snapshots, x_edges=x_edges, y_edges=y_edges, time=time_edges)

        # Indicate that the script has completed its task
        print('Done!')
        # Terminate the script
        sys.exit(0)

elif mode == 'dear-prudence':
    from scripts.archive import ursa

    # ------------------------------------
    type_plot = 'frames'
    # ------------------------------------

    if type_plot == 'image':
        input_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/temperature_LMC_100x800ckpc/'
                      '09_18_gas_temperature_LMC_100x800x800ckpc_bin2.0ckpc_2_1.npz')
        output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/temperature_LMC_100x800ckpc/'
                       '09_18_gas_temperature_lmc_100x800ckpc_faceOn.png')
        snaps = [95, 108, 114, 119, 122, 127]
        # snap 95 (z = 0.506): "primordial LMC"
        # snap 108 (z = 0.258): fully-formed, near-peak mass LMC
        # snap 114 (z = 0.165): LMC peak mass, M_vir ~ 3.52e+11
        # snap 119 (z = 0.099): after major ejection of gas
        # snap 122 (z = 0.060): LMC passes R_vir of the MW
        # snap 127 (z = 0.0): present-day (M_vir ~ 1.85e+11, M_gas ~ 1.26e+10)

        ursa.plot_image('temperature', input_path, output_path, snaps=snaps, scale=[200, 1600])

    elif type_plot == 'frames':
        input_path = ['/Users/dear-prudence/Desktop/smorgasbord/images/halo_11/'
                      '09_18_gas_temperature_halo_11_50x400ckpc_bin0.5ckpc_2_1test.npz',
                      '/Users/dear-prudence/Desktop/smorgasbord/images/halo_11/'
                      '09_18_gas_temperature_halo_11_50x400ckpc_bin0.5ckpc_2_1test.npz']
        output_path = '/Users/ursa/Desktop/smorgasbord/images/halo_11/'

        ursa.plot_frames('temperature', input_path, output_path, scale=[50, 400])

    print('Done!')
