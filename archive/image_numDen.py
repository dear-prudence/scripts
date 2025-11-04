"""
This script will construct a histogram of the mass density for a particular snapshot or array of snapshots
for a particular halo. It takes as input data, the raw particle data from the simulation run, or the isolated
halo particle data (see hestia.particles.filter_and_save_hdf5()), currently only allows for gas particles,
but should be straightforward to add functionality for other particle types. It will save data as an .npz file,
an image data file that is essentially just a two-dim numpy array (plus some additional information) that can
be plotted as pixels for an image using matplotlib.
"""
import sys
import numpy as np
from scripts.hestia import append_particles
from scripts.hestia import filter_particles
from scripts.hestia import cosmo_transform
from scripts.hestia import time_edges
from scripts.hestia import center_lmc


def make_snap(snap_i, z, run, part_type, bins, size, axs, follow_gal=None, iteration=True):
    # bins object must be a two element list
    # axes object follows the formal [x_axis, y_axis] (i.e. if plotting z vs y, axes = [1, 2])
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    if follow_gal == 'lmc':
        l_b, u_b = center_lmc(run=run, lmc_halo_id='10', snap=snap, size=size) * 1e-3  # convert from kpc to Mpc
    else:
        print('Error: Invalid reference galaxy! routine is WIP.')
        exit(1)

    key_names = ['Coordinates']
    if iteration is False:
        file_path = '/z/rschisholm/storage/snapshots_lmc/snapshot_' + snap + '/snapshot_' + snap + '.lmc.hdf5'
        print('Processing Snapshot ' + snap + '...')
        lmc_particles_h = append_particles(part_type, file_path, key_names=key_names)
        bounds = cosmo_transform({'Coordinates':
                                      np.array([[l_b[axes[0]], u_b[axes[0]]], [l_b[axes[1]], u_b[axes[1]]]])},
                                 'ckpc/h', 'ckpc', z)['Coordinates']
        lmc_particles = cosmo_transform(lmc_particles_h, 'ckpc/h', 'ckpc', z)
        return np.histogram2d(lmc_particles['Coordinates'][:, axs[0]], lmc_particles['Coordinates'][:, axs[1]],
                              bins=bins, range=bounds)
    else:
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run) + '/output_2x2.5Mpc/snapdir_'
                     + snap + '/snapshot_' + snap + '.')
        file_extension = '.hdf5'
        # Generate file paths using a loop
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        # Initialize the resulting array
        all_particles = {name: None for name in key_names}
        # Loop through the file paths and append coordinates
        print('Processing Snapshot ' + snap + '...')
        for file_path in file_paths:
            all_particles = append_particles('PartType0', file_path, key_names=key_names,
                                             existing_arrays=all_particles)
        # Filter particles based on the bounding box
        filtered_particles = cosmo_transform(filter_particles(all_particles, l_b, u_b),
                                             'ckpc/h', 'ckpc', z)
        return np.histogram2d(filtered_particles['Coordinates'][:, axs[0]],
                              filtered_particles['Coordinates'][:, axs[1]],
                              bins=bins)


# ------------------------------------
mode = 'dear-prudence'
# ------------------------------------

if mode == 'geras':
    if __name__ == "__main__":
        # ------------------------------
        simulation_run = '09_18'
        dims = [50, 400]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
        axes = [2, 1]
        n_bins = [400, 400]
        snaps = [67, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
        snap_stepSize = 1
        halo = 'lmc'
        particle = 'PartType4'
        # -----------------------------
        spatial_size = [dims[0] for i in range(3)]
        for ax in axes:
            spatial_size[ax] = dims[1]
        S = {'Coordinates': np.array(spatial_size)}
        if isinstance(snaps, int) or len(snaps) == 1:
            # single
            time_edges = time_edges(sim=simulation_run, snaps=np.array([snaps]))
            S_co = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=time_edges[0][0])
            current_snapshot, x_edges, y_edges = make_snap(snaps, float(time_edges[0][0]), simulation_run,
                                                           part_type=particle, bins=n_bins,
                                                           size=np.array(S_co['Coordinates']),
                                                           axs=axes, follow_gal='lmc', iteration=False)

            # Save data
            np.savez('/z/rschisholm/storage/snapshots_lmc/snapshot_' + str(snaps) + '/' + str(simulation_run)
                     + '_' + ('gas' if particle == 'PartType0' else 'stars')
                     + '_numDen_lmc_snap' + str(snaps) + '_'
                     + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc_'
                     + 'bin' + str(round(spatial_size[axes[0]] / float(n_bins[0]), 2)) + 'ckpc_'
                     + str(axes[0]) + '_' + str(axes[1]) + '.npz',
                     data=current_snapshot, x_edges=x_edges, y_edges=y_edges, time=time_edges)

        elif isinstance(snaps, list) or len(snaps) > 1:
            # This is the module that writes the snapshot images for an array of snapshots
            # (images are stacked on top of one another to form an image data tensor)
            time_edges = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1 * snap_stepSize))
            for i in range(snaps[1], snaps[0], -1 * snap_stepSize):
                S_co = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=time_edges[127 - i][0])
                print('time: $z = $' + str(time_edges[127 - i][0]))
                current_snapshot, x_edges, y_edges = make_snap(i, float(time_edges[127 - i][0]), simulation_run,
                                                               part_type=particle, bins=n_bins,
                                                               size=np.array(S_co['Coordinates']),
                                                               axs=axes, follow_gal=halo)
                if i == snaps[1]:
                    all_snapshots = current_snapshot
                else:
                    # Append the new snapshot
                    all_snapshots = np.dstack((all_snapshots, current_snapshot))
                print('Snapshot ' + str(i) + ': check')

            # Save data
            np.savez('/z/rschisholm/storage/images/' + simulation_run + '_'
                     + ('gas' if particle == 'PartType0' else 'stars') + '_numDen_' + halo + '_'
                     + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc_'
                     + 'bin' + str(round(spatial_size[axes[0]] / float(n_bins[0]), 2)) + 'ckpc_'
                     + str(axes[0]) + '_' + str(axes[1]) + '.npz',
                     data=all_snapshots, x_edges=x_edges, y_edges=y_edges, time=time_edges)

        # Indicate that the script has completed its task
        print('Done!')
        # Terminate the script
        sys.exit(0)

    # ------------------------------------
elif mode == 'dear-prudence':
    from scripts.archive import ursa

    # ------------------------------------
    type_plot = 'frames'
    # ------------------------------------

    if type_plot == 'image':
        input_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/starsnumDen_lmc_200x1600ckpc/'
                      '09_18_gas_massDen_LMC_100x800x800ckpc_bin1.0ckpc_2_1.npz')
        output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/stars/numDen_LMC_100x800ckpc/'
                       '09_18_gas_massDen_lmc_100x800ckpc_faceOn.png')
        snaps = [95, 108, 114, 119, 122, 127]
        # snap 95 (z = 0.506): "primordial LMC"
        # snap 108 (z = 0.258): fully-formed, near-peak mass LMC
        # snap 114 (z = 0.165): LMC peak mass, M_vir ~ 3.52e+11
        # snap 119 (z = 0.099): after major ejection of gas
        # snap 122 (z = 0.060): LMC passes R_vir of the MW
        # snap 127 (z = 0.0): present-day (M_vir ~ 1.85e+11, M_gas ~ 1.26e+10)

        ursa.plot_image('massDen', input_path, output_path, snaps=snaps, scale=[100, 800])

    elif type_plot == 'frames':
        input_paths = ['/Users/dear-prudence/Desktop/smorgasbord/images/stars/'
                       '09_18_stars_numDen_lmc_50x400ckpc_bin1.0ckpc_0_1.npz',
                       '/Users/dear-prudence/Desktop/smorgasbord/images/stars/'
                       '09_18_stars_numDen_lmc_50x400ckpc_bin1.0ckpc_2_1.npz']
        output_path = '/Users/ursa/Desktop/smorgasbord/images/stars/snaps/'

        ursa.plot_frames('numDen', input_paths, output_path, scale=[50, 400])

    elif type_plot == 'combo_frames':
        input_paths = ['/Users/dear-prudence/Desktop/smorgasbord/images/massDen_lmc_100x800ckpc/'
                       '09_18_gas_massDen_LMC_100x800x800ckpc_bin1.0ckpc_2_1.npz',
                       '/Users/dear-prudence/Desktop/smorgasbord/images/temperature_LMC_100x800ckpc/'
                       '09_18_gas_temperature_LMC_100x800x800ckpc_bin2.0ckpc_2_1.npz']
        output_path = '/Users/ursa/Desktop/smorgasbord/images/combination_massDen_temperature_lmc_100x800ckpc/'

        ursa.plot_combo_frames(['massDen', 'temperature'], input_paths, output_path, scale=[100, 800])

    print('Done!')