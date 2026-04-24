import sys
import numpy as np
from archive.hestia import append_particles
from archive.hestia import filter_particles
from archive.hestia import cosmo_transform
from archive.hestia import time_edges
from archive.hestia import center_lmc


def create_weighted_histogram(x, y, velocities, bins):
    threshold = 1
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=bins)

    # Compute the sum of temperatures in each bin
    sum_vels, _, _ = np.histogram2d(x, y, bins=(x_e, y_e), weights=velocities)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_vel = np.divide(sum_vels, hist, where=(hist != 0))
    avg_vel[hist < threshold] = 0
    return avg_vel, x_e, y_e


def levi_cevita(coord_indices):
    # Define the full set of coordinate indices
    all_indices = {0, 1, 2}  # {x, y, z}
    # Convert the input list/tuple to a set
    given_indices = set(coord_indices)
    # Find the missing index by subtracting the given indices from all indices
    missing_index = list(all_indices - given_indices)[0]
    return missing_index


def make_snap(snap_i, z, run, bins, size, axs, follow_gal=None):
    # bins object must be a two element list
    # axes object follows the formal [x_axis, y_axis] (i.e. if plotting z vs y, axes = [1, 2])
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)
    key_names = ['Coordinates', 'Velocities']
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
    return create_weighted_histogram(filtered_proper_particles['Coordinates'][:, axs[0]],
                                     filtered_proper_particles['Coordinates'][:, axs[1]],
                                     filtered_proper_particles['Velocities'][:, levi_cevita((axs[0], axs[1]))],
                                     bins=bins)


if __name__ == "__main__":
    # -------------------
    simulation_run = '09_18'
    # for this script, box needs to be a cube
    spatial_size = [50, 400, 400]  # in c-kpc
    axes = [1, 2]
    n_bins = [800, 800]
    snaps = [67, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
    snap_stepSize = 1
    # -------------------
    S = {'Coordinates': np.array(spatial_size)}
    # module that writes the panel
    time_edges = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1 * snap_stepSize))

    # module that writes the panel
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

    print(time_edges)
    # Save data
    np.savez('/z/rschisholm/storage/images/' + str(simulation_run) + '_gas_vlsr_LMC_'
             + str(spatial_size[0]) + 'x' + str(spatial_size[1]) + 'x' + str(spatial_size[2])
             + 'kpc_bin' + str(round(spatial_size[axes[0]] / float(n_bins[0]), 2)) + 'kpc_'
             + str(axes[0]) + '_' + str(axes[1]) + '.npz',
             data=all_snapshots, x_edges=x_edges, y_edges=y_edges, time=time_edges)

    # Indicate that the script has completed its task
    print('Done!')

    # Terminate the script
    sys.exit(0)
