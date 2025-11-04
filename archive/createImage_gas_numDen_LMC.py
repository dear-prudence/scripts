# This script will bin gas at all temps and return column number density over a specified number of snapshots;
# it will return two numpy data files, which can be used to create a two-panel side-by-side movie
import sys
import numpy as np
from hestia import append_particles
from scripts.hestia import filter_particles
from scripts.hestia import time_edges
from scripts.hestia import center_lmc


def make_snap(snap_i, run, bins, size, axes):
    # bins object must be a two element list
    # axes object follows the formal [x_axis, y_axis] (i.e. if plotting z vs y, axes = [1, 2])
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    key_names = ['Coordinates']
    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_'
                 + snap + '/snapshot_' + snap + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_particles = {name: None for name in key_names}
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + snap + '...')
    for file_path in file_paths:
        all_particles = append_particles(file_path, key_names=['Coordinates'], existing_arrays=all_particles)
    l_b, u_b = center_lmc(run=run, lmc_halo_id='10', snap=snap, size=size) * 10 ** -3  # convert from kpc to Mpc
    # Filter particles based on the bounding box
    filtered_particles = filter_particles(all_particles, l_b, u_b)
    return np.histogram2d(filtered_particles['Coordinates'][:, axes[0]],
                          filtered_particles['Coordinates'][:, axes[1]], bins=bins)


if __name__ == "__main__":
    # -------------------
    simulation_run = '09_18'
    # for this script, box needs to be a cube
    spatial_size = [400, 400, 400]  # in kpc
    axes = [0, 1]
    n_bins = [800, 800]
    snaps = [87, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
    snap_stepSize = 1
    # -------------------
    # module that writes the panel
    for i in range(snaps[1], snaps[0], -1 * snap_stepSize):
        current_snapshot, x_edges, y_edges = make_snap(i, simulation_run,
                                                       bins=n_bins, size=spatial_size, axes=axes)
        if i == snaps[1]:
            all_snapshots = current_snapshot
        else:
            # Append the new snapshot
            all_snapshots = np.dstack((all_snapshots, current_snapshot))
        print('Snapshot ' + str(i) + ': check')

    time_edges = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1 * snap_stepSize))
    print(time_edges)
    # Save data
    np.savez('/z/rschisholm/storage/images/' + str(simulation_run) + '_gas_numDen_LMC_'
             + str(spatial_size[0]) + 'x' + str(spatial_size[1]) + 'x' + str(spatial_size[2])
             + 'kpc_bin' + str(round(spatial_size[axes[0]] / float(n_bins[0]), 2)) + 'kpc_'
             + str(axes[0]) + '_' + str(axes[1]) + '.npz',
             data=all_snapshots, x_edges=x_edges, y_edges=y_edges, time=time_edges)

    # Indicate that the script has completed its task
    print('Done!')

    # Terminate the script
    sys.exit(0)
