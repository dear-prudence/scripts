import numpy as np
import h5py
import sys


def append_coordinates(file_path, existing_array=None):
    with h5py.File(file_path, 'r') as f:
        dset = f['PartType0']
        coords = np.array(dset['Coordinates'])

    return np.append(existing_array, coords, axis=0) if existing_array is not None else coords


def save_data_to_txt(file_path, coords):
    np.savetxt(file_path, coords, fmt='%1.4f', delimiter='\t', header='x\ty\tz', comments='')


def create_2d_histogram(x, y, ranges, bins):
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=ranges)
    return histogram, x_edges, y_edges


def manual_range():
    x_lower = 46.65
    y_lower = 48.95
    size = 0.25
    return [[x_lower, x_lower + size], [y_lower, y_lower + size]]


def follow_mw(size, snap, filename):
    # NEED TO CHANGE THIS IF PLOTTING SOMETHING OTHER THAN Y VS X
    row = 307 - int(snap)
    mw = np.loadtxt(filename)
    coordinates_mw = np.array([mw[row, 6], mw[row, 7], mw[row, 8]]) * (10 ** -3)
    # adjust these accordingly; right now its set to put the MW in the lower right corner
    # Why was I previously rounding the bounds?
    lower_x = coordinates_mw[1] - (size[0] / 8)
    lower_y = coordinates_mw[2] - (size[1] / 2)
    return [[lower_x, lower_x + size[0]], [lower_y, lower_y + size[1]]]


def make_snap(snap, num_bins):
    base_path = ('/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/output/snapdir_'
                 + str(snap) + '/snapshot_' + str(snap) + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    resulting_array = None

    # Loop through the file paths and append coordinates
    for file_path in file_paths:
        resulting_array = append_coordinates(file_path, existing_array=resulting_array)
    y_data = resulting_array[:, 1]
    z_data = resulting_array[:, 2]

    ra = follow_mw(size=[0.30, 0.15], snap=snap,
                   filename='/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/AHF_output'
                            '/HESTIA_100Mpc_8192_09_18_lgy.127_halo_307000000000003.dat')
    return create_2d_histogram(y_data, z_data, ra, bins=num_bins)


if __name__ == "__main__":
    # -------------------
    n_bins = [800, 400]
    n_snaps = 6
    # -------------------
    # all_snapshots = np.empty((n_bins, n_bins))  # Assuming 2D xy data, adjust the second dimension accordingly
    for i in range(307, 307 - n_snaps, -2):
        current_snapshot, x_edges, y_edges = make_snap(i, n_bins)
        if i == 307:
            all_snapshots = current_snapshot
        else:
            # Append the new snapshot
            all_snapshots = np.dstack((all_snapshots, current_snapshot))
        print('Snapshot ' + str(i) + ': check')

    time_edges = np.arange(307, 307 - n_snaps, -2)
    # Save data
    np.savez('/z/rschisholm/temp_storage/09_18_lastGigYear_test.npz',
             data=all_snapshots, x_edges=x_edges, y_edges=y_edges, time=time_edges)

    # Indicate that the script has completed its task
    print('Done!')

    # Terminate the script
    sys.exit(0)