import numpy as np
import h5py
import sys


def append_coordinates(file_path, existing_array=None):
    with h5py.File(file_path, 'r') as f:
        dset = f['PartType1']
        coords = np.array(dset['Coordinates'])

    return np.append(existing_array, coords, axis=0) if existing_array is not None else coords


def save_data_to_txt(file_path, coords):
    np.savetxt(file_path, coords, fmt='%1.4f', delimiter='\t', header='x\ty\tz', comments='')


def create_2d_histogram(x, y, ranges, bins=12):
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=ranges)
    return histogram, x_edges, y_edges


def manual_range():
    x_lower = 46.65
    y_lower = 48.95
    size = 0.25
    return [[x_lower, x_lower + size], [y_lower, y_lower + size]]


def follow_mw(length, snap, filename):
    # NEED TO CHANGE THIS IF PLOTTING SOMETHING OTHER THAN Y VS X
    row = 127 - int(snap)
    mw = np.loadtxt(filename)
    coordinates_mw = np.array([mw[row, 6], mw[row, 7], mw[row, 8]]) * (10 ** -3)
    lower_x = round(coordinates_mw[0] - (length / 2), 2)
    lower_y = round(coordinates_mw[1] - (length / 2), 2)
    return [[lower_x, lower_x + length], [lower_y, lower_y + length]]


def make_snap(snap):
    base_path = ('/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/output/snapdir'
                 + snap + '/snapshot_' + snap + '.')

if __name__ == "__main__":
    # Specify the common structure of file paths
    sim_name = raw_input('Simulation Run? ')
    # snap = raw_input('Snapshot? ')
    base_path = ('/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/' + sim_name + '/output_2x2.5Mpc/snapdir_'
                 + snap + '/snapshot_' + snap + '.')
    file_extension = '.hdf5'
    # output_file_path = '/store/erebos/rschisholm/' + sim_name + '/' + sim_name +'_' + snap + '_coords.dat'

    # -------------------------------------------
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    resulting_array = None

    # Loop through the file paths and append coordinates
    for file_path in file_paths:
        resulting_array = append_coordinates(file_path, existing_array=resulting_array)
        print(file_path)

    x_data = resulting_array[:, 0]
    y_data = resulting_array[:, 1]

    # ra = manual_range()
    ra = follow_mw(length=0.25, snap=snapshot,
                   filename='/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/AHF_output'
                            '/HESTIA_100Mpc_8192_09_18_lgy.127_halo_307000000000003.dat')
    histogram, x_edges, y_edges = create_2d_histogram(x_data, y_data, ra, bins=500)

    # -----------------------------

    # Save data
    np.savez('/z/rschisholm/temp_storage/' + str(sim_name) + '_snap' + str(snap) + '.npz',
             data=histogram, x_edges=x_edges, y_edges=y_edges)

    # Indicate that the script has completed its task
    print('Done!')

    # Terminate the script
    sys.exit(0)