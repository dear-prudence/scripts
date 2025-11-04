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


if __name__ == "__main__":
    # Specify the common structure of file paths
    sim_name = raw_input('Simulation Run? ')
    snap = raw_input('Snapshot? ')
    base_path = ('/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/' + sim_name + '/output_2x2.5Mpc/snapdir_'
                 + snap + '/snapshot_' + snap + '.')
    file_extension = '.hdf5'
    output_file_path = '/store/erebos/rschisholm/' + sim_name + '/' + sim_name +'_' + snap + '_coords.dat'

    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    resulting_array = None

    # Loop through the file paths and append coordinates
    for file_path in file_paths:
        resulting_array = append_coordinates(file_path, existing_array=resulting_array)
        print(file_path)

    save_data_to_txt(output_file_path, resulting_array)

    # Indicate that the script has completed its task
    print('\nData saved to ' + str(output_file_path))

    # Terminate the script
    sys.exit(0)