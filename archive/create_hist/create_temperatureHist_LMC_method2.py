from __future__ import division  # Add this line at the beginning of y
import numpy as np
import h5py
import sys
from scipy.ndimage import convolve


def append_datasets(filename, dataset_names, existing_arrays=None):
    if existing_arrays is None:
        existing_arrays = {name: None for name in dataset_names}

    with h5py.File(filename, 'r') as file:
        f = file['PartType0']
        for name in dataset_names:
            if name in f:
                data = np.array(f[name])
                existing_arrays[name] = np.append(existing_arrays[name], data, axis=0) \
                    if existing_arrays[name] is not None else data
            else:
                print('Error: ' + str(name) + ' is not a key in hdf file!')

    print('Appended ' + str(filename[-10:-6]) + '.')
    return existing_arrays


# Function to filter particles based on coordinates
def filter_particles(data, min_b, max_b):
    for key in data.keys():
        data[key] = np.array(data[key])  # Convert to numpy array for easier indexing
    print('Converted columns to numpy arrays.')
    indices_to_keep = np.where(
        (min_b[0] <= data['Coordinates'][:, 0]) & (data['Coordinates'][:, 0] <= max_b[0]) &
        (min_b[1] <= data['Coordinates'][:, 1]) & (data['Coordinates'][:, 1] <= max_b[1]) &
        (min_b[2] <= data['Coordinates'][:, 2]) & (data['Coordinates'][:, 2] <= max_b[2])
    )[0]
    print('Done with numpy filtering.')
    for key in data.keys():
        data[key] = data[key][indices_to_keep]

    return data


def create_weighted_histogram(x, y, temperature, ranges, num_bins):
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=num_bins, range=ranges)

    # Compute the sum of temperatures in each bin
    sum_temps, _, _ = np.histogram2d(x, y, bins=(x_e, y_e), weights=temperature, range=ranges)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_temps = np.divide(sum_temps, hist, where=(hist != 0))

    return avg_temps, x_e, y_e


def center_lmc(snap, filename):
    # NEED TO CHANGE THIS IF PLOTTING SOMETHING OTHER THAN Y VS X
    row = 127 - int(snap)
    mw = np.loadtxt(filename)
    return np.array([mw[row, 6], mw[row, 7], mw[row, 8]]) * (10 ** -3)


def calc_T(u, e_abundance, x_h):  # follows procedure outlined in the GIZMO documentation
    # it might be a good idea to disregard unphysical temperatures (>10^7 K); IGM should not be much hotter than this
    gamma = 5 / 3  # needs from __future__ import division when run on a machine with Python 2.x
    k_b = 1.3807 * 10 ** -16
    m_p = 1.67262 * 10 ** -24
    # values of constants in CGS taken from https://www.prl.res.in/~snaik/const.html
    unit_ratio = 10 ** 10
    mu = 4 * m_p / (1 + 3 * x_h + 4 * x_h * e_abundance)
    # internal energy per unit mass U is in (km/s)^2, and so needs the 10^6 to convert to (m/s)^2
    return (gamma - 1) * (u / k_b) * unit_ratio * mu


def smooth_histogram(histogram_data, kernel_size):
    # Create a 2D kernel for the moving average
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size**2)
    # Convolve the histogram data with the kernel
    smoothed_data = convolve(histogram_data, kernel, mode='constant', cval=0.0)
    return smoothed_data


def reduce_graininess(arr, threshold):
    new_arr = np.copy(arr)
    # Get the shape of the array
    rows, cols = arr.shape
    # Iterate over each element in the array
    for i in range(rows):
        for j in range(cols):
            # Check if the current value is an outlier
            if arr[i, j] > threshold:
                # Define the indices of neighboring elements
                neighbors = []
                if i > 0:
                    neighbors.append(arr[i - 1, j])  # Above
                if i < rows - 1:
                    neighbors.append(arr[i + 1, j])  # Below
                if j > 0:
                    neighbors.append(arr[i, j - 1])  # Left
                if j < cols - 1:
                    neighbors.append(arr[i, j + 1])  # Right

                # Replace the outlier with the average of neighboring values
                new_arr[i, j] = np.mean(neighbors)
    return new_arr


def add_temperature(data):
    temp_column = calc_T(u=np.array(data['InternalEnergy']), e_abundance=np.array(data['ElectronAbundance']),
                         x_h=np.array(data['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    data['Temperature'] = temp_column
    return data


def make_snap(snap, num_bins):
    dataset_names = ['Coordinates', 'InternalEnergy', 'ElectronAbundance', 'GFM_Metals']
    base_path = ('/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_'
                 + str(snap) + '/snapshot_' + str(snap) + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting dictionary
    full_data = {name: None for name in dataset_names}
    for file_path in file_paths:
        full_data = append_datasets(file_path, dataset_names, existing_arrays=full_data)

    # Define the bounding box
    size = np.array([150, 600, 600]) * 10 ** -3  # array in kpc
    center = center_lmc(snap, filename='/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc'
                                       '/HESTIA_100Mpc_8192_09_18.127_halo_127000000000010.dat')
    min_bound = center - (size / 2)  # Set your minimum bounds
    max_bound = center + (size / 2)  # Set your maximum bounds

    # Filter particles based on the bounding box
    filtered_particles = filter_particles(full_data, min_bound, max_bound)
    processed_data = add_temperature(filtered_particles)

    x_data = np.array(processed_data['Coordinates'])[:, 1]
    y_data = np.array(processed_data['Coordinates'])[:, 2]

    ra = [[min_bound[1], max_bound[1]], [min_bound[2], max_bound[2]]]
    hist, x_, y_ = create_weighted_histogram(x_data, y_data, processed_data['Temperature'], ranges=ra, num_bins=num_bins)
    processed_data = reduce_graininess(hist, threshold=2 * 10 ** 7)
    return processed_data, x_, y_


if __name__ == "__main__":
    # -------------------
    n_bins = 200
    snapshot = 111
    kernel_size = 3
    image, x_edges, y_edges = make_snap(snap=snapshot, num_bins=n_bins)
    # smoothed_image = smooth_histogram(image, kernel_size)
    print('Image Size: ' + str(image.shape))

    # Save data
    print('Saving Data...')
    np.savez('/z/rschisholm/storage/09_18_snap' + str(snapshot) + '_LMC_temperature2_100x600x600kpc.npz',
             data=image, x_edges=x_edges, y_edges=y_edges)  # time=time_edges)

    # Indicate that the script has completed its task
    print('Done!')

    # Terminate the script
    sys.exit(0)
