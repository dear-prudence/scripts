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


def create_histogram(x, y, ranges, num_bins):
    # Create a 2D histogram
    return np.histogram2d(x, y, bins=num_bins, range=ranges)


def manual_range():
    x_lower = 46.70
    y_lower = 49.05
    size = 0.15
    return [[x_lower, x_lower + size], [y_lower, y_lower + size]]


def follow_mw(size, snap, filename):
    # NEED TO CHANGE THIS IF PLOTTING SOMETHING OTHER THAN Y VS X
    row = 127 - int(snap)
    mw = np.loadtxt(filename)
    coordinates_mw = np.array([mw[row, 6], mw[row, 7], mw[row, 8]]) * (10 ** -3)
    # adjust these accordingly; right now its set to put the MW in the lower right corner
    # Why was I previously rounding the bounds?
    lower_x = coordinates_mw[1] - (size[0] / 8)
    lower_y = coordinates_mw[2] - (size[1] / 2)
    return [[lower_x, lower_x + size[0]], [lower_y, lower_y + size[1]]]


def calc_T(u, e_abundance, he_mass_fraction):  # follows procedure outlined in the GIZMO documentation
    gamma = 5 / 3  # needs from __future__ import division when run on a machine with Python 2.x
    k_b = 1.380649 * 10 ** -23
    m_p = 1.67262192 * 10 ** -27
    y_he = he_mass_fraction / (4 * (1 - he_mass_fraction))
    mu = m_p * ((1 + 4 * y_he) / (1 + y_he + e_abundance))
    # internal energy per unit mass U is in (km/s)^2, and so needs the 10^6 to convert to (m/s)^2
    return (mu / k_b) * (gamma - 1) * (u * 10 ** 6)


def smooth_histogram(histogram_data, kernel_size):
    # Create a 2D kernel for the moving average
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size**2)
    # Convolve the histogram data with the kernel
    smoothed_data = convolve(histogram_data, kernel, mode='constant', cval=0.0)
    return smoothed_data


def reduce_graininess(arr, threshold):
    # Create a copy of the array to avoid modifying the original data
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
                         he_mass_fraction=np.array(data['GFM_Metals'][:, 1]))
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
    size = np.array([200, 600, 600]) * 10 ** -3  # array in kpc
    center = np.array([46.94, 48.89, 49.91])
    min_bound = center - (size / 2)  # Set your minimum bounds
    max_bound = center + (size / 2)  # Set your maximum bounds

    # Filter particles based on the bounding box
    filtered_particles = filter_particles(full_data, min_bound, max_bound)
    amended_data = add_temperature(filtered_particles)
    processed_data = reduce_graininess(amended_data, threshold=2*10**7)

    x_data = np.array(processed_data['Coordinates'])[:, 1]
    y_data = np.array(processed_data['Coordinates'])[:, 2]

    # ra = follow_mw(size=[0.30, 0.15], snap=snap,
    #               filename='/store/clues/galaxy/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc'
    #                        '/HESTIA_100Mpc_8192_09_18.127_halo_127000000000003.dat')
    ra = [[min_bound[1], max_bound[1]], [min_bound[2], max_bound[2]]]
    return create_histogram(x_data, y_data, ranges=ra, num_bins=num_bins)


if __name__ == "__main__":
    # -------------------
    n_bins = 300
    snapshot = 121
    kernel_size = 3
    image, x_edges, y_edges = make_snap(snap=snapshot, num_bins=n_bins)
    # smoothed_image = smooth_histogram(image, kernel_size)
    print('Image Size: ' + str(image.shape))

    # -----------------------------------------------------------

    # Save data
    print('Saving Data...')
    np.savez('/z/rschisholm/storage/09_18_snap' + str(snapshot) + '_numDen.npz',
             data=image, x_edges=x_edges, y_edges=y_edges)  # time=time_edges)

    # Indicate that the script has completed its task
    print('Done!')

    # Terminate the script
    sys.exit(0)
