import h5py
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def create_2d_histogram(x, y, bins=10):
    """
    Create a 2D histogram from x and y coordinates.

    Parameters:
    - x: Array-like, x coordinates.
    - y: Array-like, y coordinates.
    - bins: Number of bins for the histogram.

    Returns:
    - histogram: 2D histogram array.
    - x_edges: Bin edges along the x-axis.
    - y_edges: Bin edges along the y-axis.
    """
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    return histogram, x_edges, y_edges

def plot_2d_histogram(histogram, x_edges, y_edges):
    """
    Plot a 2D histogram.

    Parameters:
    - histogram: 2D histogram array.
    - x_edges: Bin edges along the x-axis.
    - y_edges: Bin edges along the y-axis.
    """
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the size by changing the figsize parameter
    im = ax.imshow(histogram.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap='viridis', aspect='auto', norm=LogNorm())
    plt.colorbar(im, label='Number Density')
    plt.xlabel('X (Mpc)')
    plt.ylabel('Y (Mpc)')
    plt.title('Local Group (Run 09_18)')
    plt.savefig('/Volumes/enceladus/LG_simulations/snapshot_127/LG_simData.png', dpi=240)
    plt.show()


'''def append_coordinates_and_masses(file_path, existing_array=None):
    with h5py.File(file_path, 'r') as f:
        dset = f['PartType1']
        coords = np.array(dset['Coordinates'])
        masses = np.array(dset['Masses'])
        combined_array = np.transpose(np.insert(coords, 3, masses, axis=1))

    if existing_array is not None:
        # Pad the existing array with zeros if needed
        pad_length = max(0, combined_array.shape[0] - existing_array.shape[0])
        existing_array = np.vstack([existing_array, np.zeros((pad_length, existing_array.shape[1]))])

        # Pad the combined array with zeros if needed
        pad_length = max(0, existing_array.shape[0] - combined_array.shape[0])
        combined_array = np.vstack([combined_array, np.zeros((pad_length, combined_array.shape[1]))])

    return np.vstack([existing_array, combined_array])

# Specify the common structure of file paths
base_path = '/Volumes/enceladus/LG_simulations/snapshot_127/snapshot_127.'
file_extension = '.hdf5'

# Generate file paths using a loop
file_paths = [base_path + str(x) + file_extension for x in range(2)]
# Initialize the resulting array
resulting_array = None

# Loop through the file paths and append coordinates and masses
for file_path in file_paths:
    resulting_array = append_coordinates_and_masses(file_path, existing_array=resulting_array)

# Now, resulting_array contains the combined coordinates and masses'''


f = h5py.File('/Volumes/enceladus/LG_simulations/snapshot_127/snapshot_127.0.hdf5','r')

print(list(f.keys()))
# ['Config', 'Header', 'Parameters',
# 'PartType0', 'PartType1', 'PartType2', 'PartType3', 'PartType4', 'PartType5', 'PartType6']
dset = f['PartType1']
coords = np.array(dset['Coordinates'])
co = np.transpose(np.insert(coords, 3, np.array(dset['Masses']), axis=1))


x_data = co[0]
y_data = co[1]

histogram, x_edges, y_edges = create_2d_histogram(x_data, y_data, bins=500)
plot_2d_histogram(histogram, x_edges, y_edges)



