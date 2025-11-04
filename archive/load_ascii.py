import h5py
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def create_2d_histogram(x, y, ranges, bins=12):
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
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=ranges)
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
    # Create a background with a specific color (color at the minimum of the Viridis colormap)
    c_map ='viridis'
    background_color = plt.get_cmap(c_map)(0)
    ax.set_facecolor(background_color)
    # Apply PowerNorm with an exponent of 0.5 (square root)
    im = ax.imshow(histogram.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap=c_map, aspect='auto', norm=LogNorm(vmin=10))
    plt.colorbar(im, label='Number Density')
    plt.xlabel('Y (Mpc)')
    plt.ylabel('Z (Mpc)')
    plt.title('LG (Run 09_18)')
    draw_circles(plt.gca())
    plt.savefig('/Users/dear-prudence/Desktop/LG_simData_yz.png', dpi=240)
    plt.show()


def draw_circles(ax):
    m31_position = (48802.6616, 50002.7325)
    mw_position = (49055.2933, 49881.1805)
    radius = 0.1  # Adjust the radius as needed

    # Create a circle
    circle_m31 = plt.Circle(m31_position, radius,
                            color='none', ec='black', linestyle='dashed', linewidth=1, label='M31')
    circle_mw = plt.Circle(mw_position, radius,
                           color='none', ec='black', linestyle='dashed', linewidth=1, label='MW')

    # Add the circle to the plot
    ax.add_patch(circle_m31)
    ax.add_patch(circle_mw)
    # Add text annotation next to the circle
    ax.annotate('M31', xy=(m31_position[0] + radius, m31_position[1] + radius),
                color='black', fontsize=10, va='center')
    ax.annotate('MW', xy=(mw_position[0] + radius, mw_position[1] + radius),
                color='black', fontsize=10, va='center')


def append_coordinates(file_path, existing_array=None):
    with h5py.File(file_path, 'r') as f:
        dset = f['PartType1']
        coords = np.array(dset['Coordinates'])

    return np.append(existing_array, coords, axis=0) if existing_array is not None else coords


# Specify the common structure of file paths
base_path = '/Volumes/enceladus/LG_simulations/snapshot_127/snapshot_127.'
file_extension = '.hdf5'

# Generate file paths using a loop
file_paths = [base_path + str(x) + file_extension for x in range(8)]
print(file_paths)
# Initialize the resulting array
resulting_array = None

# Loop through the file paths and append coordinates
for file_path in file_paths:
    resulting_array = append_coordinates(file_path, existing_array=resulting_array)
print(resulting_array.shape)

# Now, resulting_array contains the combined coordinates and masses
x_data = resulting_array[:, 1]
y_data = resulting_array[:, 2]

ra = [[48.5, 49.5], [49.5, 50.5]]
histogram, x_edges, y_edges = create_2d_histogram(x_data, y_data, ra, bins=500)
plot_2d_histogram(histogram, x_edges, y_edges)





