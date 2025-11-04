import numpy as np
from astropy.io import fits as f
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_2d_histogram(histogram, x_edges, y_edges, snap):
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the size by changing the figsize parameter
    # Create a background with a specific color (color at the minimum of the Viridis colormap)
    c_map ='viridis'
    background_color = plt.get_cmap(c_map)(0)
    ax.set_facecolor(background_color)
    im = ax.imshow(histogram.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap=c_map, aspect='auto', norm=LogNorm(vmin=10))
    plt.colorbar(im, label='Number Density')
    plt.xlabel('X (Mpc)')
    plt.ylabel('Y (Mpc)')
    plt.title('MW (Run 09_18), Snapshot: ' + str(snap))
    # draw_circles(plt.gca())
    # plt.savefig('/Users/dear-prudence/Desktop/09_18_lastGigYear/MW_lastGigYear_09_18_snap'
    #             + str(snap) + '.png', dpi=240)
    plt.close()


# Load data
loaded_data = np.load('/Users/ursa/Desktop/09_18_lastGigYear/09_18_lastGigYear.npz')
n_snaps = 60
for i in range(307, 307 - n_snaps, -2):
    # Access individual arrays
    loaded_data_array = loaded_data['data'][:,:,int((307 - i) / 2)]
    loaded_x_edges = loaded_data['x_edges']
    loaded_y_edges = loaded_data['y_edges']
    loaded_time_edges = loaded_data['time']

    plot_2d_histogram(loaded_data_array, loaded_x_edges, loaded_y_edges, snap=i)
