import numpy as np
from astropy.io import fits as f
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# Assuming you have a list of data arrays for your plots
loaded_data = np.load('/Users/ursa/Desktop/09_18_lastGigYear_test.npz')
loaded_data_array = loaded_data['data'].T
x_edges = loaded_data['x_edges']
y_edges = loaded_data['y_edges']
# loaded_time_edges = loaded_data['time']
c_map = 'viridis'
delay = 100
data_arrays_reversed = loaded_data_array[::-1]

fontprops = fm.FontProperties(size=18)
# Set up your figure and axes
fig, ax = plt.subplots(figsize=(16, 8))
background_color = plt.get_cmap(c_map)(0)
ax.set_facecolor(background_color)
plt.title('MW + LMC (Run 09_18); ~0.95 Gyr to present')
plt.xticks(ticks=[])
plt.yticks(ticks=[])
plt.xlabel('This is a gif of the evolution of gas of the LMC-analog in simulation run 09_18 '
           'from $z=0.071$ to $z=0.0$. The scale of the image is 300 kpc across and 150 kpc tall.\n'
           'The horizontal axis is the y-coordinate and the vertical axis is the z-axis. '
           'This is in the frame of the MW-analog.', loc='left')
# -----------
# Add a scale annotation (adjust the coordinates and text as needed)
img = ax.imshow(data_arrays_reversed[0].T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap=c_map, aspect='auto', norm=LogNorm(vmin=10))  # Initialize with the first frame

# Define the update function for each frame
def update(frame):
    img.set_array(data_arrays_reversed[frame])
    return img,


# Create the animation
animation = FuncAnimation(fig, update, frames=len(data_arrays_reversed), interval=delay, blit=True)

# If you want to save the animation as a file (e.g., MP4):
# animation.save('/Users/dear-prudence/Desktop/MW_09_18_lastGigYear_expanded.gif', fps=int(np.ceil(1000 / delay)))

plt.show()
