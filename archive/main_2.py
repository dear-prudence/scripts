import numpy as np
from astropy.io import fits as f
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# Assuming you have a list of data arrays for your plots
loaded_data = np.load('/Users/ursa/Desktop/09_18_gas_numDen_LMC_400x400x400kpc_0_1.npz')
loaded_data_array = loaded_data['data'][:, :, 0].T
print(loaded_data_array.shape)
x_edges = loaded_data['x_edges']
y_edges = loaded_data['y_edges']
# loaded_time_edges = loaded_data['time']
c_map = 'viridis'


# Set up your figure and axes
fig, ax = plt.subplots(figsize=(10, 8))
background_color = plt.get_cmap(c_map)(0)
ax.set_facecolor(background_color)

# -----------
# Add a scale annotation (adjust the coordinates and text as needed)
img = ax.imshow(loaded_data_array, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap=c_map, aspect='auto', norm=LogNorm(vmin=10))  # Initialize with the first frame
plt.colorbar(img, label='Number Density')
# plt.xticks(ticks=[46.7, 46.73, 46.76, 46.79, 46.82, 46.85], labels=['-30', '0', '30', '60', '90', '120'])
# plt.yticks(ticks=[49.05, 49.08, 49.11, 49.14, 49.17, 49.20], labels=['-60', '-30', '0', '30', '60', '90'])
# plt.xlabel('X (kpc)')
# plt.ylabel('Y (kpc)')
# plt.plot([49.03, 49.13], [49.65, 49.65], c='k')
# plt.text(49.055, 49.635, s='100 kpc', c='k')

# If you want to save the animation as a file (e.g., MP4):
# animation.save('/Users/dear-prudence/Desktop/MW_09_18_lastGigYear_expanded.gif', fps=int(np.ceil(1000 / delay)))
# plt.savefig('/Users/dear-prudence/Desktop/09_18_snap121_gas_numDen.png', dpi=240)
plt.show()
