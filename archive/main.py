import numpy as np
from astropy.io import fits as f
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# Assuming you have a list of data arrays for your plots
filepath = '/Users/ursa/Desktop/09_18_snap111_LMC_temperature2_100x600x600kpc.npz'
loaded_data = np.load(filepath)
loaded_data_array = loaded_data['data'].T
x_edges = loaded_data['x_edges']
y_edges = loaded_data['y_edges']
# loaded_time_edges = loaded_data['time']
c_map = 'coolwarm'

# Set up your figure and axes
fig, ax = plt.subplots(figsize=(10, 8))
background_color = plt.get_cmap(c_map)(250)
# ax.set_facecolor(background_color)

# -----------
# Add a scale annotation (adjust the coordinates and text as needed)
img = ax.imshow(loaded_data_array, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap=c_map, aspect='auto', norm=LogNorm(vmin=5 * 10 ** 4, vmax=2 * 10 ** 6))
plt.colorbar(img, label='Temperature (K)')
# plt.xticks(ticks=[48.6, 48.7, 48.8, 48.9, 49.0, 49.1], labels=['-300', '-200', '-100', '0', '100', '200'])
# plt.yticks(ticks=[49.7, 49.8, 49.9, 50.0, 50.1, 50.2], labels=['-200', '-100', '0', '100', '200', '300'])
plt.xticks([])
plt.yticks([])
plt.title('Temperature Plot (LMC 09_18); $z = 0.213$, $t \sim -2.6$ Gyr')
plt.plot([48.58, 48.68], [49.72, 49.72], c='k')
plt.text(48.605, 49.705, s='100 kpc', c='k')
# relative velocity to MW was calculated by v = v_LMC - v_MW, using velocity data extracted from
# HESTIA_100Mpc_8192_09_18.127_halo_127000000000003.dat
# v_y = 384.81 - 434.15 = -49.34
# v_z = -49.14 - -133.31 = + 84.17
plt.arrow(48.417, 49.967, dx=-0.04934 / 3, dy=0.08417 / 3, alpha=0.5)
# dividing relative velocity by a factor for the sake of legibility
plt.xlabel('This is a temperature plot (method 2) of the LMC-analog from run 09_18 ($z = 0.213$, $t \sim -2.6$ Gyr). '
           'The box is 600 x 600 kpc (z vs y), \nwhere the sample was binned 100 kpc in the x-coordinate. '
           'Each pixel is colored according to average column temperature; \neach column is 100 x 3 x 3 kpc.'
           'The black arrow indicates the direction of velocity of the LMC relative to the MW '
           '(i.e. adjusting \nfor motion of the LG and Hubble expansion.'
           'At this time, the distance between the LMC and MW is $d_{sep} \simeq 390$ kpc,'
           'and $M_{LMC} \simeq 3.46e+11$.', loc='left', fontsize='small')

# If you want to save the animation as a file (e.g., MP4):
# animation.save('/Users/dear-prudence/Desktop/MW_09_18_lastGigYear_expanded.gif', fps=int(np.ceil(1000 / delay)))
plt.savefig('/Users/dear-prudence/Desktop/' + str(filepath[-46:-4]), dpi=240)
# Snap 121: the distance between the lmc and mw is ~230 kpc away
# Snap 117: the distance between the lmc and mw is ~300 kpc away
plt.show()
