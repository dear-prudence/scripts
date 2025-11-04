import numpy as np
from astropy.io import fits as f
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#plt.style.use(astropy_mpl_style)

image_data = f.getdata('/Volumes/enceladus/LG_simulations/09_18/HST/09_18_3_anirot_5_127_y.B.fits', ext=0)
bounds = [200, 600, 200, 600]
fig, ax = plt.subplots(figsize=(8, 8))
c_map = 'plasma'
background_color = plt.get_cmap(c_map)(0)
ax.set_facecolor(background_color)
im = ax.imshow(image_data, origin='lower', cmap=c_map, aspect='auto', norm=LogNorm(vmin=1))

plt.show()
