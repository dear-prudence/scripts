import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

type_plot = 'massDen'

if type_plot == 'numDen':
    dataL = np.load('/Users/ursa/Desktop/animation_frames/'
                    '09_18_gas_massDen_LMC_50x400x400kpc_bin0.5kpc_1_2_.npz')
    dataR = np.load('/Users/ursa/Desktop/animation_frames/'
                    '09_18_gas_massDen_LMC_50x400x400kpc_bin0.5kpc_1_2.npz')
    v_min = 1e1
    v_max = 1e5
    c_map = 'viridis'
elif type_plot == 'massDen':
    dataL = np.load('/Volumes/enceladus/hestia/09_18/gas/massDen_LMC_100x800ckpc_bin1.0ckpc/'
                    '09_18_gas_massDen_LMC_800x800x100ckpc_bin1.0ckpc_0_1.npz')
    dataR = np.load('/Volumes/enceladus/hestia/09_18/gas/massDen_LMC_100x800ckpc_bin1.0ckpc/'
                    '09_18_gas_massDen_LMC_100x800x800ckpc_bin1.0ckpc_2_1.npz')
    v_min = 1
    v_max = 1e6
    c_map = 'viridis'
elif type_plot == 'temperature':
    dataL = np.load('/Users/ursa/Desktop/images/temperature_LMC_100x800ckpc_bin2.0ckpc/'
                    '09_18_gas_temperature_LMC_800x800x100ckpc_bin2.0ckpc_0_1.npz')
    dataR = np.load('/Users/ursa/Desktop/images/temperature_LMC_100x800ckpc_bin2.0ckpc/'
                    '09_18_gas_temperature_LMC_100x800x800ckpc_bin2.0ckpc_2_1.npz')
    v_min = 5 * 10 ** 4
    v_max = 5 * 10 ** 6
    c_map = 'coolwarm'
elif type_plot == 'vlsr':
    dataL = np.load('/Users/ursa/Desktop/vlsr/09_18_gas_vlsr_LMC_400x400x50kpc_bin1.0kpc_0_1.npz')
    dataR = np.load('/Users/ursa/Desktop/vlsr/09_18_gas_vlsr_LMC_50x400x400kpc_bin1.0kpc_2_1.npz')
    v_min = -600
    v_max = 600
    c_map = 'bwr'
else:
    print('Invalid plot type!')
    exit()

imageL, imageR = dataL['data'].T, dataR['data'].T
x_edgesL, y_edgesL = dataL['x_edges'], dataL['y_edges']
x_edgesR, y_edgesR = dataR['x_edges'], dataR['y_edges']
dates = dataL['time']
# imageL = gaussian_smooth(imageR, sig=0.5)
if type_plot == 'vlsr':
    background_color = plt.get_cmap(c_map)(0)
else:
    background_color = plt.get_cmap(c_map)(0)

for i, (frameL, frameR, date) in enumerate(zip(imageL[::-1], imageR[::-1], dates[::-1]), start=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    # --------------------------------------
    # Module to define cosmetics (axes, background, labels, etc...)
    ax1.set_facecolor(background_color)
    ax2.set_facecolor(background_color)
    fig.tight_layout()
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    ax2.set_xlabel('Z'); ax2.set_ylabel('Y')
    fig.suptitle('$z = $' + '{:.{}f}'.format(date[0], 3)
                 + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(date[1]), 2), 2) + ' Gyr'
                 , x=0.5, y=0.01, ha='center', va='bottom', weight='bold')
    # --------------------------------------
    # Module to construct the image from image data
    if type_plot == 'vlsr':
        im1 = ax1.imshow(frameL, origin='lower', extent=[x_edgesL[0], x_edgesL[-1], y_edgesL[0], y_edgesL[-1]],
                         cmap=c_map, vmin=v_min, vmax=v_max)
        im2 = ax2.imshow(frameR, origin='lower', extent=[x_edgesR[0], x_edgesR[-1], y_edgesR[0], y_edgesR[-1]],
                         cmap=c_map, vmin=v_min, vmax=v_max)
    else:
        im1 = ax1.imshow(frameL, origin='lower', extent=[x_edgesL[0], x_edgesL[-1], y_edgesL[0], y_edgesL[-1]],
                         cmap=c_map, norm=LogNorm(vmin=v_min, vmax=v_max))
        im2 = ax2.imshow(frameR, origin='lower', extent=[x_edgesR[0], x_edgesR[-1], y_edgesR[0], y_edgesR[-1]],
                         cmap=c_map, norm=LogNorm(vmin=v_min, vmax=v_max))
    # --------------------------------------
    # Module to add scales/color bars ...
    if type_plot == 'numDen':
        ax1.plot([x_edgesL[-220], x_edgesL[-20]], [y_edgesL[20], y_edgesL[20]], c='white')
        ax1.text(x_edgesL[-90], y_edgesL[26], s='100 kpc', c='white')
        ax2.plot([x_edgesR[-220], x_edgesR[-20]], [y_edgesR[20], y_edgesR[20]], c='white')
        ax2.text(x_edgesR[-90], y_edgesR[26], s='100 kpc', c='white')
    elif type_plot == 'massDen':
        ax1.plot([x_edgesL[-220], x_edgesL[-20]], [y_edgesL[20], y_edgesL[20]], c='white')
        ax1.text(x_edgesL[-95], y_edgesL[26], s='200 kpc', c='white')
        ax2.plot([x_edgesR[-220], x_edgesR[-20]], [y_edgesR[20], y_edgesR[20]], c='white')
        ax2.text(x_edgesR[-95], y_edgesR[26], s='200 kpc', c='white')
        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
        # Place colorbar using ax1 as a reference
        cbar = fig.colorbar(im1, cax=cax, label='Mass Density ($M/$ckpc$^3$)')
    elif type_plot == 'temperature':
        ax1.plot([x_edgesL[-110], x_edgesL[-10]], [y_edgesL[10], y_edgesL[10]], c='white')
        ax1.text(x_edgesL[-50], y_edgesL[13], s='200 kpc', c='white')
        ax2.plot([x_edgesR[-100], x_edgesR[-10]], [y_edgesR[10], y_edgesR[10]], c='white')
        ax2.text(x_edgesR[-50], y_edgesR[13], s='200 kpc', c='white')
        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
        # Place colorbar using ax1 as a reference
        cbar = fig.colorbar(im1, cax=cax, label='Temperature (K)')
    elif type_plot == 'vlsr':
        ax1.plot([x_edgesL[-220], x_edgesL[-20]], [y_edgesL[20], y_edgesL[20]], c='white')
        ax1.text(x_edgesL[-95], y_edgesL[26], s='50 kpc', c='white')
        ax2.plot([x_edgesR[-220], x_edgesR[-20]], [y_edgesR[20], y_edgesR[20]], c='white')
        ax2.text(x_edgesR[-95], y_edgesR[26], s='50 kpc', c='white')
        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
        # Place colorbar using ax1 as a reference
        cbar = fig.colorbar(im1, cax=cax, label='vlsr')

    filename = (f'/Users/dear-prudence/Desktop/images/temperature_LMC_100x800ckpc_bin2.0ckpc/'
                f'snap_{i + 67:03d}.png')
    plt.savefig(filename, dpi=240)
    plt.close()


