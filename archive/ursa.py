import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm


def plot_image(param, input_path, output_path, snaps, scale):
    if param == 'massDen':
        c_map = 'viridis'
        v_min, v_max = 1, 1e6
        c_label = 'white'
    elif param == 'temperature':
        c_map = 'coolwarm'
        v_min, v_max = 5e4, 5e6
        c_label = 'black'
    elif param == 'AGNradiation':
        c_map = 'BuPu'
        v_min, v_max = 0, 1
        c_label = 'k'
    else:
        print('Error: plot_image(' + str(param) + ') is invalid!')
        exit(1)

    background_color = plt.get_cmap(c_map)(0)
    s_label = str(int(scale[0] * 2)) + ' kpc'

    data = np.load(input_path)
    image = np.rot90(data['data'])
    x_e, y_e = data['x_edges'], data['y_edges']
    dates = data['time']

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    # --------------------------------------
    # Module to define cosmetics (axes, background, labels, etc...)
    fig.tight_layout()
    for ax, i in zip(axes, snaps):
        ax.set_facecolor(background_color)
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(image[:, :, 127 - i], origin='upper',
                  extent=[x_e[0], x_e[-1], y_e[0], y_e[-1]], cmap=c_map,
                  norm=LogNorm(vmin=v_min, vmax=v_max))
        ax.set_title('$z = $' + '{:.{}f}'.format(dates[127 - i, 0], 3)
                     + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(dates[127 - i, 1]), 2), 2) + ' Gyr',
                     x=0.5, y=-0.06, ha='center', va='bottom', weight='bold')
        if param == 'massDen':
            ax.plot([x_e[-220], x_e[-20]], [y_e[20], y_e[20]], c=c_label)
            ax.text(x_e[-88], y_e[26], s=s_label, c=c_label)
        elif param == 'temperature':
            ax.plot([x_e[-110], x_e[-10]], [y_e[10], y_e[10]], c=c_label)
            ax.text(x_e[-43], y_e[13], s=s_label, c=c_label)
    # Create extra white space to the right of the right subplot
    # adjustment_param = 0.05
    # fig.subplots_adjust(left=adjustment_param, right=(1 - adjustment_param),
    #                     bottom=adjustment_param, top=(1 - adjustment_param))
    # fig.suptitle('Face-on images of the mass density of gas particles of the LMC (Run 09_18)',
    #              ha='left', x=0.06)
    plt.savefig(output_path, dpi=240)
    plt.close()


def plot_frames(param, input_paths, output_path, scale):
    # input_paths is a list of two files, one for the left image, and one for the right, for each frame

    if param == 'numDen':
        c_map = 'cividis'
        v_min, v_max = 1, 1e3
        c_label = 'white'
    elif param == 'massDen':
        c_map = 'viridis'
        v_min, v_max = 1, 1e6
        c_label = 'white'
    elif param == 'temperature':
        c_map = 'coolwarm'
        v_min, v_max = 5e4, 5e6
        c_label = 'black'
    elif param == 'AGNradiation':
        c_map = 'gist_heat'
        v_min, v_max = 1e-7, 1e-1
        c_label = 'white'
    elif param == 'velocity':
        c_map = 'inferno'
        v_min, v_max = 0, 300
    elif param == 'metallicity':
        c_map = 'BuPu'
        v_min, v_max = -2, -0.1
    else:
        print('Error: plot_image(' + str(param) + ') is invalid!')
        exit(1)

    data_l = np.load(input_paths[0])
    data_r = np.load(input_paths[1])
    im_l, im_r = data_l['data'].T, data_r['data'].T
    x_e_l, y_e_l = data_l['x_edges'], data_l['y_edges']
    x_e_r, y_e_r = data_r['x_edges'], data_r['y_edges']
    dates = data_l['time']

    background_color = plt.get_cmap(c_map)(0)
    s_label = str(int(scale[0] * 2)) + ' kpc'

    for i, (frameL, frameR, date) in enumerate(zip(im_l[::-1], im_r[::-1], dates[::-1]), start=1):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        ax1.set_facecolor(background_color)
        ax2.set_facecolor(background_color)
        fig.tight_layout()
        ax1.set_xticks([]);
        ax1.set_yticks([])
        ax2.set_xticks([]);
        ax2.set_yticks([])
        ax1.set_xlabel('X');
        ax1.set_ylabel('Y')
        ax2.set_xlabel('Z');
        ax2.set_ylabel('Y')
        fig.suptitle('$z = $' + '{:.{}f}'.format(date[0], 3)
                     + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(date[1]), 2), 2) + ' Gyr'
                     , x=0.5, y=0.01, ha='center', va='bottom', weight='bold')
        # --------------------------------------
        if param == 'velocity' or param == 'metallicity':
            im1 = ax1.imshow(frameL, origin='lower', extent=[x_e_l[0], x_e_l[-1], y_e_l[0], y_e_l[-1]],
                             cmap=c_map, vmin=v_min, vmax=v_max)
            im2 = ax2.imshow(frameR, origin='lower', extent=[x_e_r[0], x_e_r[-1], y_e_r[0], y_e_r[-1]],
                             cmap=c_map, vmin=v_min, vmax=v_max)
        else:
            im1 = ax1.imshow(frameL, origin='lower', extent=[x_e_l[0], x_e_l[-1], y_e_l[0], y_e_l[-1]],
                             cmap=c_map, norm=LogNorm(vmin=v_min, vmax=v_max))
            im2 = ax2.imshow(frameR, origin='lower', extent=[x_e_r[0], x_e_r[-1], y_e_r[0], y_e_r[-1]],
                             cmap=c_map, norm=LogNorm(vmin=v_min, vmax=v_max))
        # --------------------------------------
        # Module to add scales/color bars ...

        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]

        if param == 'numDen':
            cbar = fig.colorbar(im1, cax=cax, label='Relative Number Density')
        elif param == 'AGNradiation':
            cbar = fig.colorbar(im1, cax=cax, label='Relative radiation')
        elif param == 'velocity':
            cbar = fig.colorbar(im1, cax=cax, label='Magnitude of Velocity [km/s]')
        elif param == 'metallicity':
            cbar = fig.colorbar(im1, cax=cax, label='[Z/H]')
        elif param == 'massDen':
            ax1.plot([x_e_l[-220], x_e_l[-20]], [y_e_l[20], y_e_l[20]], c=c_label)
            ax1.text(x_e_l[-96], y_e_l[26], s=s_label, c=c_label)
            ax2.plot([x_e_r[-220], x_e_r[-20]], [y_e_r[20], y_e_r[20]], c=c_label)
            ax2.text(x_e_r[-96], y_e_r[26], s=s_label, c=c_label)
            cbar = fig.colorbar(im1, cax=cax, label=r'Mass Density $(M_{solar}/ckpc^3)$')
        elif param == 'temperature':
            ax1.plot([x_e_l[-110], x_e_l[-10]], [y_e_l[10], y_e_l[10]], c=c_label)
            ax1.text(x_e_l[-49], y_e_l[13], s=s_label, c=c_label)
            ax2.plot([x_e_r[-110], x_e_r[-10]], [y_e_r[10], y_e_r[10]], c=c_label)
            ax2.text(x_e_r[-49], y_e_r[13], s=s_label, c=c_label)
            # Place colorbar using ax1 as a reference
            cbar = fig.colorbar(im1, cax=cax, label='Temperature (K)')

        filename = (output_path + f'snap_{i + 67:03d}.png')
        plt.savefig(filename, dpi=240)
        plt.close()


def plot_combo_frames(params, input_paths, output_path, scale):
    # input_paths is a list of two files, one for the left image, and one for the right, for each frame

    if params == ['massDen', 'temperature']:
        c_map_l, c_map_r = 'viridis', 'coolwarm'
        v_min_l, v_max_l = 1, 1e6
        v_min_r, v_max_r = 5e4, 5e6
        c_label_l, c_label_r = 'white', 'black'
    elif params == ['temperature', 'velocity']:
        c_map_l, c_map_r = 'coolwarm', 'inferno'
        v_min_l, v_max_l = 5e4, 5e6
        v_min_r, v_max_r = 0, 300
        c_label_l, c_label_r = 'black', 'white'
    else:
        print('Error: plot_image(' + str(params) + ') is invalid!')
        exit(1)

    data_l, data_r = np.load(input_paths[0]), np.load(input_paths[1])
    im_l, im_r = data_l['data'].T, data_r['data'].T
    x_e_l, y_e_l = data_l['x_edges'], data_l['y_edges']
    x_e_r, y_e_r = data_r['x_edges'], data_r['y_edges']
    dates = data_l['time']

    background_color_l = plt.get_cmap(c_map_l)(0)
    background_color_r = plt.get_cmap(c_map_r)(0)
    s_label = str(int(scale[0] * 2)) + ' kpc'

    for i, (frameL, frameR, date) in enumerate(zip(im_l[::-1], im_r[::-1], dates[::-1]), start=1):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7))
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        ax_l.set_facecolor(background_color_l)
        ax_r.set_facecolor(background_color_r)
        fig.tight_layout()
        ax_l.set_xticks([]);
        ax_l.set_yticks([])
        ax_r.set_xticks([]);
        ax_r.set_yticks([])
        ax_l.set_xlabel('X');
        ax_l.set_ylabel('Y')
        ax_r.set_xlabel('Z');
        ax_r.set_ylabel('Y')
        fig.suptitle('$z = $' + '{:.{}f}'.format(date[0], 3)
                     + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(date[1]), 2), 2) + ' Gyr'
                     , x=0.5, y=0.01, ha='center', va='bottom', weight='bold')
        # --------------------------------------
        if params == ['temperature', 'velocity']:
            im_l = ax_l.imshow(frameL, origin='lower', extent=[x_e_l[0], x_e_l[-1], y_e_l[0], y_e_l[-1]],
                               cmap=c_map_l, norm=LogNorm(vmin=v_min_l, vmax=v_max_l))
            im_r = ax_r.imshow(frameR, origin='lower', extent=[x_e_r[0], x_e_r[-1], y_e_r[0], y_e_r[-1]],
                               cmap=c_map_r, vmin=v_min_r, vmax=v_max_r)
        else:
            im_l = ax_l.imshow(frameL, origin='lower', extent=[x_e_l[0], x_e_l[-1], y_e_l[0], y_e_l[-1]],
                               cmap=c_map_l, norm=LogNorm(vmin=v_min_l, vmax=v_max_l))
            im_r = ax_r.imshow(frameR, origin='lower', extent=[x_e_r[0], x_e_r[-1], y_e_r[0], y_e_r[-1]],
                               cmap=c_map_r, norm=LogNorm(vmin=v_min_r, vmax=v_max_r))
        # --------------------------------------
        # Module to add scales/color bars ...

        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]

        if params == ['massDen', 'temperature']:
            ax_l.plot([x_e_l[-220], x_e_l[-20]], [y_e_l[20], y_e_l[20]], c=c_label_l)
            ax_l.text(x_e_l[-96], y_e_l[26], s=s_label, c=c_label_l)
            ax_r.plot([x_e_r[-110], x_e_r[-10]], [y_e_r[10], y_e_r[10]], c=c_label_r)
            ax_r.text(x_e_r[-49], y_e_r[13], s=s_label, c=c_label_r)
            cbar = fig.colorbar(im_r, cax=cax, label='Temperature (K)')
        elif params == ['temperature', 'velocity']:
            ax_l.plot([x_e_l[-110], x_e_l[-10]], [y_e_l[10], y_e_l[10]], c=c_label_l)
            ax_l.text(x_e_l[-49], y_e_l[13], s=s_label, c=c_label_l)
            ax_r.plot([x_e_r[-110], x_e_r[-10]], [y_e_r[10], y_e_r[10]], c=c_label_r)
            ax_r.text(x_e_r[-49], y_e_r[13], s=s_label, c=c_label_r)
            cbar = fig.colorbar(im_r, cax=cax, label='Temperature (K)')

        filename = (output_path + f'snap_{i + 67:03d}.png')
        plt.savefig(filename, dpi=240)
        plt.close()


def plot_temperature_profile(num_panels, input_path, output_path, snaps, beta=1):
    # check to make sure the number of panels is equivalent to the number of snapshots
    if isinstance(snaps, list) and num_panels != len(snaps):
        print('Error: number of panels requested does not match number of snapshots provided!')
        exit(1)

    c_map = 'GnBu'
    background_color = plt.get_cmap(c_map)(0)

    data = np.load(input_path)
    image = np.rot90(data['data'])
    x_e, y_e = data['x_e'], data['y_e']
    masses = data['masses']
    n_H0 = data['n_H0']
    print('mass at snap 108: ' + str(masses[127 - 108]))
    print('nH0 at snap 108: ' + str(n_H0[127 - 108]))
    dates = data['time']

    aspect_ratio = 9 / 16 * (x_e[-1] - x_e[0]) / (y_e[-1] - y_e[0])

    if isinstance(snaps, int):
        plt.figure(figsize=(14, 8))
        plt.gca().set_facecolor(background_color)
        plt.imshow(image[:, :, 127 - snaps], origin='upper',
                   extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]), aspect=aspect_ratio,
                   cmap=c_map, norm=LogNorm(vmin=1, vmax=beta * 0.8))  # extra factor of 0.8 for aesthetics
        # plt.title('$z = $' + '{:.{}f}'.format(dates[127 - i, 0], 3)
        #           + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(dates[127 - i, 1]), 2), 2) + ' Gyr',
        #           x=0.85, y=0.0, ha='center', va='bottom', weight='bold')
        # Create extra white space to the right of the right subplot
        plt.grid(visible=True, ls='-', alpha=0.5)
        plt.xlabel('$R$ (kpc)')
        plt.ylabel('log(T)')
        plt.xticks(np.linspace(x_e[0], x_e[-1], int(np.round(x_e[-1] / 25 + 1, 0)), endpoint=True))
        plt.title(r'Temperature Profile of gas particles* of LMC Run (09_18) at $z = $'
                  + '{:.{}f}'.format(dates[127 - snaps, 0], 3)
                  + '\n*highly ionized (Neutral Hydrogen Abundance $n_{H0} < 10^{-6}$), '
                    'metal poor ($Z < 0.5$ $Z_{solar}$) gas particles', loc='left', fontsize='small')
        cax = plt.axes((0.92, 0.12, 0.02, 0.75))  # [left, bottom, width, height]
        plt.colorbar(cax=cax, label='Relative density of particles')

    else:
        if num_panels == 4:
            n_rows, n_cols = 2, 2
        elif num_panels == 6:
            n_rows, n_cols = 2, 3
        else:
            print('Error: Invalid number of panels! Routine is a WIP.')
            exit(1)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
        # Unpack the 2D array of axes into individual variables
        axes = axes.flatten()
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        fig.tight_layout()

        j = -1
        dates_temp = np.array([[0.506, -5.23], [0.258, -3.11], [0.165, -2.12], [0.099, -1.33]])

        for ax, i in zip(axes, snaps):
            j += 1

            ax.set_facecolor(background_color)
            ax.imshow(image[:, :, 127 - i], origin='upper',
                      extent=[x_e[0], x_e[-1], y_e[0], y_e[-1]], aspect=aspect_ratio,
                      cmap=c_map,
                      norm=LogNorm(vmin=1, vmax=(beta * 0.8)))  # extra factor of 0.8 for aesthetics
            """            ax.set_title('$z = $' + '{:.{}f}'.format(dates[127 - i, 0], 3)
                         + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(dates[127 - i, 1]), 2), 2) + ' Gyr',
                         x=0.85, y=0.0, ha='center', va='bottom', weight='bold')"""
            ax.set_title('$z = $' + '{:.{}f}'.format(dates_temp[j, 0], 3)
                         + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(dates_temp[j, 1]), 2), 2) + ' Gyr',
                         x=0.85, y=0.0, ha='center', va='bottom', weight='bold')
        # Create extra white space to the right of the right subplot
        adjustment_param = 0.06
        fig.subplots_adjust(left=adjustment_param, right=(1 - adjustment_param),
                            bottom=adjustment_param, top=(1 - adjustment_param))
        fig.supxlabel('$R$ (kpc)', fontsize='x-large')
        fig.supylabel('log(T)', fontsize='x-large')
        fig.suptitle('Temperature Profiles of gas particles* of LMC Run (09_18)' +
                     '\n*highly ionized (Neutral Hydrogen Abundance $n_{H0} < 10^{-6}$), '
                     'metal poor ($Z < 0.5 Z_{solar}$) gas particles', ha='left', x=0.06)

    plt.savefig(output_path, dpi=240)
    plt.show()
    plt.close()


def plot_metallicity_profile(input_path, output_path, snaps, beta=1):
    c_map = 'BuPu'

    data = np.load(input_path)
    image = np.rot90(data['data'])
    x_e, y_e = data['x_e'], data['y_e']
    dates = data['time']

    aspect_ratio = 9 / 16 * (x_e[-1] - x_e[0]) / (y_e[-1] - y_e[0])

    if isinstance(snaps, int):
        plt.figure(figsize=(14, 8))
        # plt.gca().set_facecolor(background_color)
        im = plt.imshow(image, origin='upper',
                        extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]), aspect=aspect_ratio,
                        cmap=c_map, vmin=0, vmax=1)  # extra factor of 0.8 for aesthetics
        # plt.title('$z = $' + '{:.{}f}'.format(dates[127 - i, 0], 3)
        #           + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(dates[127 - i, 1]), 2), 2) + ' Gyr',
        #           x=0.85, y=0.0, ha='center', va='bottom', weight='bold')
        # Create extra white space to the right of the right subplot
        plt.grid(visible=True, ls='-', alpha=0.5)
        plt.xlabel('$R$ (kpc)')
        plt.ylabel('log(T)')
        plt.xticks(np.linspace(x_e[0], x_e[-1], int(np.round(x_e[-1] / 25 + 1, 0)), endpoint=True))
        plt.title(r'Temperature-metallicity profile of gas particles* of LMC Run (09_18) at $z = $'
                  + '{:.{}f}'.format(0.099, 3)
                  + '\n*highly ionized (Neutral Hydrogen Abundance $n_{H0} < 10^{-6}$), '
                    'metal poor ($Z < 1.0$ $Z_{solar}$) gas particles', loc='left', fontsize='small')
        # Create extra white space to the right of the right subplot
        plt.subplots_adjust(right=0.86)
        cax = plt.axes((0.88, 0.12, 0.02, 0.75))  # [left, bottom, width, height]
        plt.colorbar(cax=cax, label=r'Metallicity ($Z/Z_{solar}$)')

    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        # Unpack the 2D array of axes into individual variables
        axes = axes.flatten()
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        fig.tight_layout()

        # dates_temp = np.array([[0.506, -5.23], [0.258, -3.11], [0.165, -2.12], [0.099, -1.33]])

        for ax, i in zip(axes, snaps):
            # ax.set_facecolor(background_color)
            im = ax.imshow(image[:, :, 127 - i], origin='upper',
                           extent=[x_e[0], x_e[-1], y_e[0], y_e[-1]], aspect=aspect_ratio,
                           cmap=c_map, vmin=0, vmax=1)
            ax.set_title('$z = $' + '{:.{}f}'.format(dates[127 - i, 0], 3)
                         + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(dates[127 - i, 1]), 2), 2) + ' Gyr',
                         x=0.85, y=0.0, ha='center', va='bottom', weight='bold')
        # Create extra white space to the right of the right subplot
        adjustment_param = 0.06
        fig.subplots_adjust(left=adjustment_param, right=(1 - adjustment_param),
                            bottom=adjustment_param, top=(1 - adjustment_param))
        fig.supxlabel('$R$ (kpc)', fontsize='x-large')
        fig.supylabel('log(T)', fontsize='x-large')
        fig.suptitle('Metallicity Profiles of gas particles* of LMC Run (09_18)' +
                     '\n*highly ionized (Neutral Hydrogen Abundance $n_{H0} < 10^{-6}$), '
                     'metal poor ($Z < 1.0 Z_{solar}$) gas particles', ha='left', x=0.06)

        # Create extra white space to the right of the right subplot
        plt.subplots_adjust(right=0.86)
        cax = plt.axes((0.88, 0.12, 0.02, 0.75))  # [left, bottom, width, height]
        fig.colorbar(im, cax=cax, label=r'Metallicity ($Z/Z_{solar}$)')

    plt.savefig(output_path, dpi=240)
    plt.show()
    plt.close()


def plot_metal_gradient(input_path, output_path, snaps):
    c_map = 'BuPu'

    data = np.load(input_path)
    H = data['data'][0, :, :]  # for some reason shape(data['data]) = (1, x_e, snaps)
    print(H.shape)
    x_e = data['x_e']
    dates = data['time']
    diff = abs(x_e[0] - x_e[1])  # since the x_e are on either side of the bins, this diff gets added to the
    # left bin edge, to place the data point on the median time of that bin
    plt.figure(figsize=(10, 6))
    print(H[:, 127 - snaps[0]])

    i = 1
    for snap in snaps:
        plt.plot(x_e[:-1] + diff, H[:, 127 - snap], c=plt.get_cmap(c_map)((i * 30) + 60),
                 label='$z = $' + '{:.{}f}'.format(dates[127 - snap, 0], 3))
        i += 1

    plt.xlim([0, 250])
    y_lim = [-4.0, 0.0]
    plt.ylim(y_lim)

    domain = [0, 18]
    ys = np.linspace(y_lim[0], y_lim[0], len(domain))
    plt.fill_between(domain, ys, facecolor='k', alpha=0.2)
    plt.text(9, -1, s='Disk', fontsize='small')
    plt.text(20, -1, s='CGM', fontsize='small')

    plt.xlabel('R [kpc]')
    plt.ylabel('[Z/H]')
    plt.legend(loc='upper right')

    plt.savefig(output_path, dpi=240)

    plt.show()
