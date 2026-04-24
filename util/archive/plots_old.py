import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_imageMap(param, snap, input_path, output_path, scale, plane='x-y'):
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(param, {})
    background_color = plt.get_cmap(settings['c_map'])(0)
    s_label = str(int(scale[1] / 4)) + ' kpc'

    last_idx = 307 if 'lastgigyear' in input_path else 127
    image = data[plane].T[last_idx - snap]
    redshift = data['redshifts'][last_idx - snap]
    lookback_time = data['lookback_times'][last_idx - snap]
    R_vir = data['virial_radii'][last_idx - snap]

    fig, ax = plt.subplots(figsize=(20, 20))

    fig.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(background_color)

    extent = (data[plane[0] + '_e'][0], data[plane[0] + '_e'][-1],
              data[plane[2] + '_e'][0], data[plane[2] + '_e'][-1])
    if settings['log'] is True:
        im = plt.imshow(image, origin='lower', extent=extent, cmap=settings['c_map'],
                        norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']))
    else:
        im = plt.imshow(image, origin='lower', extent=extent, cmap=settings['c_map'],
                        vmin=settings['v_min'], vmax=settings['v_max'])

    # --------------------------------------
    # Create extra white space to the right of the right subplot
    fig.subplots_adjust(right=0.87)
    # Create a new axis for the colorbar to the right of the subplots
    cax = fig.add_axes([0.88, 0.1105, 0.03, 0.805])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax, label=settings['bar_label'])

    # R_vir_circ = plt.Circle((0, 0), R_vir, fill=False, linestyle='dashed', alpha=0.5)
    # ax.add_artist(R_vir_circ)

    ax.plot([data[plane[0] + '_e'][-110], data[plane[0] + '_e'][-10]],
            [data[plane[2] + '_e'][10], data[plane[2] + '_e'][10]], c='k')
    ax.text(data[plane[0] + '_e'][-47], data[plane[2] + '_e'][13], s=s_label, c='k')

    plt.savefig(output_path, dpi=240)
    plt.show()
    plt.close()


def plot_imageMap_panels(param, snaps, input_path, output_path, scale, plane='x-y'):
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(param, {})
    background_color = plt.get_cmap(settings['c_map'])(0)
    s_label = str(int(scale[1] / 4)) + ' kpc'

    last_idx = 307 if 'lastgigyear' in input_path else 127
    image = data[plane].T
    redshifts = data['redshifts']
    lookback_times = data['lookback_times']
    R_virs = data['virial_radii']

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.02, wspace=0.02)
    fig.tight_layout()
    axes = gs.subplots(sharex=True, sharey=True)
    axes = axes.flatten()
    # --------------------------------------
    # Module to define cosmetics (axes, background, labels, etc...)
    # fig.tight_layout()

    # numerals = [r'$a$', r'$b$', r'$c$']
    for ax, snap in zip(axes, snaps):
        ax.set_xticks([])
        ax.set_yticks([])

        extent = (data[plane[0] + '_e'][0], data[plane[0] + '_e'][-1],
                  data[plane[2] + '_e'][0], data[plane[2] + '_e'][-1])
        if settings['log'] is True:
            im = ax.imshow(image[last_idx - snap], origin='lower', extent=extent, cmap=settings['c_map'],
                           norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']))
        else:
            im = ax.imshow(image[last_idx - snap], origin='lower', extent=extent, cmap=settings['c_map'],
                           vmin=settings['v_min'], vmax=settings['v_max'])
        # ax.set_title('$z = $' + '{:.{}f}'.format(redshifts[last_idx - snap], 3)
        #              + '$,$ \t $t = $'
        #              + '{:.{}f}'.format(-1 * round(float(lookback_times[last_idx - snap]), 2), 2) + ' Gyr',
        #              x=0.5, y=-0.08, ha='center', va='bottom', weight='bold')
        if param == 'massDen':
            # ax.plot([x_e[-220], x_e[-20]], [y_e[20], y_e[20]], c=c_label)
            # ax.text(x_e[-88], y_e[26], s=s_label, c=c_label)
            ax.text(125, -195, s='$z = $' + '{:.{}f}'.format(redshifts[last_idx - snap], 3),
                    c='k', fontsize='small')
        elif param == 'temperature':
            # ax.plot([x_e[-110], x_e[-10]], [y_e[10], y_e[10]], c=c_label)
            # ax.text(x_e[-43], y_e[13], s=s_label, c=c_label)
            pass

    # needed to recreate figure in paper
    # Create extra white space to the right of the right subplot
    # adjustment_param = 0.09
    # fig.subplots_adjust(left=adjustment_param, right=(1 - adjustment_param),
    #                     bottom=adjustment_param, top=(1 - adjustment_param))
    # fig.subplots_adjust(right=(1 - adjustment_param))
    # cax = fig.add_axes([0.92, 0.09, 0.02, 0.82])  # [left, bottom, width, height]
    # cbar = fig.colorbar(im, cax=cax, label=r'$\rho_{\rm{gas}}$' + '  ' + r'$[M_{\odot}/\rm{kpc}^{3}]$')

    # fig.suptitle('Face-on images of the mass density of gas particles of the LMC (Run 09_18)',
    #              ha='left', x=0.06)
    plt.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.show()


def plot_chisholm2025_fig1(param, snaps, input_path, output_path, scale, plane='x-y'):
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(param, {})
    background_color = plt.get_cmap(settings['c_map'])(0)
    s_label = str(int(scale[1] / 4)) + ' kpc'

    last_idx = 307 if 'lastgigyear' in input_path else 127
    image = data[plane].T
    redshifts = data['redshifts']
    lookback_times = data['lookback_times']
    R_virs = data['virial_radii']

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.02, wspace=0.02)
    fig.tight_layout()
    axes = gs.subplots(sharex=True, sharey=True)
    axes = axes.flatten()
    # --------------------------------------
    # Module to define cosmetics (axes, background, labels, etc...)
    # fig.tight_layout()

    # numerals = [r'$a$', r'$b$', r'$c$']
    for ax, snap in zip(axes, snaps):
        ax.set_xticks(np.linspace(-200, 200, num=5), minor=False)
        ax.set_xticks(np.linspace(-200, 200, num=17), minor=True)
        ax.set_yticks(np.linspace(-200, 200, num=5), minor=False)
        ax.set_yticks(np.linspace(-200, 200, num=17), minor=True)
        ax.set_xticklabels(['-200', '-100', '0', '100', ''], minor=False)
        ax.set_xticklabels([], minor=True)
        ax.set_yticklabels(['', '-100', '0', '100', '200'], minor=False)
        ax.set_yticklabels([], minor=True)

        extent = (data[plane[0] + '_e'][0], data[plane[0] + '_e'][-1],
                  data[plane[2] + '_e'][0], data[plane[2] + '_e'][-1])
        if settings['log'] is True:
            im = ax.imshow(image[last_idx - snap], origin='lower', extent=extent, cmap=settings['c_map'],
                           norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']))
        else:
            im = ax.imshow(image[last_idx - snap], origin='lower', extent=extent, cmap=settings['c_map'],
                           vmin=settings['v_min'], vmax=settings['v_max'])
        # ax.set_title('$z = $' + '{:.{}f}'.format(redshifts[last_idx - snap], 3)
        #              + '$,$ \t $t = $'
        #              + '{:.{}f}'.format(-1 * round(float(lookback_times[last_idx - snap]), 2), 2) + ' Gyr',
        #              x=0.5, y=-0.08, ha='center', va='bottom', weight='bold')
            # ax.plot([x_e[-220], x_e[-20]], [y_e[20], y_e[20]], c=c_label)
            # ax.text(x_e[-88], y_e[26], s=s_label, c=c_label)
        ax.text(119, -190, s='$z = $' + '{:.{}f}'.format(redshifts[last_idx - snap], 3), c='k')
        if snap == 95:
            ax.annotate(r'$\it{massive}$' + '\n' + r'${dwarf}$', xy=(-100, -75), color='white')
            circ_massiveDwarf = plt.Circle((0, 0), 50, fill=False, linestyle='dashed', alpha=0.5, color='white')
            ax.add_artist(circ_massiveDwarf)

            ax.annotate(r'$\it{smaller}$' + '\n' + r'${dwarf}$', xy=(125, -50), color='white')
            circ_smallerDwarf = plt.Circle((146, 11), 25, fill=False, linestyle='dashed', alpha=0.5, color='white')
            ax.add_artist(circ_smallerDwarf)
            ax.annotate('', xytext=(138, 33), xy=(123, 64), arrowprops=dict(arrowstyle="->"), alpha=0.5, color='white')

        if snap == 127:
            ax.annotate(r'$\it{neutral}$' + '\n' + r'${stream}$', xy=(80, 80), color='white')
            ax.annotate('', xytext=(70, 40), xy=(95, 75), arrowprops=dict(arrowstyle="-"), alpha=0.5, color='white')

    # needed to recreate figure in paper
    # Create extra white space to the right of the right subplot
    adjustment_param = 0.09
    fig.subplots_adjust(left=adjustment_param, right=(1 - adjustment_param),
                        bottom=adjustment_param, top=(1 - adjustment_param))
    # fig.subplots_adjust(right=(1 - adjustment_param))
    cax = fig.add_axes([0.92, 0.09, 0.02, 0.82])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax, label=r'$\rho_{\rm{gas}}$' + '  ' + r'$[M_{\odot}/\rm{kpc}^{3}]$')

    fig.supxlabel(r'$X$' + '  ' + r'$[\rm{kpc}]$')
    fig.supylabel(r'$Y$' + '  ' + r'$[\rm{kpc}]$')

    # fig.suptitle('Face-on images of the mass density of gas particles of the LMC (Run 09_18)',
    #              ha='left', x=0.06)
    plt.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.show()


def plot_imageMap_frames(param, input_path, output_path, scale, planes=None):
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(param, {})
    e_i = ['x-y', 'x-z'] if planes is None else planes

    background_color = plt.get_cmap(settings['c_map'])(0)
    s_label = str(int(scale[1] / 4)) + ' kpc'

    image_l = data[e_i[0]].T
    image_r = data[e_i[1]].T
    redshifts = data['redshifts']
    print(len(redshifts))
    lookback_times = data['lookback_times']

    for i, (frameL, frameR, z, t) in enumerate(zip(image_l[::-1], image_r[::-1],
                                                   redshifts[::-1], lookback_times[::-1]), start=0):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='k')
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        ax1.set_facecolor(background_color)
        ax2.set_facecolor(background_color)
        fig.tight_layout()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.suptitle('$z = $' + '{:.{}f}'.format(z, 3)
                     + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(t), 2), 2) + ' Gyr'
                     , x=0.5, y=0.01, ha='center', va='bottom', weight='bold', c='white')
        # --------------------------------------
        extent_l = [data[e_i[0][0] + '_e'][0], data[e_i[0][0] + '_e'][-1],
                    data[e_i[0][2] + '_e'][0], data[e_i[0][2] + '_e'][-1]]
        extent_r = [data[e_i[1][0] + '_e'][0], data[e_i[1][0] + '_e'][-1],
                    data[e_i[1][2] + '_e'][0], data[e_i[1][2] + '_e'][-1]]
        # --------------------------------------
        if settings['log'] is True:
            im1 = ax1.imshow(frameL, origin='lower', extent=extent_l, cmap=settings['c_map'],
                             norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']))
            im2 = ax2.imshow(frameR, origin='lower', extent=extent_r, cmap=settings['c_map'],
                             norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']))
        else:
            im1 = ax1.imshow(frameL, origin='lower', extent=extent_l, cmap=settings['c_map'],
                             vmin=settings['v_min'], vmax=settings['v_max'])
            im2 = ax2.imshow(frameR, origin='lower', extent=extent_r, cmap=settings['c_map'],
                             vmin=settings['v_min'], vmax=settings['v_max'])
        # --------------------------------------
        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
        cbar = fig.colorbar(im1, cax=cax, label=settings['bar_label'])

        # ax1.plot([x_e_l[-220], x_e_l[-20]], [y_e_l[20], y_e_l[20]], c=c_label)
        ax1.text(data[e_i[0][0] + '_e'][7], data[e_i[0][2] + '_e'][7], s=r'$\it{face-on}$', c='white',
                 fontsize='small')
        # ax2.plot([x_e_r[-220], x_e_r[-20]], [y_e_r[20], y_e_r[20]], c=c_label)
        ax2.text(data[e_i[1][0] + '_e'][-45], data[e_i[1][2] + '_e'][7], s=r'$\it{edge-on}$', c='white',
                 fontsize='small')

        ax1.set_facecolor(background_color)
        ax2.set_facecolor(background_color)

        final_idx = 308 if 'lastgigyear' in input_path else 127
        starting_index = final_idx - len(data['x-y'][0, 0, :])
        filename = (output_path + f'snap_{i + starting_index:03d}.png')
        plt.savefig(filename, dpi=150)
        plt.close()


def plot_imageProfile(param, input_path, output_path, axis='y'):
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(param, {})
    axis_to_edges = {'x': 'x_e', 'y': 'y_e', 'z': 'z_e'}

    y_lim = np.array([np.min(data[axis]), np.max(data[axis])])
    x_e = data[axis_to_edges[axis]]
    x = x_e[:-1] + (x_e[1] - x_e[0]) / 2

    aspect_ratio = 1 / 2 * (x_e[-1] - x_e[0]) / (y_lim[1] - y_lim[0])
    snaps = [127, 126, 125, 124, 123, 122]

    plt.figure(figsize=(12, 7))

    for i, snap in zip(range(6), snaps):
        plt.plot(x, data[axis][i], color='tab:blue',
                 label=r'$t = $' + '{:.{}f}'.format(-1 * round(data['lookback_times'][i], 2), 2) + ' Gyr',
                 alpha=1 - np.sqrt(i) * 0.4)

    # Create extra white space to the right of the right subplot
    plt.grid(visible=True, ls='-', alpha=0.4)
    plt.xlim([x_e[0], x_e[-1]])
    plt.xlabel(axis + ' (kpc)')
    plt.ylabel(settings['bar_label'])

    plt.legend(loc='upper right')

    plt.savefig(output_path, dpi=240)
    plt.show()
    plt.close()


def plot_imageMap_frames_temp(param, input_path, output_path, scale, planes=None):
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(param, {})
    e_i = ['x-y']

    background_color = plt.get_cmap(settings['c_map'])(0)
    s_label = str(int(scale[1] / 4)) + ' kpc'

    image_l = data[e_i[0]].T
    redshifts = data['redshifts']
    print(len(redshifts))
    lookback_times = data['lookback_times']

    for i, (frameL, z, t) in enumerate(zip(image_l[::-1], redshifts[::-1], lookback_times[::-1]), start=0):
        fig, ax1 = plt.subplots(figsize=(7, 7), facecolor='k')
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        ax1.set_facecolor(background_color)
        fig.tight_layout()
        ax1.set_xticks([])
        ax1.set_yticks([])
        fig.suptitle('$z = $' + '{:.{}f}'.format(z, 3)
                     + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(t), 2), 2) + ' Gyr'
                     , x=0.5, y=0.01, ha='center', va='bottom', weight='bold', c='white')
        # --------------------------------------
        extent_l = [data[e_i[0][0] + '_e'][0], data[e_i[0][0] + '_e'][-1],
                    data[e_i[0][2] + '_e'][0], data[e_i[0][2] + '_e'][-1]]
        # --------------------------------------
        if settings['log'] is True:
            im1 = ax1.imshow(frameL, origin='lower', extent=extent_l, cmap=settings['c_map'],
                             norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']))
        else:
            im1 = ax1.imshow(frameL, origin='lower', extent=extent_l, cmap=settings['c_map'],
                             vmin=settings['v_min'], vmax=settings['v_max'])
        # --------------------------------------
        # Create extra white space to the right of the right subplot
        # fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        # cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
        # cbar = fig.colorbar(im1, cax=cax, label=settings['bar_label'])

        ax1.plot([data[e_i[0][0] + '_e'][-110], data[e_i[0][0] + '_e'][-10]],
                 [data[e_i[0][2] + '_e'][7], data[e_i[0][2] + '_e'][7]], c='white', lw=1)
        ax1.text(data[e_i[0][0] + '_e'][-39], data[e_i[0][2] + '_e'][10], s='100 kpc', c='white',
                 fontsize='small')
        ax1.set_facecolor(background_color)

        final_idx = 308 if 'lastgigyear' in input_path else 127
        starting_index = final_idx - len(data['x-y'][0, 0, :])
        filename = (output_path + f'snap_{i + starting_index:03d}.png')
        plt.savefig(filename, dpi=240)
        plt.close()


def plot_imageProfile(param, input_path, output_path, axis='y'):
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(param, {})
    axis_to_edges = {'x': 'x_e', 'y': 'y_e', 'z': 'z_e'}

    y_lim = np.array([np.min(data[axis]), np.max(data[axis])])
    x_e = data[axis_to_edges[axis]]
    x = x_e[:-1] + (x_e[1] - x_e[0]) / 2

    aspect_ratio = 1 / 2 * (x_e[-1] - x_e[0]) / (y_lim[1] - y_lim[0])
    snaps = [127, 126, 125, 124, 123, 122]

    plt.figure(figsize=(12, 7))

    for i, snap in zip(range(6), snaps):
        plt.plot(x, data[axis][i], color='tab:blue',
                 label=r'$t = $' + '{:.{}f}'.format(-1 * round(data['lookback_times'][i], 2), 2) + ' Gyr',
                 alpha=1 - np.sqrt(i) * 0.4)

    # Create extra white space to the right of the right subplot
    plt.grid(visible=True, ls='-', alpha=0.4)
    plt.xlim([x_e[0], x_e[-1]])
    plt.xlabel(axis + ' (kpc)')
    plt.ylabel(settings['bar_label'])

    plt.legend(loc='upper right')

    plt.savefig(output_path, dpi=240)
    plt.show()
    plt.close()


def plot_temperature_profile(input_path, output_path, snapshot, include_isolated_sim=False, style='classic'):
    c_map = 'Blues'
    background_color = plt.get_cmap(c_map)(0)
    starting_index = 307 if 'lastgigyear' in input_path else 127

    if style == 'classic':
        plt.style.use('classic')
        plt.rcParams.update({"grid.linestyle": "--",  # Dashed grid lines
                             "grid.alpha": 0.5,  # Fainter grid
                             "axes.grid": True,  # Enable grid
                             "xtick.direction": "in",  # Ticks pointing inward
                             "ytick.direction": "in",
                             "font.size": 12,  # Standard font size for papers
                             "axes.labelsize": 14,  # Larger labels
                             "axes.titlesize": 14,
                             "xtick.labelsize": 12,
                             "ytick.labelsize": 12,
                             })
    else:
        pass
    if style == 'dark':
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "gray",
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "axes.grid": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })

    data = np.load(input_path)
    profiles = np.rot90(data['hist'])
    x_e, y_e = data['x_e'], data['y_e']
    redshifts = data['redshifts']
    T = data['profiles_T'][starting_index - snapshot]
    r = data['profiles_r'][starting_index - snapshot]
    R_vir = data['virial_radii'][starting_index - snapshot]
    column_averages = data['column_averages'][:100]
    HI_mass = data['H0_mass'][starting_index - snapshot]
    HII_mass = data['H1_mass'][starting_index - snapshot]
    temperature = data['temperature'][starting_index - snapshot]  # of HII gas
    f_H1 = data['f_H1'][starting_index - snapshot]

    print('M_HI = ' + str(HI_mass) + '\n'
          + 'M_HII = ' + str(HII_mass) + '\n'
          + 'T ~= ' + str(temperature) + '\n'
          + 'f_H1 ~= ' + str(f_H1))

    aspect_ratio = 1
    actual_aspect_ratio = aspect_ratio * (x_e[-1] - x_e[0]) / (y_e[-1] - y_e[0])
    vmax = np.max(profiles[:, :, starting_index - snapshot]) * 10

    plt.figure(figsize=(10, 7))
    # plt.gca().set_facecolor(background_color)
    plt.imshow(profiles[:, :, starting_index - snapshot], origin='upper',
               extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]), aspect=actual_aspect_ratio,
               cmap=c_map, norm=LogNorm(vmin=vmax / 1e2, vmax=vmax))

    # hestia column averages
    # plt.plot(x_e[:-1] + abs(x_e[0] + x_e[1]) / 2, column_averages, c='k', label='hestia')

    # Scott's simulations
    # if include_isolated_sim is True:
    #     column_averages_lmc, x_e_lmc = lmc_temperatureProfile()
    #     plt.plot(x_e_lmc[:-1] + abs(x_e_lmc[0] + x_e_lmc[1]) / 2, column_averages_lmc, c='k', linestyle='dashed',
    #              label='Lucchini+2024')

    # Stable corona (Salem+2015)
    plt.plot(r, np.log10(T), linestyle='solid', c='k', label='Salem+2015')

    # ine indicating virial radius
    plt.plot([R_vir, R_vir], [4.5, 6.5], linestyle='dotted', color='k', alpha=0.2)
    plt.text(R_vir + 3, 6.2, s=r'$R_{\text{vir}}$', fontsize='small')

    # Create extra white space to the right of the right subplot
    # plt.grid(visible=True, ls='-', alpha=0.4)
    plt.xlim([x_e[0], x_e[-1]])
    plt.xlabel(r'$R$ ' + '$(kpc)$')
    plt.ylabel(r'$\log T$ ' + r'$(\log K)$')
    # plt.xticks(np.linspace(x_e[0], x_e[-1], int(np.round(x_e[-1] / 25 + 1, 0)), endpoint=True))
    """    plt.title(r'Temperature Profile of gas particles* of halo_08 Run (09_18) at '
              + r'$z = $' + '{:.{}f}'.format(redshifts[127 - snapshot], 3)
              # + '\n*highly ionized (Neutral Hydrogen Abundance ' + r'$n_{H0} < $' + f'10^{h0_threshold}' + '), '
              + '\n*highly ionized ()'
              + 'diffuse (' + r'$\rho < $' + f'10^{rho_threshold}' + ') gas particles'
              + '\n' + r'$M_{\text{gas}}(r<R_{vir}\text{ kpc}) = $'
              + f'{coeff}' + r'$\times$' + f'10^{power}' + r' $M_{\odot}$, '
              + r'$T_{\text{avg}}(r<R_{vir}\text{ kpc}) =$ ' + f'10^{temperature} K', loc='left',
                fontsize='small')"""
    # cax = plt.axes((0.92, 0.12, 0.02, 0.75))  # [left, bottom, width, height]
    # plt.colorbar(cax=cax, label='Relative density of particles')
    # plt.legend(loc='lower right', fontsize='small')

    plt.savefig(output_path, dpi=240)
    plt.show()
    plt.close()


def plot_temperature_profile_dark(input_path, output_path, snapshot, include_isolated_sim=False, style='dark'):
    c_map = 'Greys_r'  # Inverted colormap: white = high density, black = low
    starting_index = 307 if 'lastgigyear' in input_path else 127

    # Set dark theme styling
    if style == 'dark':
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "gray",
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "axes.grid": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })

    data = np.load(input_path)
    profiles = np.rot90(data['hist'])
    x_e, y_e = data['x_e'], data['y_e']
    redshifts = data['redshifts']
    T = data['profiles_T'][starting_index - snapshot]
    r = data['profiles_r'][starting_index - snapshot]
    R_vir = data['virial_radii'][starting_index - snapshot]
    column_averages = data['column_averages'][:100]
    HI_mass = data['H0_mass'][starting_index - snapshot]
    HII_mass = data['H1_mass'][starting_index - snapshot]
    temperature = data['temperature'][starting_index - snapshot]
    f_H1 = data['f_H1'][starting_index - snapshot]

    print('M_HI = ' + str(HI_mass) + '\n'
          + 'M_HII = ' + str(HII_mass) + '\n'
          + 'T ~= ' + str(temperature) + '\n'
          + 'f_H1 ~= ' + str(f_H1))

    aspect_ratio = 1
    actual_aspect_ratio = aspect_ratio * (x_e[-1] - x_e[0]) / (y_e[-1] - y_e[0])
    vmax = np.max(profiles[:, :, starting_index - snapshot]) * 10

    plt.figure(figsize=(10, 7))
    plt.imshow(profiles[:, :, starting_index - snapshot], origin='upper',
               extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]), aspect=actual_aspect_ratio,
               cmap=c_map, norm=LogNorm(vmin=vmax / 1e2, vmax=vmax))

    # Temperature profile line (Salem+2015)
    plt.plot(r, np.log10(T), linestyle='solid', c='white', label='Salem+2015')

    # Virial radius line
    plt.plot([R_vir, R_vir], [4.5, 6.5], linestyle='dotted', color='white', alpha=0.3)
    plt.text(R_vir + 3, 6.2, s=r'$R_{\text{vir}}$', fontsize='small', color='white')

    plt.xlim([x_e[0], x_e[-1]])
    plt.xlabel(r'$R$ ' + '$(kpc)$')
    plt.ylabel(r'$\log T$ ' + r'$(\log \rm{K})$')

    # Optional isolated sim overlay
    #if include_isolated_sim:
    #    column_averages_lmc, x_e_lmc = lmc_temperatureProfile()
    #    plt.plot(x_e_lmc[:-1] + abs(x_e_lmc[0] + x_e_lmc[1]) / 2,
    #             column_averages_lmc, c='gray', linestyle='dashed',
    #             label='Lucchini+2024')

    # Optional legend
    # plt.legend(loc='lower right', fontsize='small')

    plt.savefig(output_path, dpi=240, facecolor='black')
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
    y_lim = [7.5, 11.5]
    plt.ylim(y_lim)

    domain = [0, 18]
    ys = np.linspace(y_lim[0], y_lim[0], len(domain))
    plt.fill_between(domain, ys, facecolor='k', alpha=0.2)
    plt.text(9, -1, s='Disk', fontsize='small')
    plt.text(20, -1, s='CGM', fontsize='small')

    plt.xlabel('R [kpc]')
    plt.ylabel('log(O/H) + 12')
    plt.legend(loc='upper right')

    plt.savefig(output_path, dpi=240)

    plt.show()


def plot_tempProf_and_coronaRelation(input_path_tempProf, input_path_coronae, output_path, snapshot, style='classic'):
    """
    This function reproduces the combined temperature profile of the massive dwarf and the corona-halo mass relation
    (Fig 2 from Chisholm+2025)
    """
    from scripts.util.archive.lucchini import lmc_temperatureProfile
    from hestia import calc_virialTemperature

    c_map = 'Blues'
    starting_index = 307 if 'lastgigyear' in input_path_tempProf else 127

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6), gridspec_kw={'wspace': 0})
    # Remove extra space between subplots
    fig.subplots_adjust(wspace=0)  # Ensure no space between left and right plots
    plt.ylim([4.5, 6.5])

    # -------------------------------------------------
    # importing and processing of temperature profiles (left subplot)

    data_profiles = np.load(input_path_tempProf)
    profiles = np.rot90(data_profiles['hist'])
    x_e_profiles, y_e_profiles = data_profiles['x_e'], data_profiles['y_e']
    redshifts = data_profiles['redshifts']
    T = data_profiles['profiles_T'][starting_index - snapshot]
    r = data_profiles['profiles_r'][starting_index - snapshot]
    R_vir = data_profiles['virial_radii'][starting_index - snapshot]
    column_averages = data_profiles['column_averages'][:100]
    HI_mass = data_profiles['H0_mass'][starting_index - snapshot]
    HII_mass = data_profiles['H1_mass'][starting_index - snapshot]
    temperature = data_profiles['temperature'][starting_index - snapshot]  # of HII gas
    f_H1 = data_profiles['f_H1'][starting_index - snapshot]

    # -------------------------------------------------
    # importing and processing of coronae (right subplot)

    all_halos = np.load(input_path_coronae)
    var_dict = {'halo_id': 0, 'bool_satellite': 1, 'M_halo': 2, 'R_halo': 3, 'M_H0': 4, 'M_H1': 5, 'T_corona': 6}
    labels_dict = {'M_halo': r'$\log (M_{halo}/M_{\odot})$', 'R_halo': r'$R_{vir}$',
                   'M_H0': r'$M_{HI}$', 'M_H1': r'$M_{HII}$', 'T_corona': r'$T_{HII}$ ' + r'$[K]$'}

    data = {name: None for name in var_dict.keys()}
    for var in var_dict.keys():
        if var == 'bool_satellite':
            data[var] = (list(all_halos['09_18'][:, 1] == b'True')
                         + list(all_halos['17_11'][:, 1] == b'True')
                         + list(all_halos['37_11'][:, 1] == b'True'))
        else:
            data[var] = np.log10(np.append(np.array(all_halos['09_18'][:, var_dict[var]], dtype=np.float64),
                                           [np.array(all_halos['17_11'][:, var_dict[var]], dtype=np.float64),
                                            np.array(all_halos['37_11'][:, var_dict[var]], dtype=np.float64)]))
    colors = []
    for i in range(len(data['bool_satellite'])):
        if data['bool_satellite'][i]:
            colors.append('none')
        else:
            if data['M_H1'][i] < 8.25:
                colors.append('k')
            else:
                colors.append('k')

    # -------------------------------------------------
    # plotting of temperature profiles (left plot)

    aspect_ratio = 1
    actual_aspect_ratio = aspect_ratio * (x_e_profiles[-1] - x_e_profiles[0]) / (y_e_profiles[-1] - y_e_profiles[0])
    vmax = np.max(profiles[:, :, starting_index - snapshot]) * 10

    ax1.imshow(profiles[:, :, starting_index - snapshot], origin='upper',
               extent=(x_e_profiles[0], x_e_profiles[-1], y_e_profiles[0], y_e_profiles[-1]),
               aspect=actual_aspect_ratio, cmap=c_map, norm=LogNorm(vmin=vmax / 1e2, vmax=vmax), rasterized=True)

    # hestia column averages
    ax1.plot(x_e_profiles[:-1] + abs(x_e_profiles[0] + x_e_profiles[1]) / 2, column_averages,
             c='tab:blue', lw=2, label='Massive dwarf (this work)')

    # Scott's simulations
    column_averages_lmc, x_e_lmc = lmc_temperatureProfile()
    ax1.plot(x_e_lmc[:-1] + abs(x_e_lmc[0] + x_e_lmc[1]) / 2, column_averages_lmc, c='k', linestyle='dashed',
             label='Isolated LMC-analog (Lucchini+ 2024)')

    # Stable corona (Salem+2015)
    ax1.plot(r, np.log10(T), linestyle='dotted', c='k', label='Equilibrium profile (Salem+ 2015)')

    # line indicating virial radius
    ax1.plot([R_vir, R_vir], [4.5, 6.5], linestyle='solid', color='k', alpha=0.3)
    ax1.text(R_vir + 3, 6.3, s=r'$R_{\rm{vir}}$', fontsize='large')

    # line indicating R_max
    R_max = 12.88  # kpc
    ax1.plot([12.88, 12.88], [4.5, 6.5], linestyle='solid', color='k', alpha=0.3)
    ax1.text(R_max + 3, 6.3, s=r'$R_{\rm{max}}$', fontsize='large')

    ax1.legend(loc='lower right', fontsize='medium')
    # Define tick positions
    # xticks_major = [20, 60, 100, 140]
    # Apply to the left subplot (ax1)
    # ax1.set_xticks(xticks_major)
    # Create extra white space to the right of the right subplot
    # plt.grid(visible=True, ls='-', alpha=0.4)
    ax1.set_xlim([x_e_profiles[0], x_e_profiles[-1]])

    ax1.set_xticks(np.linspace(20, 140, num=4), minor=False)
    ax1.set_xticks(np.linspace(0, 170, num=18), minor=True)
    ax1.set_yticks(np.linspace(4.5, 6.5, num=5), minor=False)
    ax1.set_yticks(np.linspace(4.5, 6.5, num=17), minor=True)
    ax1.set_xticklabels(['20', '60', '100', '140'], minor=False)
    ax1.set_xticklabels([], minor=True)
    ax1.set_yticklabels(['4.5', '5.0', '5.5', '6.0', '6.5'], minor=False)
    ax1.set_yticklabels([], minor=True)

    ax1.set_xlabel(r'$R$ ' + r'$[\rm{kpc}]$')
    ax1.set_ylabel(r'$\log (T/\rm{K})$')

    # -------------------------------------------------
    # plotting of coronae (right plot)

    # plotting the halos
    ax2.scatter(data['M_halo'], data['T_corona'], fc=colors, ec='k', s=25, rasterized=True)

    # plotting massive dwarf from this work
    idx_lmc_analog = 3
    plt.scatter(data['M_halo'][idx_lmc_analog], data['T_corona'][idx_lmc_analog], marker='D',
                s=80, ec='tab:blue', c='tab:blue', label='Massive dwarf', rasterized=True)

    # plotting the virial theorem line
    x_arr = np.linspace(min(data['M_halo']), max(data['M_halo']) + 1.0)
    x_l = np.linspace(min(x_arr), max(x_arr))
    y_l = np.log10(calc_virialTemperature(10 ** x_arr))
    ax2.plot(x_l, y_l, linestyle='dotted', color='k', alpha=1,
             label=('Virial theorem--\n' + r'$T_{\rm{vir}} = \mu m_p G M / 2 k_B R_{\rm{vir}}$'))

    ax2.set_xlim([10.25, 12.75])
    ax2.set_xlabel(labels_dict['M_halo'])
    # plt.grid(visible=True, alpha=0.5)

    ax2.set_xticks(np.linspace(10.5, 12.5, num=5), minor=False)
    ax2.set_xticks(np.linspace(10.3, 12.7, num=25), minor=True)
    ax2.set_xticklabels(['10.5', '11.0', '11.5', '12.0', '12.5'], minor=False)
    ax2.set_xticklabels([], minor=True)

    ax2.legend(loc='lower right', scatterpoints=1, fontsize='medium')

    ax2.axvspan(xmin=10.25, xmax=11, color='gray', alpha=0.3, hatch='//')

    # Ensure y-ticks and labels appear on the right side of ax2
    ax2.yaxis.tick_right()  # Move tick marks to the right
    ax2.yaxis.set_label_position("right")  # Move tick labels (numbers) to the right
    ax2.yaxis.set_ticks_position("both")  # Ensure ticks appear on both left & right sides

    # -------------------------------------------------
    # Manually adjust subplot positions if needed
    # ax1.set_position([0.1, 0.1, 0.4, 0.8])  # [left, bottom, width, height]
    # ax2.set_position([0.5, 0.1, 0.4, 0.8])  # Shift right plot leftward

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# global variables

# For Fig 1
# plt.style.use('classic')
"""
plt.rcParams.update({
                     "text.usetex": True,  # Use LaTeX for all text rendering
                     # "font.family": "sans-serif",  # Use a sans-serif font
                     # "font.sans-serif": ["Computer Modern Sans Serif"],
                     "font.family": "serif",  # Use a serif font
                     "font.serif": "Computer Modern Roman",
                     "mathtext.fontset": "stixsans",
                     "xtick.direction": "in",  # Ticks pointing inward
                     "xtick.top": True,
                     "xtick.labeltop": False,
                     "ytick.direction": "in",
                     "ytick.right": True,
                     "ytick.labelright": False,
                     # "axes.labelpad": 0.0,
                     "font.size": 11,  # Standard font size for papers
                     "axes.labelsize": 12,  # Larger labels
                     "axes.titlesize": 12,
                     "xtick.labelsize": 11,
                     "ytick.labelsize": 11,
                     "figure.subplot.wspace": 0.0,
                     "figure.constrained_layout.wspace": 0.0
                     })
"""
# for Fig 2
"""
plt.style.use('classic')
plt.rcParams.update({"grid.linestyle": "--",  # Dashed grid lines
                     # "font.family": "sans-serif",  # Use a sans-serif font
                     # "font.sans-serif": "geneva",
                     "text.usetex": True,  # Use LaTeX for all text rendering
                     # "font.family": "sans-serif",  # Use a sans-serif font
                     # "font.sans-serif": ["Computer Modern Sans Serif"],
                     "font.family": "serif",  # Use a serif font
                     "font.serif": "Computer Modern Roman",
                     "mathtext.fontset": "stixsans",
                     "grid.alpha": 0.4,  # Fainter grid
                     "axes.grid": True,  # Enable grid
                     "xtick.direction": "in",  # Ticks pointing inward
                     "ytick.labelright": True,
                     "ytick.direction": "in",
                     "font.size": 11,  # Standard font size for papers
                     "axes.labelsize": 12,  # Larger labels
                     "axes.titlesize": 12,
                     "xtick.labelsize": 12,
                     "ytick.labelsize": 12,
                     "figure.subplot.wspace": 0.0,
                     "figure.constrained_layout.wspace": 0.0
                     })
"""

param_settings = {'numDen': {'c_map': 'cividis', 'log': True, 'v_min': 1, 'v_max': 1e1,
                             'c_label': 'white', 'bar_label': 'Relative Number Density'},
                  'massDen': {'c_map': 'viridis', 'log': True, 'v_min': 1e1, 'v_max': 1e6,
                              'c_label': 'white', 'bar_label': r'$\rho$ ' + r'$[M_{\odot}/ckpc^3]$'},
                  # 'massDen': {'c_map': 'magma_r', 'log': True, 'v_min': 1e7, 'v_max': 1e10,
                  #             'c_label': 'white', 'bar_label': r'$\rho$ ' + r'$[M_{\odot}/ckpc^3]$'},
                  'temperature': {'c_map': 'Spectral', 'log': True, 'v_min': 1e4, 'v_max': 1e6,
                                  'c_label': 'black', 'bar_label': 'Temperature (K)'},
                  'agn_radiation': {'c_map': 'gist_heat', 'log': True, 'v_min': 1e-9, 'v_max': 1e-3,
                                    'c_label': 'white', 'bar_label': 'Relative AGN radiation'},
                  'cooling_rate': {'c_map': 'gist_heat', 'log': True, 'v_min': 1e-30, 'v_max': 1e-20,
                                   'c_label': 'white', 'bar_label': 'Relative AGN radiation'},
                  'v_phi': {'c_map': 'inferno_r', 'log': False, 'v_min': -150, 'v_max': 0,
                               'c_label': None, 'bar_label': r'$v_{\phi}$'},
                  'metallicity': {'c_map': 'twilight_shifted', 'log': False, 'v_min': -3.5, 'v_max': -1.5,
                                  'c_label': None, 'bar_label': '[Z/H]'},
                  'nH0': {'c_map': 'jet_r', 'log': True, 'v_min': 1e-5, 'v_max': 1,
                          'c_label': None, 'bar_label': 'n_H0'},
                  'energyDissipation': {'c_map': 'plasma', 'log': True, 'v_min': 1e-3, 'v_max': 1e2,
                                        'c_label': None, 'bar_label': 'E_dissipated'},
                  'angularMomentum': {'c_map': 'Spectral', 'log': False, 'v_min': 0, 'v_max': 4e8,
                                      'c_label': None, 'bar_label': 'L_z'},
                  'vx': {'c_map': 'Spectral', 'log': False, 'v_min': -150, 'v_max': 150,
                         'c_label': None, 'bar_label': r'$v_x$'},
                  'vy': {'c_map': 'Spectral', 'log': False, 'v_min': -150, 'v_max': 150,
                         'c_label': None, 'bar_label': r'$v_y$'},
                  'vz': {'c_map': 'Spectral', 'log': False, 'v_min': -150, 'v_max': 150,
                         'c_label': None, 'bar_label': r'$v_z$'},
                  'lx': {'c_map': 'twilight', 'log': True, 'v_min': 10 ** 0, 'v_max': 10 ** 4,
                         'c_label': None, 'bar_label': r'$\{\vec{r}\times\vec{v}\}_x$'},
                  'ly': {'c_map': 'twilight_shifted', 'log': True, 'v_min': 1e3, 'v_max': 1e4,
                         'c_label': None, 'bar_label': r'$\{\vec{r}\times\vec{v}\}_y$'},
                  'lz': {'c_map': 'twilight_shifted', 'log': True, 'v_min': 1e3, 'v_max': 1e4,
                         'c_label': None, 'bar_label': r'$\{\vec{r}\times\vec{v}\}_z$'},
                  'Lx': {'c_map': 'twilight_shifted', 'log': True, 'v_min': 1e5, 'v_max': 1e9,
                         'c_label': None, 'bar_label': r'$L_x$'},
                  'Ly': {'c_map': 'twilight_shifted', 'log': True, 'v_min': 10 ** 7.2, 'v_max': 10 ** 8.6,
                         'c_label': None, 'bar_label': r'$L_y$'},
                  'Lz': {'c_map': 'twilight_shifted', 'log': True, 'v_min': 10 ** 7.8, 'v_max': 10 ** 8.8,
                         'c_label': None, 'bar_label': r'$L_z$'},
                  'num_H0': {'c_map': 'viridis', 'log': True, 'v_min': 1e-7, 'v_max': 1e-2,
                             'c_label': None, 'bar_label': r'$n_{H0}$ ' + r'$cm^{-3}$'},
                  'num_H1': {'c_map': 'viridis', 'log': True, 'v_min': 1e-7, 'v_max': 1e-1,
                             'c_label': None, 'bar_label': r'$n_{H0}$ ' + r'$cm^{-3}$'},
                  'column_H0': {'c_map': 'viridis', 'log': False, 'v_min': 16, 'v_max': 19,
                                'c_label': None, 'bar_label': r'$\log n_{H0}$ ' + r'$cm^{-2}$'},
                  'E_diss': {'c_map': 'inferno', 'log': True, 'v_min': 1e-2, 'v_max': 1e2,
                             'c_label': None, 'bar_label': 'E_dissipation'}
                  }
