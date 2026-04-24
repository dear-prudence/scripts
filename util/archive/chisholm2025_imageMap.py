import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


def plot():
    # ---------------------------------------
    run = '09_18'
    halo = 'halo_08'
    dims = '100x400'
    # ---------------------------------------
    snapshots = [96, 110, 119, 127]
    # ---------------------------------------
    c1 = 'massDen'
    c2 = 'temperature'
    c3 = 'stars'
    # ---------------------------------------
    cmap_massDen = 'magma'
    cmap_temperature = 'Spectral'
    cmap_stars = 'bone'
    # ---------------------------------------
    e_i = 'x-z'
    # ---------------------------------------
    input_path_c1 = (f'/Users/dear-prudence/smorgasbord/images/{run}_{halo}/gas/'
                     f'{c1}/{run}_{halo}_gas_{c1}_{dims}kpc.npz')
    input_path_c2 = (f'/Users/dear-prudence/smorgasbord/images/{run}_{halo}/gas/'
                     f'{c2}/{run}_{halo}_gas_{c2}_{dims}kpc.npz')
    input_path_c3 = (f'/Users/dear-prudence/smorgasbord/images/{run}_{halo}/{c3}/'
                     f'massDen/{run}_{halo}_{c3}_massDen_{dims}kpc.npz')

    data_c1 = np.load(input_path_c1)
    data_c2 = np.load(input_path_c2)
    data_c3 = np.load(input_path_c3)

    image_c1 = data_c1[e_i].T
    image_c2 = data_c2[e_i].T
    image_c3 = data_c3[e_i].T
    image = [image_c1, image_c2, image_c3]

    redshifts = data_c1['redshifts']
    lookback_times = data_c1['lookback_times']

    c_bar_labels = [r'$\rho_{\rm{gas}}$ ' + r'$[$' + 'M' + r'$_{\odot}/$' + 'kpc' + r'$^{3}$' + r'$]$',
                    'Temperature ' + r'$[$' + 'K' + r'$]$',
                    r'$\Sigma_{\rm{stars}}$ ' + r'$[$' + 'M' + r'$_{\odot}/$' + 'kpc' + r'$^{2}$' + r'$]$']

    font_prop = formatting()
    plt.rcParams.update({'lines.dashed_pattern': (6, 6)})

    # lines.dashed_pattern: 3.7, 1.6

    last_idx = 127

    fig = plt.figure(figsize=(11.5, 15))
    # Adjust the layout *here* — before plotting anything
    # fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
    gs = fig.add_gridspec(4, 3, hspace=0.02, wspace=0.02)
    fig.tight_layout()
    axes = gs.subplots(sharex=True, sharey=True)
    axes = axes.flatten()
    # --------------------------------------
    # Module to define cosmetics (axes, background, labels, etc...)
    col_imshows = [None, None, None]  # One for each column

    # Loop over rows and columns
    for i in range(4):  # rows -> snapshots
        for j in range(3):  # cols -> images
            ax = axes[i * 3 + j]
            current_image = image[j]
            current_snap = snapshots[i]

            extent = (data_c1[e_i[0] + '_e'][0], data_c1[e_i[0] + '_e'][-1],
                      data_c1[e_i[2] + '_e'][0], data_c1[e_i[2] + '_e'][-1])
            if j == 0:
                v_min, v_max = 1e1, 1e6
                c_map = cmap_massDen
                # plt.rcParams.update({'xtick.color': 'k', 'ytick.color': 'k'})
            elif j == 1:
                v_min, v_max = 1e4, 1e6
                c_map = cmap_temperature
                # plt.rcParams.update({'xtick.color': 'k', 'ytick.color': 'k'})
            elif j == 2:
                v_min, v_max = 1e3, 1e8
                c_map = cmap_stars
                # plt.rcParams.update({'xtick.color': 'white', 'ytick.color': 'white'})
                background_color = plt.get_cmap(c_map)(0)
                ax.set_facecolor(background_color)

            ax.set_xticks([-200, -100, 0, 100, 200], labels=['', '-100', '0', '100', '200'], fontproperties=font_prop)
            ax.set_yticks([-200, -100, 0, 100, 200], labels=['', '-100', '0', '100', '200'], fontproperties=font_prop)
            ax.set_xticks(np.linspace(-200, 200, 21), minor=True)
            ax.set_yticks(np.linspace(-200, 200, 21), minor=True)

            im = ax.imshow(current_image[last_idx - current_snap], origin='lower', extent=extent, cmap=c_map,
                           norm=LogNorm(vmin=v_min, vmax=v_max))
            if i == 0:
                col_imshows[j] = im  # save the top image in each column for the colorbar

            virial_radii = [177.1, 167.6, 154.7, 145.8]  # kpc
            if j == 1:
                circle1 = plt.Circle((0, 0), virial_radii[i], fill=False, color='k', linestyle='dashed', lw=0.5)
                ax.add_patch(circle1)
                ax.text(virial_radii[i] / np.sqrt(2), -1 * virial_radii[i] / np.sqrt(2) - 20,
                        s=r'R$_{\text{vir}}$', fontproperties=font_prop)
            elif j == 2 and i == 0:
                circle2 = plt.Circle((0, 0), 50, fill=False, color='white', linestyle='dashed', lw=0.5)
                ax.add_patch(circle2)
                ax.text(-153, 45, s='LMC-analog', color='white', fontproperties=font_prop)

                circle2 = plt.Circle((139, -37), 25, fill=False, color='white',
                                     linestyle='dashed', lw=0.5)
                ax.add_patch(circle2)
                ax.text(70, -2, s='SMC-analog', color='white', fontproperties=font_prop)

                ax.arrow(125, -60, -15, -25, color='white', lw=0.5, head_width=3.0)
                ax.text(50, -110, s='approx. vel.', color='white', fontproperties=font_prop)

    col_positions = [0.13, 0.39, 0.65]  # x positions of each column
    width = 0.245
    height = 0.01
    for j in range(3):
        cax = fig.add_axes([col_positions[j], 0.9, width, height])
        cbar = fig.colorbar(col_imshows[j], cax=cax, orientation='horizontal')
        cbar.solids.set_rasterized(True)

        # cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(c_bar_labels[j], fontproperties=font_prop, labelpad=5)
        cbar.ax.xaxis.set_label_position('top')
        if j == 0:
            tick_positions = [1e1, 1e2, 1e3, 1e4, 1e5]
            tick_labels = ['10' + r'$^1$', '10' + r'$^2$', '10' + r'$^3$', '10' + r'$^4$', '10' + r'$^5$']
        elif j == 1:
            tick_positions = [1e4, 1e5, 1e6]
            tick_labels = ['10' + r'$^4$', '10' + r'$^5$', '10' + r'$^6$']
        elif j == 2:
            tick_positions = [1e4, 1e5, 1e6, 1e7, 1e8]
            tick_labels = ['10' + r'$^5$', '10' + r'$^5$', '10' + r'$^6$', '10' + r'$^7$', '10' + r'$^8$']

        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels, font_properties=font_prop)

    # After subplot loop
    fig.text(0.5, 0.08, 'x-coordinate ' + r'$[$' + 'kpc' + r'$]$',
             ha='center', va='center', fontproperties=font_prop)
    fig.text(0.08, 0.5, 'z-coordinate ' + r'$[$' + 'kpc' + r'$]$',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)

    dynamicalTime = - 1.33

    fig.text(0.91, 0.79, f'z={redshifts[last_idx - snapshots[0]]}, '
                         r't$_{\text{dym}}$ = '
             + f'-{round(lookback_times[last_idx - snapshots[0]] + dynamicalTime, 3)} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)
    fig.text(0.91, 0.60, f'z={redshifts[last_idx - snapshots[1]]}, '
                         r't$_{\text{dym}}$ = '
             + f'-{round(lookback_times[last_idx - snapshots[1]] + dynamicalTime, 3)} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)
    fig.text(0.91, 0.40, f'z={redshifts[last_idx - snapshots[2]]}, '
                         r't$_{\text{dym}}$ = '
             + f'-{round(lookback_times[last_idx - snapshots[2]] + dynamicalTime, 3)} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)
    fig.text(0.91, 0.2, f'z={redshifts[last_idx - snapshots[3]]}, '
                        r't$_{\text{dym}}$ = '
             + f'-{round(lookback_times[last_idx - snapshots[3]] + dynamicalTime, 3)} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)

    output_path = '/Users/ursa/smorgasbord/images/09_18_halo_08/'
    filename = (output_path + 'chisholm2025_imageMap.pdf')
    plt.savefig(filename, dpi=500, bbox_inches='tight')


def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='util/fonts/AVHersheySimplexMedium.otf', size=12)

    # plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams.update({  # "grid.linestyle": "--",  # Dashed grid lines
        'axes.unicode_minus': False,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        'mathtext.fontset': 'cm',
        # 'text.usetex': True,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        'xtick.major.size': 6,
        'xtick.major.width': 0.8,
        'xtick.minor.size': 3,
        'xtick.minor.width': 0.8,
        'ytick.major.size': 7,
        'ytick.major.width': 0.8,
        'ytick.minor.size': 4,
        'ytick.minor.width': 0.8,
        # 'xtick.labelweight': 'light',
        # 'ytick.labelweight': 'light'
        # "figure.subplot.wspace": 0.0,
        # "figure.constrained_layout.wspace": 0.0
        'legend.frameon': False,
    })
    return font_prop


# Image Map parameter settings
param_settings = {'numDen': {'c_map': 'cividis', 'log': True, 'v_min': 1, 'v_max': 1e1,
                             'c_label': 'white', 'bar_label': 'Relative Number Density'},
                  'massDen': {'c_map': 'Spectral_r', 'log': True, 'v_min': 1e1, 'v_max': 1e7,
                              'c_label': 'white', 'bar_label': r'$\rho[$ ' + 'M' + r'$_{\odot}$' + '/kpc' + r'$^3]$'},
                  # 'massDen': {'c_map': 'magma_r', 'log': True, 'v_min': 1e7, 'v_max': 1e10,
                  #             'c_label': 'white', 'bar_label': r'$\rho$ ' + r'$[M_{\odot}/ckpc^3]$'},
                  'temperature': {'c_map': 'RdBu_r', 'log': True, 'v_min': 1e4, 'v_max': 1e6,
                                  'c_label': 'black', 'bar_label': 'Temperature [K]'},
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
                  'num_H0': {'c_map': 'viridis', 'log': True, 'v_min': 1e-7, 'v_max': 1e-2,
                             'c_label': None, 'bar_label': r'$n_{H0}$ ' + r'$cm^{-3}$'},
                  'num_H1': {'c_map': 'viridis', 'log': True, 'v_min': 1e-7, 'v_max': 1e-1,
                             'c_label': None, 'bar_label': r'$n_{H0}$ ' + r'$cm^{-3}$'},
                  'column_H0': {'c_map': 'viridis', 'log': False, 'v_min': 16, 'v_max': 19,
                                'c_label': None, 'bar_label': r'$\log n_{H0}$ ' + r'$cm^{-2}$'},
                  'E_diss': {'c_map': 'inferno', 'log': True, 'v_min': 1e-2, 'v_max': 1e2,
                             'c_label': None, 'bar_label': 'E_dissipation'}
                  }
