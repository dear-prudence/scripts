import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_imageMap_frames(parameter, input_path, output_path, scale, planes=None):
    # Current plotting settings
    # ---------------------------------
    interpolation = None
    # ---------------------------------
    # Load the .npz file
    data = np.load(input_path)
    # Unpack all items in the .npz file as variables
    locals().update(data)

    settings = param_settings.get(parameter, {})
    e_i = ['x-y', 'x-z'] if planes is None else planes

    background_color = plt.get_cmap(settings['c_map'])(0)
    s_label = str(int(scale[1] / 4)) + ' kpc'

    image_l = data[e_i[0]].T
    image_r = data[e_i[1]].T
    # image_l = data[e_i[0]]
    # image_r = data[e_i[1]]
    redshifts = data['redshifts']
    print(len(redshifts))
    lookback_times = data['lookback_times']

    for i, (frameL, frameR, z, t) in enumerate(zip(image_l[::-1], image_r[::-1],
                                                   redshifts[::-1], lookback_times[::-1]), start=0):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        # ax1.grid(True, alpha=0.3, color='k', linestyle='dashed', lw=0.5)
        # ax2.grid(True, alpha=0.3, color='k', linestyle='dashed', lw=0.5)
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        ax1.set_facecolor(background_color)
        ax2.set_facecolor(background_color)
        fig.tight_layout()
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # ax1.xaxis.set_ticklabels([])
        # ax1.yaxis.set_ticklabels([])
        # ax2.xaxis.set_ticklabels([])
        # ax2.yaxis.set_ticklabels([])
        fig.suptitle('$z = $' + '{:.{}f}'.format(z, 3)
                     + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(t), 2), 2) + ' Gyr'
                     , x=0.5, y=0.01, ha='center', va='bottom', weight='bold', c='k')
        # --------------------------------------
        extent_l = [data[e_i[0][0] + '_e'][0], data[e_i[0][0] + '_e'][-1],
                    data[e_i[0][2] + '_e'][0], data[e_i[0][2] + '_e'][-1]]
        extent_r = [data[e_i[1][0] + '_e'][0], data[e_i[1][0] + '_e'][-1],
                    data[e_i[1][2] + '_e'][0], data[e_i[1][2] + '_e'][-1]]
        # --------------------------------------
        if settings['log'] is True:
            im1 = ax1.imshow(frameL, origin='lower', extent=extent_l, cmap=settings['c_map'],
                             norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']), interpolation=interpolation,
                             rasterized=True)
            im2 = ax2.imshow(frameR, origin='lower', extent=extent_r, cmap=settings['c_map'],
                             norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']), interpolation=interpolation,
                             rasterized=True)
        else:
            im1 = ax1.imshow(frameL, origin='lower', extent=extent_l, cmap=settings['c_map'],
                             vmin=settings['v_min'], vmax=settings['v_max'], interpolation=interpolation)
            im2 = ax2.imshow(frameR, origin='lower', extent=extent_r, cmap=settings['c_map'],
                             vmin=settings['v_min'], vmax=settings['v_max'], interpolation=interpolation)
        # --------------------------------------
        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
        cbar = fig.colorbar(im1, cax=cax, label=settings['bar_label'])

        # ax1.plot([x_e_l[-220], x_e_l[-20]], [y_e_l[20], y_e_l[20]], c=c_label)
        # ax1.text(data[e_i[0][0] + '_e'][7], data[e_i[0][2] + '_e'][7], s=r'$\it{face-on}$', c='white',
        #          fontsize='small')
        # ax2.plot([x_e_r[-220], x_e_r[-20]], [y_e_r[20], y_e_r[20]], c=c_label)
        # ax2.text(data[e_i[1][0] + '_e'][-45], data[e_i[1][2] + '_e'][7], s=r'$\it{edge-on}$', c='white',
        #          fontsize='small')

        ax1.set_facecolor(background_color)
        ax2.set_facecolor(background_color)

        final_idx = 308 if 'lastgigyear' in input_path else 127
        starting_index = final_idx - len(data['x-y'][0, 0, :])
        filename = (output_path + f'snap_{i + starting_index:03d}.png')
        plt.savefig(filename, dpi=150)
        plt.close()


param_settings = {'numDen': {'c_map': 'cividis', 'log': True, 'v_min': 1, 'v_max': 1e1,
                             'c_label': 'white', 'bar_label': 'Relative Number Density'},
                  'massDen': {'c_map': 'magma_r', 'log': True, 'v_min': 1e1, 'v_max': 1e7,
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