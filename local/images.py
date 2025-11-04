import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm


class ImageMaps:
    def __init__(self, data, part_type, parameter, planes, snapshot, snapshots,
                 bool_centerPot, bool_centerH0, bool_centerBar, bool_centralBH, run,
                 dark_mode, output_path):
        self.data = data
        self.part_type = part_type
        self.parameter = parameter
        self.planes = planes
        self.snapshot = snapshot
        self.snapshots = snapshots if snapshots is not None else [95, 104, 111, 116, 121, 127]
        self.bool_centerPot = bool_centerPot
        self.bool_centerH0 = bool_centerH0
        self.bool_centerBar = bool_centerBar
        self.bool_centralBH = bool_centralBH
        self.run = run
        self.dark_mode = dark_mode
        self.output_path = output_path

    def plot_cover(self):
        settings = param_settings.get(self.part_type, {}).get(self.parameter, {})
        print(settings)
        e_i = 'x-y'
        starting_index = 127

        image = self.data[e_i]
        redshift = self.data['redshifts'][starting_index - self.snapshot]
        lookback_time = self.data['lookback_times'][starting_index - self.snapshot]

        extent = (self.data[e_i[0] + '_e'][0], self.data[e_i[0] + '_e'][-1],
                  self.data[e_i[2] + '_e'][0], self.data[e_i[2] + '_e'][-1])

        if self.dark_mode:
            plt.style.use('dark_background')

        fig = plt.figure(figsize=(12, 12))
        c_map = 'binary_r'

        fig.tight_layout()
        # X, Y = np.meshgrid(self.data[e_i[0] + '_e'], self.data[e_i[2] + '_e'])
        # Z = image[:, :, starting_index - self.snapshot].T
        print(settings.keys())
        # plt.pcolormesh(X, Y, Z, cmap=c_map, norm=LogNorm(vmin=1, vmax=1e6), shading='auto', edgecolor='none')
        plt.imshow(image[:, :, starting_index - self.snapshot].T, origin='lower', extent=extent, cmap=c_map,
                   vmin=settings['v_min'], vmax=settings['v_max'])
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        plt.xticks([])
        plt.yticks([])

        # work on generalizing file name
        filename = (self.output_path + 'coverImage_massDen_100x400kpc.pdf')
        plt.savefig(filename, dpi=800)
        plt.close()

    def plot_frames(self):
        settings = param_settings.get(self.part_type, {}).get(self.parameter, {})
        e_i = self.planes if self.planes is not None else ['x-y', 'x-z']
        font_prop = formatting()

        print('building image maps from .npz file')

        background_color = plt.get_cmap(settings['c_map'])(0)

        image_l = self.data[e_i[0]].T
        image_r = self.data[e_i[1]].T
        redshifts = self.data['redshifts']
        lookback_times = self.data['lookback_times']

        for i, (frameL, frameR, z, t) in enumerate(zip(image_l[::-1], image_r[::-1],
                                                       redshifts[::-1], lookback_times[::-1]), start=0):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # --------------------------------------
            # Module to define cosmetics (axes, background, labels, etc...)
            fig.tight_layout()
            for ax, j in zip([ax1, ax2], [0, 1]):
                ax.set_facecolor(background_color)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f'{e_i[j][0]}-coordinate ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
                ax.set_ylabel(f'{e_i[j][2]}-coordinate ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
            fig.suptitle('$z = $' + '{:.{}f}'.format(z, 3)
                         + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(t), 2), 2) + ' Gyr'
                         , x=0.5, y=0.01, ha='center', va='bottom', weight='bold', c='k')

            if self.bool_centerPot:
                # only works with e_i = ['x_y', 'x_z']
                ax1.scatter(0, 0, marker='s', c='w', s=10)
                ax2.scatter(0, 0, marker='s', c='w', s=10)
            if self.bool_centerH0:
                # only works with e_i = ['x_y', 'x_z']
                ax1.scatter(self.data['center_h0'][len(redshifts) - 1 - i, 0],
                            self.data['center_h0'][len(redshifts) - 1 - i, 1], marker='+', c='tab:blue')
                ax2.scatter(self.data['center_h0'][len(redshifts) - 1 - i, 0],
                            self.data['center_h0'][len(redshifts) - 1 - i, 2], marker='+', c='tab:blue')
            if self.bool_centerBar:
                # only works with e_i = ['x_y', 'x_z']
                ax1.scatter(self.data['center_bar'][len(redshifts) - 1 - i, 0],
                            self.data['center_bar'][len(redshifts) - 1 - i, 1], marker='*', c='y')
                ax2.scatter(self.data['center_bar'][len(redshifts) - 1 - i, 0],
                            self.data['center_bar'][len(redshifts) - 1 - i, 2], marker='*', c='y')
            if self.bool_centralBH:
                # only works with e_i = ['x_y', 'x_z']
                ax1.scatter(self.data['central_BH'][len(redshifts) - 1 - i, 0],
                            self.data['central_BH'][len(redshifts) - 1 - i, 1], marker='x', c='k')
                ax2.scatter(self.data['central_BH'][len(redshifts) - 1 - i, 0],
                            self.data['central_BH'][len(redshifts) - 1 - i, 2], marker='x', c='k')

            # --------------------------------------
            extent_l = [self.data[e_i[0][0] + '_e'][0], self.data[e_i[0][0] + '_e'][-1],
                        self.data[e_i[0][2] + '_e'][0], self.data[e_i[0][2] + '_e'][-1]]
            extent_r = [self.data[e_i[1][0] + '_e'][0], self.data[e_i[1][0] + '_e'][-1],
                        self.data[e_i[1][2] + '_e'][0], self.data[e_i[1][2] + '_e'][-1]]
            # --------------------------------------
            if settings['log'] is True:
                im1 = ax1.imshow(frameL, origin='lower', extent=extent_l, cmap=settings['c_map'],
                                 norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']),
                                 rasterized=True)
                im2 = ax2.imshow(frameR, origin='lower', extent=extent_r, cmap=settings['c_map'],
                                 norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']),
                                 rasterized=True)
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
            cbar = fig.colorbar(im1, cax=cax)
            cbar.ax.set_ylabel(settings['bar_label'], fontproperties=font_prop)

            final_idx = 307 if self.run == '09_18_lastgigyear' else 127
            starting_index = final_idx - len(self.data['x-y'][0, 0, :]) + 1
            filename = (self.output_path + f'snap_{i + starting_index:03d}.png')
            plt.savefig(filename, dpi=240,
                        facecolor=('k' if self.dark_mode else 'white'))
            plt.close()

    def plot_panels(self):
        settings = param_settings.get(self.part_type, {}).get(self.parameter, {})
        e_i = ['x-y']

        background_color = plt.get_cmap(settings['c_map'])(0)

        image = self.data[e_i[0]].T
        redshifts = self.data['redshifts']
        lookback_times = self.data['lookback_times']

        font_prop = formatting()

        last_idx = 127

        if self.dark_mode:
            plt.style.use('dark_background')

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.02, wspace=0.02)
        fig.tight_layout()
        axes = gs.subplots(sharex=True, sharey=True)
        axes = axes.flatten()
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        # fig.tight_layout()

        for ax, snap in zip(axes, self.snapshots):
            length = 40
            ax.set_xticks(np.array([-0.5, -0.25, 0, 0.25, 0.5]) * length)
            ax.set_yticks(np.array([-0.5, -0.25, 0, 0.25, 0.5]) * length)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            ax.set_facecolor(background_color)

            extent = (self.data[e_i[0][0] + '_e'][0], self.data[e_i[0][0] + '_e'][-1],
                      self.data[e_i[0][2] + '_e'][0], self.data[e_i[0][2] + '_e'][-1])
            if settings['log'] is True:
                im = ax.imshow(image[last_idx - snap], origin='lower', extent=extent, cmap=settings['c_map'],
                               norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']))
            else:
                im = ax.imshow(image[last_idx - snap], origin='lower', extent=extent, cmap=settings['c_map'],
                               vmin=settings['v_min'], vmax=settings['v_max'])
            ax.text(12.5, -19.5, s='$z = $' + '{:.{}f}'.format(redshifts[last_idx - snap], 3),
                    c='white', fontsize='small')

            if self.bool_centralBH:
                # only works with e_i = ['x_y', 'x_z']
                ax.scatter(self.data['central_BH'][127 - snap, 0],
                           self.data['central_BH'][127 - snap, 1], marker='*', s=10)

        # needed to recreate figure in paper
        # Create extra white space to the right of the right subplot
        adjustment_param = 0.09
        fig.subplots_adjust(left=adjustment_param, right=(1 - adjustment_param),
                            bottom=adjustment_param, top=(1 - adjustment_param))
        fig.subplots_adjust(right=(1 - adjustment_param))
        cax = fig.add_axes([0.92, 0.09, 0.02, 0.82])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(settings['bar_label'], fontproperties=font_prop, labelpad=0)

        filename = (self.output_path + f'panels_{self.parameter}_snaps{self.snapshots[0]}-{self.snapshots[-1]}.pdf')
        plt.savefig(filename, dpi=240, bbox_inches='tight')
        plt.show()


def dispatch_plot(plot_class, plot_type, input_path, output_path, partType=None, parameter=None, planes=None,
                  snapshot=None, snapshots=None, dark_mode=False,
                  bool_centerPot=False, bool_centerH0=False, bool_centerBar=False, bool_centralBH=False, run=None):
    data = np.load(input_path, allow_pickle=True)

    if plot_class == 'imageMaps':
        image_map = ImageMaps(data, partType, parameter, planes, snapshot, snapshots,
                              bool_centerPot, bool_centerH0, bool_centerBar, bool_centralBH,
                              run, dark_mode, output_path)
        dispatcher = {
            'cover': image_map.plot_cover,
            'frames': image_map.plot_frames,
            'panels': image_map.plot_panels,
        }
    else:
        print(f'Error: {plot_class} is an invalid plot class!')
        exit(1)

    return dispatcher[plot_type]()


# -------------------------------------------------------------
def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='scripts/local/fonts/AVHersheySimplexMedium.otf', size=12)

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
param_settings = {'gas': {
    'numDen': {'c_map': 'cividis', 'log': True, 'v_min': 1, 'v_max': 1e1,
               'c_label': 'white', 'bar_label': 'Relative Number Density'},
    'massDen': {'c_map': 'viridis', 'log': True, 'v_min': 1e1, 'v_max': 1e6,
                'c_label': 'white', 'bar_label': r'$\rho[$ ' + 'M' + r'$_{\odot}$' + '/kpc' + r'$^3]$'},
    'temperature': {'c_map': 'Spectral_r', 'log': True, 'v_min': 3e3, 'v_max': 3e5,
                    'c_label': 'black', 'bar_label': 'Temperature ' + r'$[$' + 'K' + r'$]$'},
    'agn_radiation': {'c_map': 'gist_heat', 'log': True, 'v_min': 1e-9, 'v_max': 1e-3,
                      'c_label': 'white', 'bar_label': 'Relative AGN radiation'},
    'cooling_rate': {'c_map': 'gist_heat', 'log': True, 'v_min': 1e-30, 'v_max': 1e-20,
                     'c_label': 'white', 'bar_label': 'Relative AGN radiation'},
    'v_phi': {'c_map': 'inferno_r', 'log': False, 'v_min': -150, 'v_max': 0,
              'c_label': None, 'bar_label': r'$v_{\phi}$'},
    'metallicity': {'c_map': 'Spectral', 'log': False, 'v_min': -2.5, 'v_max': -1,
                    'c_label': None, 'bar_label': r'$[$' + 'Z/H' + r'$]$'},
    'energyDissipation': {'c_map': 'plasma', 'log': True, 'v_min': 1e-3, 'v_max': 1e2,
                          'c_label': None, 'bar_label': 'E_dissipated'},
    'angularMomentum': {'c_map': 'Spectral', 'log': False, 'v_min': 0, 'v_max': 4e8,
                        'c_label': None, 'bar_label': 'L_z'},
    'v_x': {'c_map': 'turbo', 'log': False, 'v_min': -150, 'v_max': 50,
            'c_label': None, 'bar_label': r'$v_x$'},
    'v_y': {'c_map': 'turbo', 'log': False, 'v_min': -50, 'v_max': 100,
            'c_label': None, 'bar_label': r'$v_y$'},
    'v_z': {'c_map': 'turbo', 'log': False, 'v_min': -150, 'v_max': 50,
            'c_label': None, 'bar_label': r'$v_z$'},
    'Lx': {'c_map': 'Spectral', 'log': True, 'v_min': 1e6, 'v_max': 1e8,
           'c_label': None, 'bar_label': r'$L_x$'},
    'Ly': {'c_map': 'Spectral', 'log': True, 'v_min': 1e7, 'v_max': 1e9,
           'c_label': None, 'bar_label': r'$L_y$'},
    'Lz': {'c_map': 'Spectral', 'log': True, 'v_min': 10 ** 8, 'v_max': 1e9,
           'c_label': None, 'bar_label': r'$L_z$'},
    'num_H0': {'c_map': 'Spectral', 'log': True, 'v_min': 1e-6, 'v_max': 1e-2,
               'c_label': None, 'bar_label': r'$n_{H0}$ ' + r'$cm^{-3}$'},
    'num_H1': {'c_map': 'viridis', 'log': True, 'v_min': 1e-7, 'v_max': 1e-1,
               'c_label': None, 'bar_label': r'$n_{H0}$ ' + r'$cm^{-3}$'},
    'column_H0': {'c_map': 'Blues', 'log': True, 'v_min': 1e16, 'v_max': 1e19,
                  'c_label': None, 'bar_label': r'$\log n_{H0}$ ' + r'$cm^{-2}$'},
    'E_diss': {'c_map': 'inferno', 'log': True, 'v_min': 1e-2, 'v_max': 1e2,
               'c_label': None, 'bar_label': 'E_dissipation'}
},
    'stars': {'massDen': {'c_map': 'magma', 'log': True, 'v_min': 1e7, 'v_max': 1e9,
                          'c_label': 'white', 'bar_label': r'$\Sigma[$ ' + 'M' + r'$_{\odot}$' + '/kpc' + r'$^2]$'},
              'surfaceBrightness': {'c_map': 'bone_r', 'log': False, 'v_min': -16, 'v_max': -13,
                                    'c_label': 'white', 'bar_label': r'$\Sigma[$ ' + 'mag/kpc' + r'$^2]$'},
              'metallicity': {'c_map': 'Spectral', 'log': False, 'v_min': -2.5, 'v_max': -1.5,
                              'c_label': None, 'bar_label': r'$[$' + 'Z/H' + r'$]$'},
              'potential': {'c_map': 'magma', 'log': True, 'v_min': 1e5, 'v_max': 1e9,
                              'c_label': None, 'bar_label': r'$[$' + 'Potential' + r'$]$'},
              },
    'dm': {'massDen': {'c_map': 'viridis', 'log': True, 'v_min': 1e5, 'v_max': 1e9,
                       'c_label': 'white', 'bar_label': r'$\Sigma[$ ' + 'M' + r'$_{\odot}$' + '/kpc' + r'$^2]$'}}
}
