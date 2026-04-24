import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm

from scripts.util.utils import add_ticks


class Kinematics:
    def __init__(self, data, projection, parameter, smoothing, dark_mode, output_path):
        self.data = data
        self.dark_mode = dark_mode
        self.output_path = output_path
        self.projection = projection
        self.parameter = parameter
        self.smoothing = smoothing

    def accretionHistory(self):
        font_prop = formatting()

        bool_BHmdot = True
        bool_BHgas = False

        if '09_18_lastgigyear' in self.output_path:
            times = self.data['lookback_times']
            x_lim = [0, 1.5]
            x_label = 'Lookback Time ' + r'$[$' + 'Gyr' + r'$]$'
        else:
            times = self.data['redshifts']
            x_lim = [0, 1.5]
            x_label = 'z'

        fig, ax = plt.subplots(figsize=(5, 5))

        if bool_BHmdot:
            y_lim = [0, 0.5]
            ax.plot(times, self.data['M_dot'], label='BH growth rate',
                    linewidth=0.8, color='tab:blue')
        elif bool_BHgas:
            y_lim = [4, 7]
            ax.plot(times, np.log10(self.data['M_BHgas']), label='BH growth rate',
                    linewidth=0.8, color='tab:blue')
        else:
            y_lim = [4, 12]

            ax.plot(times, np.log10(self.data['M_halo']), label='halo mass',
                    linewidth=0.8, color='tab:blue')
            ax.plot(times, np.log10(self.data['M_dm']), label='dm mass',
                    linewidth=0.8, color='tab:blue', linestyle='dashed', dashes=(6, 6))
            ax.plot(times, np.log10(self.data['M_gas']), label='gas mass',
                    linewidth=0.8, color='tab:green')
            ax.plot(times, np.log10(self.data['M_star']), label='stellar mass',
                    linewidth=0.8, color='tab:red')
            ax.plot(times, np.log10(self.data['M_bh']), label='bh mass',
                    linewidth=0.8, color='k')

        # Other formatting stuff
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])

        x_ticksMajor = np.arange(0, 13, 0.5)
        x_ticksMinor = np.arange(0, 13, 0.1)
        x_ticks = x_ticksMajor[(x_ticksMajor >= x_lim[0]) & (x_ticksMajor <= x_lim[1])]
        plt.xticks(x_ticks, [str(x).replace("-", '-') for x in x_ticks], fontproperties=font_prop)
        plt.xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)

        y_ticksMajor = np.arange(0, 13, 1)
        y_ticksMinor = np.arange(0, 13, 0.2)
        y_ticks = y_ticksMajor[(y_ticksMajor >= y_lim[0]) & (y_ticksMajor <= y_lim[1])]
        plt.yticks(y_ticks, [str(y).replace("-", '-') for y in y_ticks], fontproperties=font_prop)
        plt.yticks(y_ticksMinor[(y_ticksMinor >= y_lim[0]) & (y_ticksMinor <= y_lim[1])], minor=True)

        plt.gca().invert_xaxis()

        # Set font weight for tick labels
        ax.set_xlabel(x_label, fontproperties=font_prop)
        if bool_BHmdot:
            ax.set_ylabel('M dot ' + r'$[$' + 'solar masses / yr' + r'$]$', fontproperties=font_prop)
        elif bool_BHgas:
            ax.set_ylabel('log Mass ' + r'$[$' + 'log solar masses' + r'$]$', fontproperties=font_prop)
        else:
            ax.set_ylabel('log Mass ' + r'$[$' + 'log solar masses' + r'$]$', fontproperties=font_prop)

        # plt.text(8, 5, s=r'halo 16; M $\sim$ 1.5e11', fontproperties=font_prop)

        plt.legend(loc='lower left', prop=font_prop)
        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()

    def rotCurve(self):
        G = 4.300917270e-6  # kpc / M_solar (km/s)2
        font_prop = formatting()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.data['r'], np.sqrt(G * self.data['M_tot'] / self.data['r']), label='total',
                linewidth=0.8, color='tab:green')
        ax.plot(self.data['r'], np.sqrt(G * self.data['M_dm'] / self.data['r']), label='dm',
                linewidth=0.8, color='tab:blue')
        ax.plot(self.data['r'], np.sqrt(G * self.data['M_gas'] / self.data['r']), label='gas',
                linewidth=0.8, color='tab:purple')
        ax.plot(self.data['r'], np.sqrt(G * self.data['M_star'] / self.data['r']), label='stars',
                linewidth=0.8, color='tab:red')

        x_lim = [0.20, 200]
        y_lim = [0, 160]
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])
        # aspect_ratio = np.log10(abs(x_lim[1] - x_lim[0])) / abs(y_lim[1] - y_lim[0])
        # ax.set_aspect(aspect_ratio)
        plt.xscale('log')

        plt.xticks([1, 10, 100], labels=['1.0', '10', '100'], fontproperties=font_prop)
        # plt.xticks(np.linspace(x_lim[0], x_lim[1], 31), minor=True)
        plt.yticks(np.linspace(y_lim[0], y_lim[1], 9), fontproperties=font_prop)
        plt.yticks(np.linspace(y_lim[0], y_lim[1], 17), minor=True)
        # plt.gca().invert_xaxis()
        # Set font weight for tick labels
        ax.set_xlabel('Radius ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
        ax.set_ylabel(r'V$_{\text{c}}$(R) ' + r'$[$' + 'km/s' + r'$]$', fontproperties=font_prop)

        # plt.text(8, 5, s=r'halo 16; M $\sim$ 1.5e11', fontproperties=font_prop)

        plt.legend(loc='upper left', prop=font_prop)
        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()

    def orbits(self):
        font_prop = formatting()
        plt.rcParams.update({'xtick.top': False})

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(-1 * self.data['hestia_times'], self.data['hestia_distances'],
                c='tab:blue', linestyle='solid', lw=1.2, label='halo41-halo01 distance (HESTIA)')

        # Other formatting stuff
        x_lim = [-2, 0]
        y_lim = [0, 40]

        from scripts.util.utils import add_ticks

        add_ticks(ax,
                  np.arange(-4, 0, 0.5), np.arange(-4, 0, 0.1), x_lim,
                  np.arange(0, 800, 5), np.arange(0, 800, 1), y_lim,
                  x_label='Lookback Time ' + r'$[$' + 'Gyr' + r'$]$', y_label='Distance ' + r'$[$' + 'kpc' + r'$]$')

        # halo_01 R_vir line
        ax.plot([x_lim[0], x_lim[1]], [319, 319], linestyle='dashed', c='tab:blue', linewidth=0.8, dashes=(6, 6),
                alpha=0.5)
        ax.text(-1.1, 325, s=r'"MW" virial radius', c='tab:blue', fontproperties=font_prop)
        # lucchini MW R_vir line
        ax.plot([x_lim[0], x_lim[1]], [206, 206], linestyle='dashed', c='tab:green', linewidth=0.8, dashes=(6, 6),
                alpha=0.5)
        ax.text(-1.8, 215, s=r'"MW" virial radius', c='tab:green', fontproperties=font_prop)

        ax.axvspan(-0.3, -0.1, color='tab:blue', linestyle='', alpha=0.2)
        ax.plot([-0.3, -0.3], [y_lim[0], y_lim[1]],
                linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
        ax.plot([-0.1, -0.1], [y_lim[0], y_lim[1]],
                linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
        ax.text(-0.4, 330, s='sampled time span', c='tab:blue', rotation='vertical', font_properties=font_prop)

        plt.legend(prop=font_prop)

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()

    def bhSloshing(self):
        from scipy.interpolate import interp1d
        from matplotlib.collections import LineCollection
        # -------------------------------
        t_dyn0 = -1.517  # zero dynamical time, snap 119
        kind_interpolator = 'cubic'  # or 'linear', 'quadratic', 'cubic'
        show_satellite = False

        if '09_18_lastgigyear' in self.output_path:
            domain = [0, 1.5]  # t in Gyr
        else:
            domain = [-2.5, 1.5]  # t in Gyr

        if self.parameter == 'energy':
            range_ = [3.5, 5.0]  # y_lim in log(km^2 / s^2)
        elif self.parameter == 'norm':
            range_ = [0, 3.5]  # y_lim in kpc
        # -------------------------------
        font_prop = formatting()

        # checks to see if energy plot is requested for an 09_18_lastgigyear halo, 'Potential' not given
        print(self.output_path)
        # if self.projection == '1-dim' and self.parameter == 'energy' and '09_18_lastgigyear' in self.output_path:
        #     print('Error: cannot plot 1-dim energy for 09_18_lastgigyear (unphysical potential)!')
        #     exit(1)

        # *energy* or *spatial norm* of bh as a function of time
        if self.projection == '1-dim':
            lookback_times = self.data['lookback_times']
            norms, energies = self.data['bh_norms'], self.data['bh_energies']
            fig, ax = plt.subplots(figsize=(5, 5))

            if 'halo_08' in self.output_path:  # if this the LMC-SMC system from chisholm+2025
                n, t_dyn0 = 1, 1.517  # zero dynamical time, snap 119
            else:
                n, t_dyn0 = 0, 0

            try:
                if show_satellite:
                    satelliteDict_code = {'halo_08': 'smc', 'halo_10': 'halo_03'}
                    satelliteDict_name = {'halo_08': 'SMC', 'halo_10': 'MW'}
                    haloIdx_inOutputPath = self.output_path.index('halo')
                    halo = self.output_path[haloIdx_inOutputPath:haloIdx_inOutputPath + 7]

                    base_path = f'/Users/ursa/smorgasbord/kinematics/09_18_{halo}/orbits'
                    sat_orbits = np.load(f'{base_path}/orbitalDistance_09_18_{halo}-{satelliteDict_code[halo]}.npz')

                    # Create second y-axis on the right
                    ax2 = ax.twinx()
                    ax2.plot(-1 * sat_orbits['hestia_times'] - t_dyn0, sat_orbits['hestia_distances'],
                             '--', label=f'{satelliteDict_name[halo]}-analog orbit', c='tab:blue', dashes=(6, 6),
                             lw=0.8)
                    ax2.set_ylabel('', color='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')

                    if satelliteDict_name[halo] == 'SMC':
                        ax2.set_ylim([0, 125])
                        ax2.set_yticks([0, 25, 50, 75, 100, 125],
                                       labels=['0', '25', '50', '75', '100', '125'], fontproperties=font_prop)
                        ax2.set_yticks(np.arange(0, 125, 5), minor=True)
                    elif satelliteDict_name[halo] == 'MW':
                        ax2.set_ylim([0, 300])
                        ax2.set_yticks([0, 100, 200, 300, 400, 500],
                                       labels=['0', '100', '200', '300', '400', '500'], fontproperties=font_prop)
                        ax2.set_yticks(np.arange(0, 500, 20), minor=True)
                    else:
                        print('Error: invalid satellite requested!')
                        exit(1)
                    ax2.set_ylabel('displacement from halo center ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)

            except KeyError:  # if no relevant satellite is present (i.e. has an entry in satelliteDict{})
                pass

            """            x_ticks = np.array([-3.5, -3.0, -2.5, -2.0, -1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5])
            x_labels = np.array(['-3.5', '-3.0', '-2.5', '-2.0', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5'])
            y_ticks = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ])
            y_labels = np.array(['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'])

            ax.set_xlim(domain)
            plt.gca().invert_xaxis()
            mask = (x_ticks >= domain[0]) & (x_ticks <= domain[1])
            ax.set_xticks(x_ticks[mask], labels=x_labels[mask], fontproperties=font_prop)
            ax.set_xticks(np.arange(domain[0], domain[1], 0.1), minor=True)
            ax.set_xlabel('lookback time ' + r'$[$' + 'Gyr' + r'$]$', fontproperties=font_prop)

            range_ = [0, 8]
            ax.set_ylim(range_)
            mask = (y_ticks >= range_[0]) & (y_ticks <= range_[1])
            ax.set_yticks(y_ticks[mask], labels=[str(y_labels[mask])], fontproperties=font_prop)
            ax.set_yticks(np.arange(range_[0], range_[1], 0.2), minor=True)"""

            range_ = [0, 10]
            x_ticksMajor = np.arange(-3.5, 2, 0.5)
            x_ticksMinor = np.arange(-3.5, 1.5, 0.1)
            x_ticks = x_ticksMajor[(x_ticksMajor >= domain[0]) & (x_ticksMajor <= domain[1])]
            y_ticksMajor = np.arange(0, 10, 2)
            y_ticksMinor = np.arange(0, 10, 0.2)
            y_ticks = y_ticksMajor[(y_ticksMajor >= range_[0]) & (y_ticksMajor <= range_[1])]

            ax.set_xlim(domain)
            ax.set_xticks(x_ticks, [str(x) for x in x_ticks],
                          fontproperties=font_prop)
            ax.set_xticks(x_ticksMinor[(x_ticksMinor >= domain[0]) & (x_ticksMinor <= domain[1])], minor=True)
            ax.set_ylim(range_)
            ax.set_yticks(y_ticks, [str(y) for y in y_ticks],
                          fontproperties=font_prop)
            ax.set_yticks(y_ticksMinor[(y_ticksMinor >= range_[0]) & (y_ticksMinor <= range_[1])], minor=True)

            if self.parameter == 'energy':
                # ax.set_ylabel(r'log($\epsilon$-$\Phi_{\text{min}}$  km$^{-2}$ s$^{2}$)', fontproperties=font_prop)
                ax.set_ylim([-100, 0])

                if not self.smoothing:
                    print(self.data['lookback_times'] - t_dyn0)
                    ax.plot(self.data['lookback_times'] - t_dyn0, -1 * np.sqrt(-1 * self.data['bh_energies']),
                            linestyle='-', c='k', lw=0.8,
                            label='total mechanical energy\nper unit mass of\nLMC-analog central BH', )
                else:
                    print(f'Error: absent smoothing functionality for {halo}/{self.projection}/{self.parameter}!')
                    exit(1)

            elif self.parameter == 'norm':
                ax.set_ylabel(r'distance from most bound particle (kpc)', fontproperties=font_prop)

                if not self.smoothing:
                    ax.errorbar(self.data['lookback_times'] - t_dyn0, self.data['bh_norms'],
                                yerr=self.data['epsilon'],
                                ecolor='k', elinewidth=0.8, capsize=1.5, linestyle='',
                                label='euclidian distance from\nmost bound particle of\nLMC-analog central BH\n'
                                      r'$\sigma$ = $\epsilon_{\text{soft}}$')
                else:
                    h_interp = interp1d(lookback_times, norms, kind=kind_interpolator)
                    h_time_smooth = np.linspace(min(lookback_times), max(lookback_times), 1000)
                    h_smooth = h_interp(h_time_smooth)
                    h_smooth_errUp = interp1d(lookback_times, norms + self.data['epsilon'],
                                              kind=kind_interpolator)(h_time_smooth)
                    h_smooth_errDown = interp1d(lookback_times, norms - self.data['epsilon'],
                                                kind=kind_interpolator)(h_time_smooth)
                    ax.plot(h_time_smooth - t_dyn0, h_smooth, '-', c='k', lw=0.8,
                            label='LMC-analog central BH')
                    ax.fill_between(h_time_smooth - t_dyn0, h_smooth_errUp, h_smooth_errDown,
                                    alpha=0.5, color='k')
                    plt.gca().invert_xaxis()

            elif self.parameter == 'velocity':
                ax.set_ylabel(r'|v|  $[$km/s$]$', fontproperties=font_prop)
                ax.set_ylim([-40, 40])
                # ax.set_yticks([0, 10, 20, 30, 40, 50, 60], labels=['', '10', '20', '30', '40', '50', '60'],
                #               fontproperties=font_prop)
                # ax.set_yticks(np.arange(0, 60, 2), minor=True)

                if not self.smoothing:
                    print(self.data['lookback_times'] - t_dyn0)
                    ax.plot(self.data['lookback_times'] - t_dyn0, self.data['bh_velocity'],
                            linestyle='-', c='k', lw=0.8)
                else:
                    print(f'Error: absent smoothing functionality for {halo}/{self.projection}/{self.parameter}!')
                    exit(1)

            else:
                print('Error: invalid given parameter!')
                exit(1)

            # aspect_ratio = abs(domain[1] - domain[0]) / abs(range_[1] - range_[0])
            # ax.set_aspect(aspect_ratio)

            # Combine legends from both axes
            if show_satellite and halo in satelliteDict_code:
                lines_1, labels_1 = ax.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', prop=font_prop)
            else:
                ax.legend(loc='upper right', prop=font_prop)

        if self.projection == '2-dim':
            # sloshing is primarily in the x-y plane (for halo_08)
            fig, ax = plt.subplots(figsize=(8, 7))

            if self.parameter == 'norm':
                if self.smoothing:
                    if 'halo_08' in self.output_path:  # if this the LMC-SMC system from chisholm+2025
                        n, t_dyn0 = 1, 1.517  # zero dynamical time, snap 119
                        cbar_label = r'dynamical time $[$Gyr$]$'
                    else:
                        n, t_dyn0 = 0, 0
                        cbar_label = r'lookback time $[$Gyr$]$'
                    fx = interp1d(-1 ** n * self.data['lookback_times'] - t_dyn0,
                                  self.data['bh_coords'][:, 0], kind=kind_interpolator,
                                  fill_value=np.array([0.]), bounds_error=False)
                    fy = interp1d(-1 ** n * self.data['lookback_times'] - t_dyn0,
                                  self.data['bh_coords'][:, 1], kind=kind_interpolator,
                                  fill_value=np.array([0.]), bounds_error=False)
                    # t_smooth = np.linspace(min(-1 ** n * self.data['lookback_times'] - t_dyn0),
                    #                        max(-1 ** n * self.data['lookback_times'] - t_dyn0), 9999)
                    t_smooth = np.linspace(min(-1 ** n * self.data['lookback_times'] - t_dyn0), 0, 9999)
                    x_smooth, y_smooth = fx(t_smooth), fy(t_smooth)

                    # Create line segments [(x0, y0), (x1, y1)], ...
                    points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Create the LineCollection
                    norm = plt.Normalize(t_smooth.min(), t_smooth.max())
                    lc = LineCollection(segments, cmap='Spectral', norm=norm, lw=0.8)
                    ax.add_collection(lc)
                    lc.set_array(t_smooth)  # THIS is the missing step

                    ax.add_collection(lc)
                    # cbar = plt.colorbar(lc, ax=ax)
                    cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.77])  # [left, bottom, width, height] in figure coords
                    cbar = fig.colorbar(lc, cax=cbar_ax, orientation='vertical')
                    # plt.text(1.8, -1.8, s='SMC pericenters at +0.05 Gyr and +0.95 Gyr', rotation='vertical',
                    #          fontproperties=font_prop)
                    cbar.set_label(cbar_label, fontproperties=font_prop)

                    cbar.set_ticks([-1.5, -1.0, -0.5, 0.0])
                    cbar.set_ticks(np.linspace(-1.5, 0.0, 16), minor=True)
                    cbar.set_ticklabels(['1.5', '1.0', '0.5', '0.0'], fontproperties=font_prop)

                else:
                    ax.scatter(self.data['bh_coords'][:, 0], self.data['bh_coords'][:, 1],
                               s=10, c='k', alpha=0.8)

                x_lim = [-4, 4]
                y_lim = [-4, 4]
                x_ticks = np.array([-4, -3, -2, -1, 0.0, 1, 2, 3, 4])
                x_labels = np.array(['-4.0', '-3.0', '-2.0', '-1.0', '0.0', '1.0', '2.0', '3.0', '4.0'])
                y_ticks = np.array([-4, -3, -2, -1, 0.0, 1, 2, 3, 4])
                y_labels = np.array(['-4.0', '-3.0', '-2.0', '-1.0', '0.0', '1.0', '2.0', '3.0', '4.0'])

                # plt.rcParams.update({'xtick.labeltop': True, 'ytick.labelright': True})

                ax.set_xlim(x_lim)
                mask = (x_ticks >= x_lim[0]) & (x_ticks <= x_lim[1])
                ax.set_xticks(x_ticks[mask], labels=x_labels[mask], fontproperties=font_prop)
                ax.set_xticks(np.arange(x_lim[0], x_lim[1], 0.1), minor=True)
                ax.set_xlabel(r'x-coordinate $[$kpc$]$', fontproperties=font_prop)

                ax.set_ylim(y_lim)
                mask = (y_ticks >= y_lim[0]) & (y_ticks <= y_lim[1])
                ax.set_yticks(y_ticks[mask], labels=y_labels[mask], fontproperties=font_prop)
                ax.set_yticks(np.arange(y_lim[0], y_lim[1], 0.1), minor=True)
                ax.set_ylabel(r'y-coordinate $[$kpc$]$', fontproperties=font_prop)
                ax.set_aspect('equal')

                plt.rcParams.update({'lines.dashed_pattern': (6, 6)})
                ax.errorbar(0, -1, xerr=0.186 / 2, lw=0.8, capsize=2, color='k', alpha=0.8)
                ax.text(-0.2, -1.15, s=r'$\sim\epsilon_{\text{bh}}$ ', rotation='horizontal', color='black',
                        fontproperties=font_prop)
                ax.annotate("", xytext=(-0.44, -1.5), xy=(0.44, -1.5),
                            arrowprops=dict(arrowstyle="<->", linestyle='dashed', lw=0.5, alpha=0.8))
                ax.text(-0.4, -1.65, s=r'$\sim$1$^{\circ}$ in LMC', rotation='horizontal', color='black',
                        fontproperties=font_prop)

            elif self.parameter == 'L':
                if 'halo_08' in self.output_path:  # if this the LMC-SMC system from chisholm+2025
                    n, t_dyn0 = 1, 1.517  # zero dynamical time, snap 119
                    cbar_label = r'dynamical time $[$Gyr$]$'
                else:
                    n, t_dyn0 = 0, 0
                    cbar_label = r'lookback time $[$Gyr$]$'
                fx = interp1d(-1 ** n * self.data['lookback_times'] - t_dyn0,
                              np.sqrt(self.data['bh_energies']), kind=kind_interpolator,
                              fill_value=np.array([0.]), bounds_error=False)
                fy = interp1d(-1 ** n * self.data['lookback_times'] - t_dyn0,
                              self.data['bh_L'], kind=kind_interpolator,
                              fill_value=np.array([0.]), bounds_error=False)
                # t_smooth = np.linspace(min(-1 ** n * self.data['lookback_times'] - t_dyn0),
                #                        max(-1 ** n * self.data['lookback_times'] - t_dyn0), 9999)
                t_smooth = np.linspace(min(-1 ** n * self.data['lookback_times'] - t_dyn0), 0, 9999)
                x_smooth, y_smooth = fx(t_smooth), fy(t_smooth)

                # Create line segments [(x0, y0), (x1, y1)], ...
                points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create the LineCollection
                norm = plt.Normalize(t_smooth.min(), t_smooth.max())
                lc = LineCollection(segments, cmap='Spectral', norm=norm, lw=0.8)
                ax.add_collection(lc)
                lc.set_array(t_smooth)  # THIS is the missing step
                print(self.data['bh_L'].min())
                print(self.data['bh_L'].max())
                x_lim = [70, 90]
                y_lim = [-100, 200]
                plt.xlim(x_lim)
                plt.ylim(y_lim)

                ax.add_collection(lc)
                # cbar = plt.colorbar(lc, ax=ax)
                cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.77])  # [left, bottom, width, height] in figure coords
                cbar = fig.colorbar(lc, cax=cbar_ax, orientation='vertical')
                # plt.text(1.8, -1.8, s='SMC pericenters at +0.05 Gyr and +0.95 Gyr', rotation='vertical',
                #          fontproperties=font_prop)
                cbar.set_label(cbar_label, fontproperties=font_prop)

                cbar.set_ticks([-1.5, -1.0, -0.5, 0.0])
                cbar.set_ticks(np.linspace(-1.5, 0.0, 16), minor=True)
                cbar.set_ticklabels(['1.5', '1.0', '0.5', '0.0'], fontproperties=font_prop)

            # x_ticks = np.array([-2.0, -1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            # x_labels = np.array(['-2.0', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0'])
            # y_ticks = np.array([-2.0, -1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            # y_labels = np.array(['-2.0', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0'])

        elif self.projection == '3-dim':
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            x_lim = [-3.5, 3.5]
            y_lim = [-3.5, 3.5]
            z_lim = [-1.0, 1.0]

            fx = interp1d(-1 * self.data['lookback_times'] - t_dyn0,
                          self.data['bh_coords'][:, 0], kind=kind_interpolator)
            fy = interp1d(-1 * self.data['lookback_times'] - t_dyn0,
                          self.data['bh_coords'][:, 1], kind=kind_interpolator)
            fz = interp1d(-1 * self.data['lookback_times'] - t_dyn0,
                          self.data['bh_coords'][:, 2], kind=kind_interpolator)
            t_smooth = np.linspace(min(-1 * self.data['lookback_times'] - t_dyn0),
                                   max(-1 * self.data['lookback_times'] - t_dyn0), 5000)
            x_smooth, y_smooth, z_smooth = fx(t_smooth), fy(t_smooth), fz(t_smooth)

            # Build 3D segments
            points = np.array([x_smooth, y_smooth, z_smooth]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create a Line3DCollection with colormap based on time
            norm = plt.Normalize(t_smooth.min(), t_smooth.max())
            lc = Line3DCollection(segments, cmap='Spectral', norm=norm, lw=0.8)
            lc.set_array(t_smooth)

            ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')
            ax.add_collection3d(lc)
            cbar = plt.colorbar(lc, ax=ax)
            cbar.set_label(r'dynamical time $[$Gyr$]$', fontproperties=font_prop)

            ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.set_zlim(z_lim[0], z_lim[1])
            ax.set_aspect('equal')

            ax.set_xlabel(r'x-coordinate $[$kpc$]$', fontproperties=font_prop)
            ax.set_ylabel(r'y-coordinate $[$kpc$]$', fontproperties=font_prop)
            ax.set_zlabel(r'z-coordinate $[$kpc$]$', fontproperties=font_prop)

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()

    def mbpSloshing(self):
        from scipy.interpolate import CubicSpline
        from matplotlib.collections import LineCollection
        # -------------------------------
        kind_interpolator = 'cubic'  # or 'linear', 'quadratic', 'cubic'
        # -------------------------------
        font_prop = formatting()

        fig, ax = plt.subplots(figsize=(8, 7))

        cbar_label = r'lookback time $[$Gyr$]$'
        # fx = interp1d(-1 * self.data['lookback_times'], self.data['mbp_coords'][:, 0],
        #               kind=kind_interpolator, fill_value=np.array([0.]), bounds_error=False)
        # fy = interp1d(-1 * self.data['lookback_times'], self.data['mbp_coords'][:, 1],
        #               kind=kind_interpolator, fill_value=np.array([0.]), bounds_error=False)
        # t_smooth = np.linspace(min(-1 * self.data['lookback_times']), 0, 9999)
        # x_smooth, y_smooth = fx(t_smooth), fy(t_smooth)

        delta_t = self.data['lookback_times'][1] - self.data['lookback_times'][0]  # Gyr / unit_time
        vels = (self.data['mbp_vels']
                * 3.240756e-17  # kpc / km
                / (3.168809e-8 * 1e-9 / delta_t))  # (Gyr / s) (unit_time / Gyr)
        t = -self.data['lookback_times'][::-1]
        vx = vels[:, 0][::-1]
        vy = vels[:, 1][::-1]

        sx = CubicSpline(t, self.data['mbp_coords'][:, 0][::-1],
                         bc_type=((1, -vx[0]), (1, -vx[-1])))
        sy = CubicSpline(t, self.data['mbp_coords'][:, 1][::-1],
                         bc_type=((1, -vy[0]), (1, -vy[-1])))

        t_fine = np.linspace(min(-1 * self.data['lookback_times']), max(-1 * self.data['lookback_times']), 9999)
        x_fine = sx(t_fine)
        y_fine = sy(t_fine)

        # Create line segments [(x0, y0), (x1, y1)], ...
        print(np.array([x_fine, y_fine]).T[0])
        points = np.array([x_fine, y_fine]).T.reshape(-1, 1, 2)
        # print(points[0])
        # print(f'{x_fine[-1]}, {y_fine[-1]}')
        # print(self.data['mbp_coords'][:, 0])
        print(self.data['redshifts'])
        # print(self.data['lookback_times'])
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        print(segments[0])

        # Create the LineCollection
        norm = plt.Normalize(t_fine.min(), t_fine.max())
        lc = LineCollection(segments, cmap='Spectral', norm=norm, lw=0.8)
        ax.add_collection(lc)
        lc.set_array(t_fine)  # THIS is the missing step
        ax.add_collection(lc)

        # cbar = plt.colorbar(lc, ax=ax)
        cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.77])  # [left, bottom, width, height] in figure coords
        cbar = fig.colorbar(lc, cax=cbar_ax, orientation='vertical')
        cbar.set_label(cbar_label, fontproperties=font_prop)

        cbar.set_ticks([-1.5, -1.0, -0.5, 0.0])
        cbar.set_ticks(np.linspace(-1.5, 0.0, 16), minor=True)
        cbar.set_ticklabels(['1.5', '1.0', '0.5', '0.0'], fontproperties=font_prop)

        lim = [-2, 2]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')

        x_ticksMajor = np.arange(-8, 8, 0.5)
        x_ticksMinor = np.arange(-8, 8, 0.1)
        x_ticks = x_ticksMajor[(x_ticksMajor >= lim[0]) & (x_ticksMajor <= lim[1])]
        y_ticksMajor = np.arange(-8, 8, 0.5)
        y_ticksMinor = np.arange(-8, 8, 0.1)
        y_ticks = y_ticksMajor[(y_ticksMajor >= lim[0]) & (y_ticksMajor <= lim[1])]

        ax.set_xticks(x_ticks, [str(x).replace("-", '-') if x % 1 == 0 else '' for x in x_ticks],
                      fontproperties=font_prop)
        ax.set_xticks(x_ticksMinor[(x_ticksMinor >= lim[0]) & (x_ticksMinor <= lim[1])], minor=True)
        ax.set_yticks(y_ticks, [str(y).replace("-", '-') if y % 1 == 0 else '' for y in y_ticks],
                      fontproperties=font_prop)
        ax.set_yticks(y_ticksMinor[(y_ticksMinor >= lim[0]) & (y_ticksMinor <= lim[1])], minor=True)
        ax.set_xlabel(r'x-coordinate $[$kpc$]$', fontproperties=font_prop)
        ax.set_ylabel(r'y-coordinate $[$kpc$]$', fontproperties=font_prop)

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()

    def bhAccretion(self):
        from scipy.interpolate import interp1d
        font_prop = formatting()

        if '09_18_lastgigyear' in self.output_path:
            times = self.data['lookback_times']
            x_lim = [0.1, 0.4]
            x_label = 'Lookback Time ' + r'$[$' + 'Gyr' + r'$]$'
        else:
            times = self.data['redshifts']
            x_lim = [0, 1.5]
            x_label = 'z'

        fig, ax1 = plt.subplots(figsize=(5, 5))

        ax2 = ax1.twinx()

        t_smooth = np.linspace(max(self.data['lookback_times']), 0, 9999)
        fx = interp1d(self.data['lookback_times'], np.log10(self.data['M_BHgas']), kind='cubic',
                      fill_value=np.array([0.]), bounds_error=False)
        fm = interp1d(self.data['lookback_times'], np.log10(self.data['M_dot']), kind='cubic',
                      fill_value=np.array([0.]), bounds_error=False)
        displacement = fx(t_smooth)
        ax1.plot(t_smooth, displacement, label='BH displacement', linewidth=0.8, c='k')
        ax2.plot(-1, 0, label='BH displacement', linewidth=0.8, c='k')  # strictly for labelling purposes

        ax2.plot(t_smooth, fm(t_smooth), label='BH growth rate', linewidth=0.8, color='tab:blue')

        ax1_lim = [0, 8]
        ax2_lim = [-8, 0]

        # Other formatting stuff
        plt.xlim(x_lim[0], x_lim[1])
        x_ticksMajor = np.arange(0, 13, 0.5)
        x_ticksMinor = np.arange(0, 13, 0.1)
        x_ticks = x_ticksMajor[(x_ticksMajor >= x_lim[0]) & (x_ticksMajor <= x_lim[1])]
        ax1.set_xticks(x_ticks, [str(round(x, 1)).replace("-", '-') for x in x_ticks], fontproperties=font_prop)
        ax1.set_xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)

        ax1.set_ylim(ax1_lim[0], ax1_lim[1])
        ax2.set_ylim(ax2_lim[0], ax2_lim[1])

        ax1_ticksMajor = np.arange(0, 13, 1)
        ax1_ticksMinor = np.arange(0, 13, 0.2)
        ax1_ticks = ax1_ticksMajor[(ax1_ticksMajor >= ax1_lim[0]) & (ax1_ticksMajor <= ax1_lim[1])]
        ax1.set_yticks(ax1_ticks, [str(y).replace("-", '-') for y in ax1_ticks], fontproperties=font_prop)
        ax1.set_yticks(ax1_ticksMinor[(ax1_ticksMinor >= ax1_lim[0]) & (ax1_ticksMinor <= ax1_lim[1])], minor=True)

        ax2_ticksMajor = np.arange(-8, 1, 1)
        ax2_ticksMinor = np.arange(-8, 1, 0.2)
        ax2_ticks = ax2_ticksMajor[(ax2_ticksMajor >= ax2_lim[0]) & (ax2_ticksMajor <= ax2_lim[1])]
        ax2.set_yticks(ax2_ticks, [str(round(y, 2)).replace("-", '-') for y in ax2_ticks],
                       fontproperties=font_prop)
        ax2.set_yticks(ax2_ticksMinor[(ax2_ticksMinor >= ax2_lim[0]) & (ax2_ticksMinor <= ax2_lim[1])], minor=True)

        plt.gca().invert_xaxis()

        # Set font weight for tick labels
        ax1.set_xlabel(x_label, fontproperties=font_prop)
        ax1.set_ylabel('BH displacement from center ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
        ax2.set_ylabel('dM/dt ' + r'$[$' + 'solar masses / yr' + r'$]$', fontproperties=font_prop, c='tab:blue')

        # plt.text(8, 5, s=r'halo 16; M $\sim$ 1.5e11', fontproperties=font_prop)

        plt.legend(loc='upper left', prop=font_prop)
        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()


class GaseousComponents:
    def __init__(self, data, H_phase, snapshot, dark_mode, output_path):
        self.data = data
        self.H_phase = H_phase
        self.snap = snapshot
        self.dark_mode = dark_mode
        self.output_path = output_path

    def phaseDiagram(self):
        from scipy.ndimage import gaussian_filter
        image = self.data['data']
        x_e, y_e = self.data['x_e'], self.data['y_e']

        font_prop = formatting()

        fig, ax = plt.subplots(figsize=(5, 5))
        c_map = 'Blues'
        background_color = plt.get_cmap(c_map)(0)

        # Smooth the data before plotting
        slice_index = 127 - self.snap
        smoothed_image = gaussian_filter(image[:, :, slice_index], sigma=2.0)

        aspect_ratio = abs(x_e[-1] - x_e[0]) / abs(y_e[-1] - y_e[0])

        fig.tight_layout(pad=2)
        plt.gca().set_facecolor(background_color)
        vmax = np.max(smoothed_image)
        plt.imshow(np.rot90(smoothed_image), origin='upper',
                   extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]), aspect=aspect_ratio,
                   # cmap=c_map, norm=LogNorm(vmin=vmax * 1e-2, vmax=vmax), rasterized=True)
                   cmap=c_map, norm=PowerNorm(gamma=0.5, vmin=vmax * 1e-3, vmax=vmax), rasterized=True)

        # Other formatting stuff
        x_lim = [-8, -2]
        y_lim = [3.5, 6.5]
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])
        plt.xticks(ticks=np.linspace(-8, -2, 7), labels=['-8', '-7', '-6', '-5', '-4', '-3', '-2'],
                   fontproperties=font_prop)
        plt.xticks(np.linspace(-8, -2, 31), minor=True)
        plt.yticks(np.linspace(3.5, 6.5, 7), fontproperties=font_prop)
        plt.yticks(np.linspace(3.5, 6.5, 31), minor=True)

        ax.set_xlabel('log(n' + r'$_{\text{H}}$' + '/ cm' + r'$^{\text{-3}}$)', fontproperties=font_prop)
        ax.set_ylabel('log(T / K)', fontproperties=font_prop)
        # -----------------------
        plt.savefig(self.output_path, dpi=240)
        plt.show()

    def temperatureProfile(self):
        from scipy.ndimage import gaussian_filter

        slice_index = 127 - self.snap

        image = self.data['heatmap']
        x_e, y_e = self.data['x_e'], self.data['y_e']
        column_averages = self.data['coronaProfile'][:, slice_index]
        virial_radius = self.data['R_vir'][slice_index]
        equilibrium_r = self.data['salemR'][:, slice_index]
        equilibrium_T = self.data['salemT'][:, slice_index]

        font_prop = formatting()

        fig, ax = plt.subplots(figsize=(5, 5))
        c_map = 'Blues'
        background_color = plt.get_cmap(c_map)(0)

        # Smooth the data before plotting
        slice_index = 127 - self.snap
        smoothed_image = gaussian_filter(image[:, :, slice_index], sigma=2.0)

        aspect_ratio = abs(x_e[-1] - x_e[0]) / abs(y_e[-1] - y_e[0])

        fig.tight_layout(pad=2)
        # plt.gca().set_facecolor(background_color)
        vmax = np.max(smoothed_image)
        plt.imshow(np.rot90(smoothed_image), origin='upper',
                   extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]), aspect=aspect_ratio,
                   cmap=c_map, norm=LogNorm(vmin=vmax * 1e-2, vmax=vmax), rasterized=True)
        # cmap=c_map, norm=PowerNorm(gamma=0.5, vmin=vmax * 1e-10, vmax=vmax), rasterized=True)

        # Virial radius line
        plt.plot([virial_radius, virial_radius], [4, 7], linestyle='dashed', color='black',
                 dashes=(6, 6), lw=0.5, alpha=0.8)
        plt.text(virial_radius - 8, 4.2, s='virial radius', rotation='vertical', color='black',
                 fontproperties=font_prop)

        # hestia column averages
        ax.plot(x_e[:-1] + abs(x_e[0] + x_e[1]) / 2, column_averages,
                c='tab:blue', lw=1.5, label='Massive dwarf (this work)')

        # Stable corona (Salem+2015)
        ax.plot(equilibrium_r, np.log10(equilibrium_T), linestyle='solid', lw=1, c='tab:purple',
                label='Equilibrium profile (Salem+ 2015)')

        # Scott's simulations
        column_averages_lmc, x_e_lmc = lmc_temperatureProfile()
        ax.plot(x_e_lmc[:-1] + abs(x_e_lmc[0] + x_e_lmc[1]) / 2, column_averages_lmc, c='tab:green', linestyle='solid',
                lw=1,
                label='Isolated LMC-analog (Lucchini+ 2024)')

        # Other formatting stuff
        x_lim = [x_e[0], x_e[-1]]
        y_lim = [4, 7]
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])
        plt.xticks(np.linspace(0, 200, 6), fontproperties=font_prop)
        # plt.xticks(np.linspace(0, 200, 6))
        plt.xticks(np.linspace(0, 200, 21), minor=True)
        plt.yticks(np.linspace(4, 7, 7), fontproperties=font_prop)
        # plt.yticks(np.linspace(4, 7, 7))
        plt.yticks(np.linspace(4, 7, 31), minor=True)
        # Set font weight for tick labels
        ax.set_xlabel('Radius' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
        ax.set_ylabel('log (T / K)', fontproperties=font_prop)

        # plt.grid(True, alpha=0.25, color='k', linestyle='dashed', lw=0.5)
        # plt.legend(prop=font_prop)
        plt.legend(prop=font_prop)
        # -----------------------
        plt.savefig(self.output_path, dpi=240)
        plt.show()

    def coronaMassFunction(self):
        from archive.hestia.gas import plot_virial_temp_line
        redshift = self.data['redshift']
        sims = self.data['sim']
        halo_ids = self.data['halo_id']
        halo_masses = self.data['M_halo']
        bool_satellite = self.data['bool_satellite']
        m_h1 = self.data['M_HII']
        avg_temp = self.data['T_avg']
        mean_nH = self.data['mean_nH']
        sigma_nH = self.data['sigma_nH']
        corona_temp = self.data['mean_T']
        sigma_temp = self.data['sigma_T']

        font_prop = formatting()

        fig, ax = plt.subplots(figsize=(5, 5))

        for row in range(len(halo_masses)):
            if bool_satellite[row] == 1:
                plt.scatter(halo_masses[row], corona_temp[row], c='white', edgecolors='k', s=8)
            elif str(sims[row]) == '09_18' and str(halo_ids[row]) == '008':
                plt.scatter(halo_masses[row], corona_temp[row], c='tab:blue', s=20, marker='D', zorder=2,
                            label='LMC-analog')
                print(mean_nH[row])
                print(sigma_nH[row])
            else:
                plt.scatter(halo_masses[row], corona_temp[row], c='k', s=8)
        plt.errorbar(halo_masses, corona_temp, yerr=sigma_temp, c='k', ls='none', elinewidth=0.8, fmt='none')

        x_arr = np.linspace(min(halo_masses), max(halo_masses))
        x_l, y_l = plot_virial_temp_line(x_arr)
        plt.plot(x_l, y_l, linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8,
                 label='Virial Theorem')

        x_lim = [10.25, 12.75]
        y_lim = [4, 7]
        # y_lim = [8, 11]
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])
        plt.xticks(np.linspace(10.5, 12.5, 5), fontproperties=font_prop)
        plt.xticks(np.linspace(10.3, 12.7, 25), minor=True)
        plt.yticks(np.linspace(y_lim[0], y_lim[1], 7), fontproperties=font_prop)
        plt.yticks(np.linspace(y_lim[0], y_lim[1], 31), minor=True)
        plt.xlabel('log (M' + r'$_{\text{halo}}$' + '/M' + r'$_{\odot}$)', fontproperties=font_prop)
        plt.ylabel('log (T / K)', fontproperties=font_prop)
        plt.ylabel('log (M' + r'$_{\text{HII}}$' + '/M' + r'$_{\odot}$)', fontproperties=font_prop)

        plt.legend(prop=font_prop, loc='lower right')

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()


class Observables:
    def __init__(self, data, frame, dark_mode, output_path):
        self.data = data
        self.frame = frame
        self.dark_mode = dark_mode
        self.output_path = output_path

    def NH0(self):
        nH_map = np.log10(self.data['nH_map']).T
        lon_edges = self.data['lon_edges']
        lat_edges = self.data['lat_edges']
        r_smc = self.data['r_smc']

        font_prop = formatting()
        cmap = 'BuPu'

        if self.frame == 'faux':
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(5, 5))

            # Plot with pcolormesh — you'll need bin edges for R and Phi
            phi_edges = np.radians(lon_edges)
            r_edges = np.radians(90 + lat_edges)  # accounts for +20° now

            Phi_edges, R_edges = np.meshgrid(phi_edges, r_edges)

            # Plot
            c = ax.pcolormesh(Phi_edges, R_edges, nH_map, shading='auto', cmap=cmap, vmin=14, vmax=20)

            # Aesthetics
            ax.set_theta_zero_location("E")  # 0° longitude at right
            # ax.set_theta_direction(-1)  # Longitudes increase clockwise
            ax.set_rlabel_position(90)  # Move radial labels to a better spot

            lon_tick_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
            ax.set_xticks(np.radians(lon_tick_degrees))
            ax.set_xticklabels([r'0$^{\circ}$', r'45$^{\circ}$', r'90$^{\circ}$', r'135$^{\circ}$', r'180$^{\circ}$',
                                r'-135$^{\circ}$', r'-90$^{\circ}$', r'-45$^{\circ}$'], fontproperties=font_prop)

            # Custom radial ticks to show latitudes
            lat_tick_degrees = [-90, -60, -30, 0, 30, 45]
            ax.set_rticks(np.radians(90 + np.array(lat_tick_degrees)))
            ax.set_yticklabels(['', r'-60$^{\circ}$', r'-30$^{\circ}$',
                                r'0$^{\circ}$', r'30$^{\circ}$', ''], fontproperties=font_prop)

            ax.grid(True, linestyle='dashed', dashes=(6, 6), linewidth=0.5, color='k', alpha=0.8)

            ax.scatter(np.radians(r_smc[0]), np.radians(90 + r_smc[1]), s=25, marker='D', c='k')
            # ax.arrow(np.radians(smc_position[0]), np.radians(90 + smc_position[1]),
            #          np.radians(smc_velocity[0]), np.radians(90 + smc_velocity[1]))

            # Colorbar
            cbar_ax = fig.add_axes([0.11, 0.01, 0.8, 0.03])  # [left, bottom, width, height] in figure coords
            cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
            cbar.set_ticks([14, 15, 16, 17, 18, 19, 20])
            cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
            cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)
        elif self.frame == 'radec':
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(6, 6))

            # Plot with pcolormesh — you'll need bin edges for R and Phi

            # Plot with pcolormesh — you'll need bin edges for R and Phi
            phi_edges = np.radians(lon_edges)
            r_edges = np.radians(90 + lat_edges)  # accounts for +20° now

            Phi_edges, R_edges = np.meshgrid(phi_edges, r_edges)

            # Plot
            c = ax.pcolormesh(Phi_edges, R_edges, nH_map, shading='auto', cmap=cmap, vmin=15, vmax=21)

            # Aesthetics
            ax.set_theta_zero_location("E")  # 0° longitude at right
            # ax.set_theta_direction(-1)  # Longitudes increase clockwise
            ax.set_rlabel_position(90)  # Move radial labels to a better spot

            lon_tick_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
            ax.set_xticks(np.radians(lon_tick_degrees))
            ax.set_xticklabels([r'0$^{\circ}$', r'45$^{\circ}$', r'90$^{\circ}$', r'135$^{\circ}$', r'180$^{\circ}$',
                                r'-135$^{\circ}$', r'-90$^{\circ}$', r'-45$^{\circ}$'], fontproperties=font_prop)

            # Custom radial ticks to show latitudes
            lat_tick_degrees = [-90, -60, -30, 0]
            ax.set_rticks(np.radians(90 + np.array(lat_tick_degrees)))
            ax.set_yticklabels(['', r'-60$^{\circ}$', r'-30$^{\circ}$',
                                r'0$^{\circ}$'], fontproperties=font_prop)

            ax.grid(True, linestyle='dashed', dashes=(6, 6), linewidth=0.5, color='k', alpha=0.8)

            ax.scatter(np.radians(r_smc[0, 2]), np.radians(90 + r_smc[0, 1]), s=25, marker='D', c='k')
            # ax.scatter(np.radians(r_smc[0, 2]), np.radians(r_smc[0, 1]), s=25, marker='D', c='k')
            ax.scatter(np.radians(80), np.radians(90 + -69), s=25, marker='+', c='k')
            # ax.arrow(np.radians(smc_position[0]), np.radians(90 + smc_position[1]),
            #          np.radians(smc_velocity[0]), np.radians(90 + smc_velocity[1]))

            # Colorbar
            cbar_ax = fig.add_axes([0.11, 0.01, 0.8, 0.03])  # [left, bottom, width, height] in figure coords
            cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
            cbar.set_ticks([15, 16, 17, 18, 19, 20, 21])
            cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
            cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)

        elif self.frame == 'mag':
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot with pcolormesh — you'll need bin edges for R and Phi
            # Phi_edges, R_edges = np.meshgrid(lon_edges, lat_edges)

            # Plot
            # c = ax.pcolormesh(Phi_edges, R_edges, nH_map, shading='auto', cmap=cmap, vmin=15, vmax=21)
            c = ax.imshow(nH_map, extent=(lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]), origin='lower',
                          cmap=cmap, vmin=15, vmax=21)

            lon_tick_degrees = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
            ax.set_xticks(lon_tick_degrees)
            plt.gca().invert_xaxis()
            # ax.set_xticklabels([r'0$^{\circ}$', r'45$^{\circ}$', r'90$^{\circ}$', r'135$^{\circ}$', r'180$^{\circ}$',
            #                     r'-135$^{\circ}$', r'-90$^{\circ}$', r'-45$^{\circ}$'], fontproperties=font_prop)

            # Custom radial ticks to show latitudes
            lat_tick_degrees = [-90, -60, -30, 0, 30, 60, 90]
            ax.set_yticks(lat_tick_degrees)
            # ax.set_yticklabels(['', r'-60$^{\circ}$', r'-30$^{\circ}$',
            #                     r'0$^{\circ}$'], fontproperties=font_prop)

            ax.grid(True, linestyle='dashed', dashes=(6, 6), linewidth=0.5, color='k', alpha=0.8)

            ax.scatter(r_smc[0, 2], r_smc[0, 1], s=25, marker='D', c='k')
            # ax.scatter(np.radians(r_smc[0, 2]), np.radians(r_smc[0, 1]), s=25, marker='D', c='k')
            # ax.scatter(np.radians(80), np.radians(90 + -69), s=25, marker='+', c='k')
            # ax.arrow(np.radians(smc_position[0]), np.radians(90 + smc_position[1]),
            #          np.radians(smc_velocity[0]), np.radians(90 + smc_velocity[1]))

            # Colorbar
            cbar_ax = fig.add_axes([0.11, 0.01, 0.8, 0.03])  # [left, bottom, width, height] in figure coords
            cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
            cbar.set_ticks([15, 16, 17, 18, 19, 20, 21])
            cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
            cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()

    def bhPDF(self):
        from archive.hestia.astrometry import Measurements
        f_PDF = self.data['f_PDF'].T
        bar = self.data['bar']
        lon_e = self.data['lon_edges']
        lat_e = self.data['lat_edges']
        center_bh = self.data['center_bh']
        font_prop = formatting()

        print(f'(ra, dec, los)_bh : \t{round(center_bh[2], 3)} deg, {round(center_bh[1], 3)} deg, '
              f'{round(center_bh[0], 3)} kpc')

        fig = plt.figure(figsize=(6, 6))
        plt.xlim([lon_e[0], lon_e[-1]])
        plt.ylim([lat_e[0], lat_e[-1]])
        c = plt.imshow(f_PDF.T, origin='lower', cmap='cubehelix_r',
                       extent=(lon_e[0], lon_e[-1], lat_e[0], lat_e[-1]),
                       # norm=LogNorm(vmin=f_PDF.min(), vmax=f_PDF.max()))
                       vmin=f_PDF.min(), vmax=f_PDF.max())
        plt.gca().invert_xaxis()

        m = Measurements()
        # softening length of gas (dependent on cell, but roughly) ~ 186 pc --> 0.214 degrees
        # plt.errorbar(center_bh[2], center_bh[1], xerr=0.214, yerr=0.214, c='k', elinewidth=0.8, capsize=2,
        #              label='HESTIA : current position of bh')
        # plt.errorbar(center_gas[0, 2], center_gas[0, 1], xerr=0.214, yerr=0.214, c='tab:blue', elinewidth=0.8, capsize=2,
        #              label='HESTIA : center of cool gas' + r' (r $<$ 5 kpc, T $<$ 1e5 K)')
        # plt.errorbar(center_stars[0, 2], center_stars[0, 1], xerr=0.214, yerr=0.214, c='tab:cyan',
        #              elinewidth=0.8, capsize=2, label='HESTIA : center of stellar disk' + r' (r $<$ 5 kpc)')

        plt.rcParams.update({'lines.dashed_pattern': (6, 6)})
        from matplotlib.patches import Ellipse
        ax = plt.gca()
        # ellipse = Ellipse(xy=(m['LMC/HI/ra'],  m['LMC/HI/dec']),
        #                   width=2 * m['LMC/HI/sigma_ra'], height=2 * m['LMC/HI/sigma_dec'],
        #                   edgecolor='tab:blue', fc='none', lw=0.8, alpha=0.5, hatch='x', label='HI center (Kim+1998)')
        # ax.add_patch(ellipse)
        plt.scatter(m['LMC/HI/ra'], m['LMC/HI/dec'], c='tab:blue', marker='x', linewidths=0.8, s=64,
                    label='HI kinematical center (Kim+1998)')
        # ellipse = Ellipse(xy=(m['LMC/pm/ra'],  m['LMC/pm/dec']),
        #                   width=2 * 0.02, height=2 * 0.02,
        #                    edgecolor='tab:purple', fc='none', lw=0.8, alpha=0.5, hatch='x',
        #                   label='Stellar kinematic center (Choi+2022)')
        # ax.add_patch(ellipse)
        plt.scatter(m['LMC/pm/ra'], m['LMC/pm/dec'], c='tab:purple', marker='x', linewidths=0.8, s=64,
                    label='Stellar kinematical center (Choi+2022)')

        ellipse = Ellipse(xy=(m['LMC/bar/ra'], m['LMC/bar/dec']),
                          width=2 * 2 * np.degrees(np.arctan(m['LMC/bar/R_bar'] / m['LMC/bar/distance'])
                                                   / 2),
                          height=2 * 2 * np.degrees(np.arctan(m['LMC/bar/R_bar'] * m['LMC/bar/axisRatio']
                                                              / (2 * m['LMC/bar/distance']) * np.cos(
                              m['LMC/bar/inclination']))),
                          angle=180 - (90 + m['LMC/bar/nodes']),
                          edgecolor='tab:red', fc='none', lw=0.8, alpha=0.5, linestyle='dashed')
        # label='Extent of bar from RCSs; center aligned to mbp by def (Rathore+2025)')
        # ax.add_patch(ellipse)
        plt.plot(np.degrees(bar[1]), np.degrees(bar[2]), c='tab:red', lw=0.8, alpha=0.5, linestyle='dashed')
        print(bar)
        plt.scatter(m['LMC/bar/ra'], m['LMC/bar/dec'], c='tab:red', marker='x', linewidths=0.8, s=64,
                    label='Stellar dynamical center (Rathore+2025)')

        # plt.scatter(m['LMC/photometric/ra'],  m['LMC/photometric/dec'], c='tab:pink', marker='x', linewidths=0.8, s=64,
        #             label='photometric center (vanDerMarel+2001)')

        plt.errorbar(m['LMC/HVSs/ra'], m['LMC/HVSs/dec'],
                     xerr=m['LMC/HVSs/sigma_ra'], yerr=m['LMC/HVSs/sigma_dec'],
                     c='tab:green', elinewidth=0.8, capsize=2, label='LMC dynamical center from HVSs (Lucchini+2025)')
        # ellipse = Ellipse(xy=(m['LMC/HVSs/ra'], m['LMC/HVSs/dec']),
        #                   width=2 * m['LMC/HVSs/sigma_ra'], height=2 * m['LMC/HVSs/sigma_dec'],
        #                   edgecolor='tab:green', fc='none', lw=0.8, alpha=0.4, hatch='x',
        #                   label='LMC dynamical center from HVSs (Lucchini+2025)')
        # ax.add_patch(ellipse)

        plt.xlabel('right ascension (deg)', fontproperties=font_prop)
        plt.ylabel('declination (deg)', fontproperties=font_prop)

        x_ticksMajor = np.arange(70, 90, 1)
        x_ticksMinor = np.arange(70, 90, 0.2)
        x_lim = [73.5, 87.5]
        plt.xlim(x_lim[1], x_lim[0])
        x_ticks = x_ticksMajor[(x_ticksMajor >= x_lim[0]) & (x_ticksMajor <= x_lim[1])]
        plt.xticks(x_ticks, [f'{x:.1f}' if float(x) % 2 == 0 else '' for x in x_ticks],
                   fontproperties=font_prop)
        plt.xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)

        y_ticksMajor = np.arange(-75, -65, 0.5)
        y_ticksMinor = np.arange(-75, -65, 0.1)
        y_lim = [-74, -67]
        plt.ylim(y_lim)
        y_ticks = y_ticksMajor[(y_ticksMajor >= y_lim[0]) & (y_ticksMajor <= y_lim[1])]
        plt.yticks(y_ticks, [str(y).replace("-", '-') if float(y).is_integer() else '' for y in y_ticks],
                   fontproperties=font_prop)
        plt.yticks(y_ticksMinor[(y_ticksMinor >= y_lim[0]) & (y_ticksMinor <= y_lim[1])], minor=True)

        ax.set_aspect(abs(x_lim[1] - x_lim[0]) / abs(y_lim[1] - y_lim[0]))
        plt.legend(prop=font_prop, loc='lower left')

        # Colorbar
        cbar_ax = fig.add_axes([0.13, 0.92, 0.77, 0.03])  # [left, bottom, width, height] in figure coords
        cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([f_PDF.min(), 5e-7, 5e-6, f_PDF.max()])
        cbar.set_ticklabels(['', '1x probability', '10x probability', ''], fontproperties=font_prop)

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()

    def SSD(self):
        stars_map = self.data['L_map'].T
        lon_edges = self.data['lon_edges']
        lat_edges = self.data['lat_edges']
        r_smc = self.data['r_smc']

        font_prop = formatting()
        cmap = 'RdPu_r'

        if self.frame == 'faux':
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(6, 6))

            # Plot with pcolormesh — you'll need bin edges for R and Phi

            # Plot with pcolormesh — you'll need bin edges for R and Phi
            phi_edges = np.radians(lon_edges)
            r_edges = np.radians(90 + lat_edges)  # accounts for +20° now

            Phi_edges, R_edges = np.meshgrid(phi_edges, r_edges)

            # Plot
            c = ax.pcolormesh(Phi_edges, R_edges, stars_map, shading='auto', cmap=cmap,
                              vmin=5, vmax=17)
            ax.scatter(np.radians(r_smc[0]), np.radians(90 + r_smc[1]), s=25, marker='D', c='k')

            # Aesthetics
            ax.set_theta_zero_location("E")  # 0° longitude at right
            # ax.set_theta_direction(-1)  # Longitudes increase clockwise
            ax.set_rlabel_position(90)  # Move radial labels to a better spot

            lon_tick_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
            ax.set_xticks(np.radians(lon_tick_degrees))
            ax.set_xticklabels([r'0$^{\circ}$', r'45$^{\circ}$', r'90$^{\circ}$', r'135$^{\circ}$', r'180$^{\circ}$',
                                r'-135$^{\circ}$', r'-90$^{\circ}$', r'-45$^{\circ}$'], fontproperties=font_prop)

            # Custom radial ticks to show latitudes
            lat_tick_degrees = [-90, -60, -30, 0]
            ax.set_rticks(np.radians(90 + np.array(lat_tick_degrees)))
            ax.set_yticklabels(['', r'-60$^{\circ}$', r'-30$^{\circ}$',
                                r'0$^{\circ}$'], fontproperties=font_prop)

            ax.grid(True, linestyle='dashed', dashes=(6, 6), linewidth=0.5, color='k', alpha=0.8)

            # Colorbar
            cbar_ax = fig.add_axes([0.11, 0.04, 0.8, 0.03])  # [left, bottom, width, height] in figure coords
            cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(r'mag/arcsec$^2$', fontproperties=font_prop)
            cbar.set_ticks([6, 8, 10, 12, 14, 16])
            cbar.set_ticks(np.linspace(5, 17, 61), minor=True)
            cbar.set_ticklabels(['6', '8', '10', '12', '14', '16'], fontproperties=font_prop)
        elif self.frame == 'radec':
            fig = plt.figure(figsize=(7, 7))
            plt.xlim([lon_edges[0], lon_edges[-1]])
            plt.ylim([lat_edges[0], lat_edges[-1]])
            c = plt.imshow(stars_map, origin='lower', cmap='bone_r',
                           extent=(lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]),
                           vmin=stars_map.min(), vmax=stars_map.max())
            plt.gca().invert_xaxis()
            from matplotlib.patches import Ellipse
            from archive.hestia.astrometry import Measurements
            m = Measurements()
            ax = plt.gca()
            ellipse = Ellipse(xy=(79.25, -69.03), width=2 * 0.2, height=2 * 0.2,
                              edgecolor='tab:green', fc='none', lw=0.8, alpha=0.5, hatch='\\',
                              label='HI center (Kim+1998)')
            ax.add_patch(ellipse)
            ellipse = Ellipse(xy=(m['stars/LMC/ra'], m['stars/LMC/dec']),
                              width=2 * m['stars/LMC/sigma_ra'], height=2 * m['stars/LMC/sigma_dec'],
                              edgecolor='tab:green', fc='none', lw=0.8, alpha=0.5, hatch='/',
                              label='PM center (vanDerMarel+2014), Note: by def aligned to mbp')
            ax.add_patch(ellipse)
            # quoted from Lucchini+2025, https://arxiv.org/abs/2510.03393
            # plt.errorbar(80.72, -67.79, xerr=0.44 * 2, yerr=0.80 *2 , c='tab:green', elinewidth=0.8, capsize=2)
            ellipse = Ellipse(xy=(80.72, -67.79), width=2 * 2 * 0.44, height=2 * 2 * 0.80,
                              edgecolor='tab:green', fc='none', lw=0.8, alpha=0.4, hatch='x',
                              label='LMC dynamical center from HVSs (Lucchini+2025)')
            ax.add_patch(ellipse)

            plt.ylabel('declination (deg)', fontproperties=font_prop)
            plt.yticks(np.linspace(-67.5, -71, 8),
                       labels=['-67.5', '-68.0', '-68.5', '-69.0', '-69.5', '-70.0', '-70.5', '-71.0'],
                       fontproperties=font_prop)
            plt.yticks(np.linspace(-67.5, -71.0, 36), minor=True)
            plt.xlabel('right ascension (deg)', fontproperties=font_prop)
            plt.xticks(np.linspace(77.0, 80.5, 8),
                       labels=['77.0', '77.5', '78.0', '78.5', '79.0', '79.5', '80.0', '80.5'],
                       fontproperties=font_prop)
            plt.xticks(np.linspace(77.0, 80.5, 36), minor=True)

            plt.legend(prop=font_prop, loc='lower left')

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()


class StellarComponents:
    def __init__(self, data, frame, dark_mode, output_path):
        self.data = data
        self.frame = frame
        self.dark_mode = dark_mode
        self.output_path = output_path

    def gradZ(self):
        font_prop = formatting()

        fig, ax = plt.subplots(figsize=(5, 5))

        x_lim = [0, 3.5]
        y_lim = [-0.5, 0.5]

        rho = (self.data['rho_e'] + (self.data['rho_e'][1] - self.data['rho_e'][0] / 2))[:-1]
        ax.scatter(rho, self.data['H_Fe'],
                   marker='+', linewidths=0.8, s=50)
        ax.scatter(rho, self.data['H_alpha'],
                   marker='+', linewidths=0.8, s=50)
        # lines of best fit
        fe_label = (r'$[$' + 'Fe/H' + r'$]$' + f' = {self.data["Fe_line"][0]:.2f}({self.data["Fe_line"][1] * 100:.0f})'
                    + f' R/' + r'$[$' + 'kpc' + r'$]$ + '
                    + f'{self.data["Fe_line"][1]:.2f}(0{self.data["Fe_line"][3] * 100:.0f})')
        ax.plot(rho, self.data['Fe_line'][0] * rho + self.data['Fe_line'][1],
                linewidth=0.8, linestyle='dashed', dashes=(6, 6), label=fe_label)
        alpha_label = (r'$[$' + 'alpha/Fe' + r'$]$' + f' '
                       + f'= {self.data["alpha_line"][0]:.2f}({self.data["alpha_line"][1] * 100:.0f})'
                       + f' R/' + r'$[$' + 'kpc' + r'$]$ + '
                       + f'{self.data["alpha_line"][1]:.2f}(0{self.data["alpha_line"][3] * 100:.0f})')
        ax.plot(rho, self.data['alpha_line'][0] * rho + self.data['alpha_line'][1],
                linewidth=0.8, linestyle='dashed', dashes=(6, 6), label=alpha_label)

        add_ticks(ax, np.array([x_lim, y_lim]).flatten(), del_xTick=[1, 0.5], del_yTick=[0.5, 0.1],
                  x_label='cylindrical radius ' + r'$[$' + 'kpc' + r'$]$', y_label=r'$[$' + 'dex' + r'$]$')

        plt.legend(loc='lower left', prop=font_prop)
        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()


class Dynamics:
    def __init__(self, data, frame, dark_mode, output_path):
        self.data = data
        self.frame = frame
        self.dark_mode = dark_mode
        self.output_path = output_path

    def bhSloshing(self):
        pass


def dispatch_plot(plot_class, plot_type, input_path, output_path, parameter=None, H_phase=None, snapshot=None,
                  smoothing=None, frame=None, dark_mode=False, projection=None):
    data = np.load(input_path, allow_pickle=True)

    if plot_class == 'kinematics':
        kinematics_plots = Kinematics(data, projection, parameter, smoothing, dark_mode, output_path)
        dispatcher = {
            'history': kinematics_plots.accretionHistory,
            'rotCurve': kinematics_plots.rotCurve,
            'orbits': kinematics_plots.orbits,
            'bhSloshing': kinematics_plots.bhSloshing,
            'mbpSloshing': kinematics_plots.mbpSloshing,
            'bhAccretion': kinematics_plots.bhAccretion,
        }

    elif plot_class == 'gaseousComponents':
        gas_plots = GaseousComponents(data, H_phase, snapshot, dark_mode, output_path)
        dispatcher = {
            'phaseDiagram': gas_plots.phaseDiagram,
            'temperatureProfile': gas_plots.temperatureProfile,
            'coronaMassFunction': gas_plots.coronaMassFunction,
        }

    elif plot_class == 'observables':
        obs_plots = Observables(data, frame, dark_mode, output_path)
        dispatcher = {
            'NH0': obs_plots.NH0,
            'bhPDF': obs_plots.bhPDF,
            'SSD': obs_plots.SSD,
        }

    elif plot_class == 'stellarComponents':
        star_plots = StellarComponents(data, frame, dark_mode, output_path)
        dispatcher = {
            'gradZ': star_plots.gradZ,
        }

    elif plot_class == 'dynamics':
        dynamics_plots = Dynamics(data, frame, dark_mode, output_path)
        dispatcher = {
            'bhSloshing': dynamics_plots.bhSloshing,
        }

    else:
        exit(1)

    return dispatcher[plot_type]()


# -------------------------------------------------------------
def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='scripts/util/fonts/AVHersheySimplexMedium.otf',
                                  size=12)

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
