import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from scripts.util.utils import add_ticks
import astropy.units as u
from .utils import createFig
from .astrometry import Measurements


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

    def mbpSloshing(self):
        from scipy.interpolate import CubicSpline
        from matplotlib.collections import LineCollection
        # -------------------------------
        kind_interpolator = 'cubic'  # or 'linear', 'quadratic', 'cubic'
        # -------------------------------
        font_prop = formatting()

        fig, ax = plt.subplots(figsize=(8, 7))

        cbar_label = r'lookback time $[$Gyr$]$'

        delta_t = self.data['lookback_times'][1] - self.data['lookback_times'][0]  # Gyr / unit_time
        vels = (self.data['mbp_vels']
                * 3.240756e-17  # kpc / km
                / (3.168809e-8 * 1e-9 / delta_t))  # (Gyr / s) (unit_time / Gyr)
        t = -self.data['lookback_times'][::-1]
        vx, vy = vels[:, 0][::-1], vels[:, 1][::-1]

        sx = CubicSpline(t, self.data['mbp_coords'][:, 0][::-1],
                         bc_type=((1, -vx[0]), (1, -vx[-1])))
        sy = CubicSpline(t, self.data['mbp_coords'][:, 1][::-1],
                         bc_type=((1, -vy[0]), (1, -vy[-1])))

        t_fine = np.linspace(min(-1 * self.data['lookback_times']), max(-1 * self.data['lookback_times']), 9999)
        x_fine, y_fine = sx(t_fine), sy(t_fine)

        # Create line segments [(x0, y0), (x1, y1)], ...
        points = np.array([x_fine, y_fine]).T.reshape(-1, 1, 2)
        # print(self.data['lookback_times'])
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

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
            cbar_ax = fig.add_axes((0.11, 0.01, 0.8, 0.03))  # [left, bottom, width, height] in figure coords
            cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
            cbar.set_ticks([14, 15, 16, 17, 18, 19, 20])
            cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
            cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)
        elif self.frame == 'radec':
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(6, 6))
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
    def __init__(self, data, projection, time, zoom, dark_mode, output_path):
        self.data = data
        self.projection = projection
        self.time = time
        self.zoom = '_zoom' if zoom else ''
        self.dark_mode = dark_mode
        self.output_path = output_path

    def bhPDF(self):
        print('----------------(x, \t\t\ty, \t\t\tz, \t\t\tvx, \t\t\tvy, \t\t\tvz)')
        print(f'mu_t0, sigma_t0 : (', end='')
        for i in range(len(self.data["mu_t0"])):
            print(f'{self.data["mu_t0"][i]:.3f} +/- {self.data["sigma_t0"][i]:.3f}, ', end='')
        print(')')
        print(f'mu_t0, sigma_t0 : (', end='')
        for i in range(len(self.data["mu_t0"])):
            print(f'{self.data["mu_ti"][i]:.3f} +/- {self.data["sigma_ti"][i]:.3f}, ', end='')
        print(')')
        print(' ---------------- (ra, dec, distance)\n\t\t\t(', end='')
        for i in range(len(self.data["mu_radec"])):
            print(f'{self.data["mu_radec"][i]:.3f} +/- {self.data["sigma_radec"][i]:.3f}, ', end='')
        print(')')

        e_dict = {'x-y': ('x_e', 'y_e'), 'x-z': ('x_e', 'z_e'),
                  'y-z': ('y_e', 'y_e'), 'radec': ('ra_e', 'dec_e')}
        extent = (self.data[e_dict[self.projection][0] + self.zoom][0],
                  self.data[e_dict[self.projection][0] + self.zoom][-1],
                  self.data[e_dict[self.projection][1] + self.zoom][0],
                  self.data[e_dict[self.projection][1] + self.zoom][-1])

        fig, ax = createFig(dark_mode=self.dark_mode)

        if self.projection != 'radec':
            ax.imshow(self.data[f'H{self.time[1]}_{self.projection}'].T, origin='lower', cmap='magma',
                      extent=extent,
                      vmin=self.data[f'H{self.time[1]}_{self.projection}'].min(),
                      vmax=self.data[f'H{self.time[1]}_{self.projection}'].max())
            add_ticks(ax, extent, del_xTick=[1, 0.2],
                      x_label=f'{self.projection[0]}-coordinate' + r'$[$' + 'kpc' + r'$]$',
                      y_label=f'{self.projection[2]}-coordinate' + r'$[$' + 'kpc' + r'$]$')

        else:  # i.e if projection == 'radec'
            from .astrometry import LMCDisk
            from astropy.coordinates import ICRS
            bar = Measurements('LMC', 'bar')
            HVSs = Measurements('LMC', 'HVSs')
            HI = Measurements('LMC', 'HI')
            photo = Measurements('LMC', 'photometric')

            t = np.linspace(0, 2 * np.pi, 30)  # transforming the bar extent into equitorial coordinates
            x, y = bar.R_bar * np.cos(t), bar.R_bar * bar.axisRatio * np.sin(t)
            Bar = SkyCoord(x=x * u.kpc, y=y * u.kpc, z=np.zeros(t.shape) * u.kpc, frame=LMCDisk(LMC=bar),
                           representation_type='cartesian').transform_to(ICRS)
            ax.errorbar(HVSs.ra, HVSs.dec, xerr=HVSs.sigma_ra, yerr=HVSs.sigma_dec,
                        c='white', elinewidth=0.8, capsize=2, alpha=1)  # lucchini+2025
            ax.scatter(HI.ra, HI.dec, c='white', marker='^', linewidths=0.8, s=64, alpha=1,
                       label='HI Kim+1998')  # kim+1998
            ax.scatter(photo.ra, photo.dec, c='white', marker='*')  # vdMarel+2001
            ax.scatter(bar.ra, bar.dec, c='white', marker='x', linewidths=0.8, s=64, alpha=1,
                       label='Bar Rathore+2025')  # rathore+2025
            ax.plot(Bar.ra.deg, Bar.dec.deg, c='white', linestyle='dashed', dashes=(6, 6), linewidth=0.8)

            ax.imshow(self.data[f'H{self.time[1]}_{self.projection}{self.zoom}'].T, origin='lower', cmap='magma',
                      extent=extent,
                      vmin=self.data[f'H{self.time[1]}_{self.projection}{self.zoom}'].min(),
                      vmax=self.data[f'H{self.time[1]}_{self.projection}{self.zoom}'].max())

            add_ticks(ax, extent, del_xTick=[1, 0.2], x_label='right ascension (deg)', y_label='declination (deg)')
            ax.invert_xaxis()

        plt.savefig(self.output_path, dpi=240, bbox_inches='tight')
        plt.show()


def dispatch_plot(plot_class, plot_type, input_path, output_path, parameter=None, H_phase=None, snapshot=None,
                  smoothing=None, frame=None, dark_mode=False, projection=None, time=None, zoom=False):
    data = np.load(input_path, allow_pickle=True)

    if plot_class == 'kinematics':
        kinematics_plots = Kinematics(data, projection, parameter, smoothing, dark_mode, output_path)
        dispatcher = {
            'history': kinematics_plots.accretionHistory,
            'rotCurve': kinematics_plots.rotCurve,
            'orbits': kinematics_plots.orbits,
            'mbpSloshing': kinematics_plots.mbpSloshing,
            'bhAccretion': kinematics_plots.bhAccretion,
        }

    elif plot_class == 'observables':
        obs_plots = Observables(data, frame, dark_mode, output_path)
        dispatcher = {
            'NH0': obs_plots.NH0,
        }

    elif plot_class == 'stellarComponents':
        star_plots = StellarComponents(data, frame, dark_mode, output_path)
        dispatcher = {
            'gradZ': star_plots.gradZ,
        }

    elif plot_class == 'dynamics':
        dynamics_plots = Dynamics(data, projection, time, zoom, dark_mode, output_path)
        dispatcher = {
            'bhPDF': dynamics_plots.bhPDF,
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
