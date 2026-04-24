import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from .utils import add_ticks, add_colorbar
import astropy.units as u


class ImageMaps:
    """ class to make plots relating to "image maps" """

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
        image_l = self.data[e_i[0]]
        # print(image_l.shape)
        image_r = self.data[e_i[1]]
        z = self.data['redshifts']
        t = self.data['lookback_times']

        # for i, (frameL, frameR, z, t) in enumerate(zip(image_l, image_r,
        #                                                 redshifts, lookback_times), start=0):
        for i in range(len(z)):
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
            fig.suptitle('$z = $' + '{:.{}f}'.format(z[i], 3)
                         + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(t[i]), 2), 2) + ' Gyr'
                         , x=0.5, y=0.01, ha='center', va='bottom', weight='bold', c='k')

            if self.bool_centerPot:
                # only works with e_i = ['x_y', 'x_z']
                ax1.scatter(0, 0, marker='s', c='w', s=10)
                ax2.scatter(0, 0, marker='s', c='w', s=10)
            if self.bool_centerH0:
                # only works with e_i = ['x_y', 'x_z']
                ax1.scatter(self.data['center_h0'][len(z) - 1 - i, 0],
                            self.data['center_h0'][len(z) - 1 - i, 1], marker='+', c='tab:blue')
                ax2.scatter(self.data['center_h0'][len(z) - 1 - i, 0],
                            self.data['center_h0'][len(z) - 1 - i, 2], marker='+', c='tab:blue')

            # --------------------------------------
            extent_l = [self.data[e_i[0][2] + '_e'][0], self.data[e_i[0][2] + '_e'][-1],
                        self.data[e_i[0][0] + '_e'][0], self.data[e_i[0][0] + '_e'][-1]]
            extent_r = [self.data[e_i[1][0] + '_e'][0], self.data[e_i[1][0] + '_e'][-1],
                        self.data[e_i[1][2] + '_e'][0], self.data[e_i[1][2] + '_e'][-1]]
            # --------------------------------------
            if settings['log'] is True:
                im1 = ax1.imshow(image_l[:, :, i].T, origin='lower', extent=extent_l, cmap=settings['c_map'],
                                 norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']),
                                 rasterized=True)
                im2 = ax2.imshow(image_r[:, :, i].T, origin='lower', extent=extent_r, cmap=settings['c_map'],
                                 norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']),
                                 rasterized=True)
            else:
                im1 = ax1.imshow(image_l[:, :, i].T, origin='lower', extent=extent_l, cmap=settings['c_map'],
                                 vmin=settings['v_min'], vmax=settings['v_max'])
                im2 = ax2.imshow(image_r[:, :, i].T, origin='lower', extent=extent_r, cmap=settings['c_map'],
                                 vmin=settings['v_min'], vmax=settings['v_max'])

            import math
            orde = math.floor(np.log10(extent_l[1] - extent_l[0]))
            add_ticks(ax1, extent=extent_l,
                      del_xTick=[round((extent_l[1] - extent_l[0]) / 4, orde - 1),
                                 round((extent_l[1] - extent_l[0]) / 4, orde - 1) / 2],
                      x_label='y-coordinate $[$kpc$]$ vs. x-coordinate $[$kpc$]$', y_label='')
            add_ticks(ax2, extent=extent_r,
                      del_xTick=[round((extent_r[1] - extent_r[0]) / 4, orde - 1),
                                 round((extent_r[1] - extent_r[0]) / 4, orde - 1) / 2],
                      x_label='z-coordinate $[$kpc$]$ vs. x-coordinate $[$kpc$]$', y_label='')

            fig.subplots_adjust(left=0.05, right=0.88)
            add_colorbar(fig, im1, label=settings['bar_label'],
                         ax_adjust=(0.9, 0.125, 0.018, 0.78), clip=(settings['v_min'], settings['v_max']))
            # Create extra white space to the right of the right subplot
            # fig.subplots_adjust(right=0.87)
            # Create a new axis for the colorbar to the right of the subplots
            # cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
            # cbar = fig.colorbar(im1, cax=cax)
            # cbar.ax.set_ylabel(settings['bar_label'], fontproperties=font_prop)

            final_idx = 307 if self.run == '09_18_lastgigyear' else 127
            starting_index = final_idx - len(self.data['x-y'][0, 0, :]) + 1
            filename = (self.output_path + f'snap_{final_idx - i:03d}.png')
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

        filename = self.output_path + f'panels_{self.parameter}_snaps{self.snapshots[0]}-{self.snapshots[-1]}.pdf'
        plt.savefig(filename, dpi=240, bbox_inches='tight')
        plt.show()

    def plot_special(self):
        from matplotlib.patches import Ellipse

        if '09_18_lastgigyear' in self.output_path:
            idx, idx_end = 307, 118
        else:
            idx, idx_end = 127, 97

        settings = param_settings.get(self.part_type, {}).get(self.parameter, {})
        e_i = 'x-y'
        image = self.data[e_i]
        snap = self.snapshot
        img = image[:, :, idx - snap]
        redshifts = self.data['redshifts']
        lookback_times = self.data['lookback_times']
        extent = [self.data[e_i[2] + '_e'][0], self.data[e_i[2] + '_e'][-1],
                  self.data[e_i[0] + '_e'][0], self.data[e_i[0] + '_e'][-1]]

        font_prop = formatting()
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.tight_layout()

        def isodensityContour(I0, delI):
            dI = delI * I0  # delI % band
            # I0 = 5e6
            # dI = 0.02 * I0  # 2% band
            mask = np.abs(img - I0) < dI

            y, x = np.where(mask)
            weights = img[mask]
            # weights = np.ones(np.sum(mask))
            W = np.sum(weights)
            # centroid
            x0, y0 = np.sum(weights * x) / W, np.sum(weights * y) / W
            dx, dy = x - x0, y - y0

            M = np.array(
                [[np.sum(weights * dx ** 2), np.sum(weights * dx * dy)],
                 [np.sum(weights * dx * dy), np.sum(weights * dy ** 2)]]
            ) / W

            evals, evecs = np.linalg.eigh(M)
            # sort major --> minor
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]

            a = np.sqrt(evals[0])
            b = np.sqrt(evals[1])
            theta = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

            A = np.sqrt(2) * a
            B = np.sqrt(2) * b

            kpc_pixel = abs(extent[0] - extent[1]) / len(self.data['x_e'])
            X0 = (-(len(self.data['x_e']) / 2) + x0) * kpc_pixel
            Y0 = (-(len(self.data['x_e']) / 2) + y0) * kpc_pixel
            print(np.sqrt(X0 ** 2 + Y0 ** 2))

            ellipse = Ellipse(xy=(X0, Y0), width=2 * A * kpc_pixel, height=2 * B * kpc_pixel, angle=theta,
                              edgecolor='k', fc='none', lw=0.8, alpha=1)
            ax.add_patch(ellipse)

        log_I0s = np.arange(6.333, 8.667, 0.333)
        print(log_I0s)
        for I0 in log_I0s:
            isodensityContour(10 ** I0, 0.02)

        # cmap = settings['c_map']
        im = ax.imshow(img, origin='lower', extent=extent, cmap='Spectral',
                       norm=LogNorm(vmin=settings['v_min'], vmax=settings['v_max']),
                       rasterized=True)

        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(right=0.87)
        # Create a new axis for the colorbar to the right of the subplots
        cax = fig.add_axes([0.88, 0.125, 0.03, 0.785])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(settings['bar_label'], fontproperties=font_prop)

        ticksMajor = np.arange(-50, 50, 5)
        ticksMinor = np.arange(-50, 50, 1)
        ticks = ticksMajor[(ticksMajor >= extent[0]) & (ticksMajor <= extent[1])]
        minor_ticks = ticksMinor[(ticksMinor >= extent[0]) & (ticksMinor <= extent[1])]
        ax.set_xticks(ticks, [str(x) for x in ticks], fontproperties=font_prop)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(ticks, [str(y) for y in ticks], fontproperties=font_prop)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xlabel('x-coordinate' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
        ax.set_ylabel('y-coordinate' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)

        ax.text(-9.5, -9.5, s=f'z={redshifts[idx - snap]:.2f}, t={lookback_times[idx - snap]:.2f} Gyr',
                c='k', fontproperties=font_prop)

        filename = (self.output_path + f'/special_{self.parameter}_snap{self.snapshot}.pdf')
        plt.savefig(filename, dpi=240, bbox_inches='tight')
        plt.show()

    def tempOffset(self):
        settings = param_settings.get(self.part_type, {}).get(self.parameter, {})
        e_i = 'x-y'
        image = self.data[e_i]
        lookback_times = self.data['lookback_times']
        extent = [self.data[e_i[2] + '_e'][0], self.data[e_i[2] + '_e'][-1],
                  self.data[e_i[0] + '_e'][0], self.data[e_i[0] + '_e'][-1]]

        if '09_18_lastgigyear' in self.output_path:
            idx, idx_end = 307, 118
        else:
            idx, idx_end = 127, 97

        # I0_bar = 630 * 1e6
        # I0_disk = 10 * 1e6
        I0_bar = 1e8
        I0_disk = 1e7

        def offset(img, I0, delI):
            dI = delI * I0  # delI % band
            mask = np.abs(img - I0) < dI
            y, x = np.where(mask)
            weights = img[mask]
            W = np.sum(weights)
            # centroid
            x0, y0 = np.sum(weights * x) / W, np.sum(weights * y) / W
            kpc_pixel = abs(extent[0] - extent[1]) / len(self.data['x_e'])
            X0 = (-(len(self.data['x_e']) / 2) + x0) * kpc_pixel
            Y0 = (-(len(self.data['x_e']) / 2) + y0) * kpc_pixel
            return np.sqrt(X0 ** 2 + Y0 ** 2)  # offset

        for snap in range(idx, idx_end, -1):
            img = image[:, :, idx - snap].T
            if snap == idx:
                offsets = np.array([abs(offset(img, I0_bar, 0.02) - offset(img, I0_disk, 0.02))])
                times = np.array([lookback_times[0]])
            else:
                if snap % 3 == 0:
                    offsets = np.append(offsets, abs(offset(img, I0_bar, 0.02) - offset(img, I0_disk, 0.02)))
                    times = np.append(times, lookback_times[idx - snap])
                else:
                    pass

        font_prop = formatting()
        plt.rcParams.update({
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
        })

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.tight_layout()
        ax.plot(times, offsets, linewidth=0.8, c='tab:blue', label='Hestia LMC-SMC')

        x_lim = [2, 0]
        y_lim = [0, 3]

        ticksMajor = np.arange(0, 5, 0.5)
        ticksMinor = np.arange(0, 5, 0.1)
        ticks = ticksMajor[(ticksMajor >= extent[0]) & (ticksMajor <= extent[1])]
        minor_ticks = ticksMinor[(ticksMinor >= extent[0]) & (ticksMinor <= extent[1])]
        ax.set_xticks(ticks, [str(x) for x in ticks], fontproperties=font_prop)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(ticks, [str(y) for y in ticks], fontproperties=font_prop)
        ax.set_yticks(minor_ticks, minor=True)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('lookback time ' + r'$[$' + 'Gyr' + r'$]$', fontproperties=font_prop)
        ax.set_ylabel('bar-disk offset ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)

        ax.plot([0.62, 0.62], [y_lim[0], y_lim[1]], linewidth=0.8, c='k', linestyle='dashed', dashes=(6, 6))
        ax.text(0.71, 2, s='closest pericenter', rotation='vertical', fontproperties=font_prop)

        ax.plot(-pardy('x') + 0.62, pardy('y'), linewidth=0.8, c='tab:red', label='45 deg Pardy+16')

        plt.legend(loc='upper left', prop=font_prop)

        filename = self.output_path + f'/barOffset_{self.parameter}.pdf'
        plt.savefig(filename, dpi=240, bbox_inches='tight')
        plt.show()


def plot_scalarField(nom_pot, param='Phi', plane='x-y', extent=(-2, 2, -2, 2), t0=150 * u.Myr, ti=150 * u.Myr,
                     pixels=200,
                     bool_dark_mode=False, bool_orbit=False, verbose=False):
    from .potentials import LMCPotential, orbit
    from galpy.potential import evaluatePotentials, evaluateDensities

    def val2d(pot, param_, plane_, extent_, t=0.0, pixels_=100):
        e1 = np.linspace(extent_[0] * u.kpc, extent_[1] * u.kpc, pixels_)
        e2 = np.linspace(extent_[2] * u.kpc, extent_[3] * u.kpc, pixels_)

        E1, E2 = np.meshgrid(e1, e2, indexing='ij')

        if plane_ == 'x-y':  # along x=0
            RR = np.sqrt(E1 ** 2 + E2 ** 2)
            PHI = np.arctan2(E2, E1)
            ZZ = np.zeros_like(RR)
        elif plane_ == 'x-z':  # along y=0
            RR = np.sqrt(E1 ** 2)
            PHI = np.where(E1 >= 0, 0.0, np.pi) * u.rad
            ZZ = E2
        else:
            print(f'Error: {plane_} is an invalid scalarField plane; line {inspect.currentframe().f_lineno}')
            exit(1)

        RRf, PHIf, ZZf = RR.ravel(), PHI.ravel(), ZZ.ravel()

        if param_ == 'Phi':
            pot_flat = np.fromiter(
                (evaluatePotentials(pot, r, z, phi=phi, t=t).to(u.km ** 2 / u.s ** 2).value
                 for r, z, phi in zip(RRf, ZZf, PHIf)),
                dtype=float, count=len(RRf))
        elif param_ == 'rho':
            pot_flat = np.fromiter(
                (evaluateDensities(pot, r, z, phi=phi, t=t, forcepoisson=True).to(u.M_sun / u.pc ** 3).value
                 for r, z, phi in zip(RRf, ZZf, PHIf)),
                dtype=float, count=len(RRf))

        return pot_flat.reshape(RR.shape)

    if bool_dark_mode:
        from .utils import dark_mode
        dark_mode()

    fig, ax = plt.subplots(figsize=(5, 5))

    if nom_pot == 'lmc':
        Phi = LMCPotential(t0, LMC_model={'cdf': True}, v=verbose)
    elif nom_pot == 'lmc-smc':
        Phi = LMCPotential(t0, add_SMC=True, v=verbose)
    elif nom_pot == 'lmc-smc-mw':
        Phi = LMCPotential(t0, add_SMC=True, add_MW=True, v=verbose)
    else:
        print(f'Error: {nom_pot} is an invalid potential name; line {inspect.currentframe().f_lineno}')
        exit(1)

    if bool_dark_mode:
        fig.set_facecolor((33 / 255, 33 / 255, 33 / 255))

    img = val2d(Phi, param, plane, extent, t=ti, pixels_=pixels)
    add_ticks(ax, extent, [10, 2],
              x_label='x-coordinate' + r' $[$' + 'kpc' + r'$]$', y_label='y-coordiante' + r' $[$' + 'kpc' + r'$]$')

    if param == 'Phi':
        c = plt.imshow(img.T, origin='lower', cmap='Spectral', extent=extent,
                       vmin=img.min(), vmax=img.max())
    elif param == 'density':
        c = plt.imshow(np.log10(img.T), origin='lower', cmap='Spectral', extent=extent,
                       vmin=-3, vmax=0)
    else:
        print(f'Error: {param} is an invalid scalarField parameter ; line {inspect.currentframe().f_lineno}')
        exit(1)

    add_colorbar(fig, c, 'potential (km/s)' + r'$^2$')

    if bool_orbit:
        x, y = orbit(Phi, t0, plane=plane, obj='smc')
        ax.plot(x, y)

    # plt.savefig(f'/Users/ursa/dear-prudence/dynamics/test.pdf', dpi=240, bbox_inches='tight')
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
            'special': image_map.plot_special,
            'tempOffset': image_map.tempOffset,
        }
    else:
        print(f'Error: {plot_class} is an invalid plot class!')
        exit(1)

    return dispatcher[plot_type]()


# -------------------------------------------------------------
def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='scripts/util/fonts/AVHersheySimplexMedium.otf', size=12)

    # plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams.update({  # "grid.linestyle": "--",  # Dashed grid lines
        'axes.unicode_minus': False,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        'mathtext.fontset': 'cm',
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


def pardy(i):
    if i == 'x':
        return np.array([-0.49423076923076925, -0.4423076923076923, -0.37596153846153846, -0.29519230769230764,
                         -0.23461538461538461, -0.18557692307692308, -0.1365384615384615, -0.09326923076923077,
                         -0.05576923076923074, -0.0038461538461538325, 0.025000000000000022, 0.045192307692307754,
                         0.07115384615384623, 0.09423076923076923, 0.10576923076923084, 0.14903846153846156,
                         0.21826923076923077,
                         0.24711538461538463, 0.26730769230769236, 0.30192307692307696, 0.3538461538461539, 0.4,
                         0.4346153846153846,
                         0.5067307692307692, 0.5471153846153847, 0.5932692307692309, 0.6653846153846155,
                         0.714423076923077,
                         0.7865384615384616, 0.8586538461538462, 0.8817307692307692, 0.9307692307692308,
                         0.9884615384615385,
                         1.0548076923076923, 1.1557692307692309, 1.2192307692307693, 1.268269230769231,
                         1.3115384615384615,
                         1.3403846153846155, 1.4067307692307693, 1.4673076923076924, 1.501923076923077,
                         1.5653846153846156,
                         1.6519230769230768, 1.7096153846153848, 1.7615384615384615, 1.8048076923076923,
                         1.833653846153846,
                         1.8596153846153847, 1.9201923076923078, 2.0009615384615387, 2.055769230769231,
                         2.139423076923077])
    elif i == 'y':
        return np.array(
            [0.12226066897347171, 0.1395617070357555, 0.07900807381776237, 0.021337946943483288, 0.021337946943483288,
             0.09919261822376008, 0.16262975778546712, 0.12802768166089962, 0.1510957324106113, 0.09342560553633217,
             0.20588235294117646, 0.6701268742791234, 1.1833910034602075, 1.474625144175317, 1.6361014994232987,
             1.5092272202998847, 1.2871972318339102, 1.033448673587082, 0.6758938869665514, 0.28662053056516723,
             0.43656286043829295, 0.5922722029988466, 0.6643598615916955, 0.566320645905421, 0.44232987312572086,
             0.31257208765859285, 0.22606689734717417, 0.18858131487889274, 0.2433679354094579, 0.3760092272202999,
             0.47116493656286046, 0.497116493656286, 0.4682814302191465, 0.6268742791234141, 0.364475201845444,
             0.07612456747404844, 0.3615916955017301, 0.5778546712802768, 0.629757785467128, 0.5432525951557093,
             0.34717416378316035, 0.16551326412918105, 0.029988465974625123, 0.11937716262975778, 0.24625144175317187,
             0.1251441753171857, 0.029988465974625123, 0.13379469434832758, 0.21741637831603228, 0.11361014994232987,
             0.20299884659746248, 0.15974625144175314, 0.24625144175317187])
    else:
        print('Error: invalid argument for pardy() !')
        exit(1)


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
    'stars': {'massDen': {'c_map': 'magma', 'log': True, 'v_min': 1e6, 'v_max': 1e9,
                          'c_label': 'white', 'bar_label': r'$\Sigma[$ ' + 'M' + r'$_{\odot}$' + '/kpc' + r'$^2]$'},
              'surfaceDen': {'c_map': 'magma', 'log': True, 'v_min': 1e5, 'v_max': 1e9,
                             'c_label': 'white', 'bar_label': r'$\Sigma[$ ' + 'M' + r'$_{\odot}$' + '/kpc' + r'$^2]$'},
              'surfaceBrightness': {'c_map': 'bone_r', 'log': False, 'v_min': -16, 'v_max': -13,
                                    'c_label': 'white', 'bar_label': r'$\Sigma[$ ' + 'mag/kpc' + r'$^2]$'},
              'metallicity': {'c_map': 'Spectral', 'log': False, 'v_min': -0.5, 'v_max': 0.5,
                              'c_label': None, 'bar_label': r'$[$' + 'Z/H' + r'$]$'},
              'Fe_H': {'c_map': 'Spectral', 'log': False, 'v_min': -0.6, 'v_max': 0.4,
                       'c_label': None, 'bar_label': r'$[$' + 'Fe/H' + r'$]$'},
              'alpha_Fe': {'c_map': 'Spectral', 'log': False, 'v_min': 0.1, 'v_max': 0.3,
                           'c_label': None, 'bar_label': r'$[$' + 'alpha/Fe' + r'$]$'},
              'potential': {'c_map': 'magma', 'log': True, 'v_min': 1e5, 'v_max': 1e9,
                            'c_label': None, 'bar_label': r'$[$' + 'Potential' + r'$]$'},
              'mbp': {'c_map': 'magma', 'log': True, 'v_min': 1e5, 'v_max': 1e6,
                      'c_label': None, 'bar_label': r'$[$' + 'mbp' + r'$]$'},
              },
    'dm': {'massDen': {'c_map': 'viridis', 'log': True, 'v_min': 1e5, 'v_max': 1e9,
                       'c_label': 'white', 'bar_label': r'$\Sigma[$ ' + 'M' + r'$_{\odot}$' + '/kpc' + r'$^2]$'}}
}
