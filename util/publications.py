import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from scripts.util.archive.lucchini import lmc_temperatureProfile
from .astrometry import Measurements
from .utils import formatting


def chisholm2026_fig1():
    from .potentials import LMCPotential
    import astropy.units as u
    from galpy.potential import evaluatePotentials

    font_prop = formatting()

    plt.rcParams.update({
        'legend.labelcolor': 'white',
        # 'xtick.color': 'white',
        'xtick.labelcolor': 'black',
        # 'ytick.color': 'white',
        'ytick.labelcolor': 'black',
    })

    def val2d(pot, extent_, t=0.0, pixels_=100):
        e1 = np.linspace(extent_[0] * u.kpc, extent_[1] * u.kpc, pixels_)
        e2 = np.linspace(extent_[2] * u.kpc, extent_[3] * u.kpc, pixels_)

        E1, E2 = np.meshgrid(e1, e2, indexing='ij')

        RR = np.sqrt(E1 ** 2 + E2 ** 2)
        PHI = np.arctan2(E2, E1)
        ZZ = np.zeros_like(RR)
        RRf, PHIf, ZZf = RR.ravel(), PHI.ravel(), ZZ.ravel()

        pot_flat = np.fromiter(
            (evaluatePotentials(pot, r, z, phi=phi, t=t).to(u.km ** 2 / u.s ** 2).value
             for r, z, phi in zip(RRf, ZZf, PHIf)),
            dtype=float, count=len(RRf))
        return pot_flat.reshape(RR.shape)

    fig, ax = plt.subplots(2, 4, sharey=True, sharex=True,
                           figsize=(9.5, 4.8), gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
    axs = ax.flatten()

    Phi = LMCPotential(150 * u.Myr, LMC_model={'cdf': True}, add_SMC=True, add_MW=True, v=False)

    extent = (-1.5, 1.5, -1.5, 1.5)
    img = val2d(Phi, extent, t=150 * u.Myr, pixels_=200)

    fig.supxlabel('x-coordinate ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
    fig.supylabel('y-coordinate ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)

    # ---------------------------
    cmap = 'magma_r'
    # ---------------------------

    for ax in axs:
        c = ax.imshow(np.sqrt(-img.T), origin='lower', cmap=cmap, extent=extent,
                      vmin=320, vmax=335)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1.0, 1.5], ['-1.0', '', '0.0', '', '1.0', ''],
                      fontproperties=font_prop)
        ax.set_xticks(np.arange(-1.5, 1.5, 0.25), minor=True)
        ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1.0], ['', '-1.0', '', '0.0', '', '1.0'],
                      fontproperties=font_prop)
        ax.set_yticks(np.arange(-1.5, 1.5, 0.25), minor=True)
        ax.set_aspect(1)

    mu_ti, sigma_ti = 150 * u.Myr, 15 * u.Myr
    T = np.linspace(0, mu_ti.to(u.Myr).value + 5 * sigma_ti.to(u.Myr).value,
                    int(mu_ti.to(u.Myr).value + 5 * sigma_ti.to(u.Myr).value) + 1) * u.Myr
    from galpy.orbit import Orbit

    rng = np.random.default_rng(seed=6)
    Ti = rng.normal(mu_ti.to(u.Myr).value, sigma_ti.to(u.Myr).value, 8)
    mean_psi = [0.5, np.pi, 0, 0, 30, 0]
    sigma_psi = [0.4, np.pi / 2, 0.1, 20, 20, 20]
    # Convert (mean, std) in linear space to log-space parameters
    sigma_ln = np.sqrt(np.log(1 + (sigma_psi[0] / mean_psi[0]) ** 2))
    mu_ln = np.log(mean_psi[0]) - sigma_ln ** 2 / 2
    R = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=8) * u.kpc  # log normal
    phi = rng.uniform(0, 2 * np.pi, size=8) * u.rad  # uniform
    z = rng.normal(mean_psi[2], sigma_psi[2], size=8) * u.kpc  # gaussian
    vR = rng.normal(mean_psi[3], sigma_psi[3], size=8) * u.km / u.s  # gaussian
    vT = rng.normal(mean_psi[4], sigma_psi[4], size=8) * u.km / u.s  # gaussian (centered on v_circ(R0))
    vz = rng.normal(mean_psi[5], sigma_psi[5], size=8) * u.km / u.s  # gaussian

    vxvv = [R, vR, vT, z, vz, phi]
    o = Orbit(vxvv)
    o.integrate(T, Phi, progressbar=False)

    for i in range(8):
        t = np.linspace(0, Ti[i], 200) * u.Myr
        axs[i].plot(o.R(t)[i].to(u.kpc).value * np.cos(o.phi(t)[i].to(u.rad).value),
                    o.R(t)[i].to(u.kpc).value * np.sin(o.phi(t)[i].to(u.rad).value),
                    lw=0.8, c='white')

    # [left, bottom, width, height] in figure coords
    cax = fig.add_axes((0.92, 0.11, 0.02, 0.77))
    cbar = plt.colorbar(c, cax=cax)
    cbar_values, cbar_labels = [335, 330, 325, 320], ['335', '330', '325', '320']
    cbar.set_ticks([float(t) for t in cbar_values])
    cbar.set_ticklabels(cbar_labels, fontproperties=font_prop)
    cbar.ax.invert_yaxis()
    cbar.ax.set_ylabel(r'$\sqrt{-\Phi}$' + '  (km / s)', fontproperties=font_prop)

    basePath = f'/Users/ursa/dear-prudence/dynamics/bhPDF/lmc-smc-mw/'
    plt.savefig(f'{basePath}chisholm_fig1.pdf', dpi=240, bbox_inches='tight')


def chisholm2026_fig2():
    from astropy.coordinates import SkyCoord
    font_prop = formatting()

    plt.rcParams.update({
        'legend.labelcolor': 'white',
        'xtick.color': 'white',
        'xtick.labelcolor': 'black',
        'ytick.color': 'white',
        'ytick.labelcolor': 'black',
    })

    basePath = f'/Users/ursa/dear-prudence/dynamics/bhPDF/lmc-smc-mw/'
    fileName = f'bhPDF.lmc-smc-mw.N-100000.npz'
    data = np.load(basePath + fileName)

    bar = Measurements('LMC', 'bar')
    HVSs = Measurements('LMC', 'HVSs')
    HI = Measurements('LMC', 'HI')
    pm = Measurements('LMC', 'carbonStars')
    photo = Measurements('LMC', 'photometric')

    NGC1916 = Measurements('NGC1916', '')
    NGC1898 = Measurements('NGC1898', '')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(9.5, 5), gridspec_kw={'wspace': 0.15})
    # fig.subplots_adjust(wspace=0.1)
    # fig.tight_layout(pad=2)

    cmap = 'cubehelix'

    e_l = (data['ra_e'][0], data['ra_e'][-1], data['dec_e'][0], data['dec_e'][-1])
    e_r = (data['ra_e_zoom'][0], data['ra_e_zoom'][-1], data['dec_e_zoom'][0], data['dec_e_zoom'][-1])

    # -------- left panel --------

    from .astrometry import LMCDisk
    from astropy.coordinates import ICRS
    import astropy.units as u
    t = np.linspace(0, 2 * np.pi, 100)
    x, y = bar.R_bar * np.cos(t), bar.R_bar * bar.axisRatio * np.sin(t)
    Bar = SkyCoord(x=x * u.kpc, y=y * u.kpc, z=np.zeros(t.shape) * u.kpc, frame=LMCDisk(LMC=bar),
                   representation_type='cartesian').transform_to(ICRS)

    ax1.errorbar(HVSs.ra, HVSs.dec,
                 xerr=HVSs.sigma_ra, yerr=HVSs.sigma_dec,
                 c='white', elinewidth=0.8, capsize=2, alpha=1)  # (Lucchini+2025)
    ax1.scatter(HI.ra, HI.dec, c='white', marker='^', linewidths=0.8, s=64, alpha=1,
                label='HI kinematics')  # kim+1998
    ax1.scatter(pm.ra, pm.dec, edgecolors='white', marker='o', linewidths=0.8, s=64,
                facecolors='none', label='Carbon Stars')  # wan+2020
    ax1.scatter(photo.ra, photo.dec, c='white', marker='*', label='Photometric')  # van der Marel+2001
    ax1.scatter(bar.ra, bar.dec, c='white', marker='x', linewidths=0.8, s=64, alpha=1,
                label='RCSs (bar)')  # rathore+2025
    ax1.scatter(0, 0, marker='+', linewidths=0.8, s=84, c='white',
                label='HVSs')  # lucchini+2025, strictly for labelling purposes
    ax1.plot(Bar.ra.deg, Bar.dec.deg, c='white',
             linestyle='dashed', dashes=(6, 6), linewidth=0.8)

    ax1.imshow(data[f'Hi_radec'].T, origin='lower', cmap=cmap,
               extent=e_l, vmin=data[f'Hi_radec'].min(), vmax=data[f'Hi_radec'].max())

    ax1.set_xlim(e_l[:2])
    ax1.set_ylim(e_l[2:])
    ax1.set_aspect(abs(e_l[1] - e_l[0]) / abs(e_l[3] - e_l[2]))
    x_tM = np.arange(74, 86 + 1e-10, 2.0)  # major
    x_tm = np.arange(74, 87 + 1e-10, 1.0)  # minor
    y_tM = np.arange(-67, -73 - 1e-10, -1.0)
    y_tm = np.arange(e_l[2], e_l[3], 0.5)
    x_t, y_t = x_tM[(x_tM >= e_l[0]) & (x_tM <= e_l[1])], y_tM[(y_tM >= e_l[2]) & (y_tM <= e_l[3])]
    ax1.set_xticks(x_t, [str(round(x, 2)).replace("-", '-') for x in x_t], fontproperties=font_prop)
    ax1.set_xticks(x_tm[(x_tm >= e_l[0]) & (x_tm <= e_l[1])], minor=True)
    ax1.set_yticks(y_t, [str(round(y, 2)).replace("-", '-') for y in y_t], fontproperties=font_prop)
    ax1.set_yticks(y_tm[(y_tm >= e_l[2]) & (y_tm <= e_l[3])], minor=True)
    ax1.set_xlabel('right ascension (deg)', fontproperties=font_prop)
    ax1.set_ylabel('declination (deg)', fontproperties=font_prop)

    ax1.legend(loc='lower left', ncol=2, prop=font_prop)
    ax1.invert_xaxis()

    # -------- right panel --------

    img2 = ax2.imshow(data[f'Hi_radec_zoom'].T, origin='lower', cmap=cmap,
                      extent=e_r, vmin=data[f'Hi_radec_zoom'].min(), vmax=data[f'Hi_radec_zoom'].max())

    plt.rcParams.update({
        'text.color': 'white'
    })
    ax2.scatter(bar.ra, bar.dec, c='white', marker='x', linewidths=0.8, s=144, alpha=1)
    ax2.scatter(NGC1916.ra, NGC1916.dec, c='white', marker='*', s=64, label='NGC1916')
    ax2.text(NGC1916.ra - 0.06, NGC1916.dec + 0.03, s='NGC1916', fontproperties=font_prop)
    ax2.scatter(NGC1898.ra, NGC1898.dec, c='white', marker='*', s=64, label='NGC1898')
    ax2.text(NGC1898.ra + 0.08, NGC1898.dec - 0.09, s='NGC1898', fontproperties=font_prop)

    ax2.set_xlim(e_r[:2])
    ax2.set_ylim(e_r[2:])
    ax2.set_aspect(abs(e_r[1] - e_r[0]) / abs(e_r[3] - e_r[2]))
    x_tM, x_tm = np.arange(79, 81 + 1e-10, 1.0), np.arange(e_r[0], e_r[1], 0.2)
    y_tM, y_tm = np.arange(-69, -70 - 1e-10, -0.5), np.arange(e_r[2], e_r[3], 0.1)
    x_t, y_t = x_tM[(x_tM >= e_r[0]) & (x_tM <= e_r[1])], y_tM[(y_tM >= e_r[2]) & (y_tM <= e_r[3])]
    ax2.set_xticks(x_t, [str(round(x, 2)).replace("-", '-') for x in x_t], fontproperties=font_prop)
    ax2.set_xticks(x_tm[(x_tm >= e_r[0]) & (x_tm <= e_r[1])], minor=True)
    ax2.set_yticks(y_t, [str(round(y, 2)).replace("-", '-') for y in y_t], fontproperties=font_prop)
    ax2.set_yticks(y_tm[(y_tm >= e_r[2]) & (y_tm <= e_r[3])], minor=True)
    ax2.set_xlabel('right ascension (deg)', fontproperties=font_prop)
    ax2.set_ylabel('', fontproperties=font_prop)
    ax2.invert_xaxis()

    # [left, bottom, width, height] in figure coords
    cax = fig.add_axes((0.92, 0.153, 0.02, 0.685))
    cbar = plt.colorbar(img2, cax=cax)
    cbar_values, cbar_labels = [13, 130], ['1x', '10x']
    cbar.set_ticks([float(t) for t in cbar_values])
    cbar.set_ticklabels(cbar_labels, fontproperties=font_prop)
    cbar.ax.set_ylabel('SMBH probability amplitude (linearly-scaled)', fontproperties=font_prop)

    plt.savefig(f'{basePath}chisholm_fig2.pdf', dpi=240, bbox_inches='tight')


def figure1a():
    from scipy.interpolate import interp1d
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Ellipse
    import matplotlib.lines as mlines

    font_prop = formatting()
    m = Measurements()

    # -------------------------------------
    cmap = 'inferno'
    kind_interpolator = 'cubic'  # or 'linear', 'quadratic', 'cubic'
    x_lim = [-2, 4.999]
    y_lim = [-1.999, 5]
    # -------------------------------------

    basePath = '/Users/ursa/dear-prudence/halos/09_18_lastgigyear/'
    inputPath_left = f'{basePath}halo_38/kinematics/bhSloshing/bhSloshing.09_18_lastgigyear.halo_38.npz'
    inputPath_right = f'{basePath}halo_41/kinematics/bhSloshing/bhSloshing.09_18_lastgigyear.halo_41.npz'
    # -----------------------------------------

    halo_38 = np.load(inputPath_left)
    halo_41 = np.load(inputPath_right)
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9.5, 5), gridspec_kw={'wspace': 0})
    fig.subplots_adjust(wspace=0)  # Ensure no space between left and right plots
    fig.tight_layout(pad=2)

    cbar_label = r'lookback time $[$Gyr$]$'

    t_smooth = np.linspace(max(halo_38['lookback_times']), 0, 9999)

    x_ticksMajor = np.arange(-8, 8, 1)
    x_ticksMinor = np.arange(-8, 8, 0.2)
    x_ticks = x_ticksMajor[(x_ticksMajor >= x_lim[0]) & (x_ticksMajor <= x_lim[1])]
    y_ticksMajor = np.arange(-8, 8, 1)
    y_ticksMinor = np.arange(-8, 8, 0.2)
    y_ticks = y_ticksMajor[(y_ticksMajor >= y_lim[0]) & (y_ticksMajor <= y_lim[1])]

    # -----------------------------------------

    fx_38 = interp1d(halo_38['lookback_times'], halo_38['bh_coords'][:, 0], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)
    fy_38 = interp1d(halo_38['lookback_times'], halo_38['bh_coords'][:, 1], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)

    x_smooth, y_smooth = fx_38(t_smooth), fy_38(t_smooth)
    points_38 = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments_38 = np.concatenate([points_38[:-1], points_38[1:]], axis=1)

    # Create the LineCollection
    norm = plt.Normalize(t_smooth.min(), t_smooth.max())
    lc_38 = LineCollection(segments_38, cmap=cmap, norm=norm, lw=0.8)
    ax1.add_collection(lc_38)
    lc_38.set_array(t_smooth)  # THIS is the missing step

    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[0], y_lim[1])
    ax1.set_aspect('equal')

    ax1.set_xticks(x_ticks, [str(x).replace("-", '-') if x % 2 == 0 else '' for x in x_ticks],
                   fontproperties=font_prop)
    ax1.set_xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)
    ax1.set_yticks(y_ticks, [str(y).replace("-", '-') if y % 2 == 0 else '' for y in y_ticks],
                   fontproperties=font_prop)
    ax1.set_yticks(y_ticksMinor[(y_ticksMinor >= y_lim[0]) & (y_ticksMinor <= y_lim[1])], minor=True)

    ax1.set_ylabel(r'y-coordinate $[$kpc$]$', fontproperties=font_prop)

    ellipse = Ellipse(xy=(0, 0),
                      width=2 * m['LMC/bar/R_bar'],
                      height=2 * m['LMC/bar/R_bar'],
                      edgecolor='k', fc='lavender', lw=0.8, alpha=0.5, linestyle='dashed')
    ax1.add_patch(ellipse)

    # wrote a quick script in scratch.py to compute the position vector of halo_454 w.r.t halo_38 (09_18);
    halo_454 = np.array([  # all in kpc
        [55.40949442, 27.69370438, -7.82353457],  # snap 127 (t = 0.0 Gyr)
        [21.37590813, 0.28576893, -25.26025402],  # snap 124 (t ~ -0.5 Gyr)
        [-34.97754253, -19.16846084, -5.5476939],  # snap 121 (t ~ -1.0 Gyr)
        [-68.0686449, -31.42062865, 36.48823381],  # snap 118 (t ~ -1.5 Gyr)
    ])
    c_map = plt.cm.get_cmap(cmap)
    extent = [2.7, 2.7, 2, 2]
    for j in range(halo_454.shape[0]):
        norm = np.sqrt(halo_454[j, 0] ** 2 + halo_454[j, 1] ** 2)
        ax1.annotate("", xytext=(1.0 * halo_454[j, 0] / norm, 1.0 * halo_454[j, 1] / norm),
                     xy=(extent[j] * halo_454[j, 0] / norm, extent[j] * halo_454[j, 1] / norm),
                     arrowprops=dict(arrowstyle="->", linestyle='dashed', lw=0.8, alpha=0.8,
                                     ec=c_map((0.5 * j) / 1.5)))

    # Legend proxy (simple dashed arrow)
    proxy = mlines.Line2D(
        [], [],
        color=c_map(0.5 / 1.5),
        linestyle='dashed', lw=0.8, alpha=0.8,
        marker='$>$',
        markersize=3,
        label='direction towards SMC-like satellite\n' + r'$[$left panel$]$' + ' or MW-like host ' + r'$[$right panel$]$'
    )
    ax1.legend(handles=[proxy], loc='upper right', prop=font_prop)

    ax1.text(1, -1.7, s='(i).  with an \"SMC-like\" satellite', fontproperties=font_prop)
    # -----------------------------------------

    fx_41 = interp1d(halo_41['lookback_times'], halo_41['bh_coords'][:, 0], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)
    fy_41 = interp1d(halo_41['lookback_times'], halo_41['bh_coords'][:, 1], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)

    x_smooth, y_smooth = fx_41(t_smooth), fy_41(t_smooth)
    points_41 = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments_41 = np.concatenate([points_41[:-1], points_41[1:]], axis=1)

    # Create the LineCollection
    norm = plt.Normalize(t_smooth.min(), t_smooth.max())
    lc_41 = LineCollection(segments_41, cmap=cmap, norm=norm, lw=0.8)
    ax2.add_collection(lc_41)
    lc_41.set_array(t_smooth)

    ax2.set_xlim(x_lim[0], x_lim[1])
    ax2.set_ylim(y_lim[0], y_lim[1])
    ax2.set_aspect('equal')

    ax2.set_xticks(x_ticks, [str(x).replace("-", '-') if x % 2 == 0 else '' for x in x_ticks],
                   fontproperties=font_prop)
    ax2.set_xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)
    ax2.set_yticks(y_ticks, [str(y).replace("-", '-') if y % 2 == 0 else '' for y in y_ticks],
                   fontproperties=font_prop)
    ax2.set_yticks(y_ticksMinor[(y_ticksMinor >= y_lim[0]) & (y_ticksMinor <= y_lim[1])], minor=True)

    ellipse = Ellipse(xy=(0, 0),
                      width=2 * m['LMC/bar/R_bar'],
                      height=2 * m['LMC/bar/R_bar'],
                      edgecolor='k', fc='lavender', lw=0.8, alpha=0.5, linestyle='dashed')
    ax2.add_patch(ellipse)

    plt.text(-3.1, -2.6, s=r'x-coordinate $[$kpc$]$', fontproperties=font_prop)
    ax2.text(1.7, -1.7, s='(ii).  with a \"MW-like\" host', fontproperties=font_prop)
    # -----------------------------------------

    cbar_ax = fig.add_axes([0.96, 0.105, 0.02, 0.825])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(lc_38, cax=cbar_ax, orientation='vertical')
    cbar.ax.invert_yaxis()
    cbar.set_label(cbar_label, fontproperties=font_prop)

    cbar.set_ticks([1.5, 1.0, 0.5, 0.0])
    cbar.set_ticks(np.linspace(1.5, 0, 16), minor=True)
    cbar.set_ticklabels(['1.5', '1.0', '0.5', '0.0'], fontproperties=font_prop)

    # wrote a quick script in scratch.py to compute the position vector of halo-01 w.r.t halo_41;
    halo_01 = np.array([  # all in kpc
        [28.7976, 158.7235, -165.5649],  # snap 307 (t = 0.0 Gyr)
        [174.4389, 41.2026, -135.0526],  # snap 247 (t = -0.5 Gyr)
        [269.3728, -105.8549, -65.5906],  # snap 184 (t = -1.0 Gyr)
        [2324.8674, -250.8406, 52.4448],  # snap 119 (t = -1.5 Gyr)
    ])
    cmap = plt.cm.get_cmap(cmap)
    for j in range(halo_01.shape[0]):
        norm = np.sqrt(halo_01[j, 0] ** 2 + halo_01[j, 1] ** 2)
        ax2.annotate("", xytext=(2 * halo_01[j, 0] / norm, 2 * halo_01[j, 1] / norm),
                     xy=(4.5 * halo_01[j, 0] / norm, 4.5 * halo_01[j, 1] / norm),
                     arrowprops=dict(arrowstyle="->", linestyle='dashed', lw=0.8, alpha=0.8,
                                     ec=cmap((0.5 * j) / 1.5)))

    plt.savefig('/Users/ursa/dear-prudence/halos/09_18_lastgigyear/chisholm2026.bhSloshing.pdf',
                dpi=240, bbox_inches='tight')
    plt.show()


def figure1b():
    from scipy.interpolate import interp1d
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Ellipse
    import matplotlib.lines as mlines

    font_prop = formatting()
    m = Measurements()

    # -------------------------------------
    cmap = 'inferno'
    kind_interpolator = 'cubic'  # or 'linear', 'quadratic', 'cubic'
    x_lim = [-2, 4.999]
    y_lim = [-1.999, 5]
    z_lim = [-2, 1]
    # -------------------------------------

    basePath = '/Users/ursa/dear-prudence/halos/09_18_lastgigyear/'
    inputPath_left = f'{basePath}halo_38/kinematics/bhSloshing/bhSloshing.09_18_lastgigyear.halo_38.npz'
    inputPath_right = f'{basePath}halo_41/kinematics/bhSloshing/bhSloshing.09_18_lastgigyear.halo_41.npz'
    # -----------------------------------------

    halo_38 = np.load(inputPath_left)
    halo_41 = np.load(inputPath_right)
    # Create figure and subplots
    # fig, axs = plt.subplots(2, 2, sharey='all', sharex='row',
    #                                figsize=(9.5, 7), gridspec_kw={'wspace': 0})
    # fig.subplots_adjust(wspace=0)  # Ensure no space between left and right plots
    # fig.tight_layout(pad=2)
    # ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    """    fig = plt.figure(figsize=(9.5, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.02, wspace=0.02)
    fig.tight_layout()

    axes = gs.subplots(sharex='all', sharey='row')
    axes = axes.flatten()
    ax1, ax2, ax3, ax4 = axes"""

    fig = plt.figure(figsize=(9.5, 6.9))

    gs = fig.add_gridspec(2, 2, height_ratios=[7, 3], hspace=0.02, wspace=0.02)
    axes = gs.subplots(sharex=True, sharey='row')
    ax1, ax2, ax3, ax4 = axes.flatten()

    cbar_label = r'lookback time $[$Gyr$]$'

    t_smooth = np.linspace(max(halo_38['lookback_times']), 0, 9999)

    x_ticksMajor = np.arange(-8, 8, 1)
    x_ticksMinor = np.arange(-8, 8, 0.2)
    x_ticks = x_ticksMajor[(x_ticksMajor >= x_lim[0]) & (x_ticksMajor <= x_lim[1])]
    y_ticksMajor = np.arange(-8, 8, 1)
    y_ticksMinor = np.arange(-8, 8, 0.2)
    y_ticks = y_ticksMajor[(y_ticksMajor >= y_lim[0]) & (y_ticksMajor <= y_lim[1])]
    z_ticksMajor = np.arange(-8, 8, 1)
    z_ticksMinor = np.arange(-8, 8, 0.2)
    z_ticks = z_ticksMajor[(z_ticksMajor >= z_lim[0]) & (z_ticksMajor <= z_lim[1])]

    # -----------------------------------------

    fx_38 = interp1d(halo_38['lookback_times'], halo_38['bh_coords'][:, 0], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)
    fy_38 = interp1d(halo_38['lookback_times'], halo_38['bh_coords'][:, 1], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)
    fz_38 = interp1d(halo_38['lookback_times'], halo_38['bh_coords'][:, 2], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)

    x_smooth, y_smooth, z_smooth = fx_38(t_smooth), fy_38(t_smooth), fz_38(t_smooth)
    points_38xy = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    points_38xz = np.array([x_smooth, z_smooth]).T.reshape(-1, 1, 2)
    segments_38xy = np.concatenate([points_38xy[:-1], points_38xy[1:]], axis=1)
    segments_38xz = np.concatenate([points_38xz[:-1], points_38xz[1:]], axis=1)

    # Create the LineCollection
    norm = plt.Normalize(t_smooth.min(), t_smooth.max())
    lc_38xy = LineCollection(segments_38xy, cmap=cmap, norm=norm, lw=0.8)
    lc_38xz = LineCollection(segments_38xz, cmap=cmap, norm=norm, lw=0.8)
    ax1.add_collection(lc_38xy)
    lc_38xy.set_array(t_smooth)  # THIS is the missing step
    ax3.add_collection(lc_38xz)
    lc_38xz.set_array(t_smooth)  # THIS is the missing step

    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[0], y_lim[1])
    ax1.set_aspect('equal')

    ax1.set_xticks(x_ticks, [str(x).replace("-", '-') if x % 2 == 0 else '' for x in x_ticks],
                   fontproperties=font_prop)
    ax1.set_xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)
    ax1.set_yticks(y_ticks, [str(y).replace("-", '-') if y % 2 == 0 else '' for y in y_ticks],
                   fontproperties=font_prop)
    ax1.set_yticks(y_ticksMinor[(y_ticksMinor >= y_lim[0]) & (y_ticksMinor <= y_lim[1])], minor=True)

    ax1.set_ylabel(r'y-coordinate $[$kpc$]$', fontproperties=font_prop)

    ax3.set_xlim(x_lim[0], x_lim[1])
    ax3.set_ylim(z_lim[0], z_lim[1])
    ax3.set_aspect('equal')

    ax3.set_xticks(x_ticks, [str(x).replace("-", '-') if x % 2 == 0 else '' for x in x_ticks],
                   fontproperties=font_prop)
    ax3.set_xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)
    ax3.set_yticks(z_ticks, [str(z).replace("-", '-') if z % 2 == 0 else '' for z in z_ticks],
                   fontproperties=font_prop)
    ax3.set_yticks(z_ticksMinor[(z_ticksMinor >= z_lim[0]) & (z_ticksMinor <= z_lim[1])], minor=True)

    ax3.set_ylabel(r'z-coordinate $[$kpc$]$', fontproperties=font_prop)

    ellipse = Ellipse(xy=(0, 0),
                      width=2 * m['LMC/bar/R_bar'],
                      height=2 * m['LMC/bar/R_bar'],
                      edgecolor='k', fc='lavender', lw=0.8, alpha=0.5, linestyle='dashed')
    ax1.add_patch(ellipse)

    ellipse = Ellipse(xy=(0, 0),
                      width=2 * m['LMC/bar/R_bar'],
                      height=2 * m['LMC/bar/R_bar'] * 0.37,  # c/a from karachov+2024
                      edgecolor='k', fc='lavender', lw=0.8, alpha=0.5, linestyle='dashed')
    ax3.add_patch(ellipse)

    # wrote a quick script in scratch.py to compute the position vector of halo_454 w.r.t halo_38 (09_18);
    halo_454 = np.array([  # all in kpc
        [55.40949442, 27.69370438, -7.82353457],  # snap 127 (t = 0.0 Gyr)
        [21.37590813, 0.28576893, -25.26025402],  # snap 124 (t ~ -0.5 Gyr)
        [-34.97754253, -19.16846084, -5.5476939],  # snap 121 (t ~ -1.0 Gyr)
        [-68.0686449, -31.42062865, 36.48823381],  # snap 118 (t ~ -1.5 Gyr)
    ])
    c_map = plt.cm.get_cmap(cmap)
    extent = [2.7, 2.7, 2, 2]
    for j in range(halo_454.shape[0]):
        norm = np.sqrt(halo_454[j, 0] ** 2 + halo_454[j, 1] ** 2)
        ax1.annotate("", xytext=(1.0 * halo_454[j, 0] / norm, 1.0 * halo_454[j, 1] / norm),
                     xy=(extent[j] * halo_454[j, 0] / norm, extent[j] * halo_454[j, 1] / norm),
                     arrowprops=dict(arrowstyle="->", linestyle='dashed', lw=0.8, alpha=0.8,
                                     ec=c_map((0.5 * j) / 1.5)))

    # Legend proxy (simple dashed arrow)
    proxy = mlines.Line2D(
        [], [],
        color=c_map(0.5 / 1.5),
        linestyle='dashed', lw=0.8, alpha=0.8,
        marker='$>$',
        markersize=3,
        label='direction towards SMC-like satellite\n' + r'$[$left panel$]$' + ' or MW-like host ' + r'$[$right panel$]$'
    )
    ax1.legend(handles=[proxy], loc='upper right', prop=font_prop)

    ax1.text(0.7, -1.7, s='(i).  with an \"SMC-like\" satellite', fontproperties=font_prop)

    # -----------------------------------------

    fx_41 = interp1d(halo_41['lookback_times'], halo_41['bh_coords'][:, 0], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)
    fy_41 = interp1d(halo_41['lookback_times'], halo_41['bh_coords'][:, 1], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)
    fz_41 = interp1d(halo_41['lookback_times'], halo_41['bh_coords'][:, 2], kind=kind_interpolator,
                     fill_value=np.array([0.]), bounds_error=False)

    x_smooth, y_smooth, z_smooth = fx_41(t_smooth), fy_41(t_smooth), fz_41(t_smooth)
    points_41xy = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments_41xy = np.concatenate([points_41xy[:-1], points_41xy[1:]], axis=1)
    points_41xz = np.array([x_smooth, z_smooth]).T.reshape(-1, 1, 2)
    segments_41xz = np.concatenate([points_41xz[:-1], points_41xz[1:]], axis=1)

    # Create the LineCollection
    norm = plt.Normalize(t_smooth.min(), t_smooth.max())
    lc_41xy = LineCollection(segments_41xy, cmap=cmap, norm=norm, lw=0.8)
    ax2.add_collection(lc_41xy)
    lc_41xy.set_array(t_smooth)
    lc_41xz = LineCollection(segments_41xz, cmap=cmap, norm=norm, lw=0.8)
    ax4.add_collection(lc_41xz)
    lc_41xz.set_array(t_smooth)

    ax2.set_xlim(x_lim[0], x_lim[1])
    ax2.set_ylim(y_lim[0], y_lim[1])
    ax2.set_aspect('equal')

    ax2.set_xticks(x_ticks, [str(x).replace("-", '-') if x % 2 == 0 else '' for x in x_ticks],
                   fontproperties=font_prop)
    ax2.set_xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)
    ax2.set_yticks(y_ticks, [str(y).replace("-", '-') if y % 2 == 0 else '' for y in y_ticks],
                   fontproperties=font_prop)
    ax2.set_yticks(y_ticksMinor[(y_ticksMinor >= y_lim[0]) & (y_ticksMinor <= y_lim[1])], minor=True)

    ax4.set_xlim(x_lim[0], x_lim[1])
    ax4.set_ylim(z_lim[0], z_lim[1])
    ax4.set_aspect('equal')

    ax4.set_xticks(x_ticks, [str(x).replace("-", '-') if x % 2 == 0 else '' for x in x_ticks],
                   fontproperties=font_prop)
    ax4.set_xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)
    ax4.set_yticks(z_ticks, [str(z).replace("-", '-') if z % 2 == 0 else '' for z in z_ticks],
                   fontproperties=font_prop)
    ax4.set_yticks(z_ticksMinor[(z_ticksMinor >= z_lim[0]) & (z_ticksMinor <= z_lim[1])], minor=True)

    ellipse = Ellipse(xy=(0, 0),
                      width=2 * m['LMC/bar/R_bar'],
                      height=2 * m['LMC/bar/R_bar'],
                      edgecolor='k', fc='lavender', lw=0.8, alpha=0.5, linestyle='dashed')
    ax2.add_patch(ellipse)

    ellipse = Ellipse(xy=(0, 0),
                      width=2 * m['LMC/bar/R_bar'],
                      height=2 * m['LMC/bar/R_bar'] * 0.37,  # c/a from karachov+2024
                      edgecolor='k', fc='lavender', lw=0.8, alpha=0.5, linestyle='dashed')
    ax4.add_patch(ellipse)

    plt.text(-3.1, -2.6, s=r'x-coordinate $[$kpc$]$', fontproperties=font_prop)
    ax2.text(1.4, -1.7, s='(ii).  with a \"MW-like\" host', fontproperties=font_prop)
    # -----------------------------------------

    cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.768])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(lc_38xy, cax=cbar_ax, orientation='vertical')
    cbar.ax.invert_yaxis()
    cbar.set_label(cbar_label, fontproperties=font_prop)

    cbar.set_ticks([1.5, 1.0, 0.5, 0.0])
    cbar.set_ticks(np.linspace(1.5, 0, 16), minor=True)
    cbar.set_ticklabels(['1.5', '1.0', '0.5', '0.0'], fontproperties=font_prop)

    # wrote a quick script in scratch.py to compute the position vector of halo-01 w.r.t halo_41;
    halo_01 = np.array([  # all in kpc
        [28.7976, 158.7235, -165.5649],  # snap 307 (t = 0.0 Gyr)
        [174.4389, 41.2026, -135.0526],  # snap 247 (t = -0.5 Gyr)
        [269.3728, -105.8549, -65.5906],  # snap 184 (t = -1.0 Gyr)
        [2324.8674, -250.8406, 52.4448],  # snap 119 (t = -1.5 Gyr)
    ])
    cmap = plt.cm.get_cmap(cmap)
    for j in range(halo_01.shape[0]):
        norm = np.sqrt(halo_01[j, 0] ** 2 + halo_01[j, 1] ** 2)
        ax2.annotate("", xytext=(2 * halo_01[j, 0] / norm, 2 * halo_01[j, 1] / norm),
                     xy=(4.5 * halo_01[j, 0] / norm, 4.5 * halo_01[j, 1] / norm),
                     arrowprops=dict(arrowstyle="->", linestyle='dashed', lw=0.8, alpha=0.8,
                                     ec=cmap((0.5 * j) / 1.5)))

    plt.savefig('/Users/ursa/dear-prudence/halos/09_18_lastgigyear/chisholm2026.bhSloshing.pdf',
                dpi=240, bbox_inches='tight')
    plt.show()


# noinspection PyTupleAssignmentBalance
def figure2():
    from scipy.optimize import curve_fit
    from archive.hestia import twoD_Gaussian
    font_prop = formatting()
    m = Measurements()

    bool_gaussian = True
    # -------------------------------------
    x_lim = [73.0, 95]
    y_lim = [-74, -66]
    # -------------------------------------

    basePath = '/Users/ursa/dear-prudence/halos/09_18_lastgigyear/halo_41/observables/'
    inputPath = f'{basePath}bhPDF/09_18_lastgigyear.halo_41.snap278-6.bhPDF.radec.npz'
    # -----------------------------------------

    data = np.load(inputPath)
    f_PDF = data['f_PDF'].T
    bar = data['bar']
    lon_e = data['lon_edges']
    lat_e = data['lat_edges']

    print(f'(ra, dec, los)_bh : \t{round(data["center_bh"][2], 3)} deg, {round(data["center_bh"][1], 3)} deg, '
          f'{round(data["center_bh"][0], 3)} kpc')

    # -----------------------------------------
    if bool_gaussian:
        x_c = (lon_e[:-1] + lon_e[1:]) / 2
        y_c = (lat_e[:-1] + lat_e[1:]) / 2
        X, Y = np.meshgrid(x_c, y_c)

        lower_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

        initial_guess = (np.max(f_PDF), 80, -71, 0.5, 0.5, 0)  # amplitude, xo, yo, sigma_x, sigma_y, offset
        popt, _ = curve_fit(twoD_Gaussian, (X.ravel(), Y.ravel()), f_PDF.T.ravel(), p0=initial_guess,
                            bounds=(lower_bounds, upper_bounds), maxfev=9999)
        print(f'2-dim gaussian fit returned parameters -- \n'
              f'\tA : {popt[0]:.2e}\n'
              f'\tmu_ra : {popt[1]:.3f}\n'
              f'\tmu_dec : {popt[2]:.3f}\n'
              f'\tsigma_ra : {popt[3]:.3f}\n'
              f'\tsigma_dec : {popt[4]:.3f}\n')
    # -----------------------------------------

    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({
        'lines.dashed_pattern': (6, 6),
        'text.color': 'white',
        'xtick.color': 'white',
        'xtick.labelcolor': 'black',
        'ytick.color': 'white',
        'ytick.labelcolor': 'black',
        'legend.labelcolor': 'white'
    })
    ax = plt.gca()
    ax.invert_xaxis()

    # cmap = sns.color_palette("cubehelix", as_cmap=True, n_colors=8)
    # cmap = sns.cubehelix_palette(
    #     256, light=1,  # start near or at white
    #     dark=0, start=0.3,  # hue angle; tweak as desired
    #     # rot=-0.2,  # rotation through hue space
    #     gamma=0.7
    # )
    # cmap = sns.color_palette("magma_r", n_colors=256)
    # cmap = ListedColormap(cmap, name="cubehelix")
    # cmap = sns.color_palette("cubehelix", as_cmap=True)
    # cmap = cmap.reversed()  # ← reverse it
    # cmap = LinearSegmentedColormap(cmap)
    # Convert to a Matplotlib colormap
    # cmap = ListedColormap(cmap, name="cubehelix")

    # c = plt.imshow(np.log10(f_PDF.T), origin='lower', cmap=cmap,
    #                extent=(lon_e[0], lon_e[-1], lat_e[0], lat_e[-1]),
    #                vmin=np.log10(f_PDF.min()), vmax=np.log10(f_PDF.max()))
    c = plt.imshow(f_PDF.T, origin='lower', cmap='magma',
                   extent=(lon_e[0], lon_e[-1], lat_e[0], lat_e[-1]),
                   vmin=7e-6, vmax=7e-5)

    # ellipse = Ellipse(xy=(popt[1], popt[2]),
    #                   width=2 * popt[3],
    #                   height=2 * popt[4],
    #                   edgecolor='tab:blue', fc='none', lw=1.2, alpha=1, linestyle='solid', hatch='x')
    # ax.add_patch(ellipse)

    plt.errorbar(m['LMC/HVSs/ra'], m['LMC/HVSs/dec'],
                 xerr=m['LMC/HVSs/sigma_ra'], yerr=m['LMC/HVSs/sigma_dec'],
                 c='lightblue', elinewidth=0.8, capsize=2, alpha=0.8)  # (Lucchini+2025)
    # ellipse = Ellipse(xy=(m['LMC/HVSs/ra'], m['LMC/HVSs/dec']),
    #                   width=2 * m['LMC/HVSs/sigma_ra'], height=2 * m['LMC/HVSs/sigma_dec'],
    #                   edgecolor='tab:green', fc='none', lw=0.8, alpha=0.4, hatch='x',
    #                   label='dynamical center from HVSs')
    # ax.add_patch(ellipse)

    # ellipse = Ellipse(xy=(m['LMC/HI/ra'],  m['LMC/HI/dec']),
    #                   width=2 * m['LMC/HI/sigma_ra'], height=2 * m['LMC/HI/sigma_dec'],
    #                   edgecolor='tab:blue', fc='none', lw=0.8, alpha=0.5, hatch='x', label='HI center (Kim+1998)')
    # ax.add_patch(ellipse)
    plt.scatter(m['LMC/HI/ra'], m['LMC/HI/dec'], c='lightpink', marker='^', linewidths=0.8, s=64, alpha=0.8,
                label='HI kinematical center')  # (Kim+1998)
    # ellipse = Ellipse(xy=(m['LMC/pm/ra'],  m['LMC/pm/dec']),
    #                   width=2 * 0.02, height=2 * 0.02,
    #                    edgecolor='tab:purple', fc='none', lw=0.8, alpha=0.5, hatch='x',
    #                   label='Stellar kinematic center (Choi+2022)')
    # ax.add_patch(ellipse)
    plt.scatter(m['LMC/pm/ra'], m['LMC/pm/dec'], edgecolors='lavender', marker='o', linewidths=0.8, s=64,
                facecolors='none',
                label='Stellar kinematical center')  # (Choi+2022)

    # label='Extent of bar from RCSs; center aligned to mbp by def (Rathore+2025)')
    # ax.add_patch(ellipse)
    plt.plot(np.degrees(bar[1]), np.degrees(bar[2]), c='white', lw=0.8, alpha=0.8, linestyle='dashed')
    plt.scatter(m['LMC/bar/ra'], m['LMC/bar/dec'], c='white', marker='x', linewidths=0.8, s=64, alpha=0.8,
                label='Bar dynamical center')  # (Rathore+2025)
    plt.text(91, -71.1, s='LMC bar extent', fontproperties=font_prop)

    ax.annotate("", xytext=(m['LMC/disk/ra'], m['LMC/disk/dec']),
                xy=(m['LMC/disk/ra'] + 3 * m['LMC/disk/mu_alpha'], m['LMC/disk/dec'] + 3 * m['LMC/disk/mu_delta']),
                arrowprops=dict(arrowstyle="->", linestyle='solid', lw=0.8, alpha=0.8, color='white'))
    plt.text(m['LMC/disk/ra'] + 3 * m['LMC/disk/mu_alpha'] + 3.6,
             m['LMC/disk/dec'] + 3 * m['LMC/disk/mu_delta'] - 0.8,
             s='direction of LMC\nproper motion', fontproperties=font_prop)

    plt.scatter(0, 0, marker='+', linewidths=0.8, s=84, c='lightblue',
                label='LMC dynamical center from HVSs')  # strictly for labelling purposes

    plt.xlabel('right ascension (deg)', fontproperties=font_prop)
    plt.ylabel('declination (deg)', fontproperties=font_prop)

    x_ticksMajor = np.arange(70, 100, 2)
    x_ticksMinor = np.arange(70, 100, 0.5)
    plt.xlim(x_lim[1], x_lim[0])
    x_ticks = x_ticksMajor[(x_ticksMajor >= x_lim[0]) & (x_ticksMajor <= x_lim[1])]
    plt.xticks(x_ticks, [f'{int(x)}' if float(x) % 2 == 0 else '' for x in x_ticks],
               fontproperties=font_prop)
    plt.xticks(x_ticksMinor[(x_ticksMinor >= x_lim[0]) & (x_ticksMinor <= x_lim[1])], minor=True)

    y_ticksMajor = np.arange(-75, -60, 1)
    y_ticksMinor = np.arange(-75, -60, 0.25)
    plt.ylim(y_lim)
    y_ticks = y_ticksMajor[(y_ticksMajor >= y_lim[0]) & (y_ticksMajor <= y_lim[1])]
    plt.yticks(y_ticks, [str(int(y)).replace("-", '-') if float(y).is_integer() else '' for y in y_ticks],
               fontproperties=font_prop)
    plt.yticks(y_ticksMinor[(y_ticksMinor >= y_lim[0]) & (y_ticksMinor <= y_lim[1])], minor=True)

    # ax.set_aspect(abs(x_lim[1] - x_lim[0]) / abs(y_lim[1] - y_lim[0]) / (5 / 3))
    ax.set_aspect(abs(x_lim[1] - x_lim[0]) / abs(y_lim[1] - y_lim[0]) / (11 / 8))

    plt.legend(prop=font_prop, loc='lower left')

    # Colorbar
    import matplotlib
    background_color = matplotlib.cm.get_cmap('magma')(0)
    plt.gca().set_facecolor(background_color)
    cbar_ax = fig.add_axes([0.86, 0.11, 0.02, 0.77])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='vertical')
    # cbar_ax = fig.add_axes([0.12, 0.88, 0.78, 0.02])  # [left, bottom, width, height] in figure coords
    # cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    # cbar.set_ticks([f_PDF.min(), 7e-6, 7e-5, f_PDF.max()])
    # cbar.set_ticklabels(['', '1x', '10x', ''], fontproperties=font_prop)
    cbar.set_ticks([7e-6, 7e-5])
    cbar.set_ticklabels(['1x', '10x'], fontproperties=font_prop)
    cbar.set_label('Sample probability of SMBH', fontproperties=font_prop)

    output_path = f'{basePath}/chisholm2026.bhPDF.pdf'
    plt.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.show()


def figure2b():
    from scipy.interpolate import interp1d
    font_prop = formatting()

    bool_gaussian = False

    left_basePath = '/Users/ursa/dear-prudence/halos/09_18_lastgigyear/halo_41/kinematics/bhAccretion/'
    left_inputPath = f'{left_basePath}09_18_lastgigyear.halo_41.bhAccretion.npz'

    right_basePath = '/Users/ursa/dear-prudence/halos/09_18_lastgigyear/halo_41/observables/'
    right_inputPath = f'{right_basePath}bhPDF/09_18_lastgigyear.halo_41.snap278-6.bhPDF.radec.npz'
    # -----------------------------------------

    data = np.load(right_inputPath)
    f_PDF = data['f_PDF'].T
    bar = data['bar']
    lon_e = data['lon_edges']
    lat_e = data['lat_edges']

    print(f'(ra, dec, los)_bh : \t{round(data["center_bh"][2], 3)} deg, {round(data["center_bh"][1], 3)} deg, '
          f'{round(data["center_bh"][0], 3)} kpc')
    bhLocation(bool_gaussian, lon_e, lat_e, f_PDF)

    fig = plt.figure(figsize=(9.5, 6))

    # Outer grid: 3 rows, 2 columns
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1, 1.1],  # tweak if desired
        wspace=0.1,
        hspace=0.05
    )

    # Left column: three stacked plots sharing x
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    # Right column: single independent plot spanning all rows
    ax4 = fig.add_subplot(gs[:, 1])

    for ax in (ax1, ax2, ax3):
        ax.invert_xaxis()

    # Enforce 2:1 aspect ratio on left column
    # for ax in (ax_l1, ax_l2, ax_l3):
    #     ax.set_aspect(0.5)  # y/x = 0.5 → width:height = 2:1

    # -----------------------------------------------------------------------
    # left subplots

    xLim_l = [0.1, 0.4]

    x_label = 'Lookback Time ' + r'$[$' + 'Gyr' + r'$]$'
    ax3.set_xlabel(x_label, fontproperties=font_prop)

    data = np.load(left_inputPath)

    t_smooth = np.linspace(max(data['lookback_times']), 0, 9999)

    left_x_ticksMajor = np.arange(0, 1.5, 0.1)
    left_x_ticksMinor = np.arange(0, 1.5, 0.02)
    ax3.set_xlim(xLim_l[1], xLim_l[0])
    left_x_ticks = left_x_ticksMajor[(left_x_ticksMajor >= xLim_l[0]) & (left_x_ticksMajor <= xLim_l[1])]
    ax3.set_xticks(left_x_ticks, [str(round(float(x), 1)) for x in left_x_ticks], fontproperties=font_prop)
    ax3.set_xticks(left_x_ticksMinor[(left_x_ticksMinor >= xLim_l[0]) & (left_x_ticksMinor <= xLim_l[1])], minor=True)

    # ----------------------------------
    # ax1 (displacement)

    ax1_yLim = [0, 8]
    ax1.set_aspect(abs(xLim_l[1] - xLim_l[0]) / abs(ax1_yLim[1] - ax1_yLim[0]) / 2)

    ax1_label = 'BH displacement\n' + r'$[$' + 'kpc' + r'$]$'
    ax1.set_ylabel(ax1_label, fontproperties=font_prop)

    fx = interp1d(data['lookback_times'], data['x_bh'], kind='cubic', fill_value=np.array([0.]), bounds_error=False)
    ax1.plot(t_smooth, fx(t_smooth), linewidth=0.8, c='k')

    ax1_yTicksMajor = np.arange(0, 10, 2)
    ax1_yTicksMinor = np.arange(0, 10, 0.5)
    ax1.set_ylim(ax1_yLim[0], ax1_yLim[1])
    ax1_yTicks = ax1_yTicksMajor[(ax1_yTicksMajor >= ax1_yLim[0]) & (ax1_yTicksMajor <= ax1_yLim[1])]
    ax1.set_yticks(ax1_yTicks, [str(round(float(y), 1)) for y in ax1_yTicks], fontproperties=font_prop)
    ax1.set_yticks(ax1_yTicksMinor[(ax1_yTicksMinor >= ax1_yLim[0]) & (ax1_yTicksMinor <= ax1_yLim[1])], minor=True)

    # time sampled period
    ax1.axvspan(0.2, 0.3, color='black',
                linestyle='', alpha=0.1)
    ax1.plot([0.2, 0.2], [ax1_yLim[0], ax1_yLim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.plot([0.3, 0.3], [ax1_yLim[0], ax1_yLim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.text(0.293, 6.15, s='sampled\ntime span', rotation='horizontal', ha='left', fontproperties=font_prop)

    # ----------------------------------
    # ax2 (gas density)

    ax2_yLim = [4.5, 6.5]
    ax2.set_aspect(abs(xLim_l[1] - xLim_l[0]) / abs(ax2_yLim[1] - ax2_yLim[0]) / 2)

    ax2_label = 'log util gas density\n' + r'$[$' + 'log M' + r'$_{\odot}$' + '/kpc' + r'$^3$' + r'$]$'
    ax2.set_ylabel(ax2_label, fontproperties=font_prop)

    fm = interp1d(data['lookback_times'], np.log10(data['M_BHgas']), kind='cubic',
                  fill_value=np.array([0.]), bounds_error=False)
    ax2.plot(t_smooth, fm(t_smooth), linewidth=0.8, c='tab:blue')

    ax2_yTicksMajor = np.arange(0, 10, 1)
    ax2_yTicksMinor = np.arange(0, 10, 0.2)
    ax2.set_ylim(ax2_yLim[0], ax2_yLim[1])
    ax2_yTicks = ax2_yTicksMajor[(ax2_yTicksMajor >= ax2_yLim[0]) & (ax2_yTicksMajor <= ax2_yLim[1])]
    ax2.set_yticks(ax2_yTicks, [str(round(float(y), 1)) for y in ax2_yTicks], fontproperties=font_prop)
    ax2.set_yticks(ax2_yTicksMinor[(ax2_yTicksMinor >= ax2_yLim[0]) & (ax2_yTicksMinor <= ax2_yLim[1])], minor=True)

    # time sampled period
    ax2.axvspan(0.2, 0.3, color='black',
                linestyle='', alpha=0.1)
    ax2.plot([0.2, 0.2], [ax2_yLim[0], ax2_yLim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
    ax2.plot([0.3, 0.3], [ax2_yLim[0], ax2_yLim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)

    # ----------------------------------
    # ax3 (accretion rate)

    ax3_yLim = [-6, 0]
    ax3.set_aspect(abs(xLim_l[1] - xLim_l[0]) / abs(ax3_yLim[1] - ax3_yLim[0]) / 2)

    ax3_label = 'log accretion rate\n' + r'$[$' + 'log M' + r'$_{\odot}$' + '/yr' + r'$]$'
    ax3.set_ylabel(ax3_label, fontproperties=font_prop)

    fd = interp1d(data['lookback_times'], np.log10(data['M_dot']), kind='cubic',
                  fill_value=np.array([0.]), bounds_error=False)
    ax3.plot(t_smooth, fd(t_smooth), linewidth=0.8, c='tab:purple')

    ax3_yTicksMajor = np.arange(-10, 10, 2)
    ax3_yTicksMinor = np.arange(-10, 10, 0.5)
    ax3.set_ylim(ax3_yLim[0], ax3_yLim[1])
    ax3_yTicks = ax3_yTicksMajor[(ax3_yTicksMajor >= ax3_yLim[0]) & (ax3_yTicksMajor <= ax3_yLim[1])]
    ax3.set_yticks(ax3_yTicks, [str(round(float(y), 1)) for y in ax3_yTicks], fontproperties=font_prop)
    ax3.set_yticks(ax3_yTicksMinor[(ax3_yTicksMinor >= ax3_yLim[0]) & (ax3_yTicksMinor <= ax3_yLim[1])], minor=True)

    # time sampled period
    ax3.axvspan(0.2, 0.3, color='black',
                linestyle='', alpha=0.1)
    ax3.plot([0.2, 0.2], [ax3_yLim[0], ax3_yLim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
    ax3.plot([0.3, 0.3], [ax3_yLim[0], ax3_yLim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)

    for ax in (ax1, ax2):
        ax.tick_params(axis='x', which='both', labelbottom=False)

    # -----------------------------------------------------------------------
    # right subplot

    ax4.tick_params(
        which='both',
        color='white',  # tick marks
        labelcolor='black'  # tick labels
    )

    cmap = sns.color_palette("cubehelix", as_cmap=True)

    c = plt.imshow(f_PDF.T, origin='lower', cmap=cmap,
                   extent=(lon_e[0], lon_e[-1], lat_e[0], lat_e[-1]),
                   vmin=8e-6, vmax=8e-5)

    plt.rcParams.update({'text.color': 'white', 'legend.labelcolor': 'white'})

    ax4.errorbar(m['LMC/HVSs/ra'], m['LMC/HVSs/dec'],
                 xerr=m['LMC/HVSs/sigma_ra'], yerr=m['LMC/HVSs/sigma_dec'],
                 c='white', elinewidth=0.8, capsize=2, alpha=1)  # (Lucchini+2025)
    ax4.scatter(m['LMC/HI/ra'], m['LMC/HI/dec'], c='white', marker='^', linewidths=0.8, s=64, alpha=1,
                label='HI kinematical center')  # (Kim+1998)
    ax4.scatter(m['LMC/pm/ra'], m['LMC/pm/dec'], edgecolors='white', marker='o', linewidths=0.8, s=64,
                facecolors='none', label='Stellar kinematical center')  # (Choi+2022)

    # label='Extent of bar from RCSs; center aligned to mbp by def (Rathore+2025)')
    # ax.add_patch(ellipse)
    ax4.plot(np.degrees(bar[1]), np.degrees(bar[2]), c='white', lw=0.8, alpha=0.8, linestyle='dashed')
    ax4.scatter(m['LMC/bar/ra'], m['LMC/bar/dec'], c='white', marker='x', linewidths=0.8, s=64, alpha=1,
                label='Bar dynamical center')  # (Rathore+2025)
    ax4.text(87, -71.6, s='LMC bar extent', fontproperties=font_prop)

    ax4.annotate("", xytext=(m['LMC/disk/ra'], m['LMC/disk/dec']),
                 xy=(m['LMC/disk/ra'] + 3 * m['LMC/disk/mu_alpha'], m['LMC/disk/dec'] + 3 * m['LMC/disk/mu_delta']),
                 arrowprops=dict(arrowstyle="->", linestyle='solid', lw=0.8, alpha=1, color='white'))
    # ax4.text(m['LMC/disk/ra'] + 3 * m['LMC/disk/mu_alpha'] + 3.6,
    #          m['LMC/disk/dec'] + 3 * m['LMC/disk/mu_delta'] - 0.8,
    #          s='direction of LMC\nproper motion', fontproperties=font_prop)
    ax4.text(87, -68.1, s='direction of LMC\nproper motion', fontproperties=font_prop)

    ax4.scatter(0, 0, marker='+', linewidths=0.8, s=84, c='white',
                label='LMC dynamical center from HVSs')  # strictly for labelling purposes

    ax4.set_xlabel('right ascension (deg)', fontproperties=font_prop)
    ax4.set_ylabel('declination (deg)', fontproperties=font_prop)

    # -------------------------------
    ax4_xLim = [73.0, 88]
    ax4_yLim = [-74.1, -64]
    # -------------------------------

    ax4_xTicksMajor = np.arange(70, 100, 2)
    ax4_xTicksMinor = np.arange(70, 100, 0.5)
    ax4.set_xlim(ax4_xLim[1], ax4_xLim[0])
    ax4_xTicks = ax4_xTicksMajor[(ax4_xTicksMajor >= ax4_xLim[0]) & (ax4_xTicksMajor <= ax4_xLim[1])]
    ax4.set_xticks(ax4_xTicks, [f'{int(x)}' if float(x) % 2 == 0 else '' for x in ax4_xTicks],
                   fontproperties=font_prop)
    ax4.set_xticks(ax4_xTicksMinor[(ax4_xTicksMinor >= ax4_xLim[0]) & (ax4_xTicksMinor <= ax4_xLim[1])], minor=True)

    ax4_yTicksMajor = np.arange(-75, -60, 1)
    ax4_yTicksMinor = np.arange(-75, -60, 0.25)
    ax4.set_ylim(ax4_yLim[0], ax4_yLim[1])
    ax4_yTicks = ax4_yTicksMajor[(ax4_yTicksMajor >= ax4_yLim[0]) & (ax4_yTicksMajor <= ax4_yLim[1])]
    ax4.set_yticks(ax4_yTicks, [f'{int(y)}' if float(y) % 2 == 0 else '' for y in ax4_yTicks],
                   fontproperties=font_prop)
    ax4.set_yticks(ax4_yTicksMinor[(ax4_yTicksMinor >= ax4_yLim[0]) & (ax4_yTicksMinor <= ax4_yLim[1])], minor=True)

    ax4.set_aspect(2)

    ax4.legend(prop=font_prop, loc='upper left')

    # Colorbar
    # background_color = matplotlib.cm.get_cmap('magma')(0)
    # plt.gca().set_facecolor(background_color)
    cbar_ax = fig.add_axes([0.895, 0.11, 0.018, 0.77])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='vertical')
    # cbar_ax = fig.add_axes([0.12, 0.88, 0.78, 0.02])  # [left, bottom, width, height] in figure coords
    # cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    # cbar.set_ticks([f_PDF.min(), 7e-6, 7e-5, f_PDF.max()])
    # cbar.set_ticklabels(['', '1x', '10x', ''], fontproperties=font_prop)
    cbar.set_ticks([8e-6, 8e-5])
    cbar.set_ticklabels(['1x', '10x'], fontproperties=font_prop)
    cbar.set_label('Sample probability of SMBH', fontproperties=font_prop)

    output_path = f'{right_basePath}/chisholm2026.bhPDF.pdf'
    plt.savefig(output_path, dpi=240, bbox_inches='tight')
    # plt.show()


def bhLocation(bool_gaussian, lon_e, lat_e, f_PDF):
    from archive.hestia import twoD_Gaussian
    from scipy.optimize import curve_fit

    if bool_gaussian:
        x_c = (lon_e[:-1] + lon_e[1:]) / 2
        y_c = (lat_e[:-1] + lat_e[1:]) / 2
        X, Y = np.meshgrid(x_c, y_c)

        lower_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

        initial_guess = (np.max(f_PDF), 80, -71, 0.5, 0.5, 0)  # amplitude, xo, yo, sigma_x, sigma_y, offset
        popt, _ = curve_fit(twoD_Gaussian, (X.ravel(), Y.ravel()), f_PDF.T.ravel(), p0=initial_guess,
                            bounds=(lower_bounds, upper_bounds), maxfev=9999)
        print(f'2-dim gaussian fit returned parameters -- \n'
              f'\tA : {popt[0]:.2e}\n'
              f'\tmu_ra : {popt[1]:.3f}\n'
              f'\tmu_dec : {popt[2]:.3f}\n'
              f'\tsigma_ra : {popt[3]:.3f}\n'
              f'\tsigma_dec : {popt[4]:.3f}\n')
    else:
        pass


def imageMap():
    from scipy.interpolate import interp1d
    # ---------------------------------------
    run = '09_18'
    halo = 'halo_08'
    dims = '100x400'
    # ---------------------------------------
    snapshots = [96, 110, 118, 127]
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
    input_path_c1 = (f'/Users/dear-prudence/dear-prudence/halos/{run}/{halo}/images/gas/'
                     f'{c1}/{run}_{halo}_gas_{c1}_{dims}kpc.npz')
    input_path_c2 = (f'/Users/dear-prudence/dear-prudence/halos/{run}/{halo}/images/gas/'
                     f'{c2}/{run}_{halo}_gas_{c2}_{dims}kpc.npz')
    input_path_c3 = (f'/Users/dear-prudence/dear-prudence/halos/{run}/{halo}/images/{c3}/'
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

    smc_pos = np.array([[-19.03480445, 17.10259179, 8.30073911],
                        [-12.34195034, 31.39731278, -2.54328506],
                        [-2.91610407, 28.66926794, -11.30775597],
                        [13.37247795, 5.14369129, -11.79944825],
                        [10.27227456, -15.38941128, 6.12590731],
                        [-13.8859307, -19.40270977, 24.61455913],
                        [-31.45270922, -7.90006283, 22.80452767],
                        [-35.47289126, 1.39438566, 12.35971897],
                        [-24.33987522, 13.46742314, -10.29638605],
                        [8.5613704, 11.27178401, -22.5216539],
                        [30.61804531, 1.92726296, -14.08763956],
                        [46.03582035, -12.73650263, 9.66994082],
                        [47.50074501, -21.87394775, 31.51710815],
                        [43.21880314, -25.49851313, 44.80364754],
                        [27.05364223, -28.94136095, 59.96632003],
                        [8.39594882, -28.60417546, 65.76884192],
                        [-10.89032018, -23.55841239, 61.51212346],
                        [-26.65731923, -13.31168821, 47.96322424],
                        [-37.97696328, -2.79998848, 31.40162763],
                        [-41.34229691, 9.65042158, -7.33696094],
                        [-25.28466089, 12.92766948, -44.62858626],
                        [-0.57744305, 11.93004494, -71.1160769],
                        [19.2725703, 14.37434201, -86.34932766],
                        [38.50366483, 11.37915537, -96.47294679],
                        [58.94871676, 5.57107275, -99.35118956],
                        [77.71708405, 1.04081746, -96.90590204],
                        [92.23785577, 3.64449372, -93.17416302],
                        [102.80092067, 1.20061993, -91.62311522],
                        [117.71148627, -6.61897417, -80.73136969],
                        [127.26564305, -9.68001639, -71.31379593],
                        [133.69291089, -13.07801937, -60.08826635],
                        [138.764247, -21.23917152, -36.94212509],
                        [135.47062305, -25.85245948, 25.57180864],
                        [119.57102276, -26.80767841, 55.97740745],
                        [98.43806934, - 30.92416, 77.13987218]])

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

            virial_radii = [177.1, 167.6, 156.2, 145.8]  # kpc
            if j == 1:
                circle1 = plt.Circle((0, 0), virial_radii[i], fill=False, color='k', linestyle='dashed', lw=0.5)
                ax.add_patch(circle1)
                ax.text(virial_radii[i] / np.sqrt(2), -1 * virial_radii[i] / np.sqrt(2) - 20,
                        s=r'R$_{\text{vir}}$', fontproperties=font_prop)
            elif j == 2:
                plus_numSnaps = 3 if i != 3 else 5
                minus_numSnaps = 3 if i != 3 else 0
                fx = interp1d(range(snapshots[i] - minus_numSnaps, snapshots[i] + plus_numSnaps),
                              smc_pos[127 - snapshots[i] - minus_numSnaps:127 - snapshots[i] + plus_numSnaps, 0],
                              kind='quadratic')
                fz = interp1d(range(snapshots[i] - minus_numSnaps, snapshots[i] + plus_numSnaps),
                              smc_pos[127 - snapshots[i] - minus_numSnaps:127 - snapshots[i] + plus_numSnaps, 2],
                              kind='quadratic')
                t_smooth = np.linspace(snapshots[i] - minus_numSnaps, snapshots[i] + plus_numSnaps, 100)
                t_min, t_max = fx.x[0], fx.x[-1]
                t_smooth_clipped = np.clip(t_smooth, t_min, t_max)
                x_smooth, z_smooth = fx(t_smooth_clipped), fz(t_smooth_clipped)

                ax.plot(x_smooth, z_smooth, c='white', lw=0.5, linestyle='dashed', dashes=(6, 6))

                # === Add triangular arrows at start and end ===
                arrowprops = dict(arrowstyle='->', color='white', lw=0.5)
                # Start arrow
                ax.annotate('', xy=(x_smooth[0], z_smooth[0]), xytext=(x_smooth[1], z_smooth[1]),
                            arrowprops=arrowprops)
                # End arrow
                ax.annotate('', xy=(x_smooth[-1], z_smooth[-1]), xytext=(x_smooth[-2], z_smooth[-2]),
                            arrowprops=arrowprops)

                if i == 0:
                    # circle2 = plt.Circle((0, 0), 50, fill=False, color='white', linestyle='dashed', lw=0.5)
                    # ax.add_patch(circle2)
                    ax.text(-153, 25, s='LMC-analog', color='white', fontproperties=font_prop)

                    # circle2 = plt.Circle((139, -37), 25, fill=False, color='white',
                    #                      linestyle='dashed', lw=0.5)
                    # ax.add_patch(circle2)
                    ax.text(0, -50, s='SMC-analog', color='white', fontproperties=font_prop)

                    # ax.arrow(125, -60, -15, -25, color='white', lw=0.5, head_width=3.0)
                    ax.text(50, -110, s='approx. orbit', color='white', fontproperties=font_prop)
                elif i == 1:
                    ax.text(55, 40, s='bridge feature', color='white', fontproperties=font_prop)
                    ax.plot([50, 0], [45, 30], lw=0.5, color='white')
                elif i == 3:
                    ax.text(60, 40, s='tidal stellar\nstream', color='white', fontproperties=font_prop)
                    ax.plot([55, 20], [55, 30], lw=0.5, color='white')

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
            tick_labels = ['10' + r'$^4$', '10' + r'$^5$', '10' + r'$^6$', '10' + r'$^7$', '10' + r'$^8$']

        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels, font_properties=font_prop)

    # After subplot loop
    fig.text(0.5, 0.08, 'x-coordinate ' + r'$[$' + 'kpc' + r'$]$',
             ha='center', va='center', fontproperties=font_prop)
    fig.text(0.08, 0.5, 'z-coordinate ' + r'$[$' + 'kpc' + r'$]$',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)

    dynamicalTime = - 1.517

    fig.text(0.91, 0.79, f'z={redshifts[last_idx - snapshots[0]]:.2f}, '
                         r't$_{\text{dyn}}$ = '
             + f'-{round(lookback_times[last_idx - snapshots[0]] + dynamicalTime, 3):.2f} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)
    fig.text(0.91, 0.60, f'z={redshifts[last_idx - snapshots[1]]:.2f}, '
                         r't$_{\text{dyn}}$ = '
             + f'-{round(lookback_times[last_idx - snapshots[1]] + dynamicalTime, 3):.2f} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)
    fig.text(0.91, 0.40, f'z={redshifts[last_idx - snapshots[2]]:.2f}, '
                         r't$_{\text{dyn}}$ = '
             + f'-{round(lookback_times[last_idx - snapshots[2]] + dynamicalTime, 3):.2f} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)
    fig.text(0.91, 0.2, f'z={redshifts[last_idx - snapshots[3]]:.2f}, '
                        r't$_{\text{dyn}}$ = '
             + f'+{-1 * round(lookback_times[last_idx - snapshots[3]] + dynamicalTime, 3):.2f} Gyr',
             ha='center', va='center', rotation='vertical', fontproperties=font_prop)

    output_path = '//halos/09_18/halo_08/images/'
    filename = (output_path + 'chisholm2025_imageMap.pdf')
    plt.savefig(filename, dpi=500, bbox_inches='tight')


def streamPlot():
    input_path1 = ('/Users/dear-prudence/dear-prudence/halos/09_18/halo_08/kinematics/orbits/'
                   'orbitalDistance.09_18.halo_08-smc.npz')
    input_path2 = ('/Users/dear-prudence/dear-prudence/halos/09_18/stream/observables/NH0/'
                   '09_18.stream.snap118.NH0.faux.npz')
    input_path3 = ('/Users/dear-prudence/dear-prudence/halos/09_18/stream/observables/NH0/'
                   '09_18.stream.snap123.NH0.faux.npz')
    input_path4 = ('/Users/dear-prudence/dear-prudence/halos/09_18/stream/observables/NH0/'
                   '09_18.stream.snap127.NH0.faux.npz')

    # --------------------------------------------------------------------
    font_prop = formatting()
    # Create figure and subplots
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)  # Create first subplot (Cartesian)
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')  # Create second subplot (Polar)
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')  # Create third subplot (Polar)
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')  # Create fourth subplot (Polar)
    fig.subplots_adjust(wspace=0.1)  # Ensure no space between left and right plots
    # --------------------------------------------------------------------

    data1 = np.load(input_path1)
    # -------------------------------
    dynamical_zeroTime = -1.517  # snap 119
    besla_offsetTime = -1.1  # simulation ran for ~5.9 Gyr
    # -------------------------------

    ax1.plot(-1 * data1['hestia_times'] - dynamical_zeroTime, data1['hestia_distances'],
             c='k', linestyle='solid', lw=1.5, label='Magellanic-analog (this work)')
    ax1.plot(-1 * data1['besla_times'] - besla_offsetTime, data1['besla_distances'], label='Besla et al. 2012, model 2',
             c='tab:blue', linestyle='solid', lw=1, alpha=1)
    # ax.plot(-1 * self.data['pardy_times'], self.data['pardy_distances'], label='pardy+2018, 9:1',
    #         c='tab:green', linestyle='solid', alpha=0.5)
    ax1.plot(-1 * data1['lucchini_times'], data1['lucchini_distances'], label='Lucchini et al. 2020',
             c='tab:green', linestyle='solid', lw=1, alpha=1)

    # Other formatting stuff
    x_lim = [-7, 2]
    y_lim = [0, 200]
    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[0], y_lim[1])

    aspect_ratio = abs(x_lim[1] - x_lim[0]) / abs(y_lim[1] - y_lim[0])
    ax1.set_aspect(aspect_ratio)

    ax1.set_xticks(np.linspace(x_lim[0], x_lim[1], x_lim[1] - x_lim[0] + 1),
                   labels=['-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2'],
                   fontproperties=font_prop)
    ax1.set_xticks(np.linspace(x_lim[0], x_lim[1], 5 * (x_lim[1] - x_lim[0]) + 1), minor=True)
    ax1.set_yticks(np.linspace(0, 200, 5),
                   labels=['0', '50', '100', '150', '200'],
                   fontproperties=font_prop)
    ax1.set_yticks(np.linspace(0, 200, 21), minor=True)
    # plt.gca().invert_xaxis()
    # Set font weight for tick labels
    ax1.set_xlabel('Dynamical Time ' + r'$[$' + 'Gyr' + r'$]$', fontproperties=font_prop)
    ax1.set_ylabel('Distance ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
    ax1.tick_params(axis='x', which='both', bottom=True, top=False)

    # --- Top x-axis ---
    redshifts = np.array(np.linspace(0, 1, 11))
    # _, lookback_times = get_lookbackTimes(run=None, snaps=None, redshifts=redshifts)
    ax_top = ax1.secondary_xaxis('top')  # share the same x-axis scale
    ax_top.set_xlabel('z', fontproperties=font_prop)
    # ax_top.set_xticks(-1 * np.array(lookback_times) + 1.33)
    ax_top.set_xticklabels(['0', '0.1', '', '', '', '0.5', '', '', '', '', '1.0'],
                           fontproperties=font_prop)

    # zero dynamical time line
    ax1.plot([0, 0], [y_lim[0], y_lim[1]], linestyle='dashed', color='black',
             dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.text(0.1, 80, s='zero dynamical time', rotation='vertical', color='black',
             fontproperties=font_prop)

    # snap 119; z = 0.099, t_dym = 1.33 Gyr - 1.33 Gyr
    ax1.text(0.1, 55, s='(i)', rotation='horizontal', color='black', fontproperties=font_prop)
    # snap 123; z = 0.046, t_dym = 0.64 Gyr - 1.33 Gyr
    ax1.plot([-1 * (0.64 - 1.33), -1 * (0.64 - 1.33)], [10, 50], linestyle='dashed', color='black',
             dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.text(-1 * (0.64 - 1.33) - 0.1, 55, s='(ii)', rotation='horizontal', color='black',
             fontproperties=font_prop)

    # snap 127; z = 0.0, t_dym = 0.0 Gyr - 1.33 Gyr
    ax1.plot([-1 * (0.0 - 1.33), -1 * (0.0 - 1.33)], [10, 50], linestyle='dashed', color='black',
             dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.text(-1 * (0.0 - 1.33) - 0.1, 55, s='(iii)', rotation='horizontal', color='black',
             fontproperties=font_prop)

    # plt.grid(True, alpha=0.25, color='k', linestyle='dashed', lw=0.5)
    ax1.legend(prop=font_prop)

    # ---------------------------------------------------------------
    data2 = np.load(input_path2)

    nH_map = np.log10(data2['nH_map']).T
    lon_edges = data2['lon_edges']
    lat_edges = data2['lat_edges']
    smc_position = data2['smc_position']

    cmap = 'BuPu'

    # Plot with pcolormesh — you'll need bin edges for R and Phi
    phi_edges = np.radians(lon_edges)
    r_edges = np.radians(90 + lat_edges)  # accounts for +20° now

    Phi_edges, R_edges = np.meshgrid(phi_edges, r_edges)

    # Plot
    c = ax2.pcolormesh(Phi_edges, R_edges, nH_map, shading='auto', cmap=cmap, vmin=14, vmax=20, rasterized=True)

    # Aesthetics
    ax2.set_theta_zero_location("E")  # 0° longitude at right
    # ax.set_theta_direction(-1)  # Longitudes increase clockwise
    ax2.set_rlabel_position(90)  # Move radial labels to a better spot

    lon_tick_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
    ax2.set_xticks(np.radians(lon_tick_degrees))
    ax2.set_xticklabels([r'0$^{\circ}$' + '\n+x', r'45$^{\circ}$', r'90$^{\circ}$' + ', +y', r'135$^{\circ}$',
                         r'180$^{\circ}$' + '\n-x',
                         r'-135$^{\circ}$', r'-90$^{\circ}$' + ', -y', r'-45$^{\circ}$'], fontproperties=font_prop)

    # Custom radial ticks to show latitudes
    lat_tick_degrees = [-90, -60, -30, 0, 30]
    ax2.set_rticks(np.radians(90 + np.array(lat_tick_degrees)))
    ax2.set_yticklabels(['', r'-60$^{\circ}$', r'-30$^{\circ}$',
                         r'0$^{\circ}$', ''], fontproperties=font_prop)

    ax2.grid(True, linestyle='dashed', dashes=(6, 6), linewidth=0.5, color='k', alpha=0.8)

    # SMC-analog
    ax2.scatter(np.radians(smc_position[0]), np.radians(90 + smc_position[1]), s=35, marker='+', linewidths=1.0, c='k',
                label='SMC-analog')
    # ax2.text(0.945 * np.pi, 1.5, s='SMC-analog', fontproperties=font_prop)
    # LMC-analog
    ax2.scatter(0, 0, s=18, marker='D', color='k', label='LMC-analog')
    # ax2.text(1.3 * np.pi, 0.5, s='LMC-analog', fontproperties=font_prop)
    # ax2.annotate(text='', xytext=(np.radians(smc_position[0]), np.radians(90 + smc_position[1])),
    #              xy=(np.radians(smc_position[0]) + 0.5, np.radians(90 + smc_position[1]) + 0.1),
    #              arrowprops=dict(arrowstyle='->'), color='k')

    # ax2.legend(prop=font_prop, loc='lower right', bbox_to_anchor=(1.1, -0.15))
    ax2.text(1.1, 2.2, s='(i) z=0.114,\nt' + r'$_{\text{dyn}}$=0.0 Gyr', rotation='horizontal', color='black',
             fontproperties=font_prop, c='white')

    # ---------------------------------------------------------------
    data3 = np.load(input_path3)

    nH_map = np.log10(data3['nH_map']).T
    lon_edges = data3['lon_edges']
    lat_edges = data3['lat_edges']
    smc_position = data3['smc_position']

    cmap = 'BuPu'

    # Plot with pcolormesh — you'll need bin edges for R and Phi
    phi_edges = np.radians(lon_edges)
    r_edges = np.radians(90 + lat_edges)  # accounts for +20° now

    Phi_edges, R_edges = np.meshgrid(phi_edges, r_edges)

    # Plot
    c = ax3.pcolormesh(Phi_edges, R_edges, nH_map, shading='auto', cmap=cmap, vmin=14, vmax=20, rasterized=True)

    # Aesthetics
    ax3.set_theta_zero_location("E")  # 0° longitude at right
    # ax.set_theta_direction(-1)  # Longitudes increase clockwise
    ax3.set_rlabel_position(90)  # Move radial labels to a better spot

    lon_tick_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
    ax3.set_xticks(np.radians(lon_tick_degrees))
    ax3.set_xticklabels([r'0$^{\circ}$' + '\n+x', r'45$^{\circ}$', r'90$^{\circ}$' + ', +y', r'135$^{\circ}$',
                         r'180$^{\circ}$' + '\n-x',
                         r'-135$^{\circ}$', r'-90$^{\circ}$' + ', -y', r'-45$^{\circ}$'], fontproperties=font_prop)

    # Custom radial ticks to show latitudes
    lat_tick_degrees = [-90, -60, -30, 0, 30]
    ax3.set_rticks(np.radians(90 + np.array(lat_tick_degrees)))
    ax3.set_yticklabels(['', r'-60$^{\circ}$', r'-30$^{\circ}$',
                         r'0$^{\circ}$', ''], fontproperties=font_prop)

    ax3.grid(True, linestyle='dashed', dashes=(6, 6), linewidth=0.5, color='k', alpha=0.8)

    # SMC-analog
    ax3.scatter(np.radians(smc_position[0]), np.radians(90 + smc_position[1]), s=35, marker='+', linewidths=1.0, c='k',
                label='SMC-analog')
    # ax2.text(0.945 * np.pi, 1.5, s='SMC-analog', fontproperties=font_prop)
    # LMC-analog
    ax3.scatter(0, 0, s=18, marker='D', color='k', label='LMC-analog')
    # ax2.text(1.3 * np.pi, 0.5, s='LMC-analog', fontproperties=font_prop)

    # ax3.legend(prop=font_prop, loc='lower right', bbox_to_anchor=(1.1, -0.15))
    ax3.text(1.1, 2.2, s=f'(ii) z=0.046,\nt' + r'$_{\text{dyn}}$=+0.87 Gyr', rotation='horizontal', color='black',
             fontproperties=font_prop, c='white')

    # ---------------------------------------------------------------
    data4 = np.load(input_path4)

    nH_map = np.log10(data4['nH_map']).T
    lon_edges = data4['lon_edges']
    lat_edges = data4['lat_edges']
    smc_position = data4['smc_position']

    cmap = 'BuPu'

    # Plot with pcolormesh — you'll need bin edges for R and Phi
    phi_edges = np.radians(lon_edges)
    r_edges = np.radians(90 + lat_edges)  # accounts for +20° now

    Phi_edges, R_edges = np.meshgrid(phi_edges, r_edges)

    # Plot
    c = ax4.pcolormesh(Phi_edges, R_edges, nH_map, shading='auto', cmap=cmap, vmin=14, vmax=20, rasterized=True)

    # Aesthetics
    ax4.set_theta_zero_location("E")  # 0° longitude at right
    # ax.set_theta_direction(-1)  # Longitudes increase clockwise
    ax4.set_rlabel_position(90)  # Move radial labels to a better spot

    lon_tick_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
    ax4.set_xticks(np.radians(lon_tick_degrees))
    ax4.set_xticklabels([r'0$^{\circ}$' + '\n+x', r'45$^{\circ}$', r'90$^{\circ}$' + ', +y', r'135$^{\circ}$',
                         r'180$^{\circ}$' + '\n-x',
                         r'-135$^{\circ}$', r'-90$^{\circ}$' + ', -y', r'-45$^{\circ}$'], fontproperties=font_prop)

    # Custom radial ticks to show latitudes
    lat_tick_degrees = [-90, -60, -30, 0, 30]
    ax4.set_rticks(np.radians(90 + np.array(lat_tick_degrees)))
    ax4.set_yticklabels(['', r'-60$^{\circ}$', r'-30$^{\circ}$',
                         r'0$^{\circ}$', ''], fontproperties=font_prop)

    ax4.grid(True, linestyle='dashed', dashes=(6, 6), linewidth=0.5, color='k', alpha=0.8)

    # SMC-analog
    ax4.scatter(np.radians(smc_position[0]), np.radians(90 + smc_position[1]), s=35, marker='+', linewidths=1.0, c='k',
                label='SMC-analog')
    # ax2.text(0.945 * np.pi, 1.5, s='SMC-analog', fontproperties=font_prop)
    # LMC-analog
    ax4.scatter(0, 0, s=18, marker='D', color='k', label='LMC-analog')
    # ax2.text(1.3 * np.pi, 0.5, s='LMC-analog', fontproperties=font_prop)

    ax4.legend(prop=font_prop, loc='lower right', bbox_to_anchor=(1.1, -0.24))
    ax4.text(1.1, 2.2, s='(iii) z=0.0,\nt' + r'$_{\text{dyn}}$=+1.51 Gyr', rotation='horizontal', color='black',
             fontproperties=font_prop, c='white')

    # Colorbar
    cbar_ax = fig.add_axes((0.1, 0.05, 0.65, 0.02))  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
    cbar.set_ticks([14, 15, 16, 17, 18, 19, 20])
    cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
    cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)

    plt.savefig('/Users/dear-prudence/dear-prudence/halos/09_18/stream/observables/chisholm2025_streamPlot.pdf',
                dpi=360, bbox_inches='tight', facecolor=(33 / 255, 33 / 255, 33 / 255))


def gasPlot():
    snap = 118
    # -----------------------------------------
    inputPath_left = ('/Users/dear-prudence/dear-prudence/halos/09_18/halo_08/gaseous_components/temperatureProfile/'
                      '09_18.halo_08.temperatureProfile.npz')
    inputPath_right = ('/Users/dear-prudence/dear-prudence/coronaMassFunction/'
                       f'coronaMassFunction.snap{snap}.npz')
    # -----------------------------------------
    data = np.load(inputPath_left)
    slice_index = 127 - snap
    image_l = data['heatmap']
    x_el, y_el = data['x_e'], data['y_e']
    column_averages_l = data['coronaProfile'][:, slice_index]
    virial_radius_l = data['R_vir'][slice_index]
    equilibrium_r_l = data['salemR'][:, slice_index]
    equilibrium_T_l = data['salemT'][:, slice_index]
    # -----------------------------------------
    data = np.load(inputPath_right)
    sims_r = data['sim']
    halo_ids_r = data['halo_id']
    halo_masses_r = data['M_halo']
    bool_satellite_r = data['bool_satellite']
    avg_temp_r = data['T_avg']
    mean_nH_r = data['mean_nH']
    sigma_nH_r = data['sigma_nH']
    corona_temp_r = data['mean_T']
    sigma_temp_r = data['sigma_T']
    # -----------------------------------------
    font_prop = formatting()
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9.5, 5), gridspec_kw={'wspace': 0})
    # Remove extra space between subplots
    fig.subplots_adjust(wspace=0)  # Ensure no space between left and right plots
    plt.ylim([4, 7])  # in log(K)

    # -----------------------------------------
    c_map = 'Blues'
    background_color = plt.get_cmap(c_map)(0)

    # Smooth the data before plotting
    slice_index = 127 - snap
    smoothed_image = gaussian_filter(image_l[:, :, slice_index], sigma=2.0)

    aspect_ratio = abs(x_el[-1] - x_el[0]) / abs(y_el[-1] - y_el[0])

    fig.tight_layout(pad=2)
    # plt.gca().set_facecolor(background_color)
    vmax = np.max(smoothed_image)
    ax1.imshow(np.rot90(smoothed_image), origin='upper',
               extent=(x_el[0], x_el[-1], y_el[0], y_el[-1]), aspect=aspect_ratio,
               cmap=c_map, norm=LogNorm(vmin=vmax * 10 ** -1.6, vmax=vmax), rasterized=True)

    plt.rcParams.update({'lines.dashed_pattern': (6, 6)})

    # Virial radius line
    ax1.plot([virial_radius_l, virial_radius_l], [4, 7], linestyle='dashed', color='black',
             dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.text(virial_radius_l - 8, 4.35, s='virial radius', rotation='vertical', color='black',
             fontproperties=font_prop)
    ax1.plot([virial_radius_l / 10, virial_radius_l / 10], [4, 7], linestyle='dashed', color='black',
             dashes=(6, 6), lw=0.5, alpha=0.8)
    # ax1.text(virial_radius_l / 10 - 8, 4.2, s=r'0.1 x R$_{\text{vir}}$', rotation='vertical', color='black',
    #          fontproperties=font_prop)
    # ax1.arrow(virial_radius_l / 10, 4.3, 9 * virial_radius_l / 10, 0, lw=0.5, head_width=0.1, color='k', shape='full')
    ax1.annotate("", xytext=(virial_radius_l / 10, 4.2), xy=(virial_radius_l, 4.2),
                 arrowprops=dict(arrowstyle="<->", linestyle='dashed', lw=0.5, alpha=0.8))
    ax1.text(70, 4.25, s='extent of CGM', rotation='horizontal', color='black',
             fontproperties=font_prop)

    # warm/cool phase lines
    ax1.plot([0, 200], [5, 5], linestyle='dashed', color='black',
             dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.text(40, 5.05, s='warm-phase', rotation='horizontal', color='black',
             fontproperties=font_prop)
    ax1.text(40, 4.9, s='cool-phase', rotation='horizontal', color='black',
             fontproperties=font_prop)

    # hestia column averages
    ax1.plot(x_el[:-1] + abs(x_el[0] + x_el[1]) / 2, column_averages_l,
             c='tab:blue', lw=1.5, label='Massive dwarf (this work)')

    # Scott's simulations
    column_averages_lmc, x_e_lmc = lmc_temperatureProfile()
    ax1.plot(x_e_lmc[:-1] + abs(x_e_lmc[0] + x_e_lmc[1]) / 2, column_averages_lmc, c='tab:green', linestyle='solid',
             lw=1,
             label='Isolated LMC-analog (Lucchini et al. 2024)')

    # Stable corona (Salem+2015)
    ax1.plot(equilibrium_r_l, np.log10(equilibrium_T_l), linestyle='solid',
             color='k', lw=0.5, alpha=1,
             label='Equilibrium profile (Salem et al. 2015)')

    # Other formatting stuff
    x_lim = [x_el[0], x_el[-1]]
    y_lim = [4, 7]
    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[0], y_lim[1])
    ax1.set_xticks(np.linspace(0, 200, 6), labels=['0', '40', '80', '120', '160', ''],
                   fontproperties=font_prop)
    # plt.xticks(np.linspace(0, 200, 6))
    ax1.set_xticks(np.linspace(0, 200, 21), minor=True)
    ax1.set_yticks(np.linspace(4, 7, 7), labels=['4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0'],
                   fontproperties=font_prop)
    # plt.yticks(np.linspace(4, 7, 7))
    ax1.set_yticks(np.linspace(4, 7, 31), minor=True)
    # Set font weight for tick labels
    ax1.set_xlabel('Radius ' + r'$[$' + 'kpc' + r'$]$', fontproperties=font_prop)
    ax1.set_ylabel('log (T / K)', fontproperties=font_prop)
    ax1.legend(prop=font_prop)
    # -----------------------------------------
    x_lim = [10.25, 12.75]
    y_lim = [4, 7]

    i, j = 0, 0
    for row in range(len(halo_masses_r)):
        # if the data point is a satellite
        if bool_satellite_r[row] == 1:
            if i < 1:
                ax2.scatter(halo_masses_r[row], corona_temp_r[row], c='white', edgecolors='k', s=12, label='Satellite')
                i += 1
            else:
                ax2.scatter(halo_masses_r[row], corona_temp_r[row], c='white', edgecolors='k', s=12)
        elif str(sims_r[row]) == '09_18' and str(halo_ids_r[row]) == '008':
            pass
            # print(mean_nH_r[row])
            # print(sigma_nH_r[row])
        else:
            if j < 1:
                ax2.scatter(halo_masses_r[row], corona_temp_r[row], c='k', s=12, label='LG/field')
                j += 1
            else:
                ax2.scatter(halo_masses_r[row], corona_temp_r[row], c='k', s=12)
    massive_dwarf_idx = 3
    ax2.scatter(halo_masses_r[massive_dwarf_idx], corona_temp_r[massive_dwarf_idx],
                c='tab:blue', s=25, marker='D', zorder=2, label='Massive Dwarf')
    ax2.errorbar(halo_masses_r, corona_temp_r, yerr=sigma_temp_r, c='k', ls='none', elinewidth=0.8, fmt='none')

    lucchini_fiducial = (1.75e11, 3e5)
    ax2.scatter(np.log10(lucchini_fiducial[0]), np.log10(lucchini_fiducial[1]), c='tab:green', s=16, marker='D',
                label='Lucchini et al. 2024')

    # from watkins et al. 2024 from globular clusters of M_vir
    lmc_mass_estimate = np.array([1.8 - 0.54, 1.8 + 1.05]) * 1e11
    ax2.axvspan(np.log10(lmc_mass_estimate[0]), np.log10(lmc_mass_estimate[1]), color='black',
                linestyle='', alpha=0.2)
    ax2.plot([np.log10(lmc_mass_estimate[0]), np.log10(lmc_mass_estimate[0])], [y_lim[0], y_lim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
    ax2.plot([np.log10(lmc_mass_estimate[1]), np.log10(lmc_mass_estimate[1])], [y_lim[0], y_lim[1]],
             linestyle='dashed', color='k', dashes=(6, 6), lw=0.5, alpha=0.8)
    ax2.text(11, 5.9, s='estimated LMC mass', rotation='vertical', font_properties=font_prop)

    ax2.set_xlim(x_lim[0], x_lim[1])
    ax2.set_ylim(y_lim[0], y_lim[1])
    ax2.set_xticks(np.linspace(10.5, 12.5, 5), labels=['10.5', '11.0', '11.5', '12.0', '12.5'],
                   fontproperties=font_prop)
    ax2.set_xticks(np.linspace(10.3, 12.7, 25), minor=True)
    ax2.set_yticks(np.linspace(y_lim[0], y_lim[1], 7), labels=['4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0'],
                   fontproperties=font_prop)
    ax2.set_yticks(np.linspace(y_lim[0], y_lim[1], 31), minor=True)
    ax2.set_xlabel('log (M' + r'$_{\text{halo}}$' + '/M' + r'$_{\odot}$)', fontproperties=font_prop)
    ax2.set_ylabel('log (T / K)', fontproperties=font_prop)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()  # move ticks (and tick labels)
    ax2.tick_params(axis='y', which='both', left=True, right=True, labelleft=False, labelright=True)

    aspect_ratio = abs(x_lim[1] - x_lim[0]) / abs(y_lim[1] - y_lim[0])
    ax2.set_aspect(aspect_ratio)

    ax2.legend(prop=font_prop, loc='lower right')

    output_path = '//coronaMassFunction/chisholm2025_gasPlot.pdf'
    plt.savefig(output_path, dpi=240, bbox_inches='tight', facecolor=(33 / 255, 33 / 255, 33 / 255))
    plt.show()


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
