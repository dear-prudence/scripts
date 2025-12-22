import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def figure1():
    from scipy.interpolate import interp1d
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Ellipse
    from scripts.hestia.astrometry import Measurements
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


# noinspection PyTupleAssignmentBalance
def figure2():
    from scipy.optimize import curve_fit
    from scripts.hestia.astrometry import Measurements
    from scripts.hestia.image import twoD_Gaussian
    import seaborn as sns
    from matplotlib.patches import Ellipse
    from matplotlib.colors import LinearSegmentedColormap
    font_prop = formatting()
    m = Measurements()

    bool_gaussian = False
    # -------------------------------------
    x_lim = [73.0, 89]
    y_lim = [-74, -64]
    # x_lim = [73, 89]
    # y_lim = [-74, -66]
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

    fig = plt.figure(figsize=(5, 7.5))
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
                 c='lightblue', elinewidth=0.8, capsize=2, alpha=0.8,
                 label='LMC dynamical center from HVSs')  # (Lucchini+2025)
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
    plt.scatter(m['LMC/pm/ra'], m['LMC/pm/dec'], edgecolors='lavender', marker='o', linewidths=0.8, s=64, facecolors='none',
                label='Stellar kinematical center')  # (Choi+2022)

    # label='Extent of bar from RCSs; center aligned to mbp by def (Rathore+2025)')
    # ax.add_patch(ellipse)
    plt.plot(np.degrees(bar[1]), np.degrees(bar[2]), c='white', lw=0.8, alpha=0.8, linestyle='dashed')
    plt.scatter(m['LMC/bar/ra'], m['LMC/bar/dec'], c='white', marker='x', linewidths=0.8, s=64, alpha=0.8,
                label='Bar dynamical center')  # (Rathore+2025)

    ax.annotate("", xytext=(m['LMC/disk/ra'], m['LMC/disk/dec']),
                 xy=(m['LMC/disk/ra'] + 2 * m['LMC/disk/mu_alpha'], m['LMC/disk/dec'] + 2 * m['LMC/disk/mu_delta']),
                 arrowprops=dict(arrowstyle="->", linestyle='solid', lw=0.8, alpha=0.8, color='white'))

    plt.xlabel('right ascension (deg)', fontproperties=font_prop)
    plt.ylabel('declination (deg)', fontproperties=font_prop)

    x_ticksMajor = np.arange(70, 90, 2)
    x_ticksMinor = np.arange(70, 90, 0.5)
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
    ax.set_aspect(abs(x_lim[1] - x_lim[0]) / abs(y_lim[1] - y_lim[0]) / (8 / 10))

    plt.legend(prop=font_prop, loc='upper left')

    # Colorbar
    # cbar_ax = fig.add_axes([0.86, 0.11, 0.02, 0.77])  # [left, bottom, width, height] in figure coords
    # cbar = fig.colorbar(c, cax=cbar_ax, orientation='vertical')
    cbar_ax = fig.add_axes([0.12, 0.88, 0.78, 0.02])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    # cbar.set_ticks([f_PDF.min(), 7e-6, 7e-5, f_PDF.max()])
    # cbar.set_ticklabels(['', '1x', '10x', ''], fontproperties=font_prop)
    cbar.set_ticks([7e-6, 7e-5])
    cbar.set_ticklabels(['1x', '10x'], fontproperties=font_prop)
    cbar.set_label('Sample probability of CMBH', fontproperties=font_prop)

    output_path = f'{basePath}/chisholm2026.bhPDF.pdf'
    plt.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------
def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='scripts/local/fonts/AVHersheySimplexMedium.otf',
                                  size=12)

    # plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams.update({  # "grid.linestyle": "--",  # Dashed grid lines
        'axes.unicode_minus': False,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        'mathtext.fontset': 'cm',
        'lines.dashed_pattern': (6, 6),
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
