import matplotlib.pyplot as plt
import numpy as np


def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='local/fonts/AVHersheySimplexMedium.otf', size=12)

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


snaps = [118, 119, 120, 121, 122, 123, 124, 125, 126]
files = [f'/Users/dear-prudence/dear-prudence/halos/09_18/stream/observables/NH0/09_18.stream.snap{snap}.NH0.faux.npz' for snap in
         snaps]

font_prop = formatting()
cmap = 'BuPu'

fig, axes = plt.subplots(3, 3, subplot_kw={'projection': 'polar'}, figsize=(12, 12))

for ax, fname in zip(axes.ravel(), files):
    data = np.load(fname)
    nH_map = np.log10(data['nH_map']).T
    lon_edges = data['lon_edges']
    lat_edges = data['lat_edges']
    smc_position = data['smc_position']

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

    ax.scatter(np.radians(smc_position[0]), np.radians(90 + smc_position[1]), s=25, marker='D', c='k')
    # ax.arrow(np.radians(smc_position[0]), np.radians(90 + smc_position[1]),
    #          np.radians(smc_velocity[0]), np.radians(90 + smc_velocity[1]))
    from hestia import get_lookbackTimes

    t = -1 * get_lookbackTimes(run=None, snaps=None, redshifts=np.array([float(data['redshift'])]))[1] + 1.52
    ax.text(1.1, 2.2, s=f'z={data["redshift"]},\nt' + r'$_{\text{dyn}}$=+' + f'{t[0]:.2f} Gyr',
            rotation='horizontal', color='black',
            fontproperties=font_prop)

    # Colorbar
    cbar_ax = fig.add_axes([0.11, 0.01, 0.8, 0.03])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
    cbar.set_ticks([14, 15, 16, 17, 18, 19, 20])
    cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
    cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)

plt.savefig('/Users/dear-prudence/Desktop/nh0.png', dpi=240, bbox_inches='tight')
# plt.show()
