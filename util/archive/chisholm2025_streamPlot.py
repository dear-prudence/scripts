import matplotlib.pyplot as plt
import numpy as np
from hestia import get_lookbackTimes


def plot():
    input_path1 = '/Users/dear-prudence/smorgasbord/kinematics/orbits/orbitalDistance.09_18.halo_08-smc.npz'
    input_path2 = ('/Users/dear-prudence/smorgasbord/images/09_18_stream/skyProjections/'
                    '09_18_stream_gas_column_H0_snap119.npz')
    input_path3 = ('/Users/dear-prudence/smorgasbord/images/09_18_stream/skyProjections/'
                    '09_18_stream_gas_column_H0_snap123.npz')
    input_path4 = ('/Users/dear-prudence/smorgasbord/images/09_18_stream/skyProjections/'
                    '09_18_stream_gas_column_H0_snap127.npz')

    # --------------------------------------------------------------------
    font_prop = formatting()
    # Create figure and subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9.5, 5), gridspec_kw={'wspace': 0})
    fig = plt.figure(figsize=(10, 10))
    # Create first subplot (Cartesian)
    ax1 = fig.add_subplot(2, 2, 1)
    # Create second subplot (Polar)
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    # Create third subplot (Polar)
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    # Create fourth subplot (Polar)
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    # Remove extra space between subplots
    fig.subplots_adjust(wspace=0.1)  # Ensure no space between left and right plots
    # --------------------------------------------------------------------

    data1 = np.load(input_path1)
    # -------------------------------
    dynamical_zeroTime = -1.33  # snap 119
    # -------------------------------

    ax1.plot(-1 * data1['hestia_times'] - dynamical_zeroTime, data1['hestia_distances'],
            c='k', linestyle='solid', lw=1.5, label='Magellanic-analog (this work)')
    ax1.plot(-1 * data1['besla_times'], data1['besla_distances'], label='Besla et al. 2012, model 2',
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

    # --- Top x-axis ---
    redshifts = np.array(np.linspace(0, 1, 11))
    _, lookback_times = get_lookbackTimes(sim=None, snaps=None, redshifts=redshifts)
    ax_top = ax1.secondary_xaxis('top')  # share the same x-axis scale
    ax_top.set_xlabel('z', fontproperties=font_prop)
    ax_top.set_xticks(-1 * np.array(lookback_times) + 1.33)
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
    ax2.text(1.1, 2.2, s='(i) z=0.099,\nt' + r'$_{\text{dym}}$=0.0 Gyr', rotation='horizontal', color='black',
             fontproperties=font_prop)

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
    ax3.text(1.1, 2.2, s='(ii) z=0.046,\nt' + r'$_{\text{dym}}$=+0.69 Gyr', rotation='horizontal', color='black',
             fontproperties=font_prop)

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
    ax4.text(1.1, 2.2, s='(iii) z=0.0,\nt' + r'$_{\text{dym}}$=+1.33 Gyr', rotation='horizontal', color='black',
             fontproperties=font_prop)

    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.65, 0.02])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
    cbar.set_ticks([14, 15, 16, 17, 18, 19, 20])
    cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
    cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)

    plt.savefig('/Users/dear-prudence/smorgasbord/images/09_18_stream/chisholm2025_streamPlot.pdf',
                dpi=360, bbox_inches='tight')


def formatting():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='util/fonts/AVHersheySimplexMedium.otf', size=12)

    # plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams.update({  # "grid.linestyle": "--",  # Dashed grid lines
        'axes.unicode_minus': False,
        "xtick.top": False,
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