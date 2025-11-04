import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from hestia.geometry import get_lookbackTimes
from scipy.ndimage import gaussian_filter
from scripts.local.archive.lucchini import lmc_temperatureProfile
from hestia.gas import plot_virial_temp_line


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
    _, lookback_times = get_lookbackTimes(run=None, snaps=None, redshifts=redshifts)
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
    cbar_ax = fig.add_axes([0.1, 0.05, 0.65, 0.02])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'log N$_{\text{H I}}$ $[$cm$^{-2}]$', fontproperties=font_prop)
    cbar.set_ticks([14, 15, 16, 17, 18, 19, 20])
    cbar.set_ticks(np.linspace(14, 20, 31), minor=True)
    cbar.set_ticklabels(['14', '15', '16', '17', '18', '19', '20'], fontproperties=font_prop)

    plt.savefig('/Users/dear-prudence/dear-prudence/halos/09_18/stream/observables/chisholm2025_streamPlot.pdf',
                dpi=360, bbox_inches='tight', facecolor=(33/255, 33/255, 33/255))


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

    x_arr = np.linspace(min(halo_masses_r), x_lim[1])
    x_l, y_l = plot_virial_temp_line(x_arr)
    ax2.plot(x_l, y_l, linestyle='solid', color='k', lw=0.5, alpha=1,
             label='Virial Theorem\n T=' + r'$\mu$' + r'm$_{\text{p}}$'
                   + r'GM/2k$_{\text{b}}$' + r'R$_{\text{vir}}$')

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
    plt.savefig(output_path, dpi=240, bbox_inches='tight', facecolor=(33/255, 33/255, 33/255))
    plt.show()


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
        # change this back !!!
        'xtick.labelcolor': 'white',
        'ytick.labelcolor': 'white',
        'axes.labelcolor': 'white',
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
