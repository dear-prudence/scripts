import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from scripts.util.archive.lucchini import lmc_temperatureProfile
from hestia import plot_virial_temp_line


def plot():
    snap = 127
    # -----------------------------------------
    inputPath_left = ('/Users/dear-prudence/smorgasbord/gaseous_components/09_18_halo_08/temperatureProfile/'
                      'temperatureProfile.npz')
    inputPath_right = ('/Users/dear-prudence/smorgasbord/gaseous_components/coronaMassFunction/'
                       f'coronaMassRelation_snap{snap}.npz')
    # -----------------------------------------
    data = np.load(inputPath_left)
    image_l = data['data']
    x_el, y_el = data['x_e'], data['y_e']
    column_averages_l = data['column_averages']
    virial_radius_l = data['virial_radius']
    equilibrium_r_l = data['equilibrium_r']
    equilibrium_T_l = data['equilibrium_T']
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
               cmap=c_map, norm=LogNorm(vmin=vmax * 1e-2, vmax=vmax), rasterized=True)

    # Virial radius line
    ax1.plot([virial_radius_l, virial_radius_l], [4, 7], linestyle='dashed', color='black',
             dashes=(6, 6), lw=0.5, alpha=0.8)
    ax1.text(virial_radius_l - 8, 4.2, s='virial radius', rotation='vertical', color='black',
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
                label='Lucchini+2024')

    lmc_mass_estimate = np.array([1.55 - 0.26, 1.55 + 0.26]) * 1e11
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

    output_path = '/Users/ursa/smorgasbord/gaseous_components/chisholm2025_gasPlot.pdf'
    plt.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.show()


def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='util/fonts/AVHersheySimplexMedium.otf', size=12)

    # plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams.update({  # "grid.linestyle": "--",  # Dashed grid lines
        'axes.unicode_minus': False,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        'mathtext.fontset': 'dejavuserif',
        'mathtext.default': 'it',
        # 'text.usetex': True,
        # 'mathtext.tt': 'monospace',
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
