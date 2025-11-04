import numpy as np
from matplotlib.colors import LogNorm


def plot_orbits(run, subject_halo, reference_halo):
    import matplotlib.pyplot as plt

    base_path = '/Users/ursa/smorgasbord/kinematics/orbits/'
    input_file = 'orbitalDistance_' + run + '_' + reference_halo + '-' + subject_halo + '.npz'
    output_file = 'orbitalDistance_' + run + '_' + reference_halo + '-' + subject_halo + '.png'

    input_path = base_path + input_file
    output_path = base_path + output_file

    data = np.load(input_path)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(data['hestia_times'], data['hestia_distances'], c='k', linestyle='solid', label='hestia')
    ax.plot(data['besla_times'], data['besla_distances'], label='besla+2012, model 2',
            c='tab:blue', linestyle='solid', alpha=0.5)
    ax.plot(data['pardy_times'], data['pardy_distances'], label='pardy+2018, 9:1',
            c='tab:green', linestyle='solid', alpha=0.5)
    ax.plot(data['lucchini_times'], data['lucchini_distances'], label='lucchini+2020',
            c='tab:purple', linestyle='solid', alpha=0.5)
    ax.set_ylabel('Distance [kpc]')
    # ax.tick_params(axis='y')
    ax.set_xlabel(r'$t_{\rm lookback}$' + ' [Gyr]')
    plt.legend(loc='upper right')

    # Other formatting stuff
    # plt.xscale('log')
    plt.gca().invert_xaxis()
    x_lim = [7, 0]
    y_lim = [0, 200]
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    # plt.title('09_18 LMC mass accretion curves by particle type \n*stars and wind particles',
    # fontsize='small', loc='left')
    # -----------------------
    plt.savefig(output_path, dpi=240)
    plt.show()


def plot_phaseDiagram(run, halo, snap):
    import matplotlib.pyplot as plt

    base_path = f'/Users/ursa/smorgasbord/gaseous_components/{run}_{halo}/phaseDiagram/'
    input_file = 'phaseDiagram_H.npz'
    output_file = 'phaseDiagram_H.png'

    input_path = base_path + input_file
    output_path = base_path + output_file

    data = np.load(input_path)
    print(data.keys())
    image = data['data']
    x_e, y_e = data['x_e'], data['y_e']

    fig, ax = plt.subplots(figsize=(7, 7))
    c_map = 'Blues'
    background_color = plt.get_cmap(c_map)(0)

    fig.tight_layout()
    plt.gca().set_facecolor(background_color)
    plt.imshow(np.rot90(image[:, :, 127 - snap]), origin='upper', extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]),
               aspect=7/2,
               cmap=c_map, norm=LogNorm(vmin=1e-2, vmax=1), rasterized=True)

    # Other formatting stuff
    # plt.xscale('log')
    x_lim = [-8, 0]
    y_lim = [3.5, 6.5]
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    # plt.title('09_18 LMC mass accretion curves by particle type \n*stars and wind particles',
    # fontsize='small', loc='left')
    # -----------------------
    plt.savefig(output_path, dpi=240)
    plt.show()
