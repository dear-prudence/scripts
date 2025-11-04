import numpy as np


def get_sigma(run, halo, snaps):
    if halo == 'lmc':
        halo_id = '10'
    elif halo == 'mw':
        halo_id = '03'
    else:
        print('Error: Invalid halo!')
        exit(1)
    col_sigma = 19
    filename = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run
                + '/AHF_output_2x2.5Mpc/HESTIA_100Mpc_8192_' + run + '.127_halo_1270000000000'
                + halo_id + '.dat')
    lmc_data = np.loadtxt(filename)
    v_sig = lmc_data[127 - snaps[1]:127 - snaps[0], col_sigma]
    return v_sig


def f(x):
    # values for alpha. beta, epsilon, taken from Gultekin+2009 (https://arxiv.org/pdf/0903.4897)
    alpha = 8.12
    beta = 4.24
    epsilon = 0.44
    return alpha + beta * np.log10(x / 200)


# ---------------------------------------------
mode = 'dear-prudence'
# ---------------------------------------------

if mode == 'geras':
    from scripts.hestia import get_accretion_curve
    from scripts.hestia import time_edges

    snaps = [31, 127]  # AHF output only goes back to snapshot 32
    sim = '09_18'
    halo = 'mw'

    bh_masses = get_accretion_curve('PartType5', sim, halo, snaps)
    v_sigma = get_sigma(sim, halo, snaps)
    time_edges = time_edges(sim=sim, snaps=np.arange(snaps[1], snaps[0], step=-1))

    np.savez('/z/rschisholm/storage/analytical_plots/' + str(sim) + '_vSigBHmass_' + halo + '.npz',
             bh_masses=bh_masses, v_sigma=v_sigma, time_edges=time_edges)


elif mode == 'dear-prudence':
    import matplotlib.pyplot as plt

    # snaps = [95, 102, 108, 115, 121, 127]
    snaps = [108]
    run = '09_18'
    halos = ['lmc', 'mw']

    fig = plt.figure(figsize=(7, 6))
    epsilon = 0.44

    for halo in halos:
        data = np.load('/Users/dear-prudence/Desktop/smorgasbord/mSigmaRelation/09_18_vSigBHmass_' + halo + '.npz')
        v_sigma = data['v_sigma']
        bh_masses = data['bh_masses']
        redshifts = data['time_edges'][:, 0]

        for snap in snaps:
            plt.scatter(v_sigma[127 - snap], np.log10(bh_masses[127 - snap]),
                        marker='+', c='k', s=100)
            plt.text(v_sigma[127 - snap] + 5, np.log10(bh_masses[127 - snap]) + 0.05,
                     s='09_18 LMC, z = ' + '{:.{}f}'.format(redshifts[127 - snap], 3)
                       if halo == 'lmc' else '09_18 MW', fontsize='small', c='k')

    x_lim = [50, 300]
    y_lim = [5.5, 9.0]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel(r'$\sigma$ $[km/s]$')
    plt.ylabel(r'log$(M_{BH} / M_{solar})$')
    plt.xscale('log')
    fig.tight_layout()

    sigma = np.linspace(50, 300, 30, endpoint=True)
    plt.plot(sigma, f(sigma), linestyle='dashed', label='Gultekin+2009')
    plt.fill_between(x=sigma, y1=f(sigma) + epsilon, y2=f(sigma) - epsilon, alpha=0.3)
    plt.legend(loc='lower right')

    plt.title(r'Halo Blackhole mass $M_{BH}$ versus velocity dispersion $\sigma$ (M-$\sigma$ relation)'
              'with pre-heating event 09_18 LMC',
              fontsize='small', loc='left')
    plt.savefig('/Users/dear-prudence/Desktop/smorgasbord/mSigmaRelation/09_18_mSigmaRelation.png', dpi=240)

    plt.show()


print('Done!')
