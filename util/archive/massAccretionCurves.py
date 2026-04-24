# This script handles the retrieving, processing, and plotting of the mass accretion curves of a specified halo;
# Currently only compatible with LMC halo at z=0.0, will add more capability later
import numpy as np


def heating_event(x, y_lim):
    heating_snap_lims = [108, 121]
    domain = x[127 - heating_snap_lims[1]:127 - heating_snap_lims[0]]
    ys = np.linspace(y_lim[1], y_lim[1], len(domain))
    plt.fill_between(domain, ys, facecolor='k', alpha=0.2)
    plt.text(0.23, 5.35, s='Heating Event', fontsize='small')


# ------------------------------------
machine = 'dear-prudence'
mode = 'components'
# 'particles' for cumulative accretion curves of all particle types
# 'smbh' for accretion rate(s) of host SMBH
# ------------------------------------

if machine == 'geras':
    from hestia import get_accretion_curve, get_smbh_curve
    from hestia import time_edges

    snaps = [31, 127]  # AHF output only goes back to snapshot 32
    sim = '09_18'
    halo = 'halo_15'

    time_edges = time_edges(sim=sim, snaps=np.arange(snaps[1], snaps[0], step=-1))

    if mode == 'components':
        bhs = get_accretion_curve('PartType5', sim, halo, snaps)
        gas = get_accretion_curve('PartType0', sim, halo, snaps)
        dms = get_accretion_curve('PartType1', sim, halo, snaps)
        sta = get_accretion_curve('PartType4', sim, halo, snaps)
        np.savez('/z/rschisholm/storage/analytical_plots/massAccretionCurves/'
                 + str(sim) + '_massAccretionCurves_components_' + halo + '.npz',
                 bhs=bhs, gas=gas, dms=dms, sta=sta, time_edges=time_edges)

    elif mode == 'smbh':
        mass, m_dots, m_dots_bondi, m_dots_eddington = get_smbh_curve(sim, halo, snaps)
        np.savez('/z/rschisholm/storage/analytical_plots/massAccretionCurves/'
                 + str(sim) + '_massAccretionCurves_smbh_' + halo + '.npz',
                 mass=mass, m_dot=m_dots, bondi=m_dots_bondi, eddington=m_dots_eddington, time_edges=time_edges)

elif machine == 'dear-prudence':
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    halo = 'halo_23'

    data = np.load('/Users/dear-prudence/Desktop/smorgasbord/massAccretionCurves/'
                   '09_18_massAccretionCurves_' + mode + '_' + halo + '.npz')
    redshifts = data['time_edges'][:, 0]

    fig, ax1 = plt.subplots(figsize=(9, 6))

    if mode == 'components':
        # Module for all components aside from black holes
        c1 = 'tab:blue'
        ax1.plot(redshifts, np.log10(data['dms']), c=c1, linestyle='dashdot', label='DM')
        ax1.plot(redshifts, np.log10(data['gas']), c=c1, linestyle='dashed', label='Gas')
        ax1.plot(redshifts, np.log10(data['sta']), c=c1, linestyle='dotted', label='Stars*')
        ax1.set_ylabel('log(Mass)', color=c1)
        ax1.set_ylim([7, 12])
        ax1.tick_params(axis='y', labelcolor=c1)
        ax1.set_xlabel('z')
        plt.legend(loc='lower right')
        # Module for black holes
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        c2 = 'k'
        ax2.plot(redshifts, np.log10(data['bhs']), color=c2, label='BlackHole')
        ax2.set_ylabel('log(BH Mass)', color=c2)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=c2)
        y_lim = [5.3, 7.6]
        plt.ylim(y_lim[0], y_lim[1])

        # Other formatting stuff
        ax2.xaxis.set_major_formatter(ScalarFormatter())
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlim([10, 0])
        plt.title('09_18 LMC mass accretion curves by particle type \n*stars and wind particles',
                  fontsize='small', loc='left')

    elif mode == 'smbh':
        ax1.plot(redshifts, np.log10(data['m_dot']), label='Instantaneous')
        ax1.plot(redshifts, np.log10(data['bondi']), label='Bondi')
        ax1.plot(redshifts, np.log10(data['eddington']), label='Eddington')

        ax1.set_xlabel('$z$')
        ax1.set_ylabel(r'$log(M_{dot})$ $log[M_{solar}/yr]$')
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlim([10, 0])
        plt.legend(loc='lower right')
        y_lim = [0, 3]

    fig.tight_layout()
    # plt.grid(visible=True, ls='-', axis='both', alpha=0.25)

    # add the shaded region indicating the heating event
    # heating_event(redshifts, y_lim)

    # Module to add indicator of when LMC passes R_vir of MW
    # AHF considers LMC to fall into MW halo at z=0.06 (taken from AHF output of halo 10)
    # plt.plot([0.06, 0.06], [y_lim[0], y_lim[1]], linestyle='solid', color='k', alpha=0.2)
    # plt.text(0.058, 5.8, s='Halo passes $R_{vir}$ \nof the MW', fontsize='small')
    # -----------------------
    plt.savefig('/Users/dear-prudence/Desktop/smorgasbord/massAccretionCurves/'
                '09_18_massAccretionCurves_' + mode + '_' + halo + '.png', dpi=240)
    plt.show()

print('Done!')


"""# This script handles the retrieving, processing, and plotting of the mass accretion curves of a specified halo;
# Currently only compatible with LMC halo at z=0.0, will add more capability later
import numpy as np


def heating_event(x, y_lim):
    heating_snap_lims = [108, 121]
    domain = x[127 - heating_snap_lims[1]:127 - heating_snap_lims[0]]
    ys = np.linspace(y_lim[1], y_lim[1], len(domain))
    plt.fill_between(domain, ys, facecolor='k', alpha=0.2)
    plt.text(0.23, 5.35, s='Heating Event', fontsize='small')


# ------------------------------------
machine = 'dear-prudence'
mode = 'components'
# 'particles' for cumulative accretion curves of all particle types
# 'smbh' for accretion rate(s) of host SMBH
# ------------------------------------

if machine == 'geras':
    from hestia.halos import get_accretion_curve
    from hestia.image import time_edges

    snaps = [31, 127]  # AHF output only goes back to snapshot 32
    sim = '09_18'
    halo = 'lmc'

    bhs = get_accretion_curve('PartType5', sim, halo, snaps)
    gas = get_accretion_curve('PartType0', sim, halo, snaps)
    dms = get_accretion_curve('PartType1', sim, halo, snaps)
    sta = get_accretion_curve('PartType4', sim, halo, snaps)
    time_edges = time_edges(sim=sim, snaps=np.arange(snaps[1], snaps[0], step=-1))

    np.savez('/z/rschisholm/storage/analytical_plots/' + str(sim) + '_massAccretionCurves_LMC.npz',
             bhs=bhs, gas=gas, dms=dms, sta=sta, time_edges=time_edges)

elif machine == 'dear-prudence':
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    data = np.load('/Users/dear-prudence/Desktop/smorgasbord/massAccretionCurves/09_18_massAccretionCurves_LMC.npz')
    x = data['time_edges'][:, 0]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    # Module for all components aside from black holes
    c1 = 'tab:blue'
    ax1.plot(x, np.log10(data['dms']), c=c1, linestyle='dashdot', label='DM')
    ax1.plot(x, np.log10(data['gas']), c=c1, linestyle='dashed', label='Gas')
    ax1.plot(x, np.log10(data['sta']), c=c1, linestyle='dotted', label='Stars*')
    ax1.set_ylabel('log(Mass)', color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    ax1.set_xlabel('z')
    plt.legend(loc='lower right')
    # Module for black holes
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    c2 = 'k'
    ax2.plot(x, np.log10(data['bhs']), color=c2, label='BlackHole')
    ax2.set_ylabel('log(BH Mass)', color=c2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=c2)
    y_lim = [5.3, 7.6]
    plt.ylim(y_lim[0], y_lim[1])

    # Other formatting stuff
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.xlim([10, 0])
    plt.title('09_18 LMC mass accretion curves by particle type \n*stars and wind particles',
              fontsize='small', loc='left')
    fig.tight_layout()

    # add the shaded region indicating the heating event
    heating_event(x, y_lim)

    # Module to add indicator of when LMC passes R_vir of MW
    # AHF considers LMC to fall into MW halo at z=0.06 (taken from AHF output of halo 10)
    plt.plot([0.06, 0.06], [y_lim[0], y_lim[1]], linestyle='solid', color='k', alpha=0.2)
    plt.text(0.058, 5.8, s='LMC passes $R_{vir}$ \nof the MW', fontsize='small')
    # -----------------------
    plt.savefig('/Users/dear-prudence/Desktop/09_18_massAccretionCurves_lmc_addendum.png', dpi=240)
    plt.show()

print('Done!')
"""
