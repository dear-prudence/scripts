# This script handles the retrieving, processing, and plotting of the mass accretion curves of a specified halo;
# Currently only compatible with LMC halo at z=0.0, will add more capability later
import numpy as np


def heating_event(x, y_lim):
    heating_snap_lims = [108, 121]
    domain = x[127 - heating_snap_lims[1]:127 - heating_snap_lims[0]]
    print(domain)
    ys = np.linspace(y_lim[1], y_lim[1], len(domain))
    plt.fill_between(domain, ys, facecolor='k', alpha=0.2)
    plt.text(0.23, 0.12, s='Heating Event', fontsize='small')


def bh_mdot(input_path):
    # unit conversion factor from M_solar/Gyr -> M_solar/yr
    return np.load(input_path)['time_edges'][:, 0], np.load(input_path)['m_dot'] * 1e-9


def smooth(arr):
    s_arr = np.copy(arr)
    for i in range(len(arr)):
        if i == 0:
            s_arr[i] = (arr[i] + arr[i + 1]) / 2
        elif i == len(arr) - 1:
            s_arr[i] = (arr[i] + arr[i - 1]) / 2
        else:
            s_arr[i] = (arr[i - 1] + arr[i] + arr[i + 1]) / 3
    print(s_arr)
    return s_arr


# ------------------------------------
mode = 'dear-prudence'
# ------------------------------------

if mode == 'geras':
    from hestia import get_sfr_curve

    sim = '09_18'
    snaps = [95, 127]  # AHF output only goes back to snapshot 32

    redshifts, SFRs = get_sfr_curve(sim, snaps)

    np.savez('/z/rschisholm/halos/09_18/halo_08/gaseous_components/' + str(sim) + '_SFRcurve_halo_08.npz',
             sfrs=SFRs, redshifts=redshifts)

elif mode == 'dear-prudence':
    import matplotlib.pyplot as plt

    with_bh_mdot = False

    sfr_data = np.load('/Users/ursa/smorgasbord/gaseous_components/09_18_halo_08/SFRcurve/'
                       '09_18_SFRcurve_halo_08.npz')
    redshifts_sfr = sfr_data['redshifts']  # DO NOT USE AS PRIMARY REDSHIFTS, missing snapshots
    SFRs = sfr_data['sfrs']
    # redshifts_bhs, m_dots = bh_mdot('/Users/dear-prudence/smorgasbord/gaseous_components/SFRcurve/09_18_SFRcurve_halo_08.npz')

    if with_bh_mdot is False:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(redshifts_sfr, SFRs)

        ax.set_ylabel(r'SFR ($M_{solar}/yr$)')
        ax.set_xlabel('z')

        # Other formatting stuff
        # plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlim([1, 0])
        y_lim = [0, np.max(SFRs) + 0.1]
        plt.ylim(y_lim)

        # Module to add indicator of when LMC passes R_vir of MW
        # AHF considers LMC to fall into MW halo at z=0.06 (taken from AHF output of halo 10)
        plt.plot([0.75, 0.75], [y_lim[0], y_lim[1]], linestyle='solid', color='k', alpha=0.2)
        plt.plot([0.26, 0.26], [y_lim[0], y_lim[1]], linestyle='solid', color='k', alpha=0.2)
        plt.plot([0.12, 0.12], [y_lim[0], y_lim[1]], linestyle='solid', color='k', alpha=0.2)
        # plt.text(0.058, 0.1, s='LMC passes $R_{vir}$ \nof the MW', fontsize='small')

        plt.title('09_18 LMC total* star formation rate curve'
                  '\n*total SFR is calculated as the sum of the instantaneous SFR of each gas cell '
                  'identified by the AHF as a member of the LMC',
                  fontsize='small', loc='left')
        fig.tight_layout()

    else:
        fig, ax1 = plt.subplots(figsize=(9, 6))
        # Module for all components aside from black holes
        c1 = 'tab:blue'
        ax1.plot(redshifts_sfr, SFRs, c=c1, linestyle='solid', label='SFR')
        ax1.set_ylabel(r'SFR ($M_{solar}/yr$)', color=c1)
        ax1.tick_params(axis='y', labelcolor=c1)
        ax1.set_xlabel('z')

        # Module for black holes
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        c2 = 'k'
        # this line guarantees both curves are displayed in the legend
        ax2.plot([1], [-1], color=c1, linestyle='solid', label='Stars')
        ax2.plot(redshifts_bhs, smooth(m_dots), linestyle='dashed', color=c2, label='Blackhole')
        ax2.set_ylabel(r'BH Accretion ($M_{solar}/yr$)', color=c2)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=c2)

        plt.legend(loc='upper right', fontsize='small')

        y_lim = [0, np.max(m_dots) + 0.5]
        plt.ylim(y_lim[0], y_lim[1])

        # Other formatting stuff
        # ax2.xaxis.set_major_formatter(ScalarFormatter())
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlim([10, 0])
        plt.title('09_18 LMC total* star formation rate curve and instantaneous rate of accretion of halo blackhole'
                  '\n*total SFR is calculated as the sum of the instantaneous SFR of each gas cell '
                  'identified by the AHF as a member of the LMC',
                  fontsize='small', loc='left')
        fig.tight_layout()

        # Module to add indicator of when LMC passes R_vir of MW
        # AHF considers LMC to fall into MW halo at z=0.06 (taken from AHF output of halo 10)
        plt.plot([0.06, 0.06], [y_lim[0], y_lim[1]], linestyle='solid', color='k', alpha=0.2)
        plt.text(0.058, 5.8, s='LMC passes $R_{vir}$ \nof the MW', fontsize='small')

    # add the shaded region indicating the heating event
    # heating_event(redshifts_bhs, y_lim)

    plt.savefig('/Users/dear-prudence/smorgasbord/gaseous_components/09_18_halo_08/SFRcurve/'
                '09_18_SFRcurve_halo_08.png', dpi=240)

    plt.show()

print('Done!')
