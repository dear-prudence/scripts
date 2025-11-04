import numpy as np
from scripts.hestia import time_edges
from scripts.hestia import get_radial_profile


def rid_sub_pixel(r, g, s):
    # Find the indices of the negative numbers
    neg_idx = np.where(r < 0)
    # Remove all negative numbers
    r_rid = np.delete(r, neg_idx, axis=None)
    # Remove elements in data2 at the same indices as the negative numbers in data1
    g_rid = np.delete(g, neg_idx, axis=None)
    s_rid = np.delete(s, neg_idx, axis=None)
    return r_rid, g_rid, s_rid


def make_plot(data, snaps):
    plt.figure(figsize=(8, 6))
    x_lim = [1, 200]
    y_lim = [0, 4]
    if len(data['time_edges']) == 1:
        r, g, s = rid_sub_pixel(data['radii'], data['gas'], data['stars'])
        # the AHF documentation hints that gas and stellar mass are in units of M_solar, NOT M_solar/h;
        # the values check out with gas and stellar mass values from the halo file (...10.dat)
        # if M_solar is assumed; should check this with Elena/Noam
        plt.xlabel('$R$ (kpc)')
        plt.ylabel(r'$M$ ($10^{10} M_{\odot}$)')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.plot(r, g / 1e10, label='Gas')
        plt.plot(r, s / 1e10, label='Stars')
        plt.legend(loc='lower right')
        plt.savefig('/Users/dear-prudence/Desktop/09_18_radialProfile_snap' + str(snaps) + '.png', dpi=240)
        plt.show()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        # Unpack the 2D array of axes into individual variables
        axes = axes.flatten()
        fig.text(0.5, 0.04, '$R$ (kpc)', ha='center', va='center', fontsize=16)
        fig.text(0.04, 0.5, r'$M$ ($10^{10} M_{\odot}$)', ha='center', va='center', rotation='vertical', fontsize=16)
        # --------------------------------------
        # Module to define cosmetics (axes, background, labels, etc...)
        fig.tight_layout()
        frame = 0
        dates = data['time_edges']
        for ax, snap in zip(axes, snaps):
            r, g, s = rid_sub_pixel(data['radii'][frame], data['gas'][frame], data['stars'][frame])
            ax.plot(r, g / 1e10, label='Gas')
            ax.plot(r, s / 1e10, label='Stars')
            plt.xscale('log')
            ax.set_xlim(x_lim); ax.set_ylim(y_lim)
            ax.set_title('$z = $' + '{:.{}f}'.format(dates[frame, 0], 3)
                         + '$,$ \t $t = $' + '{:.{}f}'.format(-1 * round(float(dates[frame, 1]), 2), 2)
                         + ' Gyr',
                         x=0.75, y=0.005, ha='center', va='bottom', weight='bold')
            frame += 1
        plt.legend(loc='upper right')
        # Create extra white space to the right of the right subplot
        fig.subplots_adjust(left=0.08)
        fig.subplots_adjust(right=0.92)
        fig.subplots_adjust(top=0.92)
        fig.subplots_adjust(bottom=0.08)
        # Create a new axis for the colorbar to the right of the subplots
        # cax = fig.add_axes([0.88, 0.115, 0.02, 0.8])  # [left, bottom, width, height]
        plt.savefig('/Users/dear-prudence/Desktop/09_18_radialProfiles_snapsMultiple_lmc.png', dpi=240)
        plt.show()


# ---------------------------------------------
mode = 'dear-prudence'
# ---------------------------------------------
snaps = [95, 108, 114, 119, 122, 127]
run = '09_18'
halo = 'lmc'
# ---------------------------------------------

if mode == 'dear-prudence':
    import matplotlib.pyplot as plt

    input_path = '/Users/ursa/Desktop/smorgasbord/radialProfiles/09_18_radialProfiles_snapsMultiple_lmc.npz'
    make_plot(np.load(input_path), snaps=snaps)

elif mode == 'geras':
    if isinstance(snaps, int):
        rad_profile = get_radial_profile(halo, run, snaps)
        np.savez('/z/rschisholm/storage/analytical_plots/'
                 '' + str(run) + '_radialProfile_snap' + str(snaps) + '_' + str(halo) + '.npz',
                 radii=rad_profile[0], gas=rad_profile[1], stars=rad_profile[2],
                 time_edges=time_edges(run, [snaps]))
    elif isinstance(snaps, list):
        rad_profile = get_radial_profile(halo, run, np.array(snaps))
        np.savez('/z/rschisholm/storage/analytical_plots/'
                 '' + str(run) + '_radialProfiles_snapsMultiple_' + str(halo) + '.npz',
                 radii=rad_profile[0].T, gas=rad_profile[1].T, stars=rad_profile[2].T,
                 time_edges=time_edges(run, snaps))

print('Done!')

# -----------------------------------------
'''def make_rotation_curve(radii, tangential_velocities, masses, max_radius, bin_width=0.1):
    # Bin the data by radial distance
    bins = np.arange(0, max_radius + bin_width, bin_width)
    # Compute the sum of masses and mass-weighted velocities in each bin
    bin_mass_sum, _ = np.histogram(radii, bins=bins, weights=masses)
    bin_vel_mass_sum, _ = np.histogram(radii, bins=bins, weights=tangential_velocities * masses)

    # Compute the mass-weighted mean tangential velocity in each bin
    bin_means = bin_vel_mass_sum / bin_mass_sum

    # Compute the bin centers
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    return bin_centers, bin_means


def make_snap(snap, run, max_radius=20):
    h = 0.677
    row = 127 - snap
    if snap < 100:
        snap = '0' + str(snap)
    else:
        snap = str(snap)
    redshift = np.loadtxt('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc/'
                          'HESTIA_100Mpc_8192_09_18.127_halo_127000000000010.dat')[row, 0]
    key_names = ['Coordinates', 'Velocities', 'Masses']
    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run) + '/output_2x2.5Mpc/snapdir_'
                 + str(snap) + '/snapshot_' + str(snap) + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_particles = {name: None for name in key_names}
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + snap + '...')
    for file_path in file_paths:
        all_particles = append_particles('PartType0', file_path, key_names=key_names,
                                         existing_arrays=all_particles)
    # Initial filtering to cut down on unnecessary distance calculations, also transforming h units to normal
    l_b, u_b = center_lmc(run=run, lmc_halo_id='10', snap=snap, size=np.array([50, 50, 50]) * h) * 1e-3
    print(l_b)
    print(u_b)
    pos, v_p, l_c = get_halo_params('LMC', snap)
    print('pos:' + str(pos))
    print('v_p:' + str(v_p))
    print('l_c:' + str(l_c))
    init_filtered_particles = cosmo_transform(filter_particles(all_particles, min_b=l_b, max_b=u_b),
                                              'ckpc/h', 'ckpc', z=redshift)
    print(len(init_filtered_particles['Coordinates']))
    # subtract off the position and peculiar velocity of the halo (taken from AHF output)
    print('OG pos: ' + str(init_filtered_particles['Coordinates'][0]))
    init_filtered_particles['Coordinates'] = init_filtered_particles['Coordinates'] - (pos * 1e-3)
    print('New pos: ' + str(init_filtered_particles['Coordinates'][0]))
    init_filtered_particles['Velocities'] = init_filtered_particles['Velocities'] - v_p
    r, _, _, _, v_phi, _ = cartesian_to_cylindrical_velocity(init_filtered_particles['Coordinates'][:, 0],
                                                             init_filtered_particles['Coordinates'][:, 1],
                                                             init_filtered_particles['Coordinates'][:, 2],
                                                             init_filtered_particles['Velocities'][:, 0],
                                                             init_filtered_particles['Velocities'][:, 1],
                                                             init_filtered_particles['Velocities'][:, 2], l_c)
    UNITS OF POSITION AND VELOCITIES ARE DIFFERENT AND NEED TO BE CONVERTED 
    FOR THE ROTATION ROUTINES TO FUNCTION PROPERLY; 
    MUCH MORE COMPLICATED THAN ORIGINALLY THOUGHT
    print('mean r: ' + str(np.mean(r)))
    print('mean v_phi: ' + str(np.mean(v_phi)))
    return make_rotation_curve(r, v_phi, init_filtered_particles['Masses'], max_radius=max_radius, bin_width=0.2)'''
