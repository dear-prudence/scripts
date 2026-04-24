import numpy as np
from archive.hestia import time_edges
from archive.hestia import get_rotation_curve


def rid_sub_pixel(r, v):
    # Find the indices of the negative numbers
    neg_idx = np.where(r < 0)
    # Remove all negative numbers
    r_rid = np.delete(r, neg_idx, axis=None)
    # Remove elements in data2 at the same indices as the negative numbers in data1
    v_rid = np.delete(v, neg_idx, axis=None)
    return r_rid, v_rid


def make_plot(data):
    plt.figure(figsize=(8, 6))
    plt.xlim([0, 20])
    plt.xlabel('$R$ (kpc)')
    plt.ylabel('$V_{circ}$ (km/s)')
    if len(data['time_edges']) == 1:
        r, v = rid_sub_pixel(data['radii'], data['v_circ'])
        plt.plot(r, v)
        plt.savefig('/Users/dear-prudence/Desktop/09_18_rotationCurve_lmc_snap127.png', dpi=240)
    else:
        print(data['radii'].shape)
        for i in range(len(data['time_edges'])):
            cmap = 'Blues'
            r, v = rid_sub_pixel(data['radii'][i], data['v_circ'][i])
            plt.plot(r, v, c=plt.get_cmap(cmap)((i + 1) * 45),
                     label='z = ' + '{:03}'.format(data['time_edges'][i, 0]))
            plt.legend(loc='lower right')
            plt.savefig('/Users/dear-prudence/Desktop/09_18_rotationCurves_lmc_snapsMultiple.png', dpi=240)
    plt.show()


# ---------------------------------------------
mode = 'dear-prudence'
# ---------------------------------------------
snaps = [95, 102, 108, 115, 121, 127]
run = '09_18'
halo = 'lmc'
# ---------------------------------------------

if mode == 'dear-prudence':
    import matplotlib.pyplot as plt

    input_path = '/Users/ursa/Desktop/09_18_rotationCurves_snapsMultiple_LMC.npz'
    make_plot(np.load(input_path))


elif mode == 'geras':
    if isinstance(snaps, int):
        rot_curve = get_rotation_curve(halo, run, snaps)
        np.savez('/z/rschisholm/storage/analytical_plots/'
                 '' + str(run) + '_rotationCurve_snap' + str(snaps) + '_' + str(halo) + '.npz',
                 radii=rot_curve[0], v_circ=rot_curve[1], time_edges=time_edges(run, [snaps]))
    elif isinstance(snaps, list):
        rot_curve = get_rotation_curve(halo, run, np.array(snaps))
        np.savez('/z/rschisholm/storage/analytical_plots/'
                 '' + str(run) + '_rotationCurves_snapsMultiple_' + str(halo) + '.npz',
                 radii=rot_curve[0].T, v_circ=rot_curve[1].T, time_edges=time_edges(run, snaps))

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
