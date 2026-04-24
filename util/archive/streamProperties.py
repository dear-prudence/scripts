import numpy as np
from hestia import append_particles
from hestia import calc_temperature


def snap_to_redshift(snap):
    filename = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc/'
                'HESTIA_100Mpc_8192_09_18.127_halo_127000000000001.dat')
    redshift = '{:.{}f}'.format(np.loadtxt(filename)[127 - snap, 0], 3)
    return redshift  # returns the redshift of a given snapshot in format x.xxx


def add_distances(particles):
    # Initialize an array to store distances
    distances = np.zeros(particles['Coordinates'].shape[0])
    # Compute distances for each particle
    for i in range(particles['Halo_Coordinates'].shape[0]):
        distances[i] = np.linalg.norm(particles['Halo_Coordinates'][i])
    # Add the new column to the data dictionary
    particles['Distances'] = distances
    return particles


def filter_unphysical_metallicities(data):
    for key in data.keys():
        data[key] = np.array(data[key])  # Convert to numpy array for easier indexing
    # Get the indices where metallicities are negative
    original_length = len(data['GFM_Metallicity'])
    indices_to_keep = np.where(data['GFM_Metallicity'][:] > 0)[0]
    print('Done with unphysical metallicity filtering, number of thrown-out particles: '
          + str(original_length - len(indices_to_keep)))
    for key in data.keys():
        data[key] = data[key][indices_to_keep]
    return data


def filter_disk_particles(particles, cutoff):
    radii = np.zeros(particles['Halo_Coordinates'].shape[0])
    rho = np.zeros(particles['Halo_Coordinates'].shape[0])
    for i in range(particles['Halo_Coordinates'].shape[0]):
        radii[i] = np.linalg.norm(particles['Halo_Coordinates'][i])
        rho[i] = np.linalg.norm(
            np.array([particles['Halo_Coordinates'][i, 0], particles['Halo_Coordinates'][i, 1]]))
    particles['Radii'] = radii
    particles['rho'] = rho

    original_length = len(particles['rho'])

    indices_to_keep = np.where(cutoff <= particles['rho'])[0]

    for key in particles.keys():
        particles[key] = particles[key][indices_to_keep]

    print('Done with disk filtering, number of thrown-out particles = '
          + str(original_length - len(indices_to_keep)))

    return particles


def retrieve_particles(snap, sim, part_type):
    import h5py

    snap_ = '0' + str(snap) if snap < 100 else str(snap)

    # ------ Section for stream particles ------
    file_paths = ['/z/rschisholm/storage/snapshots_stream_09_18/snapshot_' + snap_ + '.hdf5']
    with h5py.File(file_paths[0], 'r') as file:
        keys = list(file[part_type].keys())

    print('Processing Snapshot ' + snap_ + '...')
    particles = {name: None for name in keys}
    for file_path in file_paths:
        particles = append_particles(part_type, file_path, key_names=keys,
                                     existing_arrays=particles)

    print('len(all particles) = ' + str(len(particles['ParticleIDs'])))

    return param_processing(particles, sim, part_type)


def param_processing(particles, sim, part_type):
    from hestia import calc_numberDensity, calc_fH0_cloudy

    X_H = np.array(particles['GFM_Metals'][:, 0]) if sim == '09_18' else 0.76

    if part_type == 'PartType4':
        # stars have SFT > 0, wind particles have SFT < 0
        stellar_mask = particles['GFM_StellarFormationTime'] > 0
        particles = {key: val[stellar_mask] for key, val in particles.items()}

    particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                e_abundance=np.array(particles['ElectronAbundance']),
                                                x_h=X_H)

    particles = filter_unphysical_metallicities(particles)

    particles['n_H'] = np.log10(calc_numberDensity(particles['Density'] * X_H))
    particles['f_H0'] = calc_fH0_cloudy(particles['Temperature']) if sim == '09_18_lastgigyear' \
        else particles['NeutralHydrogenAbundance']
    particles['n_H0'] = particles['n_H'] + np.log10(particles['f_H0'])

    dossier = {'n_H0': np.average(particles['n_H0'], weights=particles['Masses'] * particles['f_H0']),
               'M_H0': np.sum(particles['Masses'] * X_H * particles['f_H0'])}
    print('T_avg = ' + str(np.average(particles['Temperature'], weights=particles['Masses'] * particles['f_H0'])))

    return dossier


def package_data(snaps, particle):
    from hestia import get_lookbackTimes
    sim = '09_18'

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4'}
    part_type = part_to_type[particle]

    # this whole block exists because redshifts in lgy are rounded, and I can't find the raw numbers, approximation
    if snaps[0] < 118:
        z_lgy, _ = get_lookbackTimes(sim, [117, snaps[1]])
        z, _ = get_lookbackTimes('09_18', [snaps[0], 117])
        redshifts = np.append(np.linspace(z_lgy[0], z_lgy[-1], num=snaps[1] - 117),
                              np.linspace(z[0], z[-1], num=(117 - snaps[0])))
    else:
        z_lgy, _ = get_lookbackTimes(sim, snaps)
        redshifts = np.linspace(z_lgy[0], z_lgy[-1], num=(snaps[1] - snaps[0]))
    _, lookback_times = get_lookbackTimes(None, None, redshifts=redshifts)
    print('redshifts = ' + str(redshifts))
    print('len(lookback_times) = ' + str(len(lookback_times)))

    for snap_i in range(snaps[1], snaps[0], -1):

        # Transform S coordinates
        z_value = float(redshifts[0])
        print('-------- $z = {}$ --------'.format(z_value))

        if snap_i == snaps[1]:
            dossier = retrieve_particles(snap_i, sim, part_type)
        else:
            # Append new snapshot data
            dossie = retrieve_particles(snap_i, sim, part_type)
            for key in dossier:
                dossier[key] = np.append(np.array(dossier[key]), dossie[key])

        print('Snapshot ' + str(snap_i) + ': check')

    # Combine all dictionaries into one
    data_to_save = dossier
    data_to_save['redshifts'] = redshifts
    data_to_save['lookback_times'] = lookback_times

    # Save data
    output_path = '/z/rschisholm/storage/snapshots_stream_09_18/streamProperties_old.npz'
    np.savez(output_path, **data_to_save)

    # Indicate that the script has completed its task
    print('Done!\n----------------------------------------------')
    print('scp -P 2222 rschisholm@geras.aip.de:' + output_path +
          ' /Users/dear-prudence/Desktop/smorgasbord/isolateStream/')
    print('----------------------------------------------')


def plotting():
    import matplotlib.pyplot as plt

    input_path = '/Users/ursa/Desktop/smorgasbord/isolateStream/streamProperties.npz'
    dossier = np.load(input_path)
    print(dossier.keys())
    # redshifts = dossier['redshifts']
    lookback_times = dossier['lookback_times']
    print(lookback_times)
    # lookback_times = dossier['redshift'][:, 1]
    # n_H0 = dossier['n_H0']
    M_H0 = dossier['M_H0']
    print('n_H0 = ' + str(dossier['n_H0'][0]))
    print(M_H0)
    # M_H0 = 10 ** dossier['M_H0']
    fig, ax1 = plt.subplots(figsize=(7, 7))
    plt.style.use('classic')
    plt.rcParams.update({"grid.linestyle": "--",  # Dashed grid lines
                         "grid.alpha": 0.5,  # Fainter grid
                         "axes.grid": False,  # Enable grid
                         "xtick.direction": "in",  # Ticks pointing inward
                         "ytick.direction": "in",
                         "font.size": 12,  # Standard font size for papers
                         "axes.labelsize": 14,  # Larger labels
                         "axes.titlesize": 14,
                         "xtick.labelsize": 12,
                         "ytick.labelsize": 12,
                         })

    x_lim = [0, 3.12]
    y_lim_l = [1e7, 1e9]
    # y_lim_r = [1e-5, 1e-3]
    y_lim_r = y_lim_l

    c1 = 'k'
    c2 = 'k'

    # plt.tight_layout()
    # ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax1.set_xlim(x_lim)
    ax1.set_aspect('auto')
    ax1.tick_params(axis='y', labelcolor=c1)
    # ax2.tick_params(axis='y', labelcolor=c2)
    ax1.set_xlabel(r'Lookback Time $t$  ' + r'$[Gyr]$')
    ax1.set_ylabel(r'$M_{HI}$  ' + r'$[M_{\odot}]$', color=c1)
    # ax2.set_ylabel(r'$n_{HI}$  ' + r'$cm^{-3}$', color=c2)  # we already handled the x-label with ax1
    ax1.set_yscale('log')
    # ax2.set_yscale('log')
    ax1.set_ylim(y_lim_l)
    # ax2.set_ylim(y_lim_r)

    ax1.plot(lookback_times, M_H0, label='M_H0', c=c1)
    # ax2.plot(lookback_times, 10 ** n_H0, color=c2, linestyle='dotted', label='n_H0')

    plt.plot([2.65, 2.65], [y_lim_r[0], y_lim_r[1]], linestyle='dotted', color='k', alpha=0.2)
    # plt.text(2.63, y_lim_r[0] * 1.1, s='Start of major\nstripping event', fontsize='small')
    plt.text(2.63, y_lim_r[0] * 1.1, s=r'$(i)$', fontsize='small')
    plt.plot([1.52, 1.52], [y_lim_r[0], y_lim_r[1]], linestyle='dotted', color='k', alpha=0.2)
    # plt.text(1.49, y_lim_r[0] * 1.1, s='Third Perihelion,\nend of Besla+2012 dwarf\ninteraction in isolation',
    #          fontsize='small')
    plt.text(1.49, y_lim_r[0] * 1.1, s=r'$(ii)$', fontsize='small')
    plt.plot([0.52, 0.52], [y_lim_r[0], y_lim_r[1]], linestyle='dotted', color='k', alpha=0.2)
    # plt.text(0.49, y_lim_r[0] * 1.1, s='Present-day in\nBesla+2012 sims', fontsize='small')
    plt.text(0.49, y_lim_r[0] * 1.1, s=r'$(iii)$', fontsize='small')

    # Other formatting stuff
    plt.gca().invert_xaxis()
    # plt.title('Magellanic Trailing Stream-analog H0 Mass overtime '
    #           '(not including gas that has fallen into the LMC disk)', fontsize='small', loc='left')
    # -----------------------
    plt.savefig('/Users/dear-prudence/Desktop/smorgasbord/isolateStream/streamProperties.png', dpi=200)
    plt.show()


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
snaps_ = [87, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
particle_type = 'gas'
# ------------------------------------

if machine == 'geras':
    package_data(snaps_, particle_type)

elif machine == 'dear-prudence':
    plotting()
