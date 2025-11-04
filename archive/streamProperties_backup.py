import sys
import numpy as np
from scripts.hestia import append_particles
from scripts.hestia import calc_temperature


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


def param_processing(particles, part_type):
    if part_type == 'PartType4':
        # stars have SFT > 0, wind particles have SFT < 0
        stellar_mask = particles['GFM_StellarFormationTime'] > 0
        particles = {key: val[stellar_mask] for key, val in particles.items()}

    particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                e_abundance=np.array(particles['ElectronAbundance']),
                                                x_h=np.array(particles['GFM_Metals'][:, 0]))

    particles = filter_unphysical_metallicities(particles)
    # particles = filter_disk_particles(particles, cutoff=0)

    particles['H_Mass'] = particles['GFM_Metals'][:, 0] * particles['Masses']
    m_H = np.log10(np.sum(particles['H_Mass']))
    m_H0 = np.log10(np.sum(particles['H_Mass'] * particles['NeutralHydrogenAbundance']))
    m_H1 = np.log10(np.sum(particles['H_Mass'] * (1 - particles['NeutralHydrogenAbundance'])))

    # Proton mass in grams
    m_p = 1.673e-24  # g
    mu = 0.59
    M_sun = 1.989e33  # Solar mass in grams
    kpc_to_cm = 3.086e21  # kpc to cm
    Rho = np.average(particles['Density']) * M_sun / (kpc_to_cm ** 3)

    # Calculate number density
    avg_n = Rho / (mu * m_p)  # rho (float): Mass density in g/cm^3.

    E_diss = np.average(particles['EnergyDissipation'],
                        weights=(particles['H_Mass'] * particles['NeutralHydrogenAbundance']))
    cool = np.average(particles['GFM_CoolingRate'],
                      weights=(particles['H_Mass'] * particles['NeutralHydrogenAbundance']))
    agn = np.average(particles['GFM_AGNRadiation'],
                     weights=(particles['H_Mass'] * particles['NeutralHydrogenAbundance']))

    particles['log_num_H_density'] = (np.log10(particles['Density'] * particles['GFM_Metals'][:, 0]
                                               * M_sun / (kpc_to_cm ** 3 * mu * m_p)))

    hist, x_e = np.histogram(particles['log_num_H_density'], bins=100, range=(-6, 4))

    return m_H, m_H0, m_H1, avg_n, E_diss, hist, x_e, cool, agn


def retrieve_particles(snap_i, part_type):
    import h5py

    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    key_path = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_127/snapshot_127.0.hdf5'
    with h5py.File(key_path, 'r') as file:
        keys = list(file[part_type].keys()) + ['Halo_Coordinates']

    base_path = ('/z/rschisholm/storage/snapshots_stream/snapshot_' + snap + '.hdf5')
    file_paths = [base_path]
    all_particles = {name: None for name in keys}

    print('Processing Snapshot ' + snap + '...')
    for file_path in file_paths:
        all_particles = append_particles(part_type, file_path, key_names=keys,
                                         existing_arrays=all_particles)

    print('len(all particles) = ' + str(len(all_particles['Masses'])))

    return param_processing(all_particles, part_type)


def package_data(snaps, particle):
    from scripts.hestia import time_edges

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4'}
    part_type = part_to_type[particle]

    time_e = time_edges(sim='09_18', snaps=np.arange(snaps[1], snaps[0], step=-1))

    for i in range(snaps[1], snaps[0], -1):

        # Transform S coordinates
        z_value = float(time_e[snaps[1] - i][0])
        print('-------- $z = {}$ --------'.format(z_value))

        m_H, m_H0, m_H1, avg_n, E_diss, hist, x_e, cool, agn = retrieve_particles(i, part_type)

        if i == snaps[1]:
            H_mass = np.array([m_H])
            H0_mass = np.array([m_H0])
            H1_mass = np.array([m_H1])
            num_den = np.array([avg_n])
            energy_diss = np.array([E_diss])
            hists = np.array([hist])
            cools = np.array([cool])
            agns = np.array([agn])
        else:
            # Append new snapshot data
            H_mass = np.append(H_mass, m_H)
            H0_mass = np.append(H0_mass, m_H0)
            H1_mass = np.append(H1_mass, m_H1)
            num_den = np.append(num_den, avg_n)
            energy_diss = np.append(energy_diss, E_diss)
            hists = np.vstack((hists, hist))
            cools = np.append(cools, cool)
            agns = np.append(agns, agn)

        print('Snapshot ' + str(i) + ': check')

        print('avg_n = ' + str(avg_n))

    # Combine all dictionaries into one
    data_to_save = {'redshift': time_e, 'H_mass': H_mass, 'H0_mass': H0_mass, 'H1_mass': H1_mass, 'num_den': num_den,
                    'energy_diss': energy_diss, 'hists': hists, 'x_e': x_e, 'cools': cools, 'agns': agns}

    # Save data
    np.savez('/z/rschisholm/storage/snapshots_stream/streamProperties.npz', **data_to_save)

    # Indicate that the script has completed its task
    print('Done!')
    # Terminate the script
    sys.exit(0)


def plotting(snaps):
    import matplotlib.pyplot as plt

    input_path = '/Users/ursa/Desktop/smorgasbord/isolateStream/streamProperties.npz'
    data = np.load(input_path)

    lookback_times = data['redshift'][127 - snaps[1]:127 - snaps[0], 1]
    H_mass = data['H_mass'][127 - snaps[1]:127 - snaps[0]]
    H0_mass = data['H0_mass'][127 - snaps[1]:127 - snaps[0]]
    H1_mass = data['H1_mass'][127 - snaps[1]:127 - snaps[0]]
    num_den = data['num_den'][127 - snaps[1]:127 - snaps[0]]
    # energy_diss = data['energy_diss'][127 - snaps[1]:127 - snaps[0]]
    # cools = data['cools'][127 - snaps[1]:127 - snaps[0]]
    agns = data['agns'][127 - snaps[1]:127 - snaps[0]]
    hist = data['hists'][0]
    x_e = data['x_e']
    print(num_den)

    fig, ax = plt.subplots(figsize=(9, 6))
    # ax.plot(lookback_times, 10 ** H_mass, label='H_mass')
    ax.plot(lookback_times, H0_mass, label='H0_mass')
    # ax.plot(lookback_times, 10 ** H1_mass, label='H1_mass')
    # ax.plot(lookback_times, 10 ** H0_mass / 10 ** H_mass, label='n_H0')
    # ax.plot(lookback_times, 10 ** H1_mass / 10 ** H_mass, label='n_H1')
    # ax.plot(lookback_times, np.log10(energy_diss) + 9, linestyle='dashed', label='Energy_Diss')
    # ax.plot(lookback_times, np.log10(cools) + 23, linestyle='dashed', label='CoolingRate')
    # ax.plot(lookback_times, np.log10(agns) + 13, linestyle='dashed', label='AGN_radiation', alpha=0.5)
    # ax.plot(lookback_times, np.log10(num_den * (10 ** H0_mass / 10 ** H_mass)))
    # plt.step(x_e[:-1], hist, where='mid')
    ax.set_ylabel(r'$\log(M_{H0})$')
    # plt.yscale('log')
    # ax.tick_params(axis='y')
    ax.set_xlabel(r'Lookback Time $t$')
    # z = np.polyfit(lookback_times, H0_mass, deg=1)
    # a = z[0]
    # b = z[1]
    # x_arr = np.linspace(min(lookback_times), max(lookback_times))
    # y_arr = a * x_arr + b
    # plt.plot(x_arr, y_arr, linestyle='dashed', color='tab:blue', label='Linear Fit', alpha=0.5)

    plt.legend(loc='upper right')

    plt.plot([1.52, 1.52], [6.3, 8.3], linestyle='dotted', color='k', alpha=0.2)
    plt.text(1.49, 8.1, s='Third Perihelion,\nend of Besla+2012 dwarf\ninteraction in isolation', fontsize='small')
    plt.plot([0.52, 0.52], [6.3, 8.3], linestyle='dotted', color='k', alpha=0.2)
    plt.text(0.49, 7.8, s='Present-day in\nBesla+2012 sims', fontsize='small')

    plt.ylim([6.3, 8.3])

    plt.style.use('classic')

    # Other formatting stuff
    # plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.title('Magellanic Trailing Stream-analog H0 Mass overtime '
              '(not including gas that has fallen into the LMC disk)', fontsize='small', loc='left')
    # -----------------------
    plt.savefig('/Users/dear-prudence/Desktop/smorgasbord/isolateStream/stream_H0_mass.png', dpi=200)
    plt.show()


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
snaps_ = [107, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
particle_type = 'gas'
# ------------------------------------

if machine == 'geras':
    package_data(snaps_, particle_type)

elif machine == 'dear-prudence':
    plotting(snaps_)
