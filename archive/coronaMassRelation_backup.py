import numpy as np


def snap_to_redshift(snap):
    filename = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc/'
                'HESTIA_100Mpc_8192_09_18.127_halo_127000000000001.dat')
    redshift = '{:.{}f}'.format(np.loadtxt(filename)[127 - snap, 0], 3)
    return redshift  # returns the redshift of a given snapshot in format x.xxx


def add_temperature(particles):
    from archive.hestia import calc_temperature

    temp_column = calc_temperature(u=np.array(particles['InternalEnergy']), e_abundance=np.array(particles['ElectronAbundance']),
                                   x_h=np.array(particles['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    particles['Temperature'] = temp_column
    return particles


def delta_b(log_M_vir):
    rho_b = 6.2317  # M_solar/kpc
    m = (10 - 3.5) / (12.3 - 11.3)
    l_b = m * (log_M_vir - 11.3) + 3.5 if log_M_vir > 11 else 1.55
    u_b = 10 * l_b
    return l_b * rho_b, u_b * rho_b


def add_distances(particles):
    # Initialize an array to store distances
    distances = np.zeros(particles['Coordinates'].shape[0])
    # Compute distances for each particle
    for i in range(particles['Halo_Coordinates'].shape[0]):
        distances[i] = np.linalg.norm(particles['Halo_Coordinates'][i])
    # Add the new column to the data dictionary
    particles['Distances'] = distances
    return particles


def virial_temperature(M_vir, H0=67.7, delta_vir=200, mu=0.59):
    """
    Calculate the virial temperature of a halo from its virial mass.

    Parameters:
    M_vir (float): Virial mass of the halo in solar masses (M_sun).
    H0 (float): Hubble constant in km/s/Mpc (default: 67.7).
    delta_vir (float): Virial overdensity parameter (default: 200).
    mu (float): Mean molecular weight (default: 0.59 for ionized gas).

    Returns:
    float: Virial temperature in Kelvin.
    """
    # Constants
    G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
    k_B = 1.3807e-16  # Boltzmann constant in erg K^-1
    m_p = 1.673e-24  # Proton mass in grams
    Msun = 1.989e33  # Solar mass in grams
    H0_cgs = H0 * (1e5 / 3.086e24)  # Hubble constant in s^-1

    # Critical density of the universe
    rho_crit = (3 * H0_cgs**2) / (8 * np.pi * G)  # g/cm^3

    # Convert virial mass to grams
    M_vir_g = M_vir * Msun

    # Calculate virial radius (in cm)
    R_vir = ((3 * M_vir_g) / (4 * np.pi * delta_vir * rho_crit))**(1/3)

    # Calculate virial temperature
    T_vir = (mu * m_p * G * M_vir_g) / (2 * k_B * R_vir)

    return T_vir


def plot_virial_temp_line(mass):
    x = np.linspace(min(mass), max(mass))
    y = np.log10(virial_temperature(10 ** mass))

    return x, y


def filter_particles_by_param(data, parameter, inq, threshold, usage=False):
    if usage is True:
        print('Filtering particles by ' + str(parameter) + '...')
        for key in data.keys():
            data[key] = np.array(data[key])  # Convert to numpy array for easier indexing
        if inq == '<=':
            indices_to_keep = np.where(data[parameter] <= threshold)[0]
        elif inq == '<':
            indices_to_keep = np.where(data[parameter] < threshold)[0]
        elif inq == '>':
            indices_to_keep = np.where(data[parameter] > threshold)[0]
        else:
            print('Error: Invalid inq value!')
            exit(1)
        print('Done with ' + str(parameter) + ' filtering.')
        for key in data.keys():
            data[key] = data[key][indices_to_keep]
    return data


def temperature_profile(snap, redshift_, sim_run, halo_id, r_vir, M_vir):
    from archive.hestia import rid_h_units, transform_haloFrame
    from archive.hestia import center_halo
    from archive.hestia import append_particles, filter_particles
    import h5py

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    h = 0.677
    rho_b = 6.2317  # M_solar/kpc

    size = np.array([200, 200, 200]) * h  # in kpc_h units

    l_b, u_b = center_halo(run=sim_run, halo_id=halo_id, snap=snap, size=size) * 1e-3  # these are in _h units!
    # l_b and u_b carve a cube with side length 2x specified in "size", to account for the gaps in spatial
    # distribution of particles after the halo coordinate transformation

    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim_run + '/output_2x2.5Mpc/snapdir_'
                 + snap_ + '/snapshot_' + snap_ + '.')
    file_extension = '.hdf5'

    with h5py.File(base_path + '0' + file_extension, 'r') as file:
        keys = file['PartType0'].keys()

    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    all_particles = {name: None for name in keys}
    print('Processing Snapshot ' + snap_ + '...')
    for file_path in file_paths:
        all_particles = append_particles('PartType0', file_path, key_names=keys,
                                         existing_arrays=all_particles)

    processed_particles = add_temperature(
        add_distances(
            transform_haloFrame(sim_run, halo_id, snap,
                                rid_h_units(
                           filter_particles(all_particles, l_b, u_b), float(redshift_), 'PartType0'))))

    # -------------------------------
    rho_l_b, rho_u_b = delta_b(np.log10(M_vir))
    # -------------------------------

    filtered_particles = {name: None for name in processed_particles.keys()}
    indices_to_keep = np.where((processed_particles['Density'] > rho_l_b) &
                               (processed_particles['Density'] < rho_u_b))[0]

    for key in processed_particles.keys():
        filtered_particles[key] = processed_particles[key][indices_to_keep]

    print('len(filtered_particles) = ' + str(len(filtered_particles['ParticleIDs'])))

    mass_enclosed_radius = r_vir / h  # in kpc
    radius_mask = filtered_particles['Distances'] <= mass_enclosed_radius

    corona_particles = {name: None for name in filtered_particles.keys()}

    try:
        for key in corona_particles.keys():
            corona_particles[key] = processed_particles[key][radius_mask]

        print('len(corona_particles) = ' + str(len(corona_particles['ParticleIDs'])))

        f_h1 = corona_particles['GFM_Metals'][:, 0] * (1 - corona_particles['NeutralHydrogenAbundance'])

        H1_mass = np.sum(corona_particles['Masses'] * f_h1)
        avg_temp = np.average(corona_particles['Temperature'], weights=f_h1)
        std_temp = np.std(corona_particles['Temperature'])
        average_nH0 = np.average(corona_particles['NeutralHydrogenAbundance'] * corona_particles['GFM_Metals'][:, 0])
        average_agn = np.average(corona_particles['GFM_AGNRadiation'])

    # for when there are no corona particles
    except IndexError:
        print('len(corona_particles) = ' + '0')
        H1_mass = np.NaN
        avg_temp = np.NaN
        std_temp = np.NaN
        average_nH0 = np.NaN
        average_agn = np.NaN

    return H1_mass, avg_temp, std_temp, average_nH0, average_agn


def package_data(snap, d_b, m_min):
    rho_b = 6.2317  # M_solar/kpc

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    redshift_ = snap_to_redshift(snap)
    sims = ['09_18', '17_11', '37_11']
    all_halos = {sim: None for sim in sims}

    how_many_halos = 30

    for sim in sims:
        print('Working on ' + sim + '...')
        halos_file = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim + '/AHF_output_2x2.5Mpc/'
                      'HESTIA_100Mpc_8192_' + sim + '.' + snap_ + '.z' + redshift_ + '.AHF_halos')
        halos_data = np.loadtxt(halos_file)

        halo_ids = halos_data[:how_many_halos, 0].astype(int)  # gets the ids of the highest mass halos
        print(halo_ids)

        halo_masses = halos_data[:how_many_halos, 3]  # gets the masses of the highest-mass halos
        halo_radii = halos_data[:how_many_halos, 11]  # gets the virial radii of the highest-mass halos

        print('first ' + str(how_many_halos) + ' halos-- masses--\n' + str(halo_masses))

        i = 0  # while loop that checks for if the mass of halo_i > minimum mass
        j = 0
        while halo_masses[i] > m_min and i < how_many_halos:
            halo_id = snap_ + '000000000' + '%03d' % (j + 1)

            # somewhat temporary fix for the fact that not all integers are mapped to halos
            # (e.g. 17_11 is missing 127...001)
            try:
                np.loadtxt('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim
                           + '/AHF_output_2x2.5Mpc/HESTIA_100Mpc_8192_' + sim + '.127_halo_' + halo_id + '.dat')
                # halo_id = str(halo_ids[i])
                # if halo is above minimum mass, extract R_vir
                # using cutoffs from temperature profiles script, calculate mass and temperature of corona
                print('Calculating mass, avg_temp, and nH0 for halo ' + str(halo_id) + '...')
                m, t, s, n, a = temperature_profile(snap, redshift_, sim, halo_id, halo_radii[i], halo_masses[i])
                # saves data to be stored as a new row with column formatting;
                # halo_id (without snapshot prefix), M_vir, R_vir, M_corona, T_corona, n_H0
                row = np.array([halo_id, halo_masses[i], halo_radii[i], m, t, s, n, a])
                data_array = np.vstack((data_array, row)) if i != 0 else np.expand_dims(row, axis=0)
                # Convert to (1, N)

                i += 1
                j += 1

                if i == how_many_halos:
                    break
                else:
                    pass

            except IOError:
                j += 1

        all_halos[sim] = data_array

    # all_halos['nH0_threshold'] = nh0
    all_halos['delta_b'] = d_b
    all_halos['minimum_halo_mass'] = m_min

    print('all_halos[09_18] = \n' + str(all_halos['09_18']))
    print('all_halos[17_11] = \n' + str(all_halos['17_11']))
    print('all_halos[37_11] = \n' + str(all_halos['37_11']))
    # Save data
    np.savez('/z/rschisholm/storage/analytical_plots/coronas/'
             + 'coronaMassRelation_halos.npz', **all_halos)


def plotting(snap):
    import matplotlib.pyplot as plt

    input_path = '/Users/ursa/Desktop/smorgasbord/coronas/coronaMassRelation_halos.npz'
    all_halos = np.load(input_path)

    # nh0 = all_halos['nH0_threshold']
    # d_b = all_halos['delta_b']
    m_min = all_halos['minimum_halo_mass']

    sims = ['09_18', '17_11', '37_11']
    colors = {'09_18': '#04d8b2', '17_11': '#7bc8f6', '37_11': '#c79fef'}

    var_dict = {'halo_id': 0, 'M_halo': 1, 'R_halo': 2, 'M_corona': 3, 'T_corona': 4, 'n_H0': 6, 'agn': 7}

    # -----------------------------------
    x = 'M_halo'
    y = 'T_corona'
    error_bars = True
    # -----------------------------------

    fig = plt.figure(figsize=(10, 8))

    x_all = np.array([])
    y_all = np.array([])
    err_a = np.array([])
    for sim in sims:
        x_data = np.log10(all_halos[sim][:, var_dict[x]].astype(float))
        y_data = np.log10(all_halos[sim][:, var_dict[y]].astype(float))
        m_data = np.log10(all_halos[sim][:, 3].astype(float))
        agn = np.log10(all_halos[sim][:, 7].astype(float))

        # ---------------------------
        # temp fix to adjust for those halos without a reasonable corona
        # to_keep = np.where((m_data > 6))[0]
        # x_data = x_data[to_keep]
        # y_data = y_data[to_keep]
        # ---------------------------

        # ---------------------------
        # temp fix to get rid of the outlier in 37_11 with M > 1e13
        to_keep = np.where(x_data < 13)[0]
        x_data = x_data[to_keep]
        y_data = y_data[to_keep]
        # ---------------------------

        # ---------------------------
        # temp fix to adjust for those halos with nan entry
        to_keep = np.logical_not(np.isnan(y_data))
        x_data = x_data[to_keep]
        y_data = y_data[to_keep]
        # ---------------------------

        x_all = np.append(x_all, x_data)
        y_all = np.append(y_all, y_data)

        plt.scatter(x_data, y_data, label=sim, c=colors[sim])
        # plt.errorbar(x_data, y_data, yerr=err_, xerr=None, fmt='none')

    z = np.polyfit(x_all, y_all, deg=1)
    a = z[0]
    b = z[1]
    x_arr = np.linspace(min(x_all), max(x_all))
    y_arr = a * x_arr + b
    plt.plot(x_arr, y_arr, linestyle='dashed', color='tab:blue', label='Linear Fit')

    if y == 'T_corona':
        x_l, y_l = plot_virial_temp_line(x_arr)
        plt.plot(x_l, y_l, linestyle='dashed', color='k', alpha=0.5, label='Virial Theorem')

    plt.legend(loc='lower right')

    plt.xlabel(x)
    plt.ylabel(y) if y != 'M_corona' else plt.ylabel(r'$M_{HII}$')
    plt.grid(visible=True, alpha=0.5)

    plt.savefig('/Users/dear-prudence/Desktop/smorgasbord/coronas/coronaMassRelation.png', dpi=200)
    plt.show()
    plt.close()


# ---------------------------------------------
mode = 'dear-prudence'
# ---------------------------------------------
snapshot = 127
# nH0_threshold = 1e-5  # maximum neutral H fraction required to be classified as part of the corona
# delta_b = [4, 40]  # overdensity parameter to multiply the baryon critical density for IGM
minimum_halo_mass = 1e9
# ---------------------------------------------

if mode == 'geras':
    package_data(snapshot, delta_b, minimum_halo_mass)

elif mode == 'dear-prudence':
    plotting(snapshot)

print('Done!')

