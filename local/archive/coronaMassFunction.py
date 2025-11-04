import numpy as np


def snap_to_redshift(snap):
    filename = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc/'
                'HESTIA_100Mpc_8192_09_18.127_halo_127000000000001.dat')
    redshift = '{:.{}f}'.format(np.loadtxt(filename)[127 - snap, 0], 3)
    return redshift  # returns the redshift of a given snapshot in format x.xxx


def add_temperature(particles):
    from hestia import calc_temperature

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
    rho_crit = (3 * H0_cgs ** 2) / (8 * np.pi * G)  # g/cm^3

    # Convert virial mass to grams
    M_vir_g = M_vir * Msun

    # Calculate virial radius (in cm)
    R_vir = ((3 * M_vir_g) / (4 * np.pi * delta_vir * rho_crit)) ** (1 / 3)

    # Calculate virial temperature
    T_vir = (mu * m_p * G * M_vir_g) / (2 * k_B * R_vir)

    return T_vir


def plot_virial_temp_line(mass):
    x = np.linspace(min(mass), max(mass))
    y = np.log10(virial_temperature(10 ** mass))

    return x, y


def temperature_profile(run, halo_id, snap, redshift_, nH1_cutoff=-4.5, Z_cutoff=None):
    from hestia import get_halo_params
    from hestia import rid_h_units, transform_haloFrame
    from hestia import append_particles, filter_particles
    from hestia import calc_numberDensity
    import h5py

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    h = 0.677
    Z_solar = 0.0127
    # rho_b = 6.2317  # M_solar/kpc

    halo_params = get_halo_params(run, halo_id, snap, full_halo_id=True)
    halo_id, mass_h, pos_h, vel_h, l_h, r_vir_h = (halo_params['halo_id_zi'], halo_params['M_halo'],
                                                   halo_params['halo_pos'], halo_params['halo_vel'],
                                                   halo_params['halo_l'], halo_params['R_vir'])
    lb_h, ub_h = (pos_h - r_vir_h), (pos_h + r_vir_h)  # in kpc/h

    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run + '/output_2x2.5Mpc/snapdir_'
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

    unfiltered_particles = add_temperature(
        add_distances(
            transform_haloFrame(run, halo_id, snap,
                                rid_h_units(
                           filter_particles(all_particles, lb_h / 1e3, ub_h / 1e3),
                           float(redshift_), 'PartType0'))))

    unfiltered_particles['n_H1'] = np.log10(calc_numberDensity(unfiltered_particles['Density']
                                                               * unfiltered_particles['GFM_Metals'][:, 0])
                                            * (1 - unfiltered_particles['NeutralHydrogenAbundance']))

    R_bound = r_vir_h / h  # in kpc

    if Z_cutoff is None:
        mask = np.where((unfiltered_particles['n_H1'] < nH1_cutoff) &
                        (unfiltered_particles['Distances'] <= R_bound))[0]
    else:
        mask = np.where((unfiltered_particles['n_H1'] < nH1_cutoff) &
                        (unfiltered_particles['Distances'] <= R_bound) &
                        (unfiltered_particles['GFM_Metallicity'] / Z_solar < Z_cutoff))[0]
    corona_particles = {name: None for name in unfiltered_particles.keys()}
    for key in corona_particles.keys():
        corona_particles[key] = unfiltered_particles[key][mask]

    print('len(unfiltered_particles) = ' + str(len(unfiltered_particles['ParticleIDs'])))
    print('len(corona_particles) = ' + str(len(corona_particles['ParticleIDs'])))

    if len(corona_particles['ParticleIDs']) != 0:
        total_HI_mass = np.sum(corona_particles['Masses'] * corona_particles['GFM_Metals'][:, 0]
                               * corona_particles['NeutralHydrogenAbundance'])
        total_HII_mass = np.sum(corona_particles['Masses'] * corona_particles['GFM_Metals'][:, 0]
                                * (1 - corona_particles['NeutralHydrogenAbundance']))
        avg_temp = np.average(corona_particles['Temperature'],
                              weights=(1 - corona_particles['NeutralHydrogenAbundance']))
    else:
        # arbitrarily low values to since 0 would get mapped to -infinity
        total_HI_mass, total_HII_mass, avg_temp = 1, 1, 1

    print('M_HI, M_HII, T_avg = ' + str(list([total_HI_mass, total_HII_mass, avg_temp])))

    return total_HI_mass, total_HII_mass, avg_temp


def package_data(snap, how_many_halos, Z_cutoff):
    from hestia import get_lookbackTimes
    h = 0.677

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    redshift_ = snap_to_redshift(snap)
    sims = ['09_18', '17_11', '37_11']
    all_halos = {sim: None for sim in sims}

    for sim in sims:
        print('Working on ' + sim + '...')
        halos_file = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim + '/AHF_output_2x2.5Mpc/'
                                                                          'HESTIA_100Mpc_8192_' + sim + '.' + snap_
                      + '.z' + redshift_ + '.AHF_halos')
        halos_data = np.loadtxt(halos_file)

        halo_ids = halos_data[:, 0].astype(int)  # gets the ids of the highest mass halos
        halo_hosts = halos_data[:, 1].astype(int)
        halo_masses = halos_data[:, 3] / h  # gets the masses of the highest-mass halos
        halo_radii = halos_data[:, 11] / h  # gets the virial radii of the highest-mass halos

        # print('first ' + str(how_many_halos) + ' halos-- masses--\n' + str(halo_masses))

        i = 0  # while loop that checks for if the mass of halo_i > minimum mass
        j = 0
        while i < how_many_halos:
            halo_id = snap_ + '000000000' + '%03d' % (j + 1)

            # somewhat temporary fix for the fact that not all integers are mapped to halos
            # (e.g. 17_11 is missing 127...001)
            try:
                # this line is a check to see if halo_j has a .dat file
                np.loadtxt('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim
                           + '/AHF_output_2x2.5Mpc/HESTIA_100Mpc_8192_' + sim + '.127_halo_' + halo_id + '.dat')
                # halo_id = str(halo_ids[i])
                # if halo is above minimum mass, extract R_vir
                # using cutoffs from temperature profiles script, calculate mass and temperature of corona
                print('Calculating mass, avg_temp, and nH0 for halo ' + str(halo_id) + '...')
                mh0, mh1, t = temperature_profile(sim, halo_id, snap, redshift_, Z_cutoff=Z_cutoff)
                print('Is the halo a satellite?')
                bool_satellite = False if halo_hosts[j] == 0 else True
                print('Yes!' if bool_satellite is True else 'No!')
                # saves data to be stored as a new row with column formatting;
                # halo_id (without snapshot prefix), M_vir, R_vir, M_corona, T_corona, n_H0
                row = np.array([halo_id, bool_satellite, halo_masses[j], halo_radii[j], mh0, mh1, t])
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

    print('all_halos[09_18] = \n' + str(all_halos['09_18']))
    print('all_halos[17_11] = \n' + str(all_halos['17_11']))
    print('all_halos[37_11] = \n' + str(all_halos['37_11']))

    _, lookback_time = get_lookbackTimes(None, None, redshifts=float(redshift_))
    all_halos['redshift'] = redshift_
    all_halos['lookback_time'] = lookback_time

    # Save data
    output_path = ('/z/rschisholm/storage/analytical_plots/coronas/' + 'coronaMassRelation_halos_snap' + snap_
                   + ('_Zcutoff' if Z_cutoff is not None else '') + '.npz')
    np.savez(output_path, **all_halos)

    # Indicate that the script has completed its task
    print('Done!\n----------------------------------------------')
    print('scp -P 2222 rschisholm@geras.aip.de:' + output_path +
          ' /Users/dear-prudence/Desktop/smorgasbord/coronas/')
    print('----------------------------------------------')


def plotting(snap, x_axis, y_axis, z_cutoff):
    import matplotlib.pyplot as plt
    snap_ = '0' + str(snap) if snap < 100 else str(snap)

    if z_cutoff is not None:
        input_path = ('/Users/dear-prudence/Desktop/smorgasbord/coronas/coronaMassRelation_halos_snap'
                      + snap_ + '_Zcutoff.npz')
    else:
        input_path = '/Users/dear-prudence/Desktop/smorgasbord/coronas/coronaMassRelation_halos_snap' + snap_ + '.npz'
    all_halos = np.load(input_path)

    sims = ['09_18', '17_11', '37_11']
    colors = {'09_18': '#04d8b2', '17_11': '#7bc8f6', '37_11': '#c79fef'}

    var_dict = {'halo_id': 0, 'bool_satellite': 1, 'M_halo': 2, 'R_halo': 3, 'M_H0': 4, 'M_H1': 5, 'T_corona': 6}
    labels_dict = {'M_halo': r'$\log (M_{halo}/M_{\odot})$', 'R_halo': r'$R_{vir}$',
                   'M_H0': r'$M_{HI}$', 'M_H1': r'$M_{HII}$', 'T_corona': r'$T_{HII}$ ' + r'$[K]$'}

    fig = plt.figure(figsize=(7, 7))

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

    data = {name: None for name in var_dict.keys()}
    for var in var_dict.keys():
        if var == 'bool_satellite':
            data[var] = (list(all_halos['09_18'][:, 1] == b'True')
                         + list(all_halos['17_11'][:, 1] == b'True')
                         + list(all_halos['37_11'][:, 1] == b'True'))
        else:
            data[var] = np.log10(np.append(np.array(all_halos['09_18'][:, var_dict[var]], dtype=np.float64),
                                           [np.array(all_halos['17_11'][:, var_dict[var]], dtype=np.float64),
                                            np.array(all_halos['37_11'][:, var_dict[var]], dtype=np.float64)]))

        # ---------------------------
        # temp fix to adjust for those halos with nan entry
        # to_keep = np.logical_not(np.isnan(y_data))
        # x_data = x_data[to_keep]
        # y_data = y_data[to_keep]
        # ---------------------------

    colors = []
    for i in range(len(data['bool_satellite'])):
        if data['bool_satellite'][i]:
            colors.append('none')
        else:
            if data['M_H1'][i] < 8.25:
                colors.append('k')
            else:
                colors.append('k')

    print(data['M_halo'].shape)

    plt.scatter(data[x_axis], data[y_axis], fc=colors, ec='k', s=25)

    if y_axis == 'T_corona':
        x_arr = np.linspace(min(data[x_axis]), max(data[x_axis]))
        x_l, y_l = plot_virial_temp_line(x_arr)
        plt.plot(x_l, y_l, linestyle='dotted', color='k', alpha=1, label='Virial Theorem')

    x_lim = [10.25, 12.75]
    y_lim = [4.5, 7]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel(labels_dict[x_axis])
    plt.ylabel(labels_dict[y_axis])
    plt.grid(visible=True, alpha=0.5)

    plt.savefig('/Users/dear-prudence/Desktop/smorgasbord/coronas/'
                'corona_' + x_axis + '-' + y_axis + '_snap' + snap_ + '.png', dpi=200)
    plt.show()
    plt.close()


# ---------------------------------------------
mode = 'dear-prudence'
# ---------------------------------------------
snapshot = 95
how_many_halos_ = 20  # number of halos per simulation run
Z_cutoff_ = None  # in Z_solar, None if no cutoff is needed
# ---------------------------------------------
x_variable = 'M_halo'
y_variable = 'T_corona'
# ---------------------------------------------

if mode == 'geras':
    package_data(snapshot, how_many_halos_, Z_cutoff=Z_cutoff_)

elif mode == 'dear-prudence':
    plotting(snapshot, x_variable, y_variable, Z_cutoff_)

print('Done!')
