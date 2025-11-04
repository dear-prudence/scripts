"""This script will extract particles from a specified snapshot and plot their relative mass density in
temperature-coordinate space, the so-called temperature profile of a specified halo"""
from hestia import calc_temperature
from hestia import transform_haloFrame
from hestia import get_halo_params


def add_temperature(run, particles):
    X_H = particles['GFM_Metals'][:, 0] if run != '09_18_lastgigyear' else 0.76
    particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                e_abundance=np.array(particles['ElectronAbundance']), x_h=X_H)
    return particles


def add_distances(particles):
    # Initialize an array to store distances
    distances = np.zeros(particles['Coordinates'].shape[0])
    # Compute distances for each particle
    for i in range(particles['Halo_Coordinates'].shape[0]):
        distances[i] = np.linalg.norm(particles['Halo_Coordinates'][i])
    # Add the new column to the data dictionary
    particles['Distances'] = distances
    return particles


def retrieve_particles(sim, halo, snap, z, previous_halo_id):
    from hestia import calc_numberDensity, calc_fH0_cloudy
    from hestia import rid_h_units
    h = 0.677
    X_H = 0.76

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    h_pms = get_halo_params(sim, halo, snap, previous_halo_id=previous_halo_id)
    halo_id, pos_h, l_h, r_vir_h = h_pms['halo_id_zi'], h_pms['halo_pos'], h_pms['halo_l'], h_pms['R_vir']
    lb_h, ub_h = (pos_h - 1.5 * r_vir_h), (pos_h + 1.5 * r_vir_h)

    try:
        if sim == '09_18_lastgigyear':
            file_path = '/z/rschisholm/storage/snapshots_' + halo + '/snapshot_' + snap_ + '.hdf5'
        else:
            file_path = ('/z/rschisholm/storage/snapshots_' + halo + '/snapshots_lmc_traditional/snapshot_'
                         + snap_ + '.hdf5')
        with h5py.File(file_path, 'r') as file:
            keys = file['PartType0'].keys()
        all_particles = {name: None for name in keys}
        all_particles = append_particles('PartType0', file_path, key_names=keys, existing_arrays=all_particles)

    except IOError:
        print('Warning: \'/snapshots_' + halo + '/\' directory not found, extracting from original output...')
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim + '/output_2x2.5Mpc/snapdir_'
                     + snap_ + '/snapshot_' + snap_ + '.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        with h5py.File(base_path + '0' + file_extension, 'r') as file:
            keys = file['PartType0'].keys()
        all_particles = {name: None for name in keys}
        for file_path in file_paths:
            all_particles = append_particles('PartType0', file_path, key_names=keys,
                                             existing_arrays=all_particles)

    processed_particles = add_temperature(sim,
                                          add_distances(
                                              transform_haloFrame(sim, halo, snap,
                                                                  rid_h_units(
                                                             filter_particles(all_particles, lb_h / 1e3, ub_h / 1e3), z,
                                                             'PartType0'), previous_halo_id=previous_halo_id)))

    if 'NeutralHydrogenAbundance' in processed_particles.keys():
        processed_particles['f_H1'] = 1 - processed_particles['NeutralHydrogenAbundance']
        processed_particles['n_H1'] = (calc_numberDensity(processed_particles['Density']
                                                          * processed_particles['GFM_Metals'][:, 0], mu=0.59)
                                       * processed_particles['f_H1'])
    else:
        _, processed_particles['f_H1'] = calc_fH0_cloudy(processed_particles['Temperature'])
        processed_particles['n_H1'] = (calc_numberDensity(processed_particles['Density'] * X_H, mu=0.59)
                                       * processed_particles['f_H1'])
    processed_particles['R_vir'] = r_vir_h / h

    return processed_particles, halo_id


def create_plot(unfiltered_particles_, radii, nH1_cutoff=-4.5):
    h = 0.677
    Z_solar = 0.0127

    metals_cutoff = 0.1
    # metals_mask = unfiltered_particles['GFM_Metallicity'] / Z_solar < metals_cutoff
    # unfiltered_particles_ = {name: None for name in unfiltered_particles.keys()}
    # for key in unfiltered_particles_.keys():
    #     if isinstance(unfiltered_particles[key], np.ndarray):
    #         unfiltered_particles_[key] = unfiltered_particles[key][metals_mask]
    #     else:
    #         unfiltered_particles_[key] = unfiltered_particles[key]

    density_mask = np.log10(unfiltered_particles_['n_H1']) < nH1_cutoff
    particles = {name: None for name in unfiltered_particles_.keys()}
    for key in particles.keys():
        if isinstance(unfiltered_particles_[key], np.ndarray):
            particles[key] = unfiltered_particles_[key][density_mask]
        else:
            particles[key] = unfiltered_particles_[key]

    mass_enclosed_radius = particles['R_vir']
    radius_mask = particles['Distances'] <= mass_enclosed_radius

    particles_c = {name: None for name in particles.keys()}
    for key in particles.keys():
        if isinstance(unfiltered_particles_[key], np.ndarray):
            particles_c[key] = particles[key][radius_mask]
        else:
            particles_c[key] = particles[key]

    # X_H = particles['GFM_Metals'][:, 0] if 'GFM_Metals' in particles.keys() else 0.76
    X_H = 0.76

    print('len(particles) = ' + str(len(particles['ParticleIDs'])))
    print('len(particles_c) = ' + str(len(particles_c['ParticleIDs'])))

    total_HI_mass = np.sum(particles_c['Masses'] * X_H * (1 - particles_c['f_H1']))
    total_HII_mass = np.sum(particles_c['Masses'] * X_H * particles_c['f_H1'])

    avg_temp = np.average(particles_c['Temperature'], weights=particles_c['Masses'] * particles_c['f_H1'])
    average_n = np.average(particles_c['n_H1'], weights=particles_c['Masses'] * particles_c['f_H1'])

    average_fH1 = np.average(particles_c['f_H1'])

    print('M_HI, M_HII, T_avg, n_H1, fH1_avg = ' + str(list([total_HI_mass, total_HII_mass, avg_temp,
                                                             average_n, average_fH1])))

    radius_range = [0, round((1.2 * particles['R_vir']), 2)]
    log_temperature_range = [4.5, 6.5]

    bins = 300

    sum_hist, _ = np.histogram(particles['Distances'], bins=len(radii) / 2, range=(radii[0], radii[-1]),
                               weights=np.log10(particles['Temperature'])
                                       * particles['n_H1'] * particles['Masses'] * X_H)
    mass_hist, _ = np.histogram(particles['Distances'], bins=len(radii) / 2, range=(radii[0], radii[-1]),
                                weights=particles['n_H1'] * particles['Masses'] * X_H)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_hist = np.divide(sum_hist, mass_hist, where=(mass_hist != 0))

    # radial binning is per 0.5 kpc, temperature binning is half number of radial bins
    hist, x_e, y_e = np.histogram2d(particles['Distances'], np.log10(particles['Temperature']),
                                    range=np.array([radius_range, log_temperature_range]),
                                    bins=[bins, bins / 2],
                                    weights=particles['Masses'] * X_H * particles['f_H1'], normed=True)

    print('hist.shape ' + str(hist.shape))
    column_averages = np.zeros(hist.shape[0])
    print('hist[0].shape ' + str(hist[0].shape))
    print('y_e[:-1] + abs(y_e[0] - y_e[1]) / 2 ' + str(np.array(y_e[:-1] + abs(y_e[0] - y_e[1]) / 2).shape))
    print('column_averages ' + str(column_averages.shape))
    for i in range(hist.shape[0]):
        try:
            column_averages[i] = np.average(y_e[:-1] + abs(y_e[0] - y_e[1]) / 2, weights=hist[i])
        except ZeroDivisionError:
            column_averages[i] = np.NaN

    dossier = {'H0_mass': total_HI_mass, 'H1_mass': total_HII_mass, 'temperature': avg_temp, 'f_H1': average_fH1,
               'hist': hist, 'column_averages': column_averages}
    return dossier, x_e, y_e


def package_data(run, halo, snaps, nH1_cutoff):
    from hestia import calc_temperatureProfile
    from hestia import get_lookbackTimes
    h = 0.667

    redshifts, lookback_times = get_lookbackTimes(run, snaps)
    previous_halo_id = None

    for snap_i in range(snaps[1], snaps[0], -1):

        particles, halo_id = retrieve_particles(run, halo, snap_i, redshifts[snaps[1] - snap_i], previous_halo_id)

        T, r = calc_temperatureProfile(run, halo, snap_i, previous_halo_id=previous_halo_id)

        # in general, this condition is for redshift z = 0.0
        if snap_i == snaps[1]:
            dossier, x_e, y_e = create_plot(particles, r, nH1_cutoff)
            Ts, rs = T, r
            virial_radii = particles['R_vir']

        else:
            # noinspection PyUnboundLocalVariable
            dossie, _, _ = create_plot(particles, r, nH1_cutoff)
            # Append the new snapshot
            for key in dossier.keys():
                dossier[key] = np.append(dossier[key], dossie[key]) if key != 'hist' \
                    else np.dstack((dossier[key], dossie[key]))

            Ts = np.vstack((Ts, T))
            rs = np.vstack((rs, r))
            virial_radii = np.append(virial_radii, particles['R_vir'])

        previous_halo_id = halo_id

    dossier['x_e'], dossier['y_e'] = x_e, y_e
    dossier['profiles_T'], dossier['profiles_r'] = Ts, rs
    dossier['redshifts'], dossier['lookback_times'] = redshifts, lookback_times
    dossier['virial_radii'] = virial_radii

    # noinspection PyUnboundLocalVariable
    output_path = ('/z/rschisholm/storage/analytical_plots/temperatureProfiles/'
                   + run + '_gas_temperatureProfile_' + halo + '.npz')
    np.savez(output_path, **dossier)

    # Indicate that the script has completed its task
    print('Done!\n----------------------------------------------')
    print('scp -P 2222 rschisholm@geras.aip.de:' + output_path +
          ' /Users/dear-prudence/Desktop/smorgasbord/temperatureProfiles/' + halo + '/'
          + run + '_gas_temperatureProfile_' + halo + '.npz')
    print('----------------------------------------------')


def plotting(sim_run, halo, snap, include_lucchini_sim, include_coronaeRelation):
    from scripts.local.archive.plots_old import plot_temperature_profile_dark, plot_tempProf_and_coronaRelation

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/temperatureProfiles/' + halo + '/'
                  + sim_run + '_gas_temperatureProfile_' + halo + '.npz')

    output_path = ('/Users/dear-prudence/Desktop/smorgasbord/temperatureProfiles/' + halo + '/'
                   + sim_run + '_gas_temperatureProfile_' + halo + '_snapshot' + str(snap) + '.png')

    if include_coronaeRelation is False:
        plot_temperature_profile_dark(input_path, output_path, snap, include_lucchini_sim)
    else:
        snap_ = '0' + str(snap) if snap < 100 else str(snap)
        input_path_coronae = ('/Users/dear-prudence/Desktop/smorgasbord/coronas/coronaMassRelation_halos_snap'
                              + snap_ + '.npz')

        output_path = ('/Users/dear-prudence/Desktop/smorgasbord/temperatureProfiles/' + halo + '/'
                       + sim_run + '_gas_tempProfcoronae_' + halo + '_snapshot' + str(snap) + '.pdf')

        plot_tempProf_and_coronaRelation(input_path, input_path_coronae, output_path, snap, include_lucchini_sim)


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
halo_ = 'lmc'
snapshots = [125, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
nHII_cutoff = -4.5
# ------------------------------------
snapshot_ = 127
include_lucchini_lmc = True
include_coronae = False
# ------------------------------------

if machine == 'geras':
    # (sim_run, halo, snaps, size, bins_per_kpc, bool_rho_filtering, rho_threshold
    package_data(simulation_run, halo_, snapshots, nHII_cutoff)

elif machine == 'dear-prudence':
    plotting(simulation_run, halo_, snapshot_, include_lucchini_lmc, include_coronae)
