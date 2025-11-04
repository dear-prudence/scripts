import numpy as np
import argparse
from hestia.geometry import get_lookbackTimes
from hestia.particles import retrieve_particles
from hestia.halos import get_halo_params


def add_gasHeaders(particles):
    from hestia.gas import calc_temperature, calc_numberDensity
    from hestia.geometry import calc_distanceDisk

    particles['Temperature'] = calc_temperature(particles['InternalEnergy'], particles['ElectronAbundance'],
                                                particles['GFM_Metals'][:, 0])
    particles['n_H'] = calc_numberDensity(particles['Density'] * particles['GFM_Metals'][:, 0])
    particles['n_H0'] = particles['n_H'] * particles['NeutralHydrogenAbundance']
    particles['n_H1'] = particles['n_H'] * (1 - particles['NeutralHydrogenAbundance'])

    particles['Distances'] = calc_distanceDisk(particles)

    return particles


def phaseDiagram(run, halo, snaps, H_phase):
    h = 0.677

    base_path = '/halos/' + run + '/' + halo + '/gaseous_components/phaseDiagram/'
    n_bins = 400

    H_dict = {'H': 'n_H', 'H0': 'n_H0', 'HI': 'n_H0', 'H1': 'n_H1', 'HII': 'n_H1'}

    redshifts, lookback_times = get_lookbackTimes(run, snaps)
    H = np.zeros((n_bins, n_bins, snaps[1] - snaps[0]))

    for snap_i in range(snaps[1], snaps[0], -1):

        halo_dict = get_halo_params(run, halo, snap_i)
        virial_radius = halo_dict['R_vir'] / h

        particles = retrieve_particles(run, halo, snap_i, redshifts[127 - snap_i], part_type='PartType0')
        particles = add_gasHeaders(particles)

        # -----------------------------------
        log_temperature_range = [3.5, 6.5]
        log_nH_range = [-8, -2]
        radius_range = [0.1 * virial_radius, 1.2 * virial_radius]
        bool_filterIGM = True
        # -----------------------------------

        if bool_filterIGM:
            halo_mask = np.where((particles['Distances'] > radius_range[0]) &
                                 (particles['Distances'] <= radius_range[1]))
            processed_particles = {name: None for name in particles.keys()}
            for key in particles.keys():
                processed_particles[key] = particles[key][halo_mask]
        else:
            processed_particles = particles

        if H_dict[H_phase] == 'n_H':
            weights = processed_particles['Masses'] * processed_particles['GFM_Metals'][:, 0]
        elif H_dict[H_phase] == 'n_H0':
            weights = (processed_particles['Masses'] * processed_particles['GFM_Metals'][:, 0]
                       * processed_particles['NeutralHydrogenAbundance'])
        elif H_dict[H_phase] == 'n_H1':
            weights = (processed_particles['Masses'] * processed_particles['GFM_Metals'][:, 0]
                       * (1 - processed_particles['NeutralHydrogenAbundance']))
        else:
            exit(1)

        h_i, x_e, y_e = np.histogram2d(np.log10(processed_particles['n_H']),
                                       np.log10(processed_particles['Temperature']),
                                       bins=n_bins, range=np.array([log_nH_range, log_temperature_range]),
                                       weights=weights, density=True)

        H[:, :, 127 - snap_i] = h_i

    data_to_save = {'data': H, 'x_e': x_e, 'y_e': y_e}
    output_file = f'phaseDiagram_{H_phase}.npz'
    np.savez_compressed(f'/z/rschisholm{base_path}{output_file}', **data_to_save)

    return base_path + output_file


def densityProfile(run, halo, snaps, H_phase):
    h = 0.677

    base_path = '/halos/' + run + '/' + halo + '/gaseous_components/densityProfile/'
    n_bins = 400

    H_dict = {'H': 'n_H', 'H0': 'n_H0', 'HI': 'n_H0', 'H1': 'n_H1', 'HII': 'n_H1'}

    redshifts, lookback_times = get_lookbackTimes(run, snaps)
    H = np.zeros((n_bins, n_bins, snaps[1] - snaps[0]))

    for snap_i in range(snaps[1], snaps[0], -1):

        halo_dict = get_halo_params(run, halo, snap_i)
        virial_radius = halo_dict['R_vir'] / h

        particles = retrieve_particles(run, halo, snap_i, redshifts[127 - snap_i], part_type='PartType0')
        particles = add_gasHeaders(particles)

        # -----------------------------------
        radius_range = [0, 1.2 * virial_radius]
        log_nH_range = [-8, -2]
        bool_filterIGM = True
        # -----------------------------------

        if bool_filterIGM:
            halo_mask = np.where(particles['Distances'] <= radius_range[1])
            processed_particles = {name: None for name in particles.keys()}
            for key in particles.keys():
                processed_particles[key] = particles[key][halo_mask]
        else:
            processed_particles = particles

        if H_dict[H_phase] == 'n_H':
            weights = processed_particles['Masses'] * processed_particles['GFM_Metals'][:, 0]
        elif H_dict[H_phase] == 'n_H0':
            weights = (processed_particles['Masses'] * processed_particles['GFM_Metals'][:, 0]
                       * processed_particles['NeutralHydrogenAbundance'])
        elif H_dict[H_phase] == 'n_H1':
            weights = (processed_particles['Masses'] * processed_particles['GFM_Metals'][:, 0]
                       * (1 - processed_particles['NeutralHydrogenAbundance']))
        else:
            exit(1)

        h_i, x_e, y_e = np.histogram2d(np.log10(processed_particles['Distances']),
                                       np.log10(processed_particles['n_H']),
                                       bins=n_bins, range=np.array([radius_range, log_nH_range]),
                                       weights=weights, density=True)

        H[:, :, 127 - snap_i] = h_i

    data_to_save = {'data': H, 'x_e': x_e, 'y_e': y_e}
    output_file = f'phaseDiagram_{H_phase}.npz'
    np.savez_compressed(f'/z/rschisholm{base_path}{output_file}', **data_to_save)

    return base_path + output_file


def temperatureProfile(run, halo, snaps, verbose=True):
    from hestia.gas import calc_temperatureProfile
    h = 0.677

    base_path = f'/halos/{run}/{halo}/gaseous_components/temperatureProfile/'
    n_bins = 400

    H_dict = {'H': 'n_H', 'H0': 'n_H0', 'HI': 'n_H0', 'H1': 'n_H1', 'HII': 'n_H1'}

    redshifts, lookback_times = get_lookbackTimes(run, snaps)
    H = np.zeros((n_bins, n_bins, snaps[1] - snaps[0]))
    coronaProfile = np.zeros((int(n_bins / 4), snaps[1] - snaps[0]))
    R_vir = np.zeros(snaps[1] - snaps[0])
    salemT, salemR = np.zeros((int(n_bins / 2), snaps[1] - snaps[0])), np.zeros((int(n_bins / 2), snaps[1] - snaps[0]))

    for snap_i in range(snaps[1], snaps[0], -1):
        verbose and print(f'\nen train de travailler le snapshot {snap_i}...')
        halo_dict = get_halo_params(run, halo, snap_i)
        virial_radius = halo_dict['R_vir'] / h

        particles = retrieve_particles(run, halo, snap_i, part_type='PartType0', verbose=verbose)
        particles = add_gasHeaders(particles)

        # -----------------------------------
        log_temperature_range = [4, 7]
        radius_range = [0, 1.2 * virial_radius]
        mass_enclosed_radius = 1.2 * virial_radius
        log_nH_cutoff = -3
        log_temperature_cutoff = 5
        # -----------------------------------

        cgm_mask = np.where((np.log10(particles['n_H']) < log_nH_cutoff) &
                            (particles['Distances'] < mass_enclosed_radius))
        cgm_particles = {name: None for name in particles.keys()}
        for key in particles.keys():
            if isinstance(particles[key], np.ndarray):
                cgm_particles[key] = particles[key][cgm_mask]
            else:
                cgm_particles[key] = particles[key]
        verbose and print(f'\t\tnumber of cgm-associated cells : {len(cgm_particles["ParticleIDs"])}')

        corona_mask = np.where(np.log10(cgm_particles['Temperature']) > log_temperature_cutoff)
        corona_particles = {name: None for name in cgm_particles.keys()}
        for key in cgm_particles.keys():
            if isinstance(cgm_particles[key], np.ndarray):
                corona_particles[key] = cgm_particles[key][corona_mask]
            else:
                corona_particles[key] = cgm_particles[key]
        verbose and print(f'\t\tnumber of corona-associated cells : {len(corona_particles["ParticleIDs"])}')

        cgm_H0_mass = np.sum(cgm_particles['Masses'] * cgm_particles['GFM_Metals'][:, 0]
                             * (cgm_particles['NeutralHydrogenAbundance']))
        cgm_H1_mass = np.sum(cgm_particles['Masses'] * cgm_particles['GFM_Metals'][:, 0]
                             * (1 - cgm_particles['NeutralHydrogenAbundance']))

        corona_H0_mass = np.sum(corona_particles['Masses'] * corona_particles['GFM_Metals'][:, 0]
                                * (corona_particles['NeutralHydrogenAbundance']))
        corona_H1_mass = np.sum(corona_particles['Masses'] * corona_particles['GFM_Metals'][:, 0]
                                * (1 - corona_particles['NeutralHydrogenAbundance']))

        corona_temp = np.average(corona_particles['Temperature'],
                                 weights=corona_particles['Masses'] * corona_particles['GFM_Metals'][:, 0]
                                         * (1 - corona_particles['NeutralHydrogenAbundance']))
        verbose and print(f'\t\tM_H0^cgm : {cgm_H0_mass:.2e} M_solar,'
                          f'\n\t\tM_H1^cgm : {cgm_H1_mass:.2e}, M_solar,'
                          f'\n\t\tM_H0^corona : {corona_H0_mass:.2e} M_solar,'
                          f'\n\t\tM_H1^corona : {corona_H1_mass:.2e} M_solar,'
                          f'\n\t\tT^corona : {corona_temp:.3e} K')

        # radial binning is per 0.5 kpc, temperature binning is half number of radial bins
        h_i, x_e, y_e = np.histogram2d(particles['Distances'], np.log10(particles['Temperature']),
                                       range=np.array([[0, 200], log_temperature_range]),
                                       bins=n_bins,
                                       weights=particles['Masses'] * particles['GFM_Metals'][:, 0]
                                               * (1 - particles['NeutralHydrogenAbundance']), normed=True)

        h_i_corona, x_e, y_e = np.histogram2d(corona_particles['Distances'], np.log10(corona_particles['Temperature']),
                                              range=np.array([[0, 200], log_temperature_range]),
                                              bins=int(n_bins / 4),
                                              weights=corona_particles['Masses'] * corona_particles['GFM_Metals'][:, 0]
                                                      * (1 - corona_particles['NeutralHydrogenAbundance']), normed=True)

        # calculating the column averages line
        verbose and print(f'\t\tcalculating corona temperature profile')
        column_averages = np.zeros(h_i_corona.shape[0])
        for i in range(h_i_corona.shape[0]):
            try:
                column_averages[i] = np.average(y_e[:-1] + abs(y_e[0] - y_e[1]) / 2, weights=h_i_corona[i])
            except ZeroDivisionError:
                column_averages[i] = np.NaN
        coronaProfile[:, 127 - snap_i] = column_averages

        # heatmap
        H[:, :, 127 - snap_i] = h_i
        R_vir[127 - snap_i] = virial_radius
        # equlibirum temperature profile
        verbose and print('\t\tcalling calc_temepratureProfile()')
        equilibrium_T, equilibrium_r = calc_temperatureProfile(run, halo, 127)
        salemT[:, 127 - snap_i], salemR[:, 127 - snap_i] = equilibrium_T, equilibrium_r

        verbose and print(f'termine avec le snapshot {snap_i}')

    data_to_save = {'heatmap': H, 'coronaProfile': coronaProfile, 'x_e': x_e, 'y_e': y_e,
                    'R_vir': R_vir,
                    'salemT': salemT, 'salemR': salemR}
    output_file = f'{run}.{halo}.temperatureProfile.npz'
    np.savez_compressed(f'/z/rschisholm{base_path}{output_file}', **data_to_save)

    return base_path + output_file


def corona_massFunction(snap, verbose=True):
    from hestia.geometry import get_redshift
    from hestia.gas import calc_coronaProperties
    # T_corona as a function of halo_mass;
    # i. calculate T_corona from extracted particles
    # ii. calculate t_dyn = R_vir / V_vir as the crossing time
    # iii. compare t_dyn with t_cool which is given as a particle header
    # iv. color code halos based on t_dyn ~ t_cool and plot alongside line of virial temperature
    h = 0.677
    how_many_halos = 10

    base_path = '/coronaMassFunction/'

    redshift = get_redshift('09_18', snap)
    sims = ['09_18', '17_11', '37_11']

    halo_props = {
        'sim': np.array([]),
        'halo_id': np.array([]),
        'bool_satellite': np.array([]),
        'M_halo': np.array([]),
        'R_vir': np.array([]),
        'M_HI': np.array([]),
        'M_HII': np.array([]),
        'T_avg': np.array([]),
        'mean_nH': np.array([]),
        'mean_T': np.array([]),
        'sigma_nH': np.array([]),
        'sigma_T': np.array([])
    }

    for sim in sims:
        verbose and print(f'\nen train de travailler le simulation {sim}...')
        halos_file = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/'
                      + sim + '/AHF_output_2x2.5Mpc/HESTIA_100Mpc_8192_' + sim + '.127.z0.000.AHF_halos')
        halos_data = np.loadtxt(halos_file)

        halo_ids = halos_data[:, 0].astype(int)  # gets the ids of the highest mass halos
        halo_hosts = halos_data[:, 1].astype(int)
        halo_masses = halos_data[:, 3] / h  # gets the masses of the highest-mass halos
        halo_radii = halos_data[:, 11] / h  # gets the virial radii of the highest-mass halos

        i, j = 0, 0  # starting index for i halos with dat file, and j halos with rows in AHF_halos output
        while i < how_many_halos:
            halo_id = '127000000000' + '%03d' % (j + 1)
            try:
                # this line is a check to see if halo_j has a .dat file
                np.loadtxt('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim
                           + '/AHF_output_2x2.5Mpc/HESTIA_100Mpc_8192_' + sim + '.127_halo_' + halo_id + '.dat')
                # if halo is above minimum mass, extract R_vir
                # using cutoffs from temperature profiles script, calculate mass and temperature of corona
                verbose and print(f'\n\tcalling calc_coronaProperties() for {sim}/halo_{halo_id[-2:]}')
                coronaProperties = calc_coronaProperties(sim, halo_id, snap, verbose=verbose)
                for key in coronaProperties.keys():
                    halo_props[key] = np.append(halo_props[key], coronaProperties[key])

                bool_satellite = 0 if halo_hosts[j] == 0 else 1
                verbose and print('\tEst-ce que ce halo est un satellite?  '
                                  + ('Oui!' if bool_satellite == 1 else 'Non!'))

                # saves data to be stored as a new row with column formatting;
                # halo_id (without snapshot prefix), M_vir, R_vir, M_corona, T_corona, n_H0
                halo_props['sim'] = np.append(halo_props['sim'], sim)
                halo_props['halo_id'] = np.append(halo_props['halo_id'], halo_id[-3:])
                halo_props['bool_satellite'] = np.append(halo_props['bool_satellite'], bool_satellite)
                halo_props['M_halo'] = np.append(halo_props['M_halo'], np.log10(halo_masses[j]))
                halo_props['R_vir'] = np.append(halo_props['R_vir'], halo_radii[j])

                i += 1
                j += 1

                if i == how_many_halos:
                    break
                else:
                    pass

            except IOError:
                j += 1

        verbose and print(f'termine avec le simulation {sim}...')

    halo_props['redshift'] = redshift
    # for key in halo_props.keys():
    #     print(f'{key}: {halo_props[key]}\n')

    output_file = f'coronaMassFunction.snap{snap}.npz'
    np.savez_compressed(f'/z/rschisholm{base_path}{output_file}', **halo_props)

    return f'{base_path}{output_file}'


def main():
    # PARAMETERS TO CHANGE
    # ------------------------------------
    type_plot = 'temperatureProfile'
    run = '09_18'
    halo = 'halo_08'
    snaps = [67, 127]
    # ------------------------------------

    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    # Positional or optional argument for particle type to be processed
    parser.add_argument('type_plot', nargs='?', default=type_plot,
                        help='type of plot to be constructed')

    # Positional or optional argument for simulation run
    parser.add_argument('run', nargs='?', default=run,
                        help='simulation run')

    # Positional or optional argument for halo to be processed
    parser.add_argument('halo', nargs='?', default=halo,
                        help='halo to be processed')

    # Positional or optional argument for halo to be processed
    parser.add_argument('--snap', type=int, default=127, help='snapshot for corona mass function')

    # Positional or optional argument for halo to be processed
    parser.add_argument('--phase', type=str, default='H', help='which phase of Hydrogen?')

    parser.add_argument("--start", type=int, default=snaps[0], help='starting snapshot')
    parser.add_argument("--end", type=int, default=snaps[1], help='ending snapshot')

    args = parser.parse_args()

    if args.type_plot == 'coronaMassFunction':
        print('---------------------------------------------------\n'
              + 'en train de creer du graphique *' + args.type_plot + '*...\n'
              + 'snapshot -- ' + str(args.snap) + '\n'
              + '---------------------------------------------------')
    else:
        print('---------------------------------------------------\n'
              + 'en train de creer du graphique *' + args.type_plot + '*...\n'
              + 'sim run -- ' + args.run + '\n'
              + 'halo -- ' + args.halo + '\n'
              + 'start, end snapshot -- ' + str(args.start) + ', ' + str(args.end) + '\n'
              + 'phase of Hydrogen -- ' + args.phase + '\n'
              + '---------------------------------------------------')

    if args.type_plot == 'phaseDiagram':
        output_path = phaseDiagram(args.run, args.halo, [args.start, args.end], args.phase)
    elif args.type_plot == 'temperatureProfile':
        output_path = temperatureProfile(args.run, args.halo, [args.start, args.end])
    elif args.type_plot == 'coronaMassFunction':
        output_path = corona_massFunction(args.snap)
    else:
        print('Invalid type plot!')
        exit(1)

    print(f'le dossier de sortie a ete ecrit, trouvez le chemin d\'access ci-dessous:\n\n{output_path}\n')


def plotting():
    from local.plots import dispatch_plot
    from local.chisholm2025 import gasPlot
    # ------------------------------------
    type_plot = 'chisholm2025_gasPlot'
    # ------------------------------------
    run = '09_18'
    halo = 'halo_08'
    H_phase = 'H1'
    snap = 127
    # ------------------------------------

    base_path = f'//halos/{run}/{halo}/gaseous_components'

    if type_plot == 'phaseDiagram':
        dispatch_plot('gaseousComponents', type_plot,
                      input_path=base_path + f'/phaseDiagram/phaseDiagram_{H_phase}.npz',
                      output_path=base_path + f'/phaseDiagram/phaseDiagram_{H_phase}.png',
                      H_phase=H_phase, snapshot=snap)
    elif type_plot == 'temperatureProfile':
        dispatch_plot('gaseousComponents', type_plot,
                      input_path=base_path + f'/temperatureProfile/{run}.{halo}.temperatureProfile.npz',
                      output_path=base_path + f'/temperatureProfile/{run}.{halo}.temperatureProfile.snap{snap}.png',
                      H_phase=H_phase, snapshot=snap)
    elif type_plot == 'coronaMassFunction':
        dispatch_plot('gaseousComponents', type_plot,
                      input_path=f'//coronaMassFunction/coronaMassFunction.snap{snap}.npz',
                      output_path=f'//coronaMassFunction/coronaMassFunction.snap{snap}.png')
    elif type_plot == 'chisholm2025_gasPlot':
        gasPlot()


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------

if machine == 'geras':
    main()

elif machine == 'dear-prudence':
    plotting()
