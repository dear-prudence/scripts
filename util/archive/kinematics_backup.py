import numpy as np
import argparse


def orbits(run, auxiliary_halo, halo):
    from hestia import get_lookbackTimes, calc_distanceHalo
    from scipy.interpolate import interp1d

    snaps = [67, 127]
    base_path = '/halos/' + run + '/' + halo + '/kinematics/orbits/'

    lookback_times = np.array(get_lookbackTimes(run, snaps)[1])
    distances = calc_distanceHalo(run, snaps, auxiliary_halo, halo)

    kind_interpolator = 'cubic'  # or 'linear', 'quadratic', 'cubic'

    if halo == 'halo_08' and auxiliary_halo == 'smc':
        try:
            besla_orbits = np.loadtxt('/z/rschisholm' + base_path + 'orbits_besla12_model2.txt')
            pardy_orbits = np.loadtxt('/z/rschisholm' + base_path + 'orbits_pardy18_9to1.txt')
            lucchini_orbits = np.loadtxt('/z/rschisholm' + base_path + 'orbits_lucchini20.txt')
        except FileNotFoundError or OSError:
            print('Warning: No orbit files found to compare to! Is this intentional?')

        # Create interpolator -- hestia
        h_interp = interp1d(lookback_times, distances, kind=kind_interpolator)  # or 'linear', 'quadratic', 'cubic'
        # Generate a smooth time grid
        h_time_smooth = np.linspace(min(lookback_times), max(lookback_times), 500)
        h_distance_smooth = h_interp(h_time_smooth)

        # Create interpolator -- besla
        b_finalTime = 7  # in Gyr
        b_interp = interp1d(b_finalTime - besla_orbits[:, 0], besla_orbits[:, 1], kind=kind_interpolator)
        # Generate a smooth time grid
        b_time_smooth = np.linspace(min(b_finalTime - besla_orbits[:, 0]), max(b_finalTime - besla_orbits[:, 0]), 500)
        b_distance_smooth = b_interp(b_time_smooth)

        # Create interpolator -- pardy
        p_finalTime = 7  # in Gyr
        p_interp = interp1d(p_finalTime - pardy_orbits[:, 0], pardy_orbits[:, 1], kind=kind_interpolator,
                            fill_value='extrapolate')
        # Generate a smooth time grid
        p_time_smooth = np.linspace(min(p_finalTime - pardy_orbits[:, 0]), max(p_finalTime - pardy_orbits[:, 0]), 500)
        p_distance_smooth = p_interp(p_time_smooth)

        # Create interpolator -- lucchini
        l_finalTime = 5.7  # in Gyr
        l_interp = interp1d(l_finalTime - lucchini_orbits[:, 0], lucchini_orbits[:, 1], kind=kind_interpolator,
                            fill_value="extrapolate")
        # Generate a smooth time grid
        l_time_smooth = np.linspace(min(l_finalTime - lucchini_orbits[:, 0]), max(l_finalTime - lucchini_orbits[:, 0]),
                                    500)
        l_distance_smooth = l_interp(l_time_smooth)

        output_dict = {
            'hestia_times': h_time_smooth, 'hestia_distances': h_distance_smooth,
            'besla_times': b_time_smooth, 'besla_distances': b_distance_smooth,
            'pardy_times': p_time_smooth, 'pardy_distances': p_distance_smooth,
            'lucchini_times': l_time_smooth, 'lucchini_distances': l_distance_smooth
        }

    else:
        # Create interpolator -- hestia
        h_interp = interp1d(lookback_times, distances, kind=kind_interpolator)  # or 'linear', 'quadratic', 'cubic'
        # Generate a smooth time grid
        h_time_smooth = np.linspace(min(lookback_times), max(lookback_times), 500)
        h_distance_smooth = h_interp(h_time_smooth)
        output_dict = {'hestia_times': h_time_smooth, 'hestia_distances': h_distance_smooth}

    output_fileName = f'orbitalDistance_{run}_{halo}-{auxiliary_halo}.npz'

    np.savez(f'/z/rschisholm{base_path}{output_fileName}', **output_dict)

    return base_path + output_fileName


def bhPerturbation(run, halo, snaps, mbp=False, verbose=True):
    # making the plot of the offset of the central bh for a galaxy (as a function of time); for SDSS-V LVM project
    from hestia import get_redshift, get_lookbackTimes
    from hestia import get_centralBH

    base_path = (f'/halos/' + ('09_18' if run == '09_18_lastgigyear' else f'{run}')
                 + f'/{halo}/kinematics/bhPerturbation/')

    for snap in range(snaps[1], snaps[0], -1):
        redshift = float(get_redshift(run, snap))
        a = 1 / (1 + redshift)
        center = 'mbp'if mbp else 'com'
        verbose and print(f'\nen train de travailler au snapshot {snap}, z = {redshift} ...')
        try:
            bh, phi_min = get_centralBH(run, halo, snap, center=center, verbose=verbose)

            norm = np.linalg.norm(bh['Halo_Coordinates'])
            E_perMass = (bh['Potential'] / a) + (0.5 * np.linalg.norm(bh['Halo_Velocities']) ** 2)  # 2 * |v|^2 + \phi

            verbose and print(f'\tcentral bh properties:'
                              f'\n\t\t|v| = {float(np.linalg.norm(bh["Halo_Velocities"]))} km/s'
                              f'\n\t\tU_g = {float(bh["Potential"] / a)} (km/s)^2'
                              f'\n\t\tE_tot/m = {float(E_perMass)} (km/s)^2')

            try:  # if not the first instance
                redshifts = np.append(redshifts, redshift)
                bh_coords = np.vstack((bh_coords, bh['Halo_Coordinates']))
                bh_norms = np.append(bh_norms, norm)  # norms for 2-dim plot
                bh_potentials = np.append(bh_potentials, bh['Potential'] / a)  # value of potential at \vec{r}_bh
                bh_mechEnergy = np.append(bh_mechEnergy, E_perMass)  # total mechanical energy per unit mass
                phi_mins = np.append(phi_mins, phi_min)  # minimum value of total potential

            except NameError:  # if the first instance
                redshifts = redshift
                bh_coords = bh['Halo_Coordinates']  # in kpc
                bh_norms = norm  # in kpc
                bh_potentials = bh['Potential'] / a  # in (km/s)^2
                bh_mechEnergy = E_perMass  # in (km/s)^2
                phi_mins = phi_min  # in (km/s)^2

        except KeyError:
            print(f'Warning: there appears ot be no BHs in snapshot {snap}!')
        verbose and print(f'termine avec le snapshot {snap}.')

    _, lookback_times = get_lookbackTimes(run, snaps, redshifts=redshifts)

    print(f'\nbh_coords.shape = {bh_coords.shape}')
    print(f'bh_coords = {bh_coords}')

    output_file = f'bhPerturbation_{run}_{halo}_{center}.npz'
    np.savez_compressed(f'/z/rschisholm{base_path}{output_file}',
                        bh_coords=bh_coords, bh_norms=bh_norms,
                        bh_potential=bh_potentials, bh_mechEnergy=bh_mechEnergy, phi_mins=phi_mins,
                        redshifts=redshifts, lookback_times=lookback_times)

    return f'{base_path}{output_file}'


def main():
    # PARAMETERS TO CHANGE
    # ------------------------------------
    type_plot = 'orbits'
    run = '09_18'
    halo = 'halo_08'
    auxiliary_halo = 'smc'
    snaps = [67, 127]
    # ------------------------------------
    methodCenter = 'com'
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
                        help='reference halo to be processed')

    # Positional or optional argument for halo to be processed
    parser.add_argument('auxiliary_halo', nargs='?', default=auxiliary_halo,
                        help='subject halo to be processed')

    parser.add_argument("--start", type=int, default=snaps[0], help='starting snapshot')
    parser.add_argument("--end", type=int, default=(snaps[1]),
                        help='ending snapshot')

    parser.add_argument('--mbp', dest='mbp', action='store_true',
                        help='definition of center to be most bound particle (True) or center of mass (False); '
                             'all particle types')
    parser.set_defaults(mbp=False)

    args = parser.parse_args()

    print('--------------------------------------------------------------------------------\n'
          + 'Creating plot of *' + args.type_plot + '*...\n'
          + 'sim_run -- ' + args.run + '\n'
          + 'halo -- ' + args.halo + '\n'
          + (f'auxiliary halo -- {args.auxiliary_halo}\n' if args.type_plot == 'orbits' else '')
          + 'start, end snapshot (if applicable) -- ' + str(args.start) + ', ' + str(args.end) + '\n'
          + 'method of determining center -- ' + ('mbp' if args.mbp else 'com')
          + '\n--------------------------------------------------------------------------------')

    if args.type_plot == 'orbits':
        output_path = orbits(args.run, args.auxiliary_halo, args.halo)
    elif args.type_plot == 'bhSloshing':
        output_path = bhPerturbation(args.run, args.halo, [args.start, args.end], args.mbp)
    else:
        print('Invalid type plot!')
        exit(1)

    print('Finished writing *' + args.type_plot + '* data file,\n' + output_path)


def plotting():
    from scripts.util.plots import dispatch_plot
    # ------------------------------------------------
    type_plot = 'bhPerturbation'
    run = '09_18'
    halo = 'halo_15'
    smoothing = False  # apply scipy interpolation smoothing routine

    # orbits -----------------------------------------
    subject_halo = 'smc'

    # bh perturbation --------------------------------
    center = 'mbp'  # definition of "center"; center of mass 'com' or most bound particle 'mbp'
    projection = '1-dim'
    # ------------------------------------

    run_ = '09_18' if run == '09_18_lastgigyear' else run
    base_path = f'/Users/ursa/smorgasbord/kinematics/{run_}_{halo}/'

    if type_plot == 'orbits':
        dispatch_plot('kinematics', type_plot,
                      input_path=f'{base_path}orbits/orbitalDistance_{run}_{halo}-{subject_halo}.npz',
                      output_path=f'{base_path}orbits/orbitalDistance_{run}_{halo}-{subject_halo}.pdf')
    elif type_plot == 'bhPerturbation':
        dispatch_plot('kinematics', type_plot, projection=projection, smoothing=smoothing,
                      input_path=f'{base_path}bhPerturbation/'
                                 f'bhPerturbation_{run}_{halo}_{center}.npz',
                      output_path=f'{base_path}bhPerturbation/'
                                  f'bhPerturbation_{run}_{halo}_{center}_{projection}.png')


# ------------------------------------
machine = 'geras'
# ------------------------------------

if machine == 'geras':
    main()

elif machine == 'dear-prudence':
    plotting()
