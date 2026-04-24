import numpy as np
import argparse
from pathlib import Path
import os


def rotCurve(run, halo, snap):
    from archive.hestia import get_massProfile

    output_path = f'/halos/{run}/{halo}/kinematics/rotCurve/'
    output_fileName = f'{run}.{halo}.rotCurve.snap{snap}.npz'

    profile = get_massProfile(run, halo, snap, verbose=True).T
    m_dm = profile[1] - (profile[2] + profile[3])  # M_dm = M_tot - M_gas - M_star
    profile = np.vstack((profile, m_dm)).T

    output_dict = {'r': profile[:, 0], 'M_tot': profile[:, 1],
                   'M_dm': profile[:, 4], 'M_gas': profile[:, 2], 'M_star': profile[:, 3]}
    np.savez(f'/z/rschisholm{output_path}{output_fileName}', **output_dict)
    return f'{output_path}{output_fileName}'


def accretionHistory(run, halo, verbose=False):
    from archive.hestia import halo_dictionary
    from archive.hestia import retrieve_particles
    from archive.hestia.geometry import get_redshift, get_lookbackTimes, calc_distanceDisk
    h = 0.677

    output_path = f'/halos/{run}/{halo}/kinematics/history/'
    output_fileName = f'{run}.{halo}.accretionHistory.npz'

    halo_id_z0 = halo_dictionary(run, halo)

    if run != '09_18_lastgigyear':
        ahf_filePath = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc/'
                        f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        idx_i, idx_f = 67, 127
    else:
        ahf_filePath = f'/z/rschisholm/halos/{run}/{halo}/HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat'
        idx_i, idx_f = 118, 307

    ahfHalo = np.loadtxt(ahf_filePath)

    for snap in range(idx_f, idx_i, -1):
        redshift = get_redshift(run, snap)
        row = idx_f - snap

        bh = retrieve_particles(run, halo, snap, 'PartType5', verbose=verbose)
        if len(bh['ParticleIDs']) != 0:
            bh['Distances'] = calc_distanceDisk(bh)
            dist_mask = bh['Distances'] < 10  # kpc
            bh = {key: val[dist_mask] for key, val in bh.items()}
            verbose and print(f'\rretrieved central bh of {halo}, M ~ {(bh["Masses"].item()):.2e};')
        else:
            verbose and print('\rNo central blackhole at this snapshot !')
            bh['Masses'] = np.ones(1)  # arbitrarily low value

        mass = np.array([
            redshift,
            float(ahfHalo[row, 4] / h),  # M_total (M_vir)
            float(ahfHalo[row, 4] / h - ahfHalo[row, 45] / h - ahfHalo[row, 65] / h - bh['Masses']),  # M_dm
            float(ahfHalo[row, 45] / h),  # M_gas
            float(ahfHalo[row, 65] / h),  # M_star
            float(bh['Masses']),  # M_bh
            float(bh['BH_Mass']) * 1e10 / h,  # M_bh (for diagnostics)
            float(bh['BH_Density']) * h ** 2,  # averaged mass of surrounding gas, M_solar/ckpc^3
            float(bh['BH_Mdot']) * 10.22  # d/dt(M_bh) [M_solar/yr],
        ])

        verbose and print(f'\tmass_vector at z ~ {redshift} : {mass[1:]} M_solar')

        if snap == idx_f:
            massHistory = mass[np.newaxis, :]
        else:
            massHistory = np.vstack((massHistory, mass))

    output_dict = {'redshifts': massHistory[:, 0],
                   'lookback_times': get_lookbackTimes('', '', redshifts=massHistory[:, 0])[1],
                   'M_halo': massHistory[:, 1], 'M_dm': massHistory[:, 2], 'M_gas': massHistory[:, 3],
                   'M_star': massHistory[:, 4], 'M_bh': massHistory[:, 5],
                   'M_BHgas': massHistory[:, 7], 'M_dot': massHistory[:, 8]}

    try:
        os.mkdir(f'/z/rschisholm{output_path}')
        print('\toutput directory written')
    except FileExistsError:
        pass

    np.savez(f'/z/rschisholm{output_path}{output_fileName}', **output_dict)
    return f'{output_path}{output_fileName}'


def orbits(run, halo):
    from archive.hestia.geometry import get_lookbackTimes, calc_distanceHalo
    from scipy.interpolate import interp1d

    if run != '09_18_lastgigyear':
        snaps = [107, 127]
    else:
        snaps = [118, 307]
    output_path = f'/halos/{run}/{halo}/kinematics/orbits/'

    auxHalo_dict = {'halo_08': 'halo_1476',
                    'halo_38': 'halo_454',
                    'halo_41': 'halo_01'}

    _, lookback_times = get_lookbackTimes(run, range(snaps[1], snaps[0], -1))
    distances = calc_distanceHalo(run, snaps, auxHalo_dict[halo], halo)
    print(f'\t\tauxiliary halo : {auxHalo_dict[halo]}')

    kind_interpolator = 'cubic'  # or 'linear', 'quadratic', 'cubic'

    if halo == 'halo_08' and auxHalo_dict[halo] == 'smc':
        try:
            besla_orbits = np.loadtxt('/z/rschisholm' + output_path + 'orbits_besla12_model2.txt')
            pardy_orbits = np.loadtxt('/z/rschisholm' + output_path + 'orbits_pardy18_9to1.txt')
            lucchini_orbits = np.loadtxt('/z/rschisholm' + output_path + 'orbits_lucchini20.txt')
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
        print(f'len(lookback_times) : {len(lookback_times)}')
        print(f'len(distances) : {len(distances)}')
        h_interp = interp1d(lookback_times, distances, kind=kind_interpolator)  # or 'linear', 'quadratic', 'cubic'
        # Generate a smooth time grid
        h_time_smooth = np.linspace(min(lookback_times), max(lookback_times), 500)
        h_distance_smooth = h_interp(h_time_smooth)
        output_dict = {'hestia_times': h_time_smooth, 'hestia_distances': h_distance_smooth}

    output_base = '/z/rschisholm'
    output_name = f'orbitalDistance.{run}.{halo}-{auxHalo_dict[halo]}.npz'

    if not os.path.exists(output_base + output_path):
        os.makedirs(output_base + output_path)
        print('\toutput directory written')

    np.savez(f'/z/rschisholm{output_path}{output_name}', **output_dict)
    return output_path + output_name


# noinspection SpellCheckingInspection
def bhSloshing(run, halo, snaps, verbose=True):
    # making the plot of the offset of the central bh for a galaxy (as a function of time); for SDSS-V LVM project
    from archive.hestia.geometry import get_redshift, get_lookbackTimes
    from archive.hestia import get_halo_params
    from archive.hestia import get_softeningLength, retrieve_particles

    base_path = f'/halos/{run}/{halo}/kinematics/bhSloshing'

    for snap in range(snaps[1], snaps[0], -1):
        redshift = float(get_redshift(run, snap))
        a = 1 / (1 + redshift)
        h = 0.677
        verbose and print(f'\nen train de travailler au snapshot {snap}, z = {redshift} ...')

        bh = retrieve_particles(run, halo, snap, 'PartType5', verbose=verbose)
        verbose and print(f'\tretrieved central bh of {halo}, M ~ {(1e10 * bh["BH_Mass"].item() / h):.2e};')

        if len(bh['ParticleIDs']) == 0:
            print(f'Warning: there appears to be no BHs in snapshot {snap}!')
        else:
            norm = np.linalg.norm(bh['position'])
            epsilon = get_softeningLength(run, snap, 'PartType5')  # softening length in kpc

            halo_params = get_halo_params(run, halo, snap)
            R_vir, E_pot, c = (halo_params['R_vir'] / h,  # in kpc
                               halo_params['E_pot'] / halo_params['M_halo'] / h,  # in (km / s)^2
                               halo_params['cNFW'])
            print(f'E_pot : {E_pot}')
            # stars = retrieve_particles(run, halo, snap, 'PartType4')
            # phi_0 = stars['Potential'].min() / a

            # phi_bh = E_pot * R_vir / (c * norm) * np.log(1 + (c * norm / R_vir))
            phi_bh = bh['Potential'] * (1e-1 ** 2) / a - E_pot
            # unphysE_m = (bh['Potential'] / a) + (0.5 * np.linalg.norm(bh['velocity']) ** 2)  # |v|^2 / 2 + \phi
            # physE_m = unphysE_m - phi_min  # E_m - \phi_min, physical total mechanical energy, in (km/s)^2
            E_bh = phi_bh + (0.5 * np.linalg.norm(bh['velocity']) ** 2)  # |v|^2 / 2 + \phi
            vPhi = ((bh['position'][:, 0] * bh['velocity'][:, 1] - bh['velocity'][:, 0] * bh['position'][:, 1])
                     / np.sqrt(bh["position"][:, 0] ** 2 + bh["position"][:, 1] ** 2))

            verbose and print(f'\tcentral bh properties:'
                              f'\n\t\t|v| = {float(np.linalg.norm(bh["velocity"]))} km/s'
                              f'\n\t\t|phi_0| : {E_pot} (km/s)^2'
                              f'\n\t\tU_bh : {phi_bh} (km/s)^2'
                              f'\n\t\tE_bh/m = {float(E_bh)} (km/s)^2')

            if snap == snaps[1]:  # if the first instance
                redshifts = redshift
                bh_coords = bh['position']  # in kpc
                bh_norms = norm  # in kpc
                bh_pots = bh['Potential'] * (1e-1 ** 2) / a  # in (km/s)^2
                bh_Es = E_bh  # in (km/s)^2
                epsilons = epsilon
                bh_velocity = vPhi
                bh_L = bh['angularMomentum'][:, 2] / bh['Masses']

            else:  # if not the first instance
                redshifts = np.append(redshifts, redshift)
                bh_coords = np.vstack((bh_coords, bh['position']))
                bh_norms = np.append(bh_norms, norm)  # norms for 2-dim plot
                bh_pots = np.append(bh_pots, (bh['Potential'] * (1e-1 ** 2) / a) - E_pot)
                # value of potential at \vec{r}_bh
                bh_Es = np.append(bh_Es, E_bh)  # total mechanical energy per unit mass
                epsilons = np.append(epsilons, epsilon)
                bh_velocity = np.append(bh_velocity, vPhi)
                bh_L = np.append(bh_L, bh['angularMomentum'][:, 2] / bh['Masses'])

        verbose and print(f'termine avec le snapshot {snap}.')

    _, lookback_times = get_lookbackTimes(run, snaps, redshifts=redshifts)

    verbose and print(f'\nbh_norms = {bh_norms}')
    print(bh_L)
    output_file = f'bhSloshing.{run}.{halo}.npz'
    np.savez_compressed(f'/z/rschisholm{base_path}/{output_file}',
                        bh_coords=bh_coords, bh_norms=bh_norms,
                        bh_pots=bh_pots, bh_energies=bh_Es, epsilon=epsilons, bh_velocity=bh_velocity,
                        bh_L=bh_L,
                        redshifts=redshifts, lookback_times=lookback_times)

    return f'{base_path}/{output_file}'


def mbpSloshing(run, halo, snaps, verbose=True):
    # making the plot of the offset of the central bh for a galaxy (as a function of time); for SDSS-V LVM project
    from archive.hestia.geometry import get_redshift, get_lookbackTimes
    from archive.hestia import retrieve_particles
    from archive.hestia import get_mbp

    base_path = f'/halos/{run}/{halo}/kinematics/mbpSloshing'

    mbp_t0 = 37
    mbp_id = get_mbp('09_18', halo, [mbp_t0, 127], verbose=verbose)

    for snap in range(snaps[1], snaps[0], -1):
        redshift = float(get_redshift(run, snap))
        a = 1 / (1 + redshift)
        h = 0.677
        verbose and print(f'\nen train de travailler au snapshot {snap}, z = {redshift} ...')

        stars = retrieve_particles(run, halo, snap, 'PartType4', verbose=verbose)
        mbp_mask = stars['ParticleIDs'] == mbp_id
        mbp = {key: val[mbp_mask] for key, val in stars.items()}
        verbose and print(f'\tretrieved mbp of {halo}, id ~ {int(mbp["ParticleIDs"])};')

        norm = np.linalg.norm(mbp['position'])
        print(f'mbp (x,y,z) : {mbp["position"][:, 0]}, {mbp["position"][:, 1]}, {mbp["position"][:, 2]}')

        if snap == snaps[1]:  # if the first instance
            redshifts = redshift
            mbp_coords = mbp['position']  # in kpc
            mbp_vels = mbp['velocity']  # km/s
            mbp_norms = norm  # in kpc

        else:  # if not the first instance
            redshifts = np.append(redshifts, redshift)
            mbp_coords = np.vstack((mbp_coords, mbp['position']))
            mbp_vels = np.vstack((mbp_vels, mbp['velocity']))
            mbp_norms = np.append(mbp_norms, norm)  # norms for 2-dim plot

        verbose and print(f'termine avec le snapshot {snap}.')

    _, lookback_times = get_lookbackTimes(run, snaps, redshifts=redshifts)

    verbose and print(f'\nbh_norms : {mbp_norms}')
    output_file = f'mbpSloshing.{run}.{halo}.mbp{mbp_t0}.npz'
    np.savez_compressed(f'/z/rschisholm{base_path}/{output_file}',
                        mbp_coords=mbp_coords, mbp_vels=mbp_vels, mbp_norms=mbp_norms,
                        redshifts=redshifts, lookback_times=lookback_times)

    return f'{base_path}/{output_file}'


def bhAccretion(run, halo, verbose=False):
    from archive.hestia import halo_dictionary
    from archive.hestia import retrieve_particles
    from archive.hestia.geometry import get_redshift, get_lookbackTimes, calc_distanceDisk
    h = 0.677

    output_path = f'/halos/{run}/{halo}/kinematics/bhAccretion/'
    output_fileName = f'{run}.{halo}.bhAccretion.npz'

    halo_id_z0 = halo_dictionary(run, halo)

    if run != '09_18_lastgigyear':
        ahf_filePath = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc/'
                        f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        idx_i, idx_f = 67, 127
    else:
        ahf_filePath = f'/z/rschisholm/halos/{run}/{halo}/HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat'
        idx_i, idx_f = 118, 307

    ahfHalo = np.loadtxt(ahf_filePath)

    for snap in range(idx_f, idx_i, -1):
        redshift = get_redshift(run, snap)
        row = idx_f - snap

        bh = retrieve_particles(run, halo, snap, 'PartType5', verbose=verbose)
        if len(bh['ParticleIDs']) != 0:
            bh['Distances'] = calc_distanceDisk(bh)
            dist_mask = bh['Distances'] < 10  # kpc
            bh = {key: val[dist_mask] for key, val in bh.items()}
            verbose and print(f'\rretrieved central bh of {halo}, M ~ {(bh["Masses"].item()):.2e};')
        else:
            verbose and print('\rNo central blackhole at this snapshot !')
            bh['Masses'] = np.ones(1)  # arbitrarily low value

        accretion = np.array([
            redshift,
            float(ahfHalo[row, 4] / h),  # M_total (M_vir)
            float(bh['Masses']),  # M_bh
            float(bh['BH_Mass']) * 1e10 / h,  # M_bh (for diagnostics)
            float(np.linalg.norm(bh['position'])),  # bh displacement in kpc
            float(bh['BH_Density']) * h ** 2,  # averaged mass of surrounding gas, M_solar/ckpc^3
            float(bh['BH_Mdot']) * 10.22  # d/dt(M_bh) [M_solar/yr],
        ])

        verbose and print(f'\tmass_vector at z ~ {redshift} : {accretion[1:]} M_solar')

        if snap == idx_f:
            history = accretion[np.newaxis, :]
        else:
            history = np.vstack((history, accretion))

    output_dict = {'redshifts': history[:, 0],
                   'lookback_times': get_lookbackTimes('', '', redshifts=history[:, 0])[1],
                   'M_halo': history[:, 1], 'M_res': history[:, 2], 'M_bh': history[:, 3],
                   'x_bh': history[:, 4], 'M_BHgas': history[:, 5], 'M_dot': history[:, 6]}

    try:
        os.mkdir(f'/z/rschisholm{output_path}')
        print('\toutput directory written')
    except FileExistsError:
        pass

    np.savez(f'/z/rschisholm{output_path}{output_fileName}', **output_dict)
    return f'{output_path}{output_fileName}'


def main(cluster):
    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    parser.add_argument('plot_type', help='category of plot to be constructed')
    parser.add_argument('run', help='simulation run of interest')
    parser.add_argument('halo', help='reference halo to be processed, also determines satellite')
    # optional arguments regarding the snapshot range to be sampled (or singular snapshot if required)
    parser.add_argument("--snap", type=int, default=127, help='snapshot to be sampled')
    parser.add_argument("--start", type=int, default=67, help='starting snapshot')
    parser.add_argument("--end", type=int, default=127, help='ending snapshot')

    args = parser.parse_args()

    print('--------------------------------------------------------------------------------\n'
          + f'en train de creer du graphique * {args.plot_type} * ...\n'
          + f'sim_run -- {args.run}\n'
          + f'halo -- {args.halo}\n'
          + (f'snapshot -- {args.snap}\n' if args.plot_type == 'rotCurve' else '')
          # + (f'auxiliary halo -- {args.auxiliary_halo}\n' if args.type_plot == 'orbits' else '')
          + (f'start, end snapshot -- {args.start}, {args.end}\n' if args.plot_type != 'rotCurve' else '')
          + '--------------------------------------------------------------------------------')

    if args.plot_type == 'history':
        output_path = accretionHistory(args.run, args.halo, verbose=True)
    elif args.plot_type == 'rotCurve':
        output_path = rotCurve(args.run, args.halo, args.snap)
    elif args.plot_type == 'orbits':
        output_path = orbits(args.run, args.halo)
    elif args.plot_type == 'bhSloshing':
        output_path = bhSloshing(args.run, args.halo, [args.start, args.end])
    elif args.plot_type == 'mbpSloshing':
        output_path = mbpSloshing(args.run, args.halo, [args.start, args.end], verbose=True)
    elif args.plot_type == 'bhAccretion':
        output_path = bhAccretion(args.run, args.halo, verbose=True)
    else:
        print(f'Error: {args.type_plot} is an invalid plot type!')
        exit(1)

    print(f'le dossier de sortie a ete ecrit, trouvez le chemin d\'access ci-dessous:\n\n{output_path}\n')


def plotting():
    from util.plots import dispatch_plot
    # ------------------------------------------------
    type_plot = 'orbits'
    # ------------------------------------------------
    run = '09_18_lastgigyear'
    halo = 'halo_08'
    smoothing = True  # apply scipy interpolation smoothing routine
    chisholm2026_plot = False

    # rotation curves --------------------------------
    snapshot = 125

    # orbits -----------------------------------------
    subject_halo = 'halo_1476'

    # bh perturbation --------------------------------
    projection = '2-dim'
    parameter = 'norm'  # 'norm' or 'energy'

    # mbp sloshing --------------------------------
    mbp_t0 = 125

    # ------------------------------------

    if chisholm2026_plot:
        from util.chisholm2026 import figure1b
        figure1b()
        exit(0)

    home = Path.home()
    current_file = Path(__file__)
    project_root = current_file.parent.parent  # adjust as needed
    basePath = home / project_root / 'halos' / run / halo / 'kinematics' / type_plot

    if type_plot == 'history':
        dispatch_plot('kinematics', type_plot,
                      input_path=f'{basePath}/{run}.{halo}.accretionHistory.npz',
                      output_path=f'{basePath}/{run}.{halo}.accretionHistory.pdf')
    elif type_plot == 'rotCurve':
        dispatch_plot('kinematics', type_plot,
                      input_path=f'{basePath}/{run}.{halo}.rotCurve.snap{snapshot}.npz',
                      output_path=f'{basePath}/{run}.{halo}.rotCurve.snap{snapshot}.pdf')
    elif type_plot == 'orbits':
        dispatch_plot('kinematics', type_plot,
                      input_path=f'{basePath}/orbitalDistance.{run}.{halo}-{subject_halo}.npz',
                      output_path=f'{basePath}/orbitalDistance.{run}.{halo}-{subject_halo}.pdf')
    elif type_plot == 'bhSloshing':
        dispatch_plot('kinematics', type_plot, projection=projection, smoothing=smoothing, parameter=parameter,
                      input_path=f'{basePath}/bhSloshing.{run}.{halo}.npz',
                      output_path=f'{basePath}/bhSloshing.{run}.{halo}.{projection}.{parameter}.png')
    elif type_plot == 'mbpSloshing':
        dispatch_plot('kinematics', type_plot, projection=projection, smoothing=smoothing, parameter=parameter,
                      input_path=f'{basePath}/mbpSloshing.{run}.{halo}.mbp{mbp_t0}.npz',
                      output_path=f'{basePath}/mbpSloshing.{run}.{halo}.mbp{mbp_t0}.pdf')
    elif type_plot == 'bhAccretion':
        dispatch_plot('kinematics', type_plot,
                      input_path=f'{basePath}/{run}.{halo}.bhAccretion.npz',
                      output_path=f'{basePath}/{run}.{halo}.bhAccretion.pdf')


if __name__ == "__main__":
    import socket

    machine = socket.gethostname()

    if 'aip.de' in machine:  # aip cluster
        main('erebos')
    elif machine == 'scylla':  # scylla cluster
        main('scylla')
    else:  # util machine
        plotting()
