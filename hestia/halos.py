from __future__ import division
import numpy as np
from .particles import append_particles


def halo_dictionary(sim_run, halo_name):
    """
    :param sim_run: simulation run
    :param halo_name: halo shorthand
    :return: full halo id (i.e. <snapshot>000...000<halo#>)
    """
    if len(halo_name) == 15:
        print('Warning-- "halo_name" is fifteen-digit halo_id; is this intentional?')
        return halo_name
    elif 'halo' in halo_name and len(halo_name) == 7 and sim_run != '09_18_lastgigyear':
        return '1270000000000' + halo_name[-2:]
    elif 'halo' in halo_name and len(halo_name) == 7 and sim_run == '09_18_lastgigyear':
        return '3070000000000' + halo_name[-2:]
    else:
        if sim_run == '09_18':
            halo_dict = {'m31': '127000000000002', 'halo_02': '127000000000002',
                         'mw': '127000000000003', 'lmc': '127000000000008', 'stream': '127000000000008',
                         'halo_130': '127000000000130', 'halo_454': '127000000000454',
                         'smc': '127000000001384', 'halo_1384': '127000000001384'}
        elif sim_run == '17_11':
            halo_dict = {}
        elif sim_run == '37_11':
            halo_dict = {}
        elif sim_run == '09_18_lastgigyear':
            halo_dict = {'m31': '307000000000002', 'mw': '307000000000003',
                         'lmc': '307000000000008', 'stream': '307000000000008', 'smc': '307000000001476',
                         'halo_454': '307000000000540'}
        else:
            print('Error: routine is a WIP and cannot handle other simulation runs!')
            exit(1)
        return halo_dict[halo_name]


def get_ahfProfilesRange(desired_halo_id, data):
    """ Scans through the AHF_profiles file to retrieve the range of rows corresponding to a given halo of interest
    :param desired_halo_id: id of halo of interest
    :param data: output from AHF_profiles file
    :return: intial and final rows corresponding to the halo of interest
    """
    row, halo_idx, old_r = 0, 1, 0
    while halo_idx < (desired_halo_id + 1):
        r = data[row][0]
        diff = r - old_r
        if diff < -1:
            print('halo_idx = ' + str(halo_idx))
            halo_idx += 1
            print('i += 1: row ' + str(row))
        row += 1
        old_r = r
        if row > 5000:
            raise RuntimeError('Too many rows being iterated!')
        if halo_idx == desired_halo_id - 1:
            row_i = row
    row_f = row - 1
    return row_i, row_f


def get_halo_params(run, halo, snap, full_halo_id=False):
    """
    :param run: simulation run
    :param halo: halo id of interest
    :param snap: snapshot
    :param full_halo_id: True if halo id does not need to be passed through halo_dictionary
    :return: dictionary containing a variety of halo properties in h_units
            {'halo_id_zi': halo id at snapshot of interest,
            'M_halo': halo mass [M_solar/h],
            'halo_pos': co-moving halo position [kpc/h],
            'halo_vel': halo peculiar velocity [km/s],
            'halo_l': gas angular momentum unit vector (should change to stars for more consistent alignment),
            'R_vir': virial radius [kpc/h],
            'M_gas': gas mass [M_solar/h],
            'M_star': stellar mass [M_solar/h],
            'cNFW': concentration parameter for NFW dark matter profile}
    """
    halo_id_z0 = halo_dictionary(run, halo) if full_halo_id is False else halo

    if run != '09_18_lastgigyear':
        filename = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc/'
                    f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        row = 127 - int(snap)
    else:
        filename = (f'/z/rschisholm/halos/09_18_lastgigyear/{halo}/'
                    f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        row = 307 - int(snap)

    halo_data = np.loadtxt(filename)
    # two-character string identifying the lmc halo (in snap 127, it is '10')
    # halo_id = str(int(halo_data[row, 1]))[-2:]
    halo_id_zi = str(int(halo_data[row, 1]))
    halo_mass = halo_data[row, 4]  # in M_solar/h
    halo_pos = np.array([halo_data[row, 6], halo_data[row, 7], halo_data[row, 8]])  # in ckpc/h
    # (peculiar velocity, according to AHF output)
    halo_vel = np.array([halo_data[row, 9], halo_data[row, 10], halo_data[row, 11]])  # in km/s
    # angular momentum vector of gas
    halo_l = np.array([halo_data[row, 48], halo_data[row, 49], halo_data[row, 50]])
    # largest moment of inertia eigenvector for stars
    L_star = np.array([halo_data[row, 68], halo_data[row, 69], halo_data[row, 70]])
    halo_R = halo_data[row, 12]  # in kpc/h
    gas_mass = halo_data[row, 45]
    star_mass = halo_data[row, 65]
    E_pot = halo_data[row, 40]  # in (M_solar / h) (km / s)^2
    phi_0 = halo_data[row, 42]
    cNFW = halo_data[row, 43]

    halo_params = {'halo_id_zi': halo_id_zi,
                   'M_halo': halo_mass,
                   'halo_pos': halo_pos,
                   'halo_vel': halo_vel,
                   'halo_l': halo_l,
                   'L_star': L_star,
                   'R_vir': halo_R,
                   'M_gas': gas_mass,
                   'M_star': star_mass,
                   'E_pot': E_pot,
                   'phi_0': phi_0,
                   'cNFW': cNFW}
    return halo_params


def nfw_mass_profile(r, M_vir, c, delta_c=200, H0=67.7):
    """ Computes the enclosed mass M(r) for an NFW profile.
    :param r: radius [kpc]
    :param M_vir: halo mass [M_solar]
    :param c: concentration parameter
    :param delta_c: overdensity factor (default to 200)
    :param H0: present-day hubble parameter (67.7 in hestia)
    :return: enclosed mass M(r) [M_solar]
    """
    # Constants
    G = 4.30091e-6  # Gravitational constant in (kpc * km^2) / (M_sun * s^2)
    rho_crit = 3 * (H0 / 1000) ** 2 / (8 * np.pi * G)  # Critical density in M_sun/kpc^3
    # Virial radius
    R_vir = (3 * M_vir / (4 * np.pi * delta_c * rho_crit)) ** (1 / 3)
    # Scale radius
    r_s = R_vir / c
    # Characteristic density
    rho_0 = M_vir / (4 * np.pi * r_s ** 3 * (np.log(1 + c) - c / (1 + c)))
    # Enclosed mass
    M_r = 4 * np.pi * rho_0 * r_s ** 3 * (np.log(1 + r / r_s) - (r / r_s) / (1 + r / r_s))
    return M_r


def get_centralBH(run, halo, snap, verbose=True):
    from .particles import retrieve_particles, filter_particles
    from .geometry import get_redshift
    h = 0.677
    a = 1 / (1 + float(get_redshift(run, snap)))

    bh = retrieve_particles(run, halo, snap, 'PartType5', verbose=verbose)
    verbose and print(f'\rretrieved central bh of {halo}, M ~ {(1e10 * bh["BH_Mass"].item() / h):.2e};')

    # bh['Coordinates'] = bh['Coordinates'] * 1e3 / h  # from Mpc/h to kpc
    # bh['Velocities'] = bh['Velocities'] * np.sqrt(a)

    # transforming central bh to reference frame of most bound particle
    # ~ potential minimum up to a factor of the softening length, \epsilon ~ 180 pc
    # uses AHF_halos output to place origin, stars to rotate coordinates
    # bh = transform_haloFrame(run, halo, snap, bh, verbose=False)

    # I believe this might be depreciated
    if run != '09_18_lastgigyear':
        sampleCubeSideLength = 10  # half the length of the cube to sample \phi_min from, in kpc
        halo_pos = get_halo_params(run, halo, snap)['halo_pos'] / h  # in kpc
        particles = {}
        part_types = ['PartType0', 'PartType1', 'PartType4']
        part_names = {'PartType0': 'gas cells', 'PartType1': 'dm particles', 'PartType4': 'star particles'}

        for part_type in part_types:
            particles[part_type] = retrieve_particles(run, halo, snap, part_type)
            particles[part_type] = filter_particles(particles[part_type],
                                                    halo_pos - sampleCubeSideLength, halo_pos + sampleCubeSideLength)
            verbose and print(f'\t\tnumber of {part_names[part_type]} to be indexed: '
                              f'{len(particles[part_type]["ParticleIDs"])}')
            # unphysical units of a(km/s)^2
            particles[part_type]['minPot_idx'] = np.argmin(particles[part_type]['Potential'])

        minPot_partType = np.argmin(np.array([
            particles['PartType0']['Potential'][int(particles[part_type]['minPot_idx'])],
            particles['PartType1']['Potential'][int(particles[part_type]['minPot_idx'])],
            particles['PartType4']['Potential'][int(particles[part_type]['minPot_idx'])]
        ]))

        minPot_idx = particles[part_types[minPot_partType]]['minPot_idx']
        verbose and print(f'\t\tminimum of gravitational potential located at'
                          f'id: {particles[part_types[minPot_partType]]["ParticleIDs"][minPot_idx]}'
                          f'(x,y,z)_halo = {particles[part_types[minPot_partType]]["position"][minPot_idx]}')

        phi_min = particles[part_types[minPot_partType]]['Potential'][minPot_idx] / a

    else:
        phi_min = np.ones(1)

    # returns dictionary containing columns for central bh,
    # as well as computed value of phi_min, up to a factor of the softening length
    return bh, phi_min


def get_L_star(run, halo, snap, verbose=True):
    # will retrieve E_a_star for inner 10% virial radius
    from .geometry import get_redshift
    import pandas as pd

    verbose and print(f'\t\tcalculating L_star(< 0.1 * R_vir) for {run}/{halo}; snapshot {snap}')
    snap_ = '0' + str(snap) if float(snap) < 100 else str(snap)
    halo_id_z0 = halo_dictionary(run, halo)

    if run != '09_18_lastgigyear':
        filename = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc/'
                    f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        row = 127 - int(snap)
    else:
        filename = (f'/z/rschisholm/halos/09_18/{halo}/'
                    f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        row = 307 - int(snap)

    halo_data = np.loadtxt(filename)
    haloId_zi = str(int(halo_data[row, 1]))
    redshift_ = f'{get_redshift(run, snap):.3f}'

    # print(f'\t> haloId_zi = {haloId_zi}' if verbose else None)
    input_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output'
                  + ('_2x2.5Mpc/' if run != '09_18_lastgigyear' else '/')
                  + f'HESTIA_100Mpc_8192_{run}.{snap_}.z{redshift_}.AHF_disks')
    disks = pd.read_csv(input_path,
                        delim_whitespace=True, header=None, skiprows=1,
                        engine="python", dtype={0: str})  # engine="python" allows ragged rows
    idx_halo = disks.loc[disks[0].astype(str) == haloId_zi]
    row_halo = idx_halo.index[0]
    r_vir_h = disks.iloc[row_halo, 11]

    i, r_h = 0, 0
    while r_h < 0.1 * r_vir_h:  # within 10% of virial radius
        i += 1
        r_h = float(disks.iloc[row_halo + i, 0])
    # print('i = ' + str(i))
    # print('r_h = ' + str(r_h))
    # E_a_star_x = disks.iloc[row_halo + i, 27]
    # print('E_a_star_x = ' + str(E_a_star_x))

    # E_a_star [x, y, z] for inner 10% of virial radius

    # L_star [x, y, z] for inner 10% of virial radius ... when ids (22, 23, 24)
    return np.array([disks.iloc[row_halo + i, 22], disks.iloc[row_halo + i, 23], disks.iloc[row_halo + i, 24]])


# noinspection PyUnboundLocalVariable
def get_massProfile(run, halo, snap, verbose=True):
    h = 0.677
    # will obtain mass profile
    from .geometry import get_redshift
    import pandas as pd

    verbose and print(f'\t\tretrieving mass profile for {run}/{halo}; snapshot {snap}')
    snap_ = '0' + str(snap) if float(snap) < 100 else str(snap)
    halo_id_z0 = halo_dictionary(run, halo)

    if run != '09_18_lastgigyear':
        filename = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc/'
                    f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        row = 127 - int(snap)
    else:
        filename = (f'/z/rschisholm/halos/09_18/{halo}/'
                    f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
        row = 307 - int(snap)

    halo_data = np.loadtxt(filename)
    haloId_zi = str(int(halo_data[row, 1]))
    redshift_ = f'{get_redshift(run, snap):.3f}'

    # print(f'\t> haloId_zi = {haloId_zi}' if verbose else None)
    input_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output'
                  + ('_2x2.5Mpc/' if run != '09_18_lastgigyear' else '/')
                  + f'HESTIA_100Mpc_8192_{run}.{snap_}.z{redshift_}.AHF_disks')
    disks = pd.read_csv(input_path,
                        delim_whitespace=True, header=None, skiprows=1,
                        engine="python", dtype={0: str})  # engine="python" allows ragged rows
    idx_halo = disks.loc[disks[0].astype(str) == haloId_zi]
    row_halo = idx_halo.index[0]
    verbose and print(f'\t\t\t{run}/{halo} init row : {row_halo}')
    r_vir_h = disks.iloc[row_halo, 11]

    i, r_h = 0, 0
    while r_h < r_vir_h:  # within virial radius
        i += 1
        r_h = float(disks.iloc[row_halo + i, 0])
        # r_h, M_tot_h, M_gas_h, M_star_h
        row = np.array([float(disks.iloc[row_halo + i, 0]), float(disks.iloc[row_halo + i, 1]),
                        float(disks.iloc[row_halo + i, 2]), float(disks.iloc[row_halo + i, 19])]) / h
        verbose and print(f'\t\t\tr ~ {row[0]:.2f} kpc, M(< r) ~ {row[1]:.2e} M_solar')
        try:
            data = np.vstack((data, row))
        except NameError:  # first iteration, initiates data array
            data = row
    return data


def get_rotation_curve(run, halo, snaps):
    from .geometry import get_redshift
    h = 0.677

    rot_curves = np.array([])
    halo_id = None

    for snap in range(snaps[1], snaps[0], -1):
        redshift = get_redshift(run, snap)
        print('halo = ' + str(halo))
        print('run = ' + run)
        halo_params = get_halo_params(run, halo, snap, previous_halo_id=halo_id)
        halo_id = halo_params['halo_id_zi']
        print('halo_id (inside get_rotation_curve) = ' + str(halo_id))

        if run != '09_18_lastgigyear' or snap < 118:
            profile_file_path_prefix = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/'
                                        + ('09_18' if run == '09_18_lastgigyear' else str(run))
                                        + '/AHF_output_2x2.5Mpc/')
            profiles_file_path = (profile_file_path_prefix + 'HESTIA_100Mpc_8192_'
                                  + ('09_18' if run == '09_18_lastgigyear' else str(run))
                                  + '.' + "{:03}".format(snap) + '.z' + redshift + '.AHF_profiles')
        else:
            profile_file_path_prefix = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/AHF_output/'
            profiles_file_path = (profile_file_path_prefix + 'HESTIA_100Mpc_8192_'
                                  + '09_18_lastgigyear.' + "{:03}".format(snap) + '.z' + redshift + '.AHF_profiles')

        profiles_data = np.loadtxt(profiles_file_path)
        row_i, row_f = get_ahfProfilesRange(int(halo_id[-4:]), profiles_data)
        radii = profiles_data[row_i:row_f, 0] / h
        v_circ = profiles_data[row_i:row_f, 5]
        # print('Length of radii array: ' + str(len(radii)))
        # -------------------------------
        # TEMPORARY FIX FOR SNAP 127 AHF OUTPUT CONTAINING ONE FEWER ELEMENT
        # if len(radii) != 36:
        #     radii = np.append(np.array([-0.01]), radii)
        #     v_circ = np.append(np.array([0]), v_circ)
        # -------------------------------
        if snap == snaps[1]:
            rot_curves = np.array([radii, v_circ])
        else:
            len_radii_array = len(radii)
            print('len(radii_array) = ' + str(len_radii_array))
            print('rot_curves[0].shape[0] = ' + str(rot_curves[0].shape[0]))
            if len_radii_array < rot_curves[0].shape[0]:
                # adds a data point for any size radii array so that it matched rot_curves[0].shape[0]
                for i in range(rot_curves[0].shape[0] - len_radii_array):
                    radii = np.append(np.array([-0.01]), radii)
                    v_circ = np.append(np.array([0]), v_circ)
            rot_curves = np.dstack((rot_curves, [radii, v_circ]))

        print('rot_curves.shape = ' + str(rot_curves.shape))
    return rot_curves.T


def get_radial_profile(halo, run, snaps):
    h = 0.677
    # routine only works for the 100 largest halos at z=0.0 as identified by the AHF
    if halo == 'mw':
        halo_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run) + '/AHF_output_2x2.5Mpc/'
                                                                              'HESTIA_100Mpc_8192_' + str(
            run) + '.127_halo_127000000000003.dat')
    elif halo == 'lmc':
        halo_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run) + '/AHF_output_2x2.5Mpc/'
                                                                              'HESTIA_100Mpc_8192_' + str(
            run) + '.127_halo_127000000000010.dat')
    else:
        print('Error: Not a valid halo!')
        exit(1)
    halo_data = np.loadtxt(halo_path)
    if isinstance(snaps, int):
        snaps = np.array([snaps])
    rad_profs = np.array([])
    for snap in snaps:
        halo_id = str(int(halo_data[127 - snap, 1]))[-2:]
        redshift = "{:.3f}".format(halo_data[127 - snap, 0])  # string in format x.xxx
        file_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run) + '/AHF_output_2x2.5Mpc/'
                                                                              'HESTIA_100Mpc_8192_' + str(
            run) + '.' + "{:03}".format(snap) + '.z' + redshift + '.AHF_profiles')
        data = np.loadtxt(file_path)
        row_i, row_f = get_ahfProfilesRange(int(halo_id), data)
        radii = data[row_i:row_f, 0] / h
        # AHF documentation seems to imply gas and stellar mass are in units of M_solar, NOT M_solar/h
        gas = data[row_i:row_f, 24]
        stars = data[row_i:row_f, 25]
        print('Length of radii array: ' + str(len(radii)))
        # -------------------------------
        # TEMPORARY FIX FOR SNAP 127 AHF OUTPUT CONTAINING ONE FEWER ELEMENT
        if len(radii) != 37:
            radii = np.append(radii, -0.01)
            gas = np.append(gas, 0)
            stars = np.append(stars, 0)
        # -------------------------------
        if snap == snaps[0]:
            rad_profs = [radii, gas, stars]
        else:
            rad_profs = np.dstack((rad_profs, [radii, gas, stars]))
    return rad_profs


def get_extent_curve(run, snaps):
    h = 0.677
    lmc_halo_id = 10
    col_r_vir = 12
    col_r_max = 13
    filename = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run)
                + '/AHF_output_2x2.5Mpc/HESTIA_100Mpc_8192_' + str(run) + '.127_halo_1270000000000'
                + str(lmc_halo_id) + '.dat')
    lmc_data = np.loadtxt(filename)
    r_vir = lmc_data[127 - snaps[1]:127 - snaps[0], col_r_vir] / h
    r_max = lmc_data[127 - snaps[1]:127 - snaps[0], col_r_max] / h
    return r_vir, r_max


def get_sfr_curve(run, snaps):
    from .geometry import get_lookbackTimes

    arr_snaps = np.arange(snaps[1], snaps[0], step=-1)
    base_path = '/z/rschisholm/halos/09_18/halo_08/snapshot_files_withPadding/'

    key_names = ['Coordinates', 'StarFormationRate']

    avgs_sfr, arr_snaps_present = np.array([]), np.array([])
    for snap in arr_snaps:
        try:
            str_snap = '0' + str(snap) if snap < 100 else str(snap)
            print('Processing snap ' + str_snap + '...')
            file_path = base_path + '/snapshot_' + str_snap + '.hdf5'

            particles = append_particles('PartType0', file_path, key_names)
            avg_sfr = np.sum(particles['StarFormationRate'])
            avgs_sfr = np.append(avgs_sfr, avg_sfr)

            arr_snaps_present = np.append(arr_snaps_present, snap)

        except IOError:
            print('Warning: snapshot' + str_snap + '.hdf5 appear to not have been created!')

    redshifts, _ = get_lookbackTimes(run, snaps)
    return redshifts, avgs_sfr
