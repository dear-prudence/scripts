from __future__ import division
import numpy as np
import h5py
from .particles import append_particles, convert_to_supported_dtype

"""
This file contains routines involved directly and exclusively with gas cells in the hestia simulations
"""


def calc_temperature(u, e_abundance, x_h):
    """
    :param u: internal energy [(km/s)^2]
    :param e_abundance: electron abundance
    :param x_h: mass fraction of Hydrogen
    :return: temperature for gas cell [K]
    """
    gamma = 5 / 3  # needs from __future__ import division when run on a machine with Python 2.x
    k_b = 1.3807 * 1e-16
    m_p = 1.67262 * 1e-24
    unit_ratio = 1e10
    mu = 4 * m_p / (1 + 3 * x_h + 4 * x_h * e_abundance)
    # internal energy per unit mass U is in (km/s)^2, and so needs the 10^6 to convert to (m/s)^2
    return (gamma - 1) * (u / k_b) * unit_ratio * mu


def calc_numberDensity(rhoH, mu=1.00784):
    """
    :param rhoH: Hydrogen mass density [M_solar/kpc^3]
    :param mu: mean molecular mass (use mu=1.00784 for H0 and mu=0.59 for H1)
    :return: volumetric number density of Hydrogen atoms [cm^-3]
    """
    M_solar_to_g = 1.989e33  # Solar mass in grams
    kpc_to_cm = 3.086e21  # 1 kpc in cm
    amu_to_g = 1.66053906660e-24  # Atomic mass unit in grams
    # Safeguard against extremely large inputs
    rhoH = np.asarray(rhoH, dtype=np.float64)  # Ensure it's a float array or scalar
    if np.any(rhoH <= 0):
        raise ValueError("Hydrogen mass density must be positive and non-zero.")
    # Convert hydrogen mass density to g/cm^3
    rho_h_cgs = rhoH * M_solar_to_g / (kpc_to_cm ** 3)
    # Convert mass density to number density
    m_h = mu * amu_to_g  # Mass of a hydrogen atom/molecule in grams
    n_h = rho_h_cgs / m_h
    return n_h


def calc_virialTemperature(M_vir, H0=67.7, delta_vir=200, mu=0.59):
    """
    :param M_vir: halo mass [M_solar]
    :param H0: present-day hubble parameter (H0 = 67.7 km/s/Mpc)
    :param delta_vir: mass threshold (default to M_200)
    :param mu: mean molecular weight (mu = 0.59 for ionized gas)
    :return: virial temperature of halo [K]
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


def calc_temperatureProfile(run, halo, snap, max_radius=400, radius_step=2):
    """
    :param run, halo, snap, previous_halo_id: passing arguments for halo parameters
    :param max_radius: maximum radius of temperature profile [kpc], optional
    :param radius_step: step size of temperature profile [kpc], optional
    :return: equilibrium temperature profile T(r) for an NFW halo [K]
    """
    from .halos import nfw_mass_profile, get_halo_params

    gamma = 5 / 3  # needs from __future__ import division when run on a machine with Python 2.x
    mu = 0.59  # mean molecular weight, ~ 0.59 for ionized gas
    G = 6.674e-11
    k_b = 1.381e-23
    m_p = 1.672e-27
    h = 0.677

    halo_params = get_halo_params(run, halo, snap)
    halo_mass_h, cNFW = halo_params['M_halo'], halo_params['cNFW']
    # c = 24.4522  # taken from ...0008.dat file
    radii = np.linspace(1, max_radius, num=int(round(max_radius / radius_step)))  # Radius in kpc
    M_in_r = nfw_mass_profile(radii, halo_mass_h / h, cNFW)

    # equation taken from Salem+2015 (https://arxiv.org/abs/1507.07935)
    T = gamma * (mu / 3) * (G * m_p / k_b) * (M_in_r * 1.988e30 / radii / 3.086e19)
    return T, radii


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


def get_h0Center(run, halo, snap, disk_cutoff=10, verbose=False):
    from .particles import retrieve_particles

    verbose and print('\t\t\tcomputing ' + ('h0' if run != '09_18_lastgigyear' else 'cold gas') + 'center...')
    cells = retrieve_particles(run, halo, snap, 'PartType0', verbose=False)
    cells['norm'] = np.linalg.norm(cells['position'], axis=1)  # norm from halo center
    mask = cells['norm'] < disk_cutoff  # in kpc
    nuclearCells = {k: v[mask] for k, v in cells.items()}
    verbose and print(f'\t\t\tnum of nuclear gas cells : {len(nuclearCells["ParticleIDs"])}')

    if run == '09_18_lastgigyear':
        f_h0 = 0.76
        nuclearCells['T'] = calc_temperature(nuclearCells['InternalEnergy'], nuclearCells['ElectronAbundance'],
                                             x_h=f_h0)
        mask = nuclearCells['T'] < 1e4  # in K
        nuclearCells = {k: v[mask] for k, v in nuclearCells.items()}
        verbose and print(f'\t\t\tnum of cold nuclear gas cells : {len(nuclearCells["ParticleIDs"])}')
    else:
        f_h0 = nuclearCells['NeutralHydrogenAbundance']

    return np.array([
        np.average(nuclearCells['position'][:, 0], weights=nuclearCells['Masses'] * f_h0),
        np.average(nuclearCells['position'][:, 1], weights=nuclearCells['Masses'] * f_h0),
        np.average(nuclearCells['position'][:, 2], weights=nuclearCells['Masses'] * f_h0)
    ])


def get_bh_feedback(run, snaps):
    """
    NEEDS UPDATING -----------------------------------------------
    :param run:
    :param snaps:
    :return:
    """
    h = 0.677
    # lmc_halo_id = 10
    lmc_bh_id = 369831806
    masses, m_dots, bh_e_qm, bh_e_rm, bh_m_qm, bh_m_rm \
        = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for snap_i in range(snaps[1], snaps[0], -1):
        if snap_i < 100:
            snap = '0' + str(snap_i)
        else:
            snap = str(snap_i)
        key_names = ['ParticleIDs', 'BH_Mass', 'BH_Mdot', 'BH_CumEgyInjection_QM', 'BH_CumEgyInjection_RM',
                     'BH_CumMassGrowth_QM', 'BH_CumMassGrowth_RM']
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run) + '/output_2x2.5Mpc/snapdir_'
                     + snap + '/snapshot_' + snap + '.')
        file_extension = '.hdf5'
        # Generate file paths using a loop
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        # Initialize the resulting array
        all_particles = {name: None for name in key_names}
        # Loop through the file paths and append coordinates
        print('Processing Snapshot ' + snap + '...')
        for file_path in file_paths:
            try:
                all_particles = append_particles('PartType5', file_path, key_names=key_names,
                                                 existing_arrays=all_particles)
            except KeyError:
                print('Warning: ' + str(file_path[-19:-5]) + ' has no PartType5!')
        try:
            idx = int(np.where(all_particles['ParticleIDs'] == lmc_bh_id)[0])
            try:
                # unit conversion factors taken from Illustris-TNG documentation
                mass = all_particles['BH_Mass'][idx] * 1e10 / h
                # units are M_solar/Gyr
                m_dot = all_particles['BH_Mdot'][idx] * (1e10 / h) / (0.978 / h)
                # units are M_solar*ckpc^2/Gyr^2
                bh_cumEgyInjection_qm = (all_particles['BH_CumEgyInjection_QM'][idx]
                                         * (1e10 / h) * (h ** -2) / (0.978 / h) ** 2)
                bh_cumEgyInjection_rm = (all_particles['BH_CumEgyInjection_RM'][idx]
                                         * (1e10 / h) * (h ** -2) / (0.978 / h) ** 2)
                # units are M_solar
                bh_cumMassGrowth_qm = all_particles['BH_CumMassGrowth_QM'][idx] * 1e10 / h
                bh_cumMassGrowth_rm = all_particles['BH_CumMassGrowth_RM'][idx] * 1e10 / h

                masses = np.append(masses, mass)
                m_dots = np.append(m_dots, m_dot)
                bh_e_qm = np.append(bh_e_qm, bh_cumEgyInjection_qm)
                bh_e_rm = np.append(bh_e_rm, bh_cumEgyInjection_rm)
                bh_m_qm = np.append(bh_m_qm, bh_cumMassGrowth_qm)
                bh_m_rm = np.append(bh_m_rm, bh_cumMassGrowth_rm)
                print('bh cum Egy: ' + str(bh_cumEgyInjection_qm))
            except IndexError or TypeError:
                pass
        except TypeError:
            masses = np.append(masses, 0)
            m_dots = np.append(m_dots, 0)
            bh_e_qm = np.append(bh_e_qm, 0)
            bh_e_rm = np.append(bh_e_rm, 0)
            bh_m_qm = np.append(bh_m_qm, 0)
            bh_m_rm = np.append(bh_m_rm, 0)
            pass
            # hestia mass units are 1e10 * M_solar / h
    return masses, m_dots, bh_e_qm, bh_e_rm, bh_m_qm, bh_m_rm


def calc_coronaProperties(run, halo_id, snap, verbose=True):
    from .halos import get_halo_params
    from .geometry import get_redshift, calc_distanceDisk
    from .particles import retrieve_particles
    from .image import twoD_Gaussian
    from scipy.optimize import curve_fit

    redshift = get_redshift(run, snap)
    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    h = 0.677
    Z_solar = 0.0127
    spillover_factor = 1.0  # how much further past the virial radius is the corona expected to extend
    gaseousDisk_factor = 0.1

    halo_params = get_halo_params(run, halo_id, snap, full_halo_id=True)
    _, mass_h, pos_h, vel_h, l_h, r_vir_h = (halo_params['halo_id_zi'], halo_params['M_halo'],
                                             halo_params['halo_pos'], halo_params['halo_vel'],
                                             halo_params['halo_l'], halo_params['R_vir'])

    halo = 'halo_' + halo_id[-2:]
    cells = retrieve_particles(run, halo, snap, part_type='PartType0', verbose=verbose)

    cells['Distances'] = calc_distanceDisk(cells)
    cells['Temperature'] = calc_temperature(cells['InternalEnergy'],
                                            cells['ElectronAbundance'],
                                            cells['GFM_Metals'][:, 0])
    cells['n_H'] = np.log10(calc_numberDensity(cells['Density']
                                               * cells['GFM_Metals'][:, 0]))
    verbose and print(f'\t\tmean(distance, temperature, n_H) : ({np.average(cells["Distances"]):.1f} kpc, '
                      f'{np.average(cells["Temperature"]):.3e} K, {np.average(cells["n_H"]):.2f} cm^-3)')

    # filtering for cgm-associated gas cells within full set
    cgm_mask = np.where((cells['Distances'] <= spillover_factor * r_vir_h / h) &
                        (cells['Distances'] > gaseousDisk_factor * r_vir_h / h))
    cgm_particles = {name: None for name in cells.keys()}
    for key in cgm_particles.keys():
        cgm_particles[key] = cells[key][cgm_mask]
    verbose and print(f'\t\tnumber of cgm-associated cells : {len(cgm_particles["ParticleIDs"])}')

    # filtering for corona-associated gas cells within cgm subset
    corona_mask = np.where(cgm_particles['Temperature'] > 1e5)
    corona_particles = {name: None for name in cgm_particles.keys()}
    for key in corona_particles.keys():
        corona_particles[key] = cgm_particles[key][corona_mask]
    verbose and print(f'\t\tnumber of corona-associated cells : {len(corona_particles["ParticleIDs"])}')

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

    verbose and print(f'\t\t(M_HI, M_HII, T_avg) : ({total_HI_mass:.3e} M_solar, {total_HII_mass:.3e} M_solar, '
                      f'{avg_temp:.3e} K)')

    # minimum number of gas cells to accurately simulate the corona
    gasCells_threshold = 100
    if len(corona_particles['ParticleIDs']) > gasCells_threshold:
        # phase diagram
        H, x_e, y_e = np.histogram2d(cgm_particles['n_H'],
                                     np.log10(cgm_particles['Temperature']),
                                     bins=400, range=np.array([[-8, -2], [5, 7]]),
                                     weights=cgm_particles['Masses'] * cgm_particles['GFM_Metals'][:, 0]
                                             * (1 - cgm_particles['NeutralHydrogenAbundance']), density=True)
        # H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        # H_smooth = gaussian_filter(H, sigma=1)
        # H_smooth_sanitized = np.nan_to_num(H_smooth, nan=0.0, posinf=0.0, neginf=0.0)
        x_c = (x_e[:-1] + x_e[1:]) / 2
        y_c = (y_e[:-1] + y_e[1:]) / 2
        X, Y = np.meshgrid(x_c, y_c)

        lower_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

        initial_guess = (np.max(H), -5.5, 5.5, 0.5, 0.5, 0)  # amplitude, xo, yo, sigma_x, sigma_y, offset
        popt, _ = curve_fit(twoD_Gaussian, (X.ravel(), Y.ravel()), H.T.ravel(), p0=initial_guess,
                            bounds=(lower_bounds, upper_bounds), maxfev=9999)

        # Extract the fitted parameters:
        amplitude, x_nH, x_T, sigma_nH, sigma_T, offset = popt

        verbose and print(f'\t\t(log x_nH, log mean_T ) : ({x_nH:.3f} log cm^-3, {x_T:.3f} log K);'
                          f'\n\t\t(log sigma_nH, log sigma_T) : ({sigma_nH:.3f} log cm^-3, {sigma_T:.3f} log K)')

        coronaProperties = {'M_HI': np.log10(total_HI_mass),
                            'M_HII': np.log10(total_HII_mass),
                            'T_avg': np.log10(avg_temp),
                            'mean_nH': x_nH,
                            'mean_T': x_T,
                            'sigma_nH': sigma_nH,
                            'sigma_T': sigma_T}
    else:
        verbose and print(f'\t\tfewer than {gasCells_threshold} gas cells found, omitting coronal classification')
        coronaProperties = {'M_HI': np.log10(total_HI_mass),
                            'M_HII': np.log10(total_HII_mass),
                            'T_avg': np.log10(avg_temp),
                            'mean_nH': 0,
                            'mean_T': 0,
                            'sigma_nH': 0,
                            'sigma_T': 0}

    return coronaProperties


def create_stream_snapshotFiles(snap, output_path, particle_ids=None):
    """
    Writes snapshot files for the tidally-produced neutral stream produced by halo 127...008 in 09_18
    :param snap: snapshot
    :param output_path: path of hdf5 output files
    :param particle_ids: array of particle ids to include in stream (only valid for small time-steps), optional
    :return: array of particle ids included in the hdf5 file
    """
    from .geometry import transform_haloFrame, rid_h_units, get_redshift
    from .particles import filter_particles
    from .halos import get_halo_params

    # ----- TEMPORARILY DISABLED FUNCTIONALITY FOR PART_TYPE1 AND PART_TYPE4 -----

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    redshift = get_redshift('09_18', snap)

    file_path = '/z/rschisholm/storage/snapshots_lmc/snapshots_lmc_traditional/snapshot_' + snap_ + '.hdf5'
    with h5py.File(file_path, 'r') as k:
        part0_keys = list(k['PartType0'].keys())
        # part1_keys = list(k['PartType1'].keys())
        # part4_keys = list(k['PartType4'].keys())

    all_parts0 = {name: None for name in part0_keys}
    # all_parts1 = {name: None for name in part1_keys}
    # all_parts4 = {name: None for name in part4_keys}

    all_parts0 = append_particles('PartType0', file_path, key_names=part0_keys, existing_arrays=all_parts0)
    # all_parts1 = append_particles('PartType1', file_path, key_names=part1_keys, existing_arrays=all_parts1)
    # all_parts4 = append_particles('PartType4', file_path, key_names=part4_keys, existing_arrays=all_parts4)

    print('len(all_parts0) = ' + str(len(all_parts0['ParticleIDs'])))

    # halo_id = halo_dictionary('09_18', 'lmc')
    # l_b, u_b = center_halo(run='09_18', halo_id=halo_id, snap=snap,
    #                        size=np.array([150, 150, 150])) * 1e-3  # these are in _h units!
    # l_b and u_b carve a cube with side length 2x specified in "size", to account for the gaps in spatial
    # distribution of particles after the halo coordinate transformation

    halo_params = get_halo_params('09_18', 'lmc', snap)
    pos_h, r_vir_h = halo_params['halo_pos'], halo_params['R_vir']
    lb_h, ub_h = (pos_h - 2 * r_vir_h) / 1e3, (pos_h + 2 * r_vir_h) / 1e3

    print(lb_h)
    print(ub_h)

    all_parts0_phys = rid_h_units(filter_particles(all_parts0, lb_h, ub_h),
                                  z=float(redshift), part_type='PartType0')
    # all_parts1_phys = cosmo_to_physical(filter_particles(all_parts1, lb_h, ub_h),
    #                                     z=float(redshift), part_type='PartType1')
    # all_parts4_phys = cosmo_to_physical(filter_particles(all_parts4, lb_h, ub_h),
    #                                     z=float(redshift), part_type='PartType4')

    print('all_parts0_phys[Coords][0] = ' + str(all_parts0_phys['Coordinates'][0]))

    all_parts0_rot = transform_haloFrame('09_18', '127000000000008', snap, all_parts0_phys)
    # all_parts1_rot = halo_frame('09_18', '127000000000008', snap, all_parts1_phys)
    # all_parts4_rot = halo_frame('09_18', '127000000000008', snap, all_parts4_phys)

    filtered_particles0, ids0 = filter_stream_particles(all_parts0_rot, 'PartType0', particle_ids)
    # filtered_particles1, ids1 = filter_stream_particles(all_parts1_rot, 'PartType1', particle_ids)
    # filtered_particles4, ids4 = filter_stream_particles(all_parts4_rot, 'PartType4', particle_ids)

    # Convert filtered particles to supported dtypes
    filtered_particles0 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles0.items()}
    # filtered_particles1 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles1.items()}
    # filtered_particles4 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles4.items()}

    filtered_particles0['n_H'] = calc_numberDensity(
        filtered_particles0['Density'] * filtered_particles0['GFM_Metals'][:, 0])

    # Create a new HDF5 file and write the filtered particles to it
    with h5py.File(output_path, 'w') as outfile:
        # Write the filtered particles dataset
        for key, data in filtered_particles0.items():
            outfile.create_dataset('PartType0/' + key, data=data)
        # for key, data in filtered_particles1.items():
        #     outfile.create_dataset('PartType1/' + key, data=data)
        # for key, data in filtered_particles4.items():
        #     outfile.create_dataset('PartType4/' + key, data=data)

    print('Done!')
    return ids0  # np.append(ids0, np.append(ids1, ids4))


def create_stream_snapshotFiles_lastgigyear(snap, output_path, particle_ids=None, previous_halo_id=None):
    """
    Same as create_stream_snapshotFiles, but valid for 09_18_lastgigyear
    :param snap: snapshot
    :param output_path: path of hdf5 output files
    :param particle_ids: array of particle ids to include in stream (only valid for small time-steps), optional
    :param previous_halo_id: passing argument (speeds up reading of merger tree files), optional
    :return: array of particle ids included in the hdf5 file
    """
    from .geometry import transform_haloFrame, rid_h_units, get_redshift
    from .particles import filter_particles
    from .halos import get_halo_params

    # ----- TEMPORARILY DISABLED FUNCTIONALITY FOR PART_TYPE1 AND PART_TYPE4 -----

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    redshift = get_redshift('09_18_lastgigyear', snap)

    file_path = '/z/rschisholm/storage/snapshots_lmc/snapshot_' + snap_ + '.hdf5'
    with h5py.File(file_path, 'r') as k:
        part0_keys = list(k['PartType0'].keys())

    all_parts0 = {name: None for name in part0_keys}

    all_parts0 = append_particles('PartType0', file_path, key_names=part0_keys, existing_arrays=all_parts0)

    print('len(all_parts0) = ' + str(len(all_parts0['ParticleIDs'])))

    halo_id, _, pos_h, _, _, r_vir_h = get_halo_params('09_18_lastgigyear', 'lmc', snap,
                                                       previous_halo_id=previous_halo_id)
    lb_h, ub_h = (pos_h - 2 * r_vir_h) / 1e3, (pos_h + 2 * r_vir_h) / 1e3

    all_parts0_phys = rid_h_units(filter_particles(all_parts0, lb_h, ub_h),
                                  z=float(redshift), part_type='PartType0')

    print('all_parts0_phys[Coords][0] = ' + str(all_parts0_phys['Coordinates'][0]))

    all_parts0_rot = transform_haloFrame('09_18_lastgigyear', 'lmc', snap, all_parts0_phys)

    filtered_particles0, ids0 = filter_stream_particles(all_parts0_rot, 'PartType0',
                                                        particle_ids, lastgigyear=True)
    # Convert filtered particles to supported dtypes
    filtered_particles0 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles0.items()}

    X_H = 0.76  # approximation of primordial hydrogen mass fraction
    filtered_particles0['n_H'] = calc_numberDensity(filtered_particles0['Density'] * X_H)

    # Create a new HDF5 file and write the filtered particles to it
    with h5py.File(output_path, 'w') as outfile:
        # Write the filtered particles dataset
        for key, data in filtered_particles0.items():
            outfile.create_dataset('PartType0/' + key, data=data)

    print('Done!')
    return ids0, halo_id  # np.append(ids0, np.append(ids1, ids4))


def filter_stream_particles(all_parts, part_type, particle_ids, lastgigyear=False):
    """
    Selects particles for the neutrl stream using angular momentum, metallicity,
    and density cutoffs (or a subset of those)
    :param all_parts, part_type, particle_ids, lastgigyear: passing arguments
    :return: particles selected for the neutral stream
    """
    # ------------------------------------------
    # Cutoff values -- this is where to tweak them
    log_Lx_cutoff = [1, 8]
    log_Ly_cutoff = [7.6, 8.6]
    log_Lz_cutoff = [7.8, 8.8]
    Z_cutoff = [-3, -2]
    log_density_cutoff = [0, 4]
    # ------------------------------------------

    if particle_ids is None:
        print('------- Extracting stream... --------')

        for k in all_parts.keys():
            all_parts[k] = np.array(all_parts[k])  # Convert to numpy array for easier indexing

        radii = np.zeros(all_parts['Halo_Coordinates'].shape[0])
        rho = np.zeros(all_parts['Halo_Coordinates'].shape[0])
        for i in range(all_parts['Halo_Coordinates'].shape[0]):
            radii[i] = np.linalg.norm(all_parts['Halo_Coordinates'][i])
            rho[i] = np.linalg.norm(
                np.array([all_parts['Halo_Coordinates'][i, 0], all_parts['Halo_Coordinates'][i, 1]]))
        all_parts['Radii'] = radii
        all_parts['rho'] = rho

        print('Performing initial selection of '
              + ('gas' if part_type == 'PartType0' else 'stars/dm') + '...')

        if part_type == 'PartType0':
            if lastgigyear is True:
                # approximation for hydrogen mass fraction since 'GFM_Metals' is missing from mini snapshots
                X_H = 0.76
                metals = np.log10(all_parts['GFM_Metallicity'] / X_H)
            else:
                metals = np.log10(all_parts['GFM_Metallicity'] / all_parts['GFM_Metals'][:, 0])
            all_parts['Z'] = metals
        else:
            all_parts['Z'] = np.zeros(len(all_parts['ParticleIDs'])) - np.average(Z_cutoff)
            all_parts['Density'] = np.zeros(len(all_parts['ParticleIDs'])) - 10 ** np.average(log_density_cutoff)

        indices_to_keep = np.where(
            # (rho_cutoff <= all_parts['rho']) &
            # (-150 < all_parts['Halo_Coordinates'][:, 0]) &
            # (150 > all_parts['Halo_Coordinates'][:, 0]) &
            # (-150 < all_parts['Halo_Coordinates'][:, 1]) &
            # (150 > all_parts['Halo_Coordinates'][:, 1]) &
            # (-150 < all_parts['Halo_Coordinates'][:, 2]) &
            # (150 > all_parts['Halo_Coordinates'][:, 2]) &
            (all_parts['Radii'] < 200) &  # kpc
            (10 ** log_Lx_cutoff[0] < all_parts['Angular_Momenta'][:, 0]) &
            (10 ** log_Lx_cutoff[1] > all_parts['Angular_Momenta'][:, 0]) &
            (10 ** log_Ly_cutoff[0] < all_parts['Angular_Momenta'][:, 1]) &
            (10 ** log_Ly_cutoff[1] > all_parts['Angular_Momenta'][:, 1]) &
            (10 ** log_Lz_cutoff[0] < all_parts['Angular_Momenta'][:, 2]) &
            (10 ** log_Lz_cutoff[1] > all_parts['Angular_Momenta'][:, 2]) &
            (Z_cutoff[0] < all_parts['Z']) & (Z_cutoff[1] > all_parts['Z']) &
            (10 ** log_density_cutoff[0] < all_parts['Density']) &
            (10 ** log_density_cutoff[1] > all_parts['Density'])
        )[0]

        particle_ids = all_parts['ParticleIDs'][indices_to_keep]
        print('---- Done with particle filtering.')

    else:
        # if there is a given array of particle_ids
        pass

    # Create a mask to filter particles with IDs in array_ids
    mask = np.in1d(all_parts['ParticleIDs'].astype(np.int64), np.array(particle_ids, dtype=np.int64),
                   assume_unique=True)
    filtered_parts = {name: None for name in all_parts.keys()}
    for k in filtered_parts.keys():
        filtered_parts[k] = all_parts[k][mask]

    print('len(all_parts) = ' + str(len(all_parts['ParticleIDs'])))
    print('len(particle_ids) = ' + str(len(particle_ids)))
    print('len(mask) = ' + str(len(mask)))
    print('len(true(mask)) = ' + str(sum(mask)))
    print('len(filtered_particles) = ' + str(len(filtered_parts['Masses'])))

    return filtered_parts, particle_ids


def uvbPhotoionAtten(log_hDens, log_temp, redshift):
    """ Compute the reduction in the photoionisation rate at an energy of 13.6 eV at a given
    density [log cm^-3] and temperature [log K], using the Rahmati+ (2012) fitting formula.
    """
    """    import scipy.interpolate.interpolate as spi

    # Opacities for the FG09 UVB from Rahmati 2012.
    # Note: The values given for z > 5 are calculated by fitting a power law and extrapolating.
    # Gray power law: -1.12e-19*(zz-3.5)+2.1e-18 fit to z > 2.
    # gamma_UVB: -8.66e-14*(zz-3.5)+4.84e-13
    gray_opac = [2.59e-18, 2.37e-18, 2.27e-18, 2.15e-18, 2.02e-18, 1.94e-18, 1.82e-18, 1.71e-18, 1.60e-18, 2.8e-20]
    gamma_UVB = [3.99e-14, 3.03e-13, 6e-13, 5.53e-13, 4.31e-13, 3.52e-13, 2.678e-13, 1.81e-13, 9.43e-14, 1e-20]
    zz = [0, 1, 2, 3, 4, 5, 6, 7, 8, 22]

    gamma_UVB_z = spi.interp1d(zz, gamma_UVB)(redshift)[()]  # 1/s (1.16e-12 is HM01 at z=3)
    gray_opacity_z = spi.interp1d(zz, gray_opac)(redshift)[()]  # cm^2 (2.49e-18 is HM01 at z=3)
    # gray_opacity_z = - 1.12e-19 * (redshift - 3.5) + 2.1e-18
    # gamma_UVB_z = -8.66e-14 * (redshift - 3.5) + 4.84e-13

    f_bar = 0.167  # baryon fraction, Omega_b/Omega_M = 0.0456/0.2726 (Plank/iPrime)

    self_shield_dens = 6.73e-3 * (gray_opacity_z / 2.49e-18) ** (-2.0 / 3.0) * \
                       (10.0 ** log_temp / 1e4) ** 0.17 * (gamma_UVB_z / 1e-12) ** (2.0 / 3.0) * (f_bar / 0.17) ** (
                               -1.0 / 3.0)  # cm^-3

    # photoionisation rate vs density from Rahmati+ (2012) Eqn. 14.
    # (coefficients are best-fit from appendix A)
    ratio_nH_to_selfShieldDens = 10.0 ** log_hDens / self_shield_dens
    photUVBratio = 0.98 * (1 + ratio_nH_to_selfShieldDens ** 1.64) ** (-2.28) + \
                   0.02 * (1 + ratio_nH_to_selfShieldDens) ** (-0.84)

    # photUVBratio is attenuation fraction, e.g. multiply by gamma_UVB_z to get actual Gamma_photon
    return photUVBratio, gamma_UVB_z"""
    gamma_UVB = [3.99e-14, 3.03e-13]
    z = [0, 1]
    slope = (gamma_UVB[-1] - gamma_UVB[0]) / (z[-1] - z[0])
    gamma_UVB_z = slope * redshift + gamma_UVB[0]
    return z, gamma_UVB_z


def calc_neutral_fraction(nH, snap, temp=1e4):
    from .geometry import get_redshift
    """ The neutral fraction from Rahmati+ (2012) Eqn. A8. """
    redshift = float(get_redshift('09_18_lastgigyear', snap))
    # recombination rate from Rahmati+ (2012) Eqn. A3, also Hui & Gnedin (1997). [cm^3 / s] """
    lamb = 315614.0 / temp
    alpha_A = 1.269e-13 * lamb ** 1.503 / (1 + (lamb / 0.522) ** 0.47) ** 1.923

    # photoionization rate
    _, gamma_UVB_z = uvbPhotoionAtten(np.log10(nH), np.log10(temp), redshift)
    print('gamma_UVB_z = ' + str(gamma_UVB_z))

    # A6 from Theuns 98
    LambdaT = 1.17e-10 * temp ** 0.5 * np.exp(-157809.0 / temp) / (1 + np.sqrt(temp / 1e5))

    A = alpha_A + LambdaT
    B = 2 * alpha_A + gamma_UVB_z / nH + LambdaT

    return (B - np.sqrt(B ** 2 - 4 * A * alpha_A)) / (2 * A)


def calc_fH0_cloudy(temperatures):
    """table from most updated cloudy release here:
    https://gitlab.nublado.org/cloudy/cloudy/-/blob/master/tsuite/programs/collion/collion-Hydrogen.txt?ref_type=heads
    link can be found on pg. 405 of 2017 release of cloudy---
    cloudy/tsuite/programs/collion"""
    import scipy.interpolate.interpolate as spi

    print('range(temperatures) = ' + str([min(temperatures), max(temperatures)]))

    T_____ = [0.00,
              1.00,
              2.00,
              3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90,
              4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90,
              5.00, 5.10, 5.20, 5.30, 5.40, 5.50, 5.60, 5.70, 5.80, 5.90,
              6.00, 6.10, 6.20, 6.30, 6.40, 6.50, 6.60, 6.70, 6.80, 6.90,
              7.00,
              8.00,
              9.00]
    log_H0 = [-32.0,
              -32.0,
              -2.0,
              -1.76, -1.77, -0.55, -0.05, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00,
              -0.00, -0.02, -0.30, -1.06, -1.87, -2.56, -3.14, -3.62, -4.04, -4.40,
              -4.71, -4.99, -5.23, -5.45, -5.65, -5.84, -6.01, -6.17, -6.32, -6.45,
              -6.59, -6.71, -6.83, -6.94, -7.06, -7.16, -7.27, -7.37, -7.47, -7.58,
              -7.68,
              -8.70,
              -32.0]
    log_H1 = [-32.0,
              -32.0,
              -32.0,
              -32.0, -32.0, -32.0, -32.0, -32.0, -32.0, -32.0, -32.0, -32.0, -7.91,
              -3.36, -1.33, -0.30, -0.04, -0.01, -0.00, -0.00, -0.00, -0.00, -0.00,
              -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00,
              -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00,
              -0.00,
              -0.00,
              -0.00]

    log_f_H0 = spi.interp1d(T_____, log_H0)(np.log10(temperatures))
    log_f_H1 = spi.interp1d(T_____, log_H1)(np.log10(temperatures))
    return 10 ** log_f_H0, 10 ** log_f_H1
