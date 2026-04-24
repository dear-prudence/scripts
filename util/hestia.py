import numpy as np
import h5py
import astropy.units as u
import os
import inspect


def hestiaHaloDict(sim_run, halo_name):
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
                         'halo_1476': '307000000001476',
                         'halo_454': '307000000000540'}
        else:
            print('Error: routine is a WIP and cannot handle other simulation runs!')
            exit(1)
        return halo_dict[halo_name]


class Halo:
    H0 = 67.7
    Delta_vir = 200

    def __init__(self, run, halo, snap, full_halo_id=False):
        """
        :param run: simulation run
        :param halo: halo id of interest
        :param snap: snapshot
        :param full_halo_id: True if halo id does not need to be passed through halo_dictionary
        """
        halo_id_z0 = hestiaHaloDict(run, halo) if full_halo_id is False else halo

        if run != '09_18_lastgigyear':
            filename = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc/'
                        f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
            row = 127 - int(snap)
        else:
            filename = (f'/z/rschisholm/halos/09_18_lastgigyear/{halo}/'
                        f'HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
            row = 307 - int(snap)

        H = np.loadtxt(filename)
        h = 0.677
        # two-character string identifying the lmc halo (in snap 127, it is '10')
        self.run = run
        self.halo = halo
        self.snap = snap
        self.halo_id_zi = str(int(H[row, 1]))
        self.M = H[row, 4] * u.M_sun / h  # virial mass
        self.R = H[row, 12] * u.kpc / h  # virial radius
        self.pos = np.array([H[row, 6], H[row, 7], H[row, 8]]) * u.kpc / h  # in ckpc/h
        self.vel = np.array([H[row, 9], H[row, 10], H[row, 11]]) * u.km / u.s  # peculiar velocity, in km/s
        self.L = np.array([H[row, 48], H[row, 49], H[row, 50]]) * u.M_sun * u.km / u.s * u.kpc
        # angular momentum in j ~ M_solar km/s kpc
        self.E_star = np.array(
            [H[row, 68], H[row, 69], H[row, 70]])  # largest moment of inertia eigenvector for stars
        self.M_gas = H[row, 45] * u.M_sun / h  # gas mass
        self.M_star = H[row, 65] * u.M_sun / h  # stellar mass
        self.U = H[row, 40] * u.M_sun / h * u.km ** 2 / u.s ** 2  # potential energy in (M_solar) (km / s)^2
        self.phi_0 = H[row, 42]
        self.cNFW = H[row, 43]  # concentration parameter for NFW halo profile

    @property
    def T_vir(self):
        mu = 0.59
        # Constants
        G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
        k_B = 1.3807e-16  # Boltzmann constant in erg K^-1
        m_p = 1.673e-24  # Proton mass in grams
        H0_cgs = self.H0 * (1e5 / 3.086e24)  # Hubble constant in s^-1
        # Critical density of the universe
        rho_crit = (3 * H0_cgs ** 2) / (8 * np.pi * G)  # g/cm^3
        # Calculate virial radius (in cm)
        R_vir = ((3 * self.M * u.M_sun.to(u.g)) / (4 * np.pi * self.Delta_vir * rho_crit)) ** (1 / 3)
        # Calculate virial temperature
        T_vir = (mu * m_p * G * self.M * u.M_sun.to(u.g)) / (2 * k_B * R_vir)

        return T_vir

    def nfw_mass_profile(self, bool_computeRvir=True):
        r = np.linspace(0, 400, 400)
        # Constants
        G = 4.30091e-6  # Gravitational constant in (kpc * km^2) / (M_sun * s^2)
        rho_crit = 3 * (self.H0 / 100) ** 2 / (8 * np.pi * G)  # Critical density in M_sun/kpc^3
        # Virial radius (computed directly from virial mass instead of taken from AHF output)
        if bool_computeRvir:
            R_vir = (3 * self.M / (4 * np.pi * self.Delta_vir * rho_crit)) ** (1 / 3)
        else:
            R_vir = self.R
        # Scale radius
        r_s = R_vir / self.cNFW
        # Characteristic density
        rho_0 = self.M / (4 * np.pi * r_s ** 3 * (np.log(1 + self.cNFW) - self.cNFW / (1 + self.cNFW)))
        # Enclosed mass
        M_r = 4 * np.pi * rho_0 * r_s ** 3 * (np.log(1 + r / r_s) - (r / r_s) / (1 + r / r_s))
        return r, M_r

    def temperatureProfile(self):
        gamma = 5 / 3
        mu = 0.59  # mean molecular weight, ~ 0.59 for ionized gas
        G = 6.674e-11
        k_b = 1.381e-23
        m_p = 1.672e-27
        r, M_in_r = self.nfw_mass_profile()
        # equation taken from Salem+2015 (https://arxiv.org/abs/1507.07935)
        T = gamma * (mu / 3) * (G * m_p / k_b) * (M_in_r * 1.988e30 / r / 3.086e19)
        return r, T

    def L_star(self, verbose=True):
        # will retrieve E_a_star for inner 10% virial radius
        import pandas as pd

        verbose and print(f'\t\tcalculating L_star(< 0.1 * R_vir) for {self.run}/{self.halo}; snapshot {self.snap}')
        snap_ = '0' + str(self.snap) if float(self.snap) < 100 else str(self.snap)

        redshift_ = f'{get_redshift(self.run, self.snap):.3f}'

        input_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{self.run}/AHF_output'
                      + ('_2x2.5Mpc/' if self.run != '09_18_lastgigyear' else '/')
                      + f'HESTIA_100Mpc_8192_{self.run}.{snap_}.z{redshift_}.AHF_disks')
        disks = pd.read_csv(input_path,
                            delim_whitespace=True, header=None, skiprows=1,
                            engine="python", dtype={0: str})  # engine="python" allows ragged rows
        idx_halo = disks.loc[disks[0].astype(str) == self.halo_id_zi]
        row_halo = idx_halo.index[0]
        # check if virial radii match b/w disks file and halos file
        assert round(float(disks.iloc[row_halo, 11]) / (self.H0 / 100), 3) == round(float(self.R.value), 3), \
            'virial radii from AHF.disks and AHF.halos dat do not match!'

        i, r_h = 0, 0
        while r_h < 0.1 * self.R.value * self.H0 / 100:  # within 10% of virial radius
            i += 1
            r_h = float(disks.iloc[row_halo + i, 0])

        # E_a_star [x, y, z] for inner 10% of virial radius
        # L_star [x, y, z] for inner 10% of virial radius ... when ids (22, 23, 24)
        return np.array(disks.iloc[row_halo + i, 22:25]) / np.linalg.norm(np.array([disks.iloc[row_halo + i, 22:25]]))

    # noinspection PyUnboundLocalVariable
    def get_massProfile(self, verbose=True):
        # will obtain mass profile
        import pandas as pd

        verbose and print(f'\t\tretrieving mass profile for {self.run}/{self.halo}; snapshot {self.snap}')
        snap_ = '0' + str(self.snap) if float(self.snap) < 100 else str(self.snap)

        redshift_ = f'{get_redshift(self.run, self.snap):.3f}'

        input_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{self.run}/AHF_output'
                      + ('_2x2.5Mpc/' if self.run != '09_18_lastgigyear' else '/')
                      + f'HESTIA_100Mpc_8192_{self.run}.{snap_}.z{redshift_}.AHF_disks')
        disks = pd.read_csv(input_path,
                            delim_whitespace=True, header=None, skiprows=1,
                            engine="python", dtype={0: str})  # engine="python" allows ragged rows
        idx_halo = disks.loc[disks[0].astype(str) == self.halo_id_zi]
        row_halo = idx_halo.index[0]
        verbose and print(f'\t\t\t{self.run}/{self.halo} init row : {row_halo}')

        # r_vir_h = disks.iloc[row_halo, 11]
        # check if virial radii match b/w disks file and halos file
        assert round(float(disks.iloc[row_halo, 11]), 3) == round(float(self.R.value), 3), \
            'virial radii from AHF.disks and AHF.halos dat do not match!'

        i, r_h = 0, 0
        while r_h < self.R.value * self.H0 / 100:  # within virial radius
            i += 1
            r_h = float(disks.iloc[row_halo + i, 0])
            # r_h, M_tot_h, M_gas_h, M_star_h
            row = np.array([
                float(disks.iloc[row_halo + i, 0]),  # radius
                float(disks.iloc[row_halo + i, 1]),  # M_tot
                float(disks.iloc[row_halo + i, 2]),  # M_gas
                float(disks.iloc[row_halo + i, 19])  # M_star
            ]) / (self.H0 / 100)
            verbose and print(f'\t\t\tr ~ {row[0]:.2f} kpc, M(< r) ~ {row[1]:.2e} M_solar')
            try:
                data = np.vstack((data, row))
            except NameError:  # first iteration, initiates data array
                data = row
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


class Particles:
    def __init__(self, run, halo, snap, part_type, verbose=False):
        # particles[key] --> Particles.key
        particles = retrieve_particles(run, halo, snap, part_type, verbose=verbose)
        for key, value in particles.items():
            setattr(self, key, value)
        self.run = run
        self.halo = halo
        self.snap = snap
        self.part_type = part_type

    @property
    def len(self):
        return int(len(self.ParticleIDs))

    @property
    def norm(self):
        return np.linalg.norm(self.position, axis=1)

    @property
    def temperature(self):
        gamma = 5 / 3
        k_b = 1.3807 * 1e-16
        m_p = 1.67262 * 1e-24
        unit_ratio = 1e10
        mu = 4 * m_p / (1 + 3 * self.GFM_Metals[:, 0] + 4 * self.GFM_Metals[:, 0] * self.ElectronAbundance)
        # internal energy per unit mass U is in (km/s)^2, and so needs the 10^6 to convert to (m/s)^2
        return (gamma - 1) * (self.InternalEnergy / k_b) * unit_ratio * mu * u.K

    @property
    def nH(self, mu=1.00784):
        if self.part_type != 'PartType0':
            print(f'Error: {self}.nH is an invalid property for particle type {self.part_type}!\n'
                  f'util/hestia.py; line {inspect.currentframe().f_lineno}')
        f_H = self.GFM_Metals[:, 0] if self.run != '09_18_lastgigyear' else 0.76
        rho_H = self.Density * f_H  # * u.M_sun / u.kpc ** 3
        rho_cgs = rho_H.to(u.g / u.cm ** 3)
        m_h = (mu * u.u).to(u.g)  # mass of hydrogen atom/molecule in grams
        n_h = rho_cgs / m_h  # num H atoms / cm^3

        # M_solar_to_g = 1.989e33  # Solar mass in grams
        # kpc_to_cm = 3.086e21  # 1 kpc in cm
        # amu_to_g = 1.66053906660e-24  # Atomic mass unit in grams
        # rhoH = np.asarray(self.Density, dtype=np.float64)  # Ensure it's a float array or scalar
        # if np.any(rhoH <= 0):
        #     raise ValueError('Hydrogen mass density must be positive and non-zero.')
        # Convert hydrogen mass density to g/cm^3
        # rho_h_cgs = rhoH * M_solar_to_g / (kpc_to_cm ** 3)
        # Convert mass density to number density
        # m_h = mu * amu_to_g  # Mass of a hydrogen atom/molecule in grams
        # n_h = rho_h_cgs / m_h
        return n_h  # cm^-3

    def filter(self, mask):
        import copy
        specks = copy.copy(self)
        for key, val in vars(specks).items():
            if hasattr(val, '__len__') and hasattr(val, '__getitem__') and not isinstance(val, str):
                setattr(specks, key, val[mask])
        return specks


def retrieve_particles(run, halo, snap, part_type, padding=None, verbose=True):
    # for particles split across multiple hdf5 files
    h = 0.677

    def append_particles(partType, filename, key_names, exist_arr=None):
        if exist_arr is None:
            exist_arr = {name: None for name in key_names}

        with h5py.File(filename, 'r') as f:
            for k in key_names:
                if k in f[partType]:
                    data = np.array(f[partType][k])
                    exist_arr[k] = np.append(exist_arr[k], data, axis=0) if exist_arr[k] is not None else data
                else:
                    print(f'Error: {k} is not a key in hdf file!')
        return exist_arr

    def filter_particles(all_parts, lb, ub):
        filtered_parts = {}
        for k in all_parts.keys():
            all_parts[k] = np.array(all_parts[k])  # Convert to numpy array for easier indexing
        idxs_to_keep = np.where(
            (lb[0] <= all_parts['Coordinates'][:, 0]) & (all_parts['Coordinates'][:, 0] <= ub[0]) &
            (lb[1] <= all_parts['Coordinates'][:, 1]) & (all_parts['Coordinates'][:, 1] <= ub[1]) &
            (lb[2] <= all_parts['Coordinates'][:, 2]) & (all_parts['Coordinates'][:, 2] <= ub[2])
        )[0]
        for k in all_parts.keys():
            filtered_parts[k] = all_parts[k][idxs_to_keep]
        return filtered_parts

    z_ = get_redshift(run, snap)

    partType_dict = {'PartType0': 'gas cells', 'PartType1': 'dm particles',
                     'PartType4': 'star particles', 'PartType5': 'central bh'}
    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    verbose and print(f'\tretrieving {partType_dict[part_type]} for {run}/{halo}, snapshot {snap};')

    basePath = '/store/erebos/rschisholm/halos/'
    filePath = f'{run}/{halo}/snapshot_{snap_}.hdf5'

    if os.path.exists(f'{basePath}{filePath}') and padding is None:
        verbose and print('\t\tos.path exists')
        # if a processed halo directory already exists
        with h5py.File(basePath + filePath, 'r') as file:
            keys = file[part_type].keys()
            all_particles = {name: None for name in keys}
            all_particles = append_particles(part_type, basePath + filePath,
                                             key_names=keys, exist_arr=all_particles)
        processed_particles = add_units(all_particles)

        if not isinstance(processed_particles['Coordinates'], u.Quantity):
            processed_particles = add_units(processed_particles, part_type, z_, raw=False)
        if not isinstance(processed_particles['position'], u.Quantity):
            processed_particles['position'] *= u.kpc
            processed_particles['velocity'] *= (u.km / u.s)

    else:  # if no processed halo directory exists, extract particles from full simulation output
        if padding is None:
            verbose and print('\t\tos.path does not exist, extracting particles from simulation snapshot files')
        else:
            verbose and print(f'\t\trequested padding : {padding} R_vir, '
                              f'extracting particles from simulation snapshot files')

        H = Halo(run, halo, snap)
        halo_id, pos, vel, L, R_vir = H.halo_id_zi, H.pos, H.vel, H.L, H.R

        base_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/output'
                     + ('_2x2.5Mpc' if run != '09_18_lastgigyear' else '') + f'/snapdir_{snap_}/snapshot_{snap_}.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        with h5py.File(base_path + '0' + file_extension, 'r') as file:
            keys = file[part_type].keys()
            all_particles = {name: None for name in keys}
            for filePath in file_paths:
                all_particles = append_particles(part_type, filePath, key_names=keys,
                                                 exist_arr=all_particles)

        # trims (definitely) unnecessary particles to speed up calculations
        lower_bound, upper_bound = ((pos.to(u.Mpc).value - R_vir.to(u.Mpc).value) * h,
                                    (pos.to(u.Mpc).value + R_vir.to(u.Mpc).value) * h)

        filtered_particles = filter_particles(all_particles, lower_bound, upper_bound)

        processed_particles = transform_diskFrame(
            run, halo, snap, add_units(filtered_particles, part_type=part_type, redshift=z_, raw=True)
        )

    verbose and print(f'\t\textracted {len(processed_particles["ParticleIDs"])} {partType_dict[part_type]}')
    return processed_particles


def get_redshift(run, snap):
    """
    directly returns the redshift of a given snapshot, taken from AHF_output/snapshot.parameter file
    :param run: simulation run
    :param snap: array of snapshots
    :return: redshifts
    """
    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    if run == '09_18_lastgigyear' and snap < 118:
        run = '09_18'

    filepath = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output'
                + ('_2x2.5Mpc/' if run != '09_18_lastgigyear' else '/') +
                f'HESTIA_100Mpc_8192_{run}.{snap_}.parameter')

    with open(filepath, 'r') as f:
        contents = f.read()
        for line in contents.splitlines():
            try:
                if line[0] == 'z':
                    redshift = float(line[1:])
                    break
            except IndexError:
                pass

    return redshift  # returns the redshift of a given snapshot in format x.xxx


def get_lookbackTimes(run, snaps, redshifts=None):
    from astropy.cosmology import FlatLambdaCDM
    """
    :param run: simulation run
    :param snaps: 1-dim array of snapshots in consecutive order
    :param redshifts: direct input of redshifts, optional
    :return: redshifts, lookback times assuming a flat Lambda-CDM cosmology [Gyr]
    """
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.318)
    if redshifts is None:
        redshifts = np.array([])
        for snap in snaps:
            redshift = get_redshift(run, snap)
            redshifts = np.append(redshifts, redshift)

    lookback_times = cosmo.lookback_time(redshifts).to(u.Gyr)
    return redshifts, lookback_times


def add_units(particles, part_type='PartType0', redshift=0., raw=False):
    h = 0.677  # value used in the hestia simulation suite
    a = 1.0 / (1 + float(redshift))

    units_map = {
        'Coordinates': u.kpc,
        'Density': u.M_sun / u.kpc ** 3,
        'Masses': u.M_sun,
        'Velocities': u.km / u.s,
        'Potential': (u.km / u.s) ** 2
    }

    transform_map = {
        'Coordinates': {'factor': lambda a, h: 1e3 / h, 'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
        'Density': {'factor': lambda a, h: h ** 2, 'types': ['PartType0']},
        'EnergyDissipation': {'factor': lambda a, h: 1e10 / a, 'types': ['PartType0']},
        'Masses': {'factor': lambda a, h: 1e10 / h, 'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
        'Velocities': {'factor': lambda a, h: np.sqrt(a),
                       'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
        'Potential': {'factor': lambda a, h: 1 / a, 'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
    }

    if raw:
        for key, info in transform_map.items():
            if part_type in info['types'] and key in particles:
                factor = info['factor'](a, h)
                particles[key] = particles[key] * factor

    # applies units
    for key, info in units_map.items():
        if key in particles.keys():
            particles[key] *= units_map[key]
    return particles


def transform_diskFrame(run, halo, snap, particles, verbose=True):
    """
    transforms coordinates to static halo reference frame;
    should be used in conjunction with (and after) rid_h_units()!
    :param run: simulation run
    :param halo: halo name of interest
    :param snap: snapshot
    :param particles: particles to be transformed (in ordinary units)
    :param verbose: verbose
    :return: particles in frame of reference of halo [angular momentum: j ~ M_solar kpc km/s]
    """

    def rotationMatrix(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2; does not work for 180 degree rotations
        :param vec1: 3d "source" vector
        :param vec2: 3d "destination" vector
        :return mat: a transformation matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a = (np.array(vec1, dtype=np.float64) / np.linalg.norm(vec1)).reshape(3)
        b = (np.array(vec2, dtype=np.float64) / np.linalg.norm(vec2)).reshape(3)
        v, c, s = np.cross(a, b), np.dot(a, b), np.linalg.norm(np.cross(a, b))
        K = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        return np.eye(3) + K + np.dot(K, K) * ((1 - c) / (s ** 2))  # rotation matrix

    print(f'\t\ttransforming {len(particles["ParticleIDs"])} particles/cells to disk frame')

    H = Halo(run, halo, snap)
    L = H.L_star(verbose=verbose)
    # gets the rotation matrix to orient the L of the halo in the +z direction
    verbose and print(f'\t\t\tL_star : ({L[0]:3f}, {L[1]:3f}, {L[2]:3f}) --> {tuple((0, 0, 1))}')
    rot_matrix = rotationMatrix(np.array(L), np.array([0, 0, 1]))

    if len(particles['ParticleIDs']) == 1:
        coords = particles['Coordinates'] - H.pos
        vels = particles['Velocities'] - H.vel
    else:
        coords, vels = particles['Coordinates'].to(u.kpc).value, particles['Velocities'].to(u.km / u.s).value
        for i in range(coords.shape[0]):  # iterate over rows
            coords[i] = (particles['Coordinates'][i] - H.pos).to(u.kpc).value
            vels[i] = (particles['Velocities'][i] - H.vel).to(u.km / u.s).value  # Subtract halo velocity (in km/s)

    verbose and print(f'\t\t\trotating spatial coordinates')
    rot_coords = np.dot(coords, rot_matrix.T)  # Matrix multiplication
    particles['position'] = rot_coords * u.kpc

    verbose and print(f'\t\t\trotating velocities')
    rot_vels = np.dot(vels, rot_matrix.T)
    particles['velocity'] = rot_vels * u.km / u.s

    verbose and print(f'\t\t\tcomputing angular momenta')
    if len(particles['ParticleIDs']) == 1:
        angularMomenta = np.cross(rot_coords, rot_vels * particles['Masses'].to(u.M_sun).value)
    else:
        angularMomenta = particles['Velocities'].value
        for i in range(particles['Masses'].shape[0]):
            angularMomenta[i] = np.cross(rot_coords[i], rot_vels[i] * particles['Masses'][i].to(u.M_sun).value)
    particles['angularMomentum'] = angularMomenta * u.M_sun * u.kpc * u.km / u.s  # units of j ~ M_solar kpc km s^-1

    return particles


def calc_distanceHalo(run, snaps, subject_halo, reference_halo, verbose=False):
    """
    similar to calc_distanceDisk(), but uses halo positions directly from AHF_halos output file
    (avoids expensive halo reference frame calculations)
    :param sim_run, snaps, subject_halo: passing arguments
    :param reference_halo: halo of interest to calculate the distance to
    :return: distance between halos [kpc]
    """
    starting_snap = snaps[0]
    ending_snap = snaps[1]
    start_idx = 307 if run == '09_18_lastgigyear' else 127

    dist_halo = np.zeros(ending_snap - starting_snap)
    for i in range(ending_snap, starting_snap, -1):
        sh_pos = Halo(run, subject_halo, i).pos  # in kpc
        rh_pos = Halo(run, reference_halo, i).pos  # in kpc
        dist_halo[start_idx - i] = np.linalg.norm(sh_pos - rh_pos)

    verbose and print(f'\t\tdist_halo : {dist_halo}')
    return dist_halo


def cartesian_toSpherical(coords, vels=None, return_deg=False):
    """
    converts cartesian coordinates to spherical (in kpc, km/s, rad, rad/s)
    :param coords:  cartesian particle coordinates (in kpc), i.e. [\vec{x_i}, ..., \vec{x_N}]
    :param vels:  cartesian particle velocities (in km/s), i.e. [\vec{v_i}, ..., \vec{v_N}]
    :param return_deg: option to return angles in degrees (as oppossed to radians)
    :return:  spherical particle coordinates and velocities; (r, theta, phi), (v_r, v_theta, v_phi)
    """
    # Ensure coords and vels are at least 2D arrays (N,3)
    coords = np.atleast_2d(coords)
    if vels is not None:
        vels = np.atleast_2d(vels)

    # Cartesian coordinates
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # Spherical positions
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arcsin(z / r)  # polar angle (up from xy-plane)
    phi = np.arctan2(y, x)
    if return_deg:
        theta, phi = np.degrees(theta), np.degrees(phi)

    sph_coords = np.stack((r, theta, phi), axis=1)

    # If no velocities, return positions only
    if vels is None:
        return sph_coords.squeeze()

    # Cartesian velocities
    vx, vy, vz = vels[:, 0], vels[:, 1], vels[:, 2]
    # Basis vectors
    e_r = np.stack((np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)), axis=1)
    e_theta = np.stack((np.cos(theta) * np.cos(phi),
                        np.cos(theta) * np.sin(phi),
                        -np.sin(theta)), axis=1)
    e_phi = np.stack((-np.sin(phi),
                      np.cos(phi),
                      np.zeros_like(phi)), axis=1)

    # Project velocities onto spherical basis
    v_cart = np.stack((vx, vy, vz), axis=1)
    v_r = np.sum(v_cart * e_r, axis=1)
    v_theta = np.sum(v_cart * e_theta, axis=1)
    v_phi = np.sum(v_cart * e_phi, axis=1)
    sph_vels = np.stack((v_r, v_theta, v_phi), axis=1)

    return sph_coords.squeeze(), sph_vels.squeeze()  # Return both, squeezed back if N = 1


def get_softeningLength(run, snap, part_type):
    # not valid for gas (PartType0), uses adaptive softening lengths
    param_filePath = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/'
                      f'output' + ('_2x2.5Mpc' if run != '09_18_lastgigyear' else '') + '/parameters-usedvalues')
    z = get_redshift(run, snap)
    a = 1 / (1 + z)
    h = 0.677

    with open(param_filePath, 'r') as f:
        contents = f.read()
        # obtains softening particle type reference for desired part_type
        for line in contents.splitlines():
            try:
                if str(line[:24]) == f'SofteningTypeOfPartType{part_type[-1]}':
                    part_index = int(line[-1])
            except IndexError:
                pass

        for line in contents.splitlines():
            try:
                if str(line[:22]) == f'SofteningComovingType{part_index}':
                    epsilon_comoving = float(line[-8:]) * 1e3 / h  # in ckpc
                if str(line[:21]) == f'SofteningMaxPhysType{part_index}':
                    epsilon_phys = float(line[-8:]) * 1e3 / h  # in pkpc
            except IndexError:
                pass
            except NameError:
                print(f'Error: could not find particle reference type #SofteningTypeOf{part_type}!')
                exit(1)
    try:
        epsilon_minPhys = np.min(np.array([a * epsilon_comoving, epsilon_phys]))
    except NameError:
        print(f'Error: could not find softening lengths in {param_filePath}!')
        exit(1)

    return epsilon_minPhys / a  # in ckpc


def sphProjection(particles, param, axs, bins, bounds, v=False):
    from scipy.ndimage import gaussian_filter
    di = ((bounds[1] - bounds[0]) / bins[0])  # in kpc
    dj = ((bounds[3] - bounds[2]) / bins[1])  # in kpc
    da = di * dj  # kpc^2

    if particles.part_type == 'PartType0':
        # --------------------------------
        num_percentiles = 20
        # --------------------------------
        cell_radii = (3 / (4 * np.pi) * (particles.Masses / particles.Density) ** (1 / 3)).to(u.kpc).value

        img_smoothed, norm_smoothed = np.zeros((bins[0], bins[0])), np.zeros((bins[0], bins[0]))
        sigma_bins = cell_radii / di
        sigma_levels = np.percentile(sigma_bins, np.linspace(0, 100, num_percentiles))
        for i in range(len(sigma_levels) - 1):
            mask = (sigma_bins >= sigma_levels[i]) & (sigma_bins < sigma_levels[i + 1])
            layer_w, i_e, j_e = np.histogram2d(particles.position[mask, axs[0]].to(u.kpc).value,
                                               particles.position[mask, axs[1]].to(u.kpc).value,
                                               bins=bins, range=np.array(bounds).reshape((2, 2)),
                                               weights=(particles.weights.value * particles.Masses.value)[mask])
            layer_n, _, _ = np.histogram2d(particles.position[mask, axs[0]].to(u.kpc).value,
                                           particles.position[mask, axs[1]].to(u.kpc).value,
                                           bins=bins, range=np.array(bounds).reshape((2, 2)),
                                           weights=particles.Masses.value[mask])

            sigma = np.median(sigma_bins[mask])
            img_smoothed += gaussian_filter(layer_w, sigma=sigma)
            norm_smoothed += gaussian_filter(layer_n, sigma=sigma)

        img = np.divide(img_smoothed, norm_smoothed)

        v and print(f'\t\t\tcomputed volume densities for adaptively smoothed particles : '
                    f'\\mu_epsilon ~ {np.mean(cell_radii):2f} kpc\n'
                    f'\t\t\t\\mu_param : {np.average(img):1e} +/- {np.std(img):1e} {particles.weights.unit}')

    else:  # stars or dms, surface density, global smoothing

        epsilon = get_softeningLength(particles.run, particles.snap, particles.part_type)
        sigma = epsilon / di  # convert to bin units

        img_w, i_e, j_e = np.histogram2d(particles.position[:, axs[0]].to(u.kpc).value,
                                         particles.position[:, axs[1]].to(u.kpc).value,
                                         bins=bins, range=np.array(bounds).reshape((2, 2)),
                                         weights=particles.weights.value * particles.Masses.value)
        img_n, _, _ = np.histogram2d(particles.position[:, axs[0]].to(u.kpc).value,
                                     particles.position[:, axs[1]].to(u.kpc).value,
                                     bins=bins, range=np.array(bounds).reshape((2, 2)),
                                     weights=particles.Masses.value)

        img_w = gaussian_filter(img_w, sigma=sigma / 2)
        img_n = gaussian_filter(img_n, sigma=sigma / 2)

        img_w[img_w == 0] = particles.background
        img_n[img_n == 0] = 1

        if param == 'surfaceDen':
            img = img_w / da  # M_sol / kpc ^ 2
            unit = 'M_sol / kpc ^ 2'

        elif param == 'surfaceBrightness':
            img = -2.5 * np.log10(img_w / da)
            unit = 'mag / kpc ^ 2'

        elif param == 'metallicity':
            from .astrometry import Measurements
            sun = Measurements('Sun')
            img = np.log10(img_w / img_n) - np.log10(sun.Z / sun.X)  # [Z/H] - [Z/H]_sol
            unit = 'dex'

        elif param == 'Fe_H':
            from .astrometry import Measurements
            sun = Measurements('Sun')
            img = np.log10(img_w / img_n) - np.log10(sun.Fe / sun.X)  # [Fe/H] - [Fe/H]_sol
            unit = 'dex'

        elif param == 'alpha_Fe':
            from .astrometry import Measurements
            sun = Measurements('Sun')
            sun_alpha = sun.Ox + sun.Ne + sun.Mg + sun.Si
            img = np.log10(img_w / img_n) - np.log10(sun_alpha / sun.Fe)  # [alpha/Fe] - [alpha/Fe]_sol
            unit = 'dex'

        else:
            print(f'Error: {param} is an invalid param for util/hestia.py/spHProjection; '
                  f'line {inspect.currentframe().f_lineno}')
            exit(1)

        v and print(f'\t\t\tcomputed surface densities for globally smoothed particles : epsilon ~ {epsilon:2f} kpc\n'
                    f'\t\t\t\\mu_param : {np.average(img):1e} +/- {np.std(img):1e} {unit}')

    return img, i_e, j_e


def compute_barParams(run, halo, snap, bins=200, v=True):
    """
    :return:  positions angle, a, b
    """
    from scipy.ndimage import gaussian_filter
    from photutils.isophote import EllipseGeometry, Ellipse
    from photutils.aperture import EllipticalAperture
    band_dict = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
    band = 'V'

    stars = Particles(run, halo, snap, part_type='PartType4')
    fluxes = np.power(10, -0.4 * stars.GFM_StellarPhotometrics[:, band_dict[band]]) * u.dimensionless_unscaled

    bounds = (-10, 10)
    dx = (bounds[1] - bounds[0]) / bins  # kpc / pixel

    epsilon = get_softeningLength(run, snap, 'PartType4')
    sigma = epsilon / dx  # convert to bin units

    img_w, i_e, j_e = np.histogram2d(stars.position[:, 0].to(u.kpc).value,
                                     stars.position[:, 1].to(u.kpc).value,
                                     bins=bins, range=np.array([[bounds[0], bounds[1]], [bounds[0], bounds[1]]]),
                                     weights=fluxes)
    img_w[img_w == 0] = 1e-10
    img = gaussian_filter(img_w.T, sigma=sigma / 2)

    # Define initial ellipse geometry
    geometry = EllipseGeometry(x0=bins / 2, y0=bins / 2, sma=2 / dx,  # initial guess in pixels units
                               eps=0.5, pa=3 * np.pi / 4)
    ellipse = Ellipse(img, geometry)
    isolist = ellipse.fit_image(minsma=2 * epsilon / dx, maxsma=4 / dx, step=0.1)

    # Print fitted parameters at each isophote
    avg_pas = np.array([])
    v and print(f'\tellipse isofit returned parameters:')
    for iso in isolist:
        # Define elliptical aperture from fitted parameters
        aperture = EllipticalAperture(
            positions=(iso.x0, iso.y0),
            a=iso.sma,
            b=iso.sma * (1 - iso.eps),
            theta=iso.pa
        )
        # Create mask and extract pixels
        mask = aperture.to_mask(method='center')
        img_cutout = mask.multiply(img)
        pixels_inside = img_cutout[img_cutout != 0]
        mean_intensity = np.mean(pixels_inside)

        v and print(f'\t\ta : {iso.sma * dx * u.kpc:.2f}, e : {iso.eps:.3f}, theta : {np.degrees(iso.pa) * u.deg:.2f}, '
                    f'(x,y)_0 : {np.round(np.array([iso.x0, iso.y0]) * dx + bounds[0], 3) * u.kpc}, '
                    f'I: {mean_intensity:.4e}')

        # come up wiht a better way to find the correct isophote later
        if round(iso.sma * dx, 1) == 3.5:
            iso0 = iso

    return {
        'sma': iso0.sma * dx * u.kpc,
        'smi': np.sqrt((iso0.sma * dx) ** 2 * (1 - iso0.eps ** 2)) * u.kpc,
        'eps': iso0.eps,
        'pa': iso0.pa * u.rad,
        'x0': (iso0.x0 * dx + bounds[0]) * u.kpc,
        'y0': (iso0.y0 * dx + bounds[0]) * u.kpc
    }



