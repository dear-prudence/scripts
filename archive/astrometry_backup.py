from __future__ import division
import numpy as np
from archive.hestia.geometry import cartesian_toSpherical, transform_haloFrame
from numpy import sin, cos


class Measurements:
    def __init__(self):
        # (ra, dec) all in J2000 epoch unless otherwise specified
        self._data = {
            'LMC': {
                'bar': {  # stellar dynamical (completeness-corrected, red clump stars)
                    # Rathore+2025 (https://iopscience.iop.org/article/10.3847/1538-4357/ad93ae/pdf)
                    'ra': 80.27,
                    'dec': -69.65,
                    'sigma_ra': 0.02,
                    'sigma_dec': 0.02,
                    'inclination': 25.86,
                    'nodes': 121.26,
                    'distance': 49.9,
                    'R_bar': 2.13,  # kpc
                    'axisRatio': 0.54
                },
                'pm': {  # stellar kinematical center;
                    # Choi+2022 (https://iopscience.iop.org/article/10.3847/1538-4357/ac4e90/pdf)
                    'ra': 80.443,
                    'dec': -69.272,
                    'inclination': 23.396,
                    'nodes': 138.856,
                    'distance': 50.1  # Freedman+2001
                },
                'HI': {
                    'ra': 79.25,  # Kim+1998 (https://iopscience.iop.org/article/10.1086/306030/pdf)
                    'dec': -69.03,
                    'sigma_ra': 0.08,  # taken from pixel size of 20"
                    'sigma_dec': 0.08
                },
                'photometric': {  # vanDerMarel+2001 (iopscience.iop.org/article/10.1086/323100/pdf for (ra, dec),
                    # https://iopscience.iop.org/article/10.1086/323099/pdf for (i, theta))
                    'ra': 82.25,
                    'dec': -69.5,
                    'inclination': 34.7,  # +- 6.2
                    'nodes': 122.5,  # +- 8.3
                },
                'disk': {  # Kacharov+2024 (https://www.aanda.org/articles/aa/pdf/2024/12/aa51578-24.pdf)
                    # jeans modeling, gaia dr3
                    'ra': 80.29,
                    'dec': -69.25,
                    'inclination': 25.5,
                    'sigma_i': 0.2,
                    'nodes': 124,
                    'sigma_nodes': 0.4,
                    'distance': 49.9,  # adopted
                    'axisRatio': 0.23
                },
                'HVSs': {  # Lucchini+2025 (https://iopscience.iop.org/article/10.3847/2041-8213/ae109d/pdf)
                    'ra': 80.72,
                    'dec': -67.79,
                    'sigma_ra': 0.44,
                    'sigma_dec': 0.80
                },
                'cepheids': {  # Bhuyan+2024, disk of LMC from cepheid light curves
                    'ra': 80.78,  # Nikolaev+2004
                    'dec': -69.03,
                    'distance': 49.59,  # Pietrzyński+2019
                    'inclination': 22.87,
                    'nodes': 154.76
                },
                'nasa': {
                    'ra': 80.894167,
                    'dec': -69.756111,
                    'distance': 49.59,  # Pietrzyński+2019
                    'inclination': 25.86,
                    'nodes': 149.23,
                }
            },
            'SMC': {
                'stars': {  # Graczyk+2020
                    # 'ra': 13.052083,
                    'ra': 12.54,  # +- 0.30
                    'dec': -73.11,  # +- 0.15
                    'distance': 62.44
                },
                'nasa': {
                    'ra': 13.186667,
                    'dec': -72.828611,
                    'distance': 62.44  # Graczyk+2020
                }
            },
            'MW': {
                'SagA*': {  # https://ui.adsabs.harvard.edu/abs/2023AJ....165...49G/abstract
                    'ra': 266.4168,
                    'dec': -29.0078,
                    'distance': 8.178  # https://ui.adsabs.harvard.edu/abs/2019A%26A...625L..10G/abstract
                }
            }
        }

    def __getitem__(self, path: str):
        """Allow lookup using a string path, e.g. m['nasa/LMC/ra']"""
        keys = path.split('/')
        current = self._data
        for k in keys:
            try:
                current = current[k]
            except KeyError:
                raise KeyError(f"Invalid path: {'/'.join(keys)}") from None
        return current

    def keys(self, path: str = ""):
        """List available keys at a given level, e.g. m.keys('nasa/LMC')."""
        current = self._data
        if path:
            for k in path.split('/'):
                current = current[k]
        if isinstance(current, dict):
            return list(current.keys())
        return []  # terminal value, no subkeys


def deg(x): return np.degrees(x)

def rad(x): return np.radians(x)

def Ref_x(): return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

def Ref_z(): return np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

def R_x(a): return np.array([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])

def R_y(a): return np.array([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])

def R_z(a): return np.array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])


def get_smc(database='nasa', verbose=True):
    """
    Compute the SMC position and the SMC-LMC relative velocity in the LMC-disk frame
    where the LMC disk lies in the x-y plane and the disk angular-momentum axis is +z.
    """

    verbose and print('\tcomputing SMC kinemtics in LMC rest frame using the following qunatities...')

    m = Measurements()

    ra_lmc, dec_lmc, i, theta, d_lmc = (np.radians(m[f'LMC/{database}/ra']),
                                        np.radians(m[f'LMC/{database}/dec']),
                                        np.radians(m[f'LMC/{database}/inclination']),
                                        -np.radians(90) + np.radians(m[f'LMC/{database}/nodes']),
                                        m[f'LMC/{database}/distance'])  # in kpc
    verbose and print(f'\t\t(ra, dec)_lmc : ({np.degrees(ra_lmc)}, {np.degrees(dec_lmc)}) deg\n'
                      f'\t\t\ti_lmc : {np.degrees(i)} deg\n'
                      f'\t\t\ttheta_lmc : {90 + np.degrees(theta):.2f} deg\n'
                      f'\t\t\tdist_lmc : {d_lmc} kpc')

    ra_smc, dec_smc, d_smc = (np.radians(m[f'SMC/{database}/ra']),
                              np.radians(m[f'SMC/{database}/dec']),
                              m[f'SMC/{database}/distance'])  # in kpc
    verbose and print(f'\t\t(ra, dec)_smc : ({np.degrees(ra_smc)}, {np.degrees(dec_smc)}) deg\n'
                      f'\t\t\tdist_smc : {d_smc} kpc')

    # cartesian conversion of (ra, dec) coordiantes for the smc
    x_smc = np.array([
        d_smc * np.cos(dec_smc) * np.cos(ra_smc),
        d_smc * np.cos(dec_smc) * np.sin(ra_smc),
        d_smc * np.sin(dec_smc)
    ])

    x_smc = R_y(dec_lmc) @ R_z(-1 * ra_lmc) @ x_smc
    x_smc[0] -= d_lmc
    verbose and print(f'\t\trotated SMC into LMC-centered sky projection,\n'
                      f'\t\t\tx_smc^lmc : ({x_smc[0]:.2f}, {x_smc[1]:.2f}, {x_smc[2]:.2f}) kpc')

    x_smc = R_y(-i) @ R_x(theta) @ x_smc  # x is los, y is left, z is up
    x_smc = R_y(np.radians(90)) @ x_smc
    verbose and print(f'\t\trotated SMC into LMC static rest frame,\n'
                      f'\t\t\tx_smc^lmc : ({x_smc[0]:.2f}, {x_smc[1]:.2f}, {x_smc[2]:.2f}) kpc\n'
                      f'\t\t--> |x_smc^lmc| ~ d_sep : {np.linalg.norm(x_smc):.2f} kpc')

    return x_smc  # halo frame


def vrai_frame(run, particles, snap, frame='radec', tracer='bar', halo='halo_08',
               inclination=None, nodes=None, verbose=True):
    from numpy import sin, cos, arcsin, arctan2
    m = Measurements()
    h = 0.677

    def wrap_ra(a): return np.mod(a, 2 * np.pi)

    def radec_to_unit(alpha, delta):
        return np.array([
            cos(delta) * cos(alpha),
            cos(delta) * sin(alpha),
            sin(delta)
        ])
        # return np.vstack((x, y, z)).T  # returns (N,3) array of unit vectors

    def unit_to_radec(vec):
        x, y, z = vec[0], vec[1], vec[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        x, y, z = x / r, y / r, z / r

        alpha = arctan2(y, x)
        delta = arcsin(z)

        alpha = wrap_ra(alpha)
        return alpha % (2*np.pi), delta

    def rotationMatrix_backup(alpha0, delta0):
        """
        Build rotation matrix R whose columns are the util basis vectors:
          e_x' = util East direction
          e_y' = util North direction
          e_z' = direction to (alpha0, delta0)
        R maps GLOBAL -> LOCAL frame.
        """
        # Local zenith (LMC center)
        ez = np.array([
            cos(delta0) * cos(alpha0),
            cos(delta0) * sin(alpha0),
            sin(delta0)
        ])

        # Local East (increasing alpha at constant delta)
        ex = np.array([-np.sin(alpha0), np.cos(alpha0), 0.0])

        # Local North = ez × ex
        ey = np.cross(ez, ex)

        # Normalize
        # ex /= np.linalg.norm(ex)
        # ey /= np.linalg.norm(ey)
        # ez /= np.linalg.norm(ez)

        # Rotation matrix: GLOBAL -> LOCAL
        R = np.column_stack((ex, ey, ez))
        return R

    def rotationMatrix(alpha0, delta0):
        """
        Build rotation matrix R whose columns are the util basis vectors:
          e_x' = util East direction
          e_y' = util North direction
          e_z' = direction to (alpha0, delta0)
        R maps GLOBAL -> LOCAL frame.
        """
        # Local zenith (LMC center)
        ez = np.array([
            cos(delta0) * cos(alpha0),
            -cos(delta0) * sin(alpha0),
            sin(delta0)
        ])

        # Local East (increasing alpha at constant delta)
        ex = np.array([-np.sin(alpha0), np.cos(alpha0), 0.0])

        # Local North = ez × ex
        ey = np.array([
            sin(delta0) * cos(alpha0),
            sin(delta0) * sin(alpha0),
            -cos(delta0)
        ])

        # Rotation matrix: GLOBAL -> LOCAL
        R = np.column_stack((ex, ey, ez))
        return R

    def radec_to_rhophi(alpha, delta, alpha0, delta0):
        """
        Convert (RA,Dec) to (rho,phi) with
        phi = North of West as in van der Marel (2001).
        """
        R = rotationMatrix(alpha0, delta0)

        v = radec_to_unit(alpha, delta)  # (N,3)
        v_local = R.T @ v  # (E, N, Z)
        E, N, Z = v_local[0], v_local[1], v_local[2]

        rho = np.arccos(Z)
        # -phi = np.arctan2(-N, E)  # flip sign to fix handedness mismatch
        phi = np.arctan2(N, E)
        return rho, phi

    def rhophi_to_radec(rho, phi, alpha0, delta0):
        """
        Inverse transform for phi = North of West.
        """
        R = rotationMatrix(alpha0, delta0)

        E = -np.sin(rho) * np.cos(phi)
        N = -np.sin(rho) * np.sin(phi)
        Z = np.cos(rho)

        # v_local = np.vstack((E, N, Z))
        v_local = np.array([
            sin(rho) * cos(phi),
            sin(rho) * sin(phi),
            cos(rho)
        ])
        v_global = R @ v_local

        alpha, delta = unit_to_radec(v_global)
        return alpha, delta

    def Spherical(coords):
        x, y, z = coords[0], coords[1], coords[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arcsin(z / r)  # polar angle (up from xy-plane)
        phi = np.arctan2(y, x)

        return np.array([r, theta, phi])

    def Cartesian(coords):
        r, theta, phi = coords[0], coords[1], coords[2]
        x = r * cos(theta) * cos(phi)
        y = r * cos(theta) * sin(phi)
        z = r * sin(theta)

        return np.array([x, y, z])

    def rho_phi_backup(pos):
        d_lmc = m[f'LMC/{tracer}/distance']
        D = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + (d_lmc - pos[2]) ** 2)
        rho = np.arccos((d_lmc - pos[2]) / D)
        phi = np.arctan2(pos[1], pos[0])

        return np.array([D, rho, phi])

    def rho_phi(pos):
        # going from post-inclined (projected) cartesian position to rho,phi
        d_lmc = m[f'LMC/{tracer}/distance']
        D = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + (d_lmc - pos[2]) ** 2)
        rho = np.arccos((d_lmc - pos[2]) / D)
        phi = np.arctan2(pos[1], pos[0])

        return np.array([D, rho, phi])

    def get_mw(database='nasa', verbose=True):
        """
        Compute the MW position  in the LMC-disk frame
        where the LMC disk lies in the x-y plane and the disk angular-momentum axis is +z.
        """

        verbose and print('\tcomputing MW in LMC rest frame using the following quantities...')

        m = Measurements()

        lmc_ra, lmc_dec, i, theta, d_lmc = (np.radians(m[f'LMC/{database}/ra']),
                                            np.radians(m[f'LMC/{database}/dec']),
                                            np.radians(m[f'LMC/{database}/inclination']),
                                            np.radians(90) + np.radians(m[f'LMC/{database}/nodes']),
                                            m[f'LMC/{database}/distance'])  # in kpc

        i = rad(inclination) if inclination is not None else i
        theta = rad(90) + rad(nodes) if nodes is not None else theta

        verbose and print(f'\t\t(ra, dec)_lmc : ({np.degrees(lmc_ra)}, {np.degrees(lmc_dec)}) deg\n'
                          f'\t\t\ti_lmc : {np.degrees(i)} deg\n'
                          f'\t\t\ttheta_lmc : {-90 + np.degrees(theta):.2f} deg\n'
                          f'\t\t\tdist_lmc : {d_lmc} kpc')

        ra_mw, dec_mw, d_mw = (np.radians(m[f'MW/SagA*/ra']),  # Sag A* from IAU (def of galactic coords)
                               np.radians(m[f'MW/SagA*/dec']),
                               m[f'MW/SagA*/distance'])  # in kpc
        verbose and print(f'\t\t(ra, dec)_mw : ({np.degrees(ra_mw)}, {np.degrees(dec_mw)}) deg\n'
                          f'\t\t\tdist_mw : {d_mw} kpc')

        Rho, Phi = radec_to_rhophi(ra_mw, dec_mw, lmc_ra, lmc_dec)

        verbose and print(f'\t\t\t\\vec"{{mw_rho_phi"}}" : {deg(Rho):.2f} deg, {deg(Phi):.2f} deg')

        x_mw = np.array([
            d_mw * sin(Rho) * cos(Phi),
            d_mw * sin(Rho) * sin(Phi),
            d_lmc - d_mw * cos(Rho)
        ])

        verbose and print(f'\t\t\t\\vec"{{x_mw_proj"}}" : ({x_mw[0]:.2f}, {x_mw[1]:.2f}, {x_mw[2]:.2f}) kpc')

        x_mw = R_x(i) @ R_z(-theta) @ x_mw

        verbose and print(f'\t\t\t\\vec"{{x_mw_xyz\'"}}" : ({x_mw[0]:.2f}, {x_mw[1]:.2f}, {x_mw[2]:.2f}) kpc')

        x_mw = R_x(np.radians(180)) @ x_mw

        verbose and print(f'\t\t\t\\vec"{{x_mw_CCW\'"}}" : ({x_mw[0]:.2f}, {x_mw[1]:.2f}, {x_mw[2]:.2f}) kpc')

        verbose and print(f'\t\trotated MW into LMC static rest frame,\n'
                          f'\t\t\tx_mw^lmc : ({x_mw[0]:.2f}, {x_mw[1]:.2f}, {x_mw[2]:.2f}) kpc\n'
                          f'\t\t--> |x_mw^lmc| ~ d_sep : {np.linalg.norm(x_mw):.2f} kpc')

        return x_mw  # halo frame

    if halo == 'halo_08':  # if this is the LMC-SMC analog system
        SMC_pos = get_smc(database=tracer, verbose=verbose)  # SMC position in static reference frame of LMC
        SMC_vel = (217.54, 398.34, -63.50)  # km/s
        SMC_cart = SMC_pos
        SMC_pos = cartesian_toSpherical(SMC_pos)

        if run == '09_18_lastgigyear':
            smc_filePath = ('/z/rschisholm/halos/09_18/smc/'
                            'HESTIA_100Mpc_8192_09_18_lastgigyear.127_halo_307000000001476.dat')
            row = 307 - snap
        elif run == '09_18':
            smc_filePath = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc/'
                            'HESTIA_100Mpc_8192_09_18.127_halo_127000000001384.dat')
            row = 127 - snap
        else:
            print('Error: routine only valid for 09_18 or 09_18_lastgigyear!')
            exit(1)

        smc_data = np.loadtxt(smc_filePath)
        smc_pos = np.array([smc_data[row, 6], smc_data[row, 7], smc_data[row, 8]]) / h  # in kpc
        smc_vel = np.array([smc_data[row, 9], smc_data[row, 10], smc_data[row, 11]])
        smc_mass = np.array([smc_data[row, 4]]) / h
        smc = {
            'ParticleIDs': np.ones(1),  # dummy particle id
            'Coordinates': smc_pos, 'Velocities': smc_vel, 'Masses': smc_mass
        }
        # smc = transform_haloFrame('09_18_lastgigyear', 'halo_08', 255, smc)
        if run == '09_18_lastgigyear':
            smc = transform_haloFrame('09_18_lastgigyear', 'halo_08', 255, smc)
        else:
            smc = transform_haloFrame(run, 'halo_08', snap, smc)

        # reflects (mirrors) smc-analog along x-axis to better mimic the morphology of LMC-SMC orbital history
        smc['position'] = Ref_x() @ smc['position']
        # smc['position'] = R_x(np.radians(180)) @ smc['position']

        smc_pos, smc_vel = cartesian_toSpherical(smc['position'], vels=smc['velocity'])

        d_phi = SMC_pos[2] - smc_pos[2]
        verbose and print(f'\t\tSMC_pos : ({SMC_pos[0]:.2f} kpc, '
                          f'{np.degrees(SMC_pos[1]):.2f} deg, {np.degrees(SMC_pos[2]):.2f} deg)\n'
                          f'\t\tsmc_pos : ({smc_pos[0]:.2f} kpc, '
                          f'{np.degrees(smc_pos[1]):.2f} deg, {np.degrees(smc_pos[2]):.2f} deg)\n'
                          f'\t\t\t--> delta_phi : {np.degrees(d_phi)}')

    elif halo == 'halo_41':
        MW_pos_cart = get_mw(database=tracer, verbose=verbose)  # SMC position in static reference frame of LMC
        MW_pos = Spherical(MW_pos_cart)

        if run == '09_18_lastgigyear':
            mw_filePath = ('/z/rschisholm/halos/09_18/halo_01/'
                           'HESTIA_100Mpc_8192_09_18_lastgigyear.127_halo_307000000000001.dat')
            row = 307 - snap
        elif run == '09_18':
            mw_filePath = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc/'
                           'HESTIA_100Mpc_8192_09_18.127_halo_127000000000001.dat')
            row = 127 - snap
        else:
            print('Error: routine only valid for 09_18 or 09_18_lastgigyear!')
            exit(1)

        mw_data = np.loadtxt(mw_filePath)
        mw_pos = np.array([mw_data[row, 6], mw_data[row, 7], mw_data[row, 8]]) / h  # in kpc
        mw_vel = np.array([mw_data[row, 9], mw_data[row, 10], mw_data[row, 11]])
        mw_mass = np.array([mw_data[row, 4]]) / h
        mw = {
            'ParticleIDs': np.ones(1),  # dummy particle id
            'Coordinates': mw_pos, 'Velocities': mw_vel, 'Masses': mw_mass
        }
        if run == '09_18_lastgigyear':
            mw = transform_haloFrame('09_18_lastgigyear', 'halo_41', snap, mw)
        else:
            mw = transform_haloFrame(run, 'halo_41', snap, mw)

        mw_pos = Spherical(mw['position'])

        d_phi = MW_pos[2] - mw_pos[2]
        verbose and print(f'\t\tMW_pos : ({MW_pos[0]:.2f} kpc, '
                          f'{np.degrees(MW_pos[1]):.2f} deg, {np.degrees(MW_pos[2]):.2f} deg)\n'
                          f'\t\tmw_pos : ({mw_pos[0]:.2f} kpc, '
                          f'{np.degrees(mw_pos[1]):.2f} deg, {np.degrees(mw_pos[2]):.2f} deg)\n'
                          f'\t\t\t--> delta_phi : {np.degrees(d_phi)}')

        verbose and print(f'\t\t\t\\vec"{{MW_pos_CCW\'"}}" '
                          f': {MW_pos[0]:.2f} kpc, ({deg(MW_pos[1]):.2f}, {deg(MW_pos[2]):.2f}) deg')

        MW_pos = R_x(rad(180)) @ Cartesian(MW_pos)

        verbose and print(f'\t\t\t\\vec"{{MW_pos_CW\'"}}" : ({MW_pos[0]:.2f}, {MW_pos[1]:.2f}, {MW_pos[2]:.2f}) kpc')

        i, theta = (rad(m[f'LMC/{tracer}/inclination']),
                    rad(90) + rad(m[f'LMC/{tracer}/nodes']))

        i = rad(inclination) if inclination is not None else i
        theta = rad(90) + rad(nodes) if nodes is not None else theta

        MW_pos = R_z(theta) @ R_x(-i) @ MW_pos

        verbose and print(f'\t\t\t\\vec"{{mw_pos_proj\'"}}" : ({MW_pos[0]:.2f}, {MW_pos[1]:.2f}, {MW_pos[2]:.2f}) kpc')

        MW_pos = rho_phi(MW_pos)

        verbose and print(f'\t\t\t\\vec"{{mw_ang_proj\'"}}" : {MW_pos[0]:.2f} kpc, '
                          f'{deg(MW_pos[1]):.2f} deg, {deg(MW_pos[2]):.2f} deg')

    else:  # if a galaxy other than the LMC-SMC analog system
        d_phi = 0

    # if particles[position].ndim == 1:
    #     particles[position] = particles[position][np.newaxis, :]
    # particles[position] = particles[position].T
    position = particles["position"][0]

    t = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    a = float(m['LMC/bar/R_bar']) * cos(t)
    b = float(m['LMC/bar/R_bar']) * float(m['LMC/bar/axisRatio']) * sin(t)
    bar = np.vstack((a, b, np.zeros(a.shape)))  # column vectors
    print(f'bar : {bar}')

    verbose and print(f'\t\trotating particles/cells to {frame} frame...\n'
                      f'\t\te.g...\t\\vec"{{"x_init"}}" : ({position[0]:.2f}, '
                      f'{position[1]:.2f}, {position[2]:.2f}) kpc')

    ali_pos = R_z(d_phi) @ position
    verbose and print(f'\t\t\t\\vec"{{"x_aligned"}}" : ({ali_pos[0]:.2f}, '
                      f'{ali_pos[1]:.2f}, {ali_pos[2]:.2f}) kpc')
    ali_bar = R_z(d_phi) @ bar

    CW_pos = R_x(rad(180)) @ ali_pos
    verbose and print(f'\t\t\t\\vec"{{"x_CW"}}" : ({CW_pos[0]:.2f}, '
                      f'{CW_pos[1]:.2f}, {CW_pos[2]:.2f}) kpc')
    CW_bar = R_x(rad(180)) @ ali_bar
    # CW_bar = ali_bar

    i, theta = (rad(m[f'LMC/{tracer}/inclination']),
                rad(90) + rad(m[f'LMC/{tracer}/nodes']))

    i = rad(inclination) if inclination is not None else i
    theta = rad(90) + rad(nodes) if nodes is not None else theta

    proj_pos = R_z(theta) @ R_x(-i) @ CW_pos
    proj_bar = R_z(theta) @ R_x(-i) @ CW_bar
    print(f'proj_bar : {proj_bar}')

    verbose and print(f'\t\t\t\\vec"{{"x_proj"}}" : ({proj_pos[0]:.2f}, '
                      f'{proj_pos[1]:.2f}, {proj_pos[2]:.2f}) kpc')

    proj_ang = rho_phi(proj_pos)
    print(f'proj_bar.shape : {proj_bar.shape}')
    proj_angBar = np.zeros(proj_bar.shape)
    for j in range(len(proj_angBar[0])):
        proj_angBar[:, j] = rho_phi(proj_bar[:, j])
    print(f'proj_angBar(rho, phi) : {deg(proj_angBar[1]), deg(proj_angBar[2])}')

    verbose and print(f'\t\t\t\\vec"{{"r_proj"}}" : {proj_ang[0]:.2f} kpc, '
                      f'{deg(proj_ang)[1]:.2f} deg, {deg(proj_pos)[2]:.2f} deg')

    if frame == 'radec':
        lmc_ra, lmc_dec = np.radians(m[f'LMC/{tracer}/ra']), np.radians(m[f'LMC/{tracer}/dec'])

        # delta = np.arcsin(sin(lmc_dec) * cos(ang[1]) + cos(lmc_dec) * sin(ang[1]) * sin(ang[2]))
        # alpha = np.arcsin(-sin(ang[1]) * cos(ang[2]) / cos(delta)) + lmc_ra

        rho, phi = radec_to_rhophi(rad(m['MW/SagA*/ra']), rad(m['MW/SagA*/dec']), lmc_ra, lmc_dec)
        print(f'original: {m["MW/SagA*/ra"]} deg, {m["MW/SagA*/dec"]} deg')
        print(f'forward: {deg(rho)} deg, {deg(phi)} deg')
        ra, dec = rhophi_to_radec(rho, phi, lmc_ra, lmc_dec)
        print(f'back: {deg(ra)} deg, {deg(dec)} deg')

        rho, phi = radec_to_rhophi(rad(m['LMC/pm/ra']), rad(m['LMC/pm/dec']), lmc_ra, lmc_dec)
        print(f'original: {m["LMC/pm/ra"]} deg, {m["LMC/pm/dec"]} deg')
        print(f'forward: {deg(rho)} deg, {deg(phi)} deg')
        ra, dec = rhophi_to_radec(rho, phi, lmc_ra, lmc_dec)
        print(f'back: {deg(ra)} deg, {deg(dec)} deg')

        MW_pos[2], MW_pos[1] = rhophi_to_radec(MW_pos[1], MW_pos[2], lmc_ra, lmc_dec)

        # R = get_rotationMatrix(nrp, np.array([1, 0, 0]))
        # print(f'R @ nrp : {R @ nrp}')
        # MW_pos = R @ MW_pos

        # verbose and print(f'\t\t\t\\vec"{{mw_radec_cart\'"}}" : {MW_pos[0]:.2f} kpc, '
        #                   f'{MW_pos[1]:.2f} kpc, {MW_pos[2]:.2f} kpc')

        # MW_pos = Spherical(MW_pos)

        verbose and print(f'\t\t\t\\vec"{{mw_radec\'"}}" : {MW_pos[0]:.2f} kpc, '
                          f'{deg(MW_pos[1]):.2f} deg, {deg(MW_pos[2]):.2f} deg')

        sky_coords = np.zeros(proj_ang.shape)
        sky_coords[0] = proj_ang[0]
        sky_coords[2], sky_coords[1] = rhophi_to_radec(proj_ang[1], proj_ang[2], lmc_ra, lmc_dec)

        sky_bar = np.zeros(proj_angBar.shape)
        for j in range(len(sky_bar[1])):
            sky_bar[0, j] = proj_angBar[0, j]
            sky_bar[2, j], sky_bar[1, j] = rhophi_to_radec(proj_angBar[1, j], proj_angBar[2, j],
                                                           rad(m['LMC/bar/ra']), rad(m['LMC/bar/dec']))
            sky_bar[2, j], sky_bar[1, j] = deg(sky_bar[2, j]), deg(sky_bar[1, j])
        print(f'sky_bar (ra, dec): {sky_bar[2]}, {sky_bar[[1]]}')

        verbose and print(f'\t\t\t\\vec"{{"r_sky"}}" : ({sky_coords[0]:.2f} kpc, '
                          f'{deg(sky_coords[1]):.2f} deg, {deg(sky_coords[2]):.2f} deg)')

        particles['radec'] = sky_coords[np.newaxis, :]
        particles['bar'] = sky_bar

        proj_ang[1], proj_ang[2] = radec_to_rhophi(rad(sky_coords[2]), rad(sky_coords[1]), lmc_ra, lmc_dec)
        verbose and print(f'\t\t\t\\vec"{{"proj_ang_back"}}" : {proj_ang[0]:.2f} kpc, '
                          f'{deg(proj_ang)[1]:.2f} deg, {deg(proj_pos)[2]:.2f} deg')

    elif frame == 'mag':
        sky_coords = cartesian_toSpherical(projected_positions, return_deg=True)  # (r, l, b)
        sky_coords = sky_coords[np.newaxis, :] if sky_coords.ndim == 1 else sky_coords
        verbose and print(f'\t\t\t\\vec"{{"r_sky"}}" : ({sky_coords[0, 0]:.2f} kpc, '
                          f'{sky_coords[0, 1]:.2f} deg, {sky_coords[0, 2]:.2f} deg)')

        particles['mag'] = sky_coords
    else:
        print('Error: invalid sky frame! (e.g. radec, Mag, ...)')
        exit(1)

    return particles


def compute_LMCanalog_snapshot(propsectiveSnaps):
    from .geometry import transform_haloFrame, cartesian_toSpherical
    # -----------------------------------------
    SMC_pos = (3.40, -15.87, 17.46)  # kpc, SMC position in static reference frame of LMC
    SMC_vel = (-27.31, -101.73, 46.54)  # kpc, SMC velocity in static reference frame of LMC
    # -----------------------------------------
    smc_filePath = '/z/rschisholm/halos/09_18/smc/HESTIA_100Mpc_8192_09_18_lastgigyear.127_halo_307000000001476.dat'
    smc_data = np.loadtxt(smc_filePath)
    h = 0.677

    for snap in range(propsectiveSnaps[1], propsectiveSnaps[0], -1):
        row = 307 - snap
        smc_pos = np.array([smc_data[row, 6], smc_data[row, 7], smc_data[row, 8]]) / h  # in kpc
        smc_vel = np.array([smc_data[row, 9], smc_data[row, 10], smc_data[row, 11]])  # in km/s
        smc_mass = smc_data[row, 4] / h  # in M_solar

        smc = {
            'ParticleIDs': np.ones(1),  # dummy particle id
            'Coordinates': smc_pos,
            'Velocities': smc_vel,
            'Masses': smc_mass
        }

        spherical_frame = transform_haloFrame('09_18_lastgigyear', 'halo_08', snap, smc)
        smc_spherical_coords, smc_spherical_vels = cartesian_toSpherical(spherical_frame['position'],
                                                                         spherical_frame['velocity'])
        if snap == propsectiveSnaps[1]:
            smc_sph = {
                'snaps': snap,
                'r_i': smc_spherical_coords,  # r_i = (r [in kpc], theta [in rad], phi [in rad])
                'v_i': smc_spherical_vels  # v_i = (v_r [in km/s], v_theta [in rad/s], v_phi [in rad/s])
            }
        else:
            smc_sph['snaps'] = np.append(smc_sph['snaps'], snap)
            smc_sph['r_i'] = np.vstack((smc_sph['r_i'], smc_spherical_coords))
            smc_sph['v_i'] = np.vstack((smc_sph['v_i'], smc_spherical_vels))

    SMC_posSph, SMC_velSph = cartesian_toSpherical(np.array(SMC_pos), np.array(SMC_vel))

    # minimizing function,
    # f = min( (r_smc - R_SMC)^2 + (r_smc * theta_smc - R_SMC * THETA_SMC)^2
    #        + (vr_smc - VR_SMC)^2 + (r_smc * vtheta_smc - R_SMC * VTHETA_SMC)^2 )
    f = (
            (smc_sph['r_i'][:, 0] - SMC_posSph[0]) ** 2
            + (smc_sph['r_i'][:, 0] * np.absolute(smc_sph['r_i'][:, 1]) - SMC_posSph[0] * SMC_posSph[1]) ** 2
            + (smc_sph['v_i'][:, 0] - SMC_velSph[0])
            + (smc_sph['r_i'][:, 0] * np.absolute(smc_sph['v_i'][:, 1]) - SMC_posSph[0] * SMC_velSph[1]) ** 2
    )
    smc_sph['f'] = f
    order = np.argsort(f)
    smc_sph['ordered_snaps'] = smc_sph['snaps'][order]
    smc_sph['ordered_f'] = smc_sph['f'][order]

    print(f'ordered_snaps = {smc_sph["ordered_snaps"]}')
    print(f'ordered_f = {smc_sph["ordered_f"]}')

    return smc_sph[f'ordered_snaps'][0]  # snapshot corresponding to minimum of f



def newton_2d(rho, phi, init_ra, init_dec, tol=1e-2, maxiter=100):
    def f(ra, dec):
        return np.arccos(cos(dec) * cos(lmc_dec) * cos(ra - lmc_ra) + sin(dec) * sin(lmc_dec)) - rho

    def g(ra, dec):
        return np.arccos(-cos(dec) * sin(ra - lmc_ra) / sin(f(ra, dec) + rho)) - phi

    def dfda(ra, dec):
        return (1 - (f(ra, dec) + rho) ** 2) ** (-1 / 2) * cos(dec) * cos(lmc_dec) * sin(ra - lmc_ra)

    def dfdd(ra, dec):
        return (-(1 - (f(ra, dec) + rho) ** 2) ** (-1 / 2)
                * (-sin(dec) * cos(lmc_dec) * cos(ra - lmc_ra) + cos(dec) * sin(lmc_dec)))

    def dgda(ra, dec):
        return (-(1 - (g(ra, dec) + phi) ** 2) ** (-1 / 2)
                * (-cos(dec) * cos(ra - lmc_ra) * csc(f(ra, dec) + rho)
                   + cos(dec) * sin(ra - lmc_ra) * csc(f(ra, dec) + rho) * cot(f(ra, dec) + rho)) * dfda(ra, dec))

    def dgdd(ra, dec):
        return (-(1 - (g(ra, dec) + phi) ** 2) ** (-1 / 2)
                * (sin(dec) * sin(ra - lmc_ra) * csc(f(ra, dec) + rho)
                   + cos(dec) * sin(ra - lmc_ra) * csc(f(ra, dec) + rho) * cot(f(ra, dec) + rho) * dfdd(ra, dec)))

    x, y = init_ra, init_dec

    for iteration in range(maxiter):
        # Evaluate functions
        F1 = f(x, y)
        print(f'F1 : {F1}')
        print(f'gish : {-cos(y) * sin(x - lmc_ra) / sin(f(x, y) + rho)}')
        F2 = g(x, y)
        # Evaluate Jacobian entries
        J11 = dfda(x, y)
        J12 = dfdd(x, y)
        J21 = dgda(x, y)
        J22 = dgdd(x, y)

        # Construct Jacobian matrix and RHS
        J = np.array([[J11, J12],
                      [J21, J22]])
        F = np.array([F1, F2])

        # Solve J * delta = F
        try:
            delta = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            raise RuntimeError("Jacobian is singular at iteration {}".format(iteration))

        # Newton update
        x_new = x - delta[0]
        y_new = y - delta[1]

        print(f'ra : {deg(x_new)}')
        print(f'dec : {deg(y_new)}')

        # Check convergence
        if np.sqrt(delta[0] ** 2 + delta[1] ** 2) < tol:
            return x_new, y_new

        x, y = x_new, y_new
