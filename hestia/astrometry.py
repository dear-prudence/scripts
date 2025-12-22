from __future__ import division
import numpy as np
from .geometry import cartesian_toSpherical, transform_haloFrame
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
                    'sigma_ra': 0.04,
                    'dec': -69.25,
                    'sigma_dec': 0.02,
                    'inclination': 25.5,
                    'sigma_i': 0.2,
                    'nodes': 124,
                    'sigma_nodes': 0.4,
                    'distance': 49.9,  # adopted
                    'axisRatio': 0.23,
                    'mu_alpha': 1.88,  # mu_alpha*
                    'mu_delta': 0.32,  # mas/yr,
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


def vrai_frame(run, particles, snap, frame, tracer, halo, MC_dict=None, bool_bar=False, verbose=True):
    from numpy import sin, cos, arcsin, arctan2
    m = Measurements()
    h = 0.677

    def print_vector(string, vector, representation):  # prints coordinates of a vector
        if representation == 'cartesian':
            print(f'\t\t\t\\vec {string} : '
                  f'({float(vector[0]):.2f}, {float(vector[1]):.2f}, {float(vector[2]):.2f}) kpc')
        elif representation == 'spherical':
            print(f'\t\t\t\\vec {string} : '
                  f'{float(vector[0]):.2f} kpc, {deg(float(vector[1])):.2f} deg, {deg(float(vector[2])):.2f} deg')
        else:
            pass

    def rotationMatrix(alpha0, delta0):
        # R : ra,dec --> projected coordinates (cos\rho = e_los, tan\phi = e_north / e_east)
        # R^-1 = R^T : projected coordinates --> ra,dec
        R = np.array([
            [-sin(alpha0), cos(alpha0), 0],
            [sin(delta0) * cos(alpha0), sin(delta0) * sin(alpha0), -cos(delta0)],
            [cos(delta0) * cos(alpha0), cos(delta0) * sin(alpha0), sin(delta0)]
        ])
        return R

    def equatorial_to_projected(v_equat, alpha0, delta0):
        alpha, delta = v_equat[1], v_equat[2]
        v = np.array([
            cos(delta) * cos(alpha),
            cos(delta) * sin(alpha),
            sin(delta)
        ])
        R = rotationMatrix(alpha0, delta0)

        v_proj = v_equat[0] * (R @ v)  # D * R @ v

        return v_proj  # E, N, LOS

    def projected_to_equatorial(v_proj, alpha0, delta0):
        R = rotationMatrix(alpha0, delta0)
        v = R.T @ v_proj  # cartesian equatorial
        los = np.linalg.norm(v)
        v_equat = np.array([
            los,
            arctan2(v[1], v[0]) % (2 * np.pi),
            arcsin((v[2] / los))
        ])
        return v_equat  # los, ra, dec

    def Spherical(coords):
        x, y, z = coords[0], coords[1], coords[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arcsin(z / r)  # polar angle (up from xy-plane)
        phi = np.arctan2(y, x)

        return np.array([r, theta, phi])

    if MC_dict is not None:
        lmc_ra, lmc_dec, i, theta, d_lmc = (rad(MC_dict['alpha']), rad(MC_dict['delta']),
                                            rad(MC_dict['inclination']), -rad(90) + rad(MC_dict['nodes']),
                                            m[f'LMC/{tracer}/distance'])  # no uncertainty in distance
    else:
        lmc_ra, lmc_dec, i, theta, d_lmc = (rad(m[f'LMC/{tracer}/ra']), rad(m[f'LMC/{tracer}/dec']),
                                            rad(m[f'LMC/{tracer}/inclination']),
                                            -rad(90) + rad(m[f'LMC/{tracer}/nodes']),
                                            m[f'LMC/{tracer}/distance'])  # in kpc

    def get_mw():  # computes MW position in static LMC disk frame
        verbose and print('\tcomputing MW in LMC rest frame using the following quantities...\n'
                          f'\t\t\t(ra, dec, i , theta)_lmc : '
                          f'({deg(lmc_ra):.2f}, {deg(lmc_dec):.2f}, {deg(i):.2f}, {deg(theta):.2f}) deg'
                          f'\n\t\t\t\tdist_lmc : {d_lmc} kpc')

        ra_MW, dec_MW, d_MW = (rad(m[f'MW/SagA*/ra']),  # Sag A* from IAU (def of galactic coords)
                               rad(m[f'MW/SagA*/dec']),
                               m[f'MW/SagA*/distance'])  # in kpc
        verbose and print(f'\t\t\t(ra, dec)_mw : ({np.degrees(ra_MW)}, {np.degrees(dec_MW)}) deg,\n'
                          f'\t\t\t\tdist_mw : {d_MW} kpc')

        # (los, ra, dec) --> projected (x,y,z)
        x_MW = equatorial_to_projected(np.array([d_MW, ra_MW, dec_MW]), lmc_ra, lmc_dec)
        verbose and print_vector('x_MW^proj (E, N, LOS)', x_MW, representation='cartesian')
        # place LMC at LMC distance
        x_MW[2] -= d_lmc
        # projected (x,y,z) --> disk frame (x',y',z')
        x_MW = R_x(i) @ R_z(theta) @ x_MW

        verbose and print_vector("x_MW^LMC (x',y',z')", x_MW, representation='cartesian')
        verbose and print(f'\t\t--> |x_MW^LMC| ~ d_sep : {np.linalg.norm(x_MW):.2f} kpc')
        return x_MW

    #  --------------- start of main body of routine ---------------

    if halo == 'halo_08':  # if this is the LMC-SMC analog system
        SMC_pos = get_smc(database=tracer, verbose=verbose)  # SMC position in static reference frame of LMC
        SMC_vel = (217.54, 398.34, -63.50)  # km/s
        SMC_cart = SMC_pos
        SMC_pos = cartesian_toSpherical(SMC_pos)
        print('Error: halo_08 module is WIP !')
        exit(1)

    elif halo == 'halo_41':
        x_MW = get_mw()  # MW position in static disk frame of LMC
        r_MW = Spherical(x_MW)

        #  AHF file path and final snapshot depending on simulation run
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

        # block to obtain mw-analog data
        mw_data = np.loadtxt(mw_filePath)
        mw = {
            'ParticleIDs': np.ones(1),  # dummy particle id
            'Coordinates': np.array([mw_data[row, 6], mw_data[row, 7], mw_data[row, 8]]) / h,  # in kpc,
            'Velocities': np.array([mw_data[row, 9], mw_data[row, 10], mw_data[row, 11]]),
            'Masses': np.array([mw_data[row, 4]]) / h
        }
        mw = transform_haloFrame(run, 'halo_41', snap, mw, verbose=False)
        r_mw = Spherical(mw['position'])

        deltaPhi = r_MW[2] - r_mw[2]
        verbose and print(f'\t\tMW_pos : ({float(r_MW[0]):.2f} kpc, '
                          f'{deg(float(r_MW[1])):.2f} deg, {deg(float(r_MW[2])):.2f} deg)\n'
                          f'\t\tmw_pos : ({r_mw[0]:.2f} kpc, '
                          f'{deg(r_mw[1]):.2f} deg, {deg(r_mw[2]):.2f} deg)\n'
                          f'\t\t\t--> delta_phi : {deg(deltaPhi)}')

        if verbose:  # if verbose, check that MW gets transformed back to its original position
            print_vector('x_MW^LMC', x_MW, representation='cartesian')

            # disk frame (x',y',z') --> projected (x,y,z)
            x_MW = R_z(-theta) @ R_x(-i) @ x_MW
            x_MW[2] += d_lmc
            print_vector('x_MW^proj (E, N, LOS)', x_MW, representation='cartesian')
            # projected (x,y,z) -- > (los, ra, dec)
            r_MW = projected_to_equatorial(x_MW, lmc_ra, lmc_dec)
            print_vector('r_MW^radec', r_MW, representation='spherical')

            # checks if roundtrip transformation matches original
            if (round(deg(r_MW[1]), 2) != round(float(m[f'MW/SagA*/ra']), 2)
                    and round(deg(r_MW[2]), 2) != round(float(m[f'MW/SagA*/dec']), 2)):
                print('Warning: roundtrip transformation for MW does not match !')

    else:  # if a galaxy other than a lmc-smc or lmc-mw analog system
        deltaPhi = 0

    print(f'\t\trotating particles/cells to {frame} frame...')
    # Currently only capable of handling one particle vector at a time !
    particle = particles['position'][0]
    verbose and print_vector('x_p^init', particle, representation='cartesian')

    # R_z(\varphi) : aligning MW and mw-analog (or SMC and smc-analog)
    ali_pos = R_z(deltaPhi) @ particle
    verbose and print_vector('x_p^aligned', ali_pos, representation='cartesian')

    # disk frame (x',y',z') --> projected (x,y,z)
    proj_pos = R_z(-theta) @ R_x(-i) @ ali_pos  # passive...?
    proj_pos[2] += m[f'LMC/{tracer}/distance']
    verbose and print_vector('x_p^projected', proj_pos, representation='cartesian')

    if bool_bar:  # if orientation of the bar is requested
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        a = float(m['LMC/bar/R_bar']) * cos(t)
        b = float(m['LMC/bar/R_bar']) * float(m['LMC/bar/axisRatio']) * sin(t)
        bar = np.vstack((a, b, np.zeros(a.shape)))  # column vectors
        print_vector('bar[:, 0]^init', bar[:, 0], representation='cartesian')

        proj_bar = np.zeros(bar.shape)
        for k in range(len(bar[0])):  # active ...?
            proj_bar[:, k] = R_z(rad(-90) + rad(m['LMC/bar/nodes'])) @ R_x(rad(m['LMC/bar/inclination'])) @ bar[:, k]
        proj_bar[2] += m[f'LMC/bar/distance']
        verbose and print_vector('bar[:, 0]^projected', proj_bar[:, 0], representation='cartesian')

    # --------------- rotations to sky frame ---------------
    if frame == 'radec':

        sky_pos = projected_to_equatorial(proj_pos, lmc_ra, lmc_dec)  # los, ra, dec
        verbose and print_vector('r_p^sky', sky_pos, representation='spherical')
        particles['radec'] = sky_pos[np.newaxis, :]

        if verbose:
            print('\t\tchecking roundtrip rotation (equatorial <--> projected) for alternative LMC center;')
            alt_center = 'pm'
            alt_init = np.array([m[f'LMC/{alt_center}/distance'],
                                 rad(m[f'LMC/{alt_center}/ra']), rad(m[f'LMC/{alt_center}/dec'])])
            print_vector(f'r_{alt_center}^init', alt_init, representation='spherical')
            alt_proj = equatorial_to_projected(alt_init, lmc_ra, lmc_dec)
            print_vector(f'x_alt^forward', alt_proj, representation='cartesian')
            alt_sky = projected_to_equatorial(alt_proj, lmc_ra, lmc_dec)
            print_vector(f'x_alt^backwards', alt_sky, representation='spherical')

        if bool_bar:
            sky_bar = np.zeros(proj_bar.shape)
            for j in range(len(sky_bar[1])):
                sky_bar[:, j] = projected_to_equatorial(proj_bar[:, j], rad(m['LMC/bar/ra']), rad(m['LMC/bar/dec']))
            verbose and print_vector('sky_bar[0]', sky_bar[:, 0], representation='spherical')
            particles['bar'] = sky_bar

    elif frame == 'mag':
        # sky_coords = cartesian_toSpherical(projected_positions, return_deg=True)  # (r, l, b)
        # sky_coords = sky_coords[np.newaxis, :] if sky_coords.ndim == 1 else sky_coords
        # verbose and print(f'\t\t\t\\vec"{{"r_sky"}}" : ({sky_coords[0, 0]:.2f} kpc, '
        #                   f'{sky_coords[0, 1]:.2f} deg, {sky_coords[0, 2]:.2f} deg)')
        # particles['mag'] = sky_coords
        pass

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
