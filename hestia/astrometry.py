from __future__ import division
import numpy as np
from .geometry import cartesian_toSpherical, transform_haloFrame


class Measurements:
    def __init__(self):
        # (ra, dec) all in J2000 epoch unless otherwise specified
        self._data = {
            'nasa': {
                'LMC': {
                    'ra': 80.894167,
                    'dec': -69.756111,
                    'distance': 49.59,  # Pietrzyński+2019
                    'inclination': 25.86,
                    'line of nodes': 149.23,
                },
                'SMC': {
                    'ra': 13.186667,
                    'dec': -72.828611,
                    'distance': 62.44  # Graczyk+2020
                }
            },
            # stellar rotation field, vanDerMarel+2013 (https://iopscience.iop.org/article/10.1088/0004-637X/781/2/121)
            'stars': {
                'LMC': {
                    'ra': 78.76,
                    'dec': -69.19,
                    'sigma_ra': 0.52,
                    'sigma_dec': 0.25,
                    'inclination': 39.6,
                    'line of nodes': 147.4,
                    'distance': 50.1  # Freedman+2001
                },
                'SMC': {
                    'ra': 13.052083,
                    'dec': -72.828611,
                    'distance': 62.44  # Graczyk+2020
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


def get_smc(database='nasa', verbose=True):
    """
    Compute the SMC position and the SMC-LMC relative velocity in the LMC-disk frame
    where the LMC disk lies in the x-y plane and the disk angular-momentum axis is +z.
    """

    from numpy import sin, cos

    def R_x(a):
        return np.array([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])

    def R_y(a):
        return np.array([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])

    def R_z(a):
        return np.array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])

    verbose and print('\tcomputing SMC kinemtics in LMC rest frame using the following qunatities...')

    m = Measurements()

    ra_lmc, dec_lmc, i, theta, d_lmc = (np.radians(m[f'{database}/LMC/ra']),
                                        np.radians(m[f'{database}/LMC/dec']),
                                        np.radians(m[f'{database}/LMC/inclination']),
                                        -np.radians(90) + np.radians(m[f'{database}/LMC/line of nodes']),
                                        m[f'{database}/LMC/distance'])  # in kpc
    verbose and print(f'\t\t(ra, dec)_lmc : ({np.degrees(ra_lmc)}, {np.degrees(dec_lmc)}) deg\n'
                      f'\t\t\ti_lmc : {np.degrees(i)} deg\n'
                      f'\t\t\ttheta_lmc : {90 + np.degrees(theta):.2f} deg\n'
                      f'\t\t\tdist_lmc : {d_lmc} kpc')

    ra_smc, dec_smc, d_smc = (np.radians(m[f'{database}/SMC/ra']),
                              np.radians(m[f'{database}/SMC/dec']),
                              m[f'{database}/SMC/distance'])  # in kpc
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


def vrai_frame(run, particles, snap, frame='radec', database='nasa', halo='halo_08', verbose=True):
    from numpy import sin, cos
    m = Measurements()
    h = 0.677

    def Ref_x():
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def R_x(a):
        return np.array([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]]).T

    def R_y(a):
        return np.array([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]]).T

    def R_z(a):
        return np.array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]]).T

    # ----------------------------------------------
    # to be compatible with depreceated keys (will remove eventually)
    if 'position' in particles.keys():
        position = 'position'
    else:
        position = 'Halo_Coordinates'
    # ----------------------------------------------

    if halo == 'halo_08':  # if this is the LMC-SMC analog system (by default is True)
        SMC_pos = get_smc(database=database, verbose=verbose)  # SMC position in static reference frame of LMC
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

        # locked to the position of the smc in 09_18_lastgigyear, snapshot 255
        # smc_filePath = ('/z/rschisholm/halos/09_18/smc/'
        #                 'HESTIA_100Mpc_8192_09_18_lastgigyear.127_halo_307000000001476.dat')
        # row = 307 - 255

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

    else:  # if a galaxy other than the LMC-SMC analog system
        d_phi = 0

    if particles[position].ndim == 1:
        particles[position] = particles[position][np.newaxis, :]
    # reflects (mirrors) along x-axis to better mimic the morphology of LMC-SMC orbital history,
    # or in the case of other galaxies, reflects to have a CW-rotating disk
    particles[position] = particles[position] @ Ref_x().T
    # particles[position] = particles[position] @ R_x(np.radians(180)).T

    verbose and print(f'\t\trotating particles/cells to {frame} frame...\n'
                      f'\t\te.g...\t\\vec"{{"x_init"}}" : ({particles[position][0, 0]:.2f}, '
                      f'{particles[position][0, 1]:.2f}, {particles[position][0, 2]:.2f}) kpc')

    # relabels cartesian triad by a 90-degree rotation, and then aligning SMC and smc using delta_phi
    aligned_positions = particles[position] @ (R_z(d_phi) @ R_y(-np.pi / 2))
    verbose and print(f'\t\t\t\\vec"{{"x_aligned"}}" : ({aligned_positions[0, 0]:.2f}, '
                      f'{aligned_positions[0, 1]:.2f}, {aligned_positions[0, 2]:.2f}) kpc')

    # rotates the disk out of the sky plane and reorients the line of nodes
    i, theta = (np.radians(m[f'{database}/LMC/inclination']),
                -np.radians(90) + np.radians(m[f'{database}/LMC/line of nodes']))
    projected_positions = aligned_positions @ (R_y(i) @ R_x(-theta))
    verbose and print(f'\t\t\t\\vec"{{"x_projected"}}" : ({projected_positions[0, 0]:.2f}, '
                      f'{projected_positions[0, 1]:.2f}, {projected_positions[0, 2]:.2f}) kpc')

    projected_positions[:, 0] += m[f'{database}/LMC/distance']

    if frame == 'radec':
        lmc_ra, lmc_dec = np.radians(m[f'{database}/LMC/ra']), np.radians(m[f'{database}/LMC/dec'])
        sky_positions = projected_positions @ (R_y(-lmc_dec) @ R_z(lmc_ra))
        verbose and print(f'\t\t\t\\vec"{{"x_sky"}}" : ({sky_positions[0, 0]:.2f}, '
                          f'{sky_positions[0, 1]:.2f}, {sky_positions[0, 2]:.2f}) kpc')

        sky_coords = cartesian_toSpherical(sky_positions, return_deg=True)  # (r, dec, ra)
        sky_coords = sky_coords[np.newaxis, :] if sky_coords.ndim == 1 else sky_coords
        verbose and print(f'\t\t\t\\vec"{{"r_sky"}}" : ({sky_coords[0, 0]:.2f} kpc, '
                          f'{sky_coords[0, 1]:.2f} deg, {sky_coords[0, 2]:.2f} deg)')

        particles['radec'] = sky_coords

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
