from __future__ import division
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import numpy as np
from .halos import get_halo_params

"""
This file contains routines specifically involved with the geometry of the spacetime of the hestia simulation runs
(coordinate transformations, distance calculations, lookback time calculations, etc...)
"""


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
    """
    :param run: simulation run
    :param snaps: 1-dim array of snapshots in consecutive order
    :param redshifts: direct input of redshifts, optional
    :return: redshifts, lookback times assuming a flat Lambda-CDM cosmology [Gyr]
    """
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.315)
    if redshifts is None:
        redshifts = np.array([])
        for snap in snaps:
            redshift = get_redshift(run, snap)
            redshifts = np.append(redshifts, redshift)

    lookback_times = cosmo.lookback_time(redshifts).to(u.Gyr).value
    return redshifts, lookback_times


def rid_h_units(data, z=None, part_type='PartType0', reverse=False):
    """
    :param data: particles in simulations h_units*
    :param z: redshifts
    :param part_type: particle type, default to gas
    :param reverse: to return to h_units, default to False
    :return: particles in ordinary comoving units [kpc, M_solar/kpc^3, *physical units for energy dissipation,
    M_solar, km/s]
    """
    h = 0.677  # value used in the hestia simulation suite
    a = 1.0 / (1 + float(z))

    transform_map = {
        'Coordinates': {'factor': lambda a, h: 1e3 / h, 'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
        'Density': {'factor': lambda a, h: h ** 2, 'types': ['PartType0']},
        'EnergyDissipation': {'factor': lambda a, h: 1e10 / a, 'types': ['PartType0']},
        'Masses': {'factor': lambda a, h: 1e10 / h, 'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
        'Velocities': {'factor': lambda a, h: np.sqrt(a),
                       'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
        'Potential': {'factor': lambda a, h: 1 / a, 'types': ['PartType0', 'PartType1', 'PartType4', 'PartType5']},
    }

    for key, info in transform_map.items():
        if part_type in info['types'] and key in data:
            factor = info['factor'](a, h)
            if reverse:
                factor = 1.0 / factor
            data[key] = data[key] * factor
    return data


def transform_haloFrame(run, halo, snap, particles, full_halo_id=False, verbose=True):
    """
    transforms coordinates to static halo reference frame;
    should be used in conjunction with (and after) rid_h_units()!
    :param run: simulation run
    :param halo: halo name of interest
    :param snap: snapshot
    :param full_halo_id: passing argument
    :param particles: particles to be transformed (in ordinary units)
    :param verbose: verbose
    :return: particles in frame of reference of halo [angular momentum: j ~ M_solar kpc km/s]
    """
    from .halos import get_L_star
    verbose and print(f'\ttransforming {str(len(particles["ParticleIDs"]))} '
                      f'particle(s) to static halo reference frame... ')

    h = 0.677
    halo_params = get_halo_params(run, halo, snap, full_halo_id=full_halo_id)
    halo_pos_h, halo_vel, L_star, = halo_params['halo_pos'], halo_params['halo_vel'], halo_params['L_star']
    halo_pos = halo_pos_h / h

    L = get_L_star(run, halo, snap)

    # Coordinates transformation
    halo_coords = np.zeros(particles['Coordinates'].shape)
    # Compute relative coordinates for each particle
    # Iterate over every element in the 2D numpy array

    # gets the rotation matrix to orient the L of the halo in the +z direction
    rot_matrix = get_rotationMatrix(L, np.array([0, 0, 1]))
    verbose and print(f'\t\trotating spatial coordinates')
    if len(particles['ParticleIDs']) == 1:
        halo_coords = particles['Coordinates'] - halo_pos
    else:
        for i in range(halo_coords.shape[0]):  # iterate over rows
            halo_coords[i] = particles['Coordinates'][i] - halo_pos  # in kpc

    # Iterate over each vector and multiply by the rotation matrix
    rot_coords = np.dot(halo_coords, rot_matrix.T)  # Matrix multiplication
    # Add the new column to the data dictionary
    particles['position'] = rot_coords  # in kpc

    # Velocities transformation
    halo_vels = np.zeros(particles['Velocities'].shape)
    if len(particles['ParticleIDs']) == 1:
        halo_vels = particles['Velocities'] - halo_vel
    else:
        for i in range(halo_vels.shape[0]):
            halo_vels[i] = particles['Velocities'][i] - halo_vel  # Subtract halo velocity (in km/s)

    verbose and print(f'\t\trotating velocities')
    rot_vels = np.dot(halo_vels, rot_matrix.T)
    # Add transformed velocities to the particles dictionary
    particles['velocity'] = rot_vels

    # Angular momentum calculation L = r x mv
    # particle_masses = particles['Masses'][:, np.newaxis]  # Reshape to (N, 1) to broadcast with velocities,
    particle_masses = particles['Masses']
    # factor of 1 / h already added in cosmo_transform, need to update routines to simplify unit conversions
    # particle_masses = particles['Masses'][:, np.newaxis] * 1e10  # COMPATIBLE WITH COSMO_TRANSFORM NOT COSMO_TO_PHYS.
    verbose and print(f'\t\tcomputing angular momenta')

    if len(particles['ParticleIDs']) == 1:
        angularMomenta = np.cross(rot_coords, rot_vels * particle_masses)
    else:
        angularMomenta = np.zeros(particles['Velocities'].shape)
        for i in range(particles['Masses'].shape[0]):
            angularMomenta[i] = np.cross(rot_coords[i], rot_vels[i] * particle_masses[i])
    particles['angularMomentum'] = angularMomenta  # in units of j ~ M_solar kpc km s^-1

    return particles


def get_rotationMatrix(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2; does not work for 180 degree rotations
    :param vec1: 3d "source" vector
    :param vec2: 3d "destination" vector
    :return mat: a transformation matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rot_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    return rot_matrix


def calc_distanceDisk(particles):
    """
    :param particles: particles in halo rest frame [kpc]
    :return: particles: with additional column for distances (essentially spherical radial coordinate) [kpc]
    """
    # -------------------------------
    if 'position' in particles.keys():
        position = 'position'
    else:
        position = 'Halo_Coordinates'
    # -------------------------------

    distances = np.zeros(particles[position].shape[0])
    # Compute distances to center of disk for each particle
    for i in range(particles[position].shape[0]):
        distances[i] = np.linalg.norm(particles[position][i])
    return distances


def calc_distanceHalo(sim_run, snaps, subject_halo, reference_halo, verbose=False):
    """
    similar to calc_distanceDisk(), but uses halo positions directly from AHF_halos output file
    (avoids expensive halo reference frame calculations)
    :param sim_run, snaps, subject_halo: passing arguments
    :param reference_halo: halo of interest to calculate the distance to
    :return: distance between halos [kpc]
    """
    starting_snapshot = snaps[0]
    ending_snapshot = snaps[1]
    h = 0.677

    dist_halo = np.zeros(ending_snapshot - starting_snapshot)
    for i in range(ending_snapshot, starting_snapshot, -1):
        sh_pos = get_halo_params(sim_run, subject_halo, i)['halo_pos']  # in kpc_h
        rh_pos = get_halo_params(sim_run, reference_halo, i)['halo_pos']  # in kpc_h
        dist_halo[127 - i] = np.linalg.norm(sh_pos / h - rh_pos / h)

    verbose and print(f'\t\tdist_halo = {dist_halo}')
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


