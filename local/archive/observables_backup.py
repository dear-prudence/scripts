import numpy as np
import argparse
import os
import inspect
from pathlib import Path

from numpy import sin, cos, arcsin, arctan2

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
    return alpha % (2 * np.pi), delta


def rotationMatrix(alpha0, delta0):
    """
    Build rotation matrix R whose columns are the local basis vectors:
      e_x' = local East direction
      e_y' = local North direction
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

    # Local North = ez x ex
    ey = np.cross(ez, ex)

    # Normalize
    ex /= np.linalg.norm(ex)
    ey /= np.linalg.norm(ey)
    ez /= np.linalg.norm(ez)

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
    phi = -np.arctan2(N, -E)  # flip sign to fix handedness mismatch
    return rho, phi, E, N, Z


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
        -sin(rho) * cos(phi),
        -sin(rho) * sin(phi),
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


def rho_phi(pos):
    from hestia.astrometry import Measurements
    m = Measurements()
    d_lmc = m[f'LMC/disk/distance']
    r = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + (d_lmc - pos[2]) ** 2)
    rho = np.arccos((d_lmc - pos[2]) / r)
    phi = np.arctan2(pos[1], pos[0])

    return np.array([r, rho, phi])


def spherical(position):
    if position.ndim == 1:
        r = np.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)
        theta = 90 - np.degrees(np.arccos(position[2] / r))  # polar angle
        phi = np.degrees(np.arctan2(position[1], position[0]))  # azimuthal angle
    else:
        r = np.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2 + position[:, 2] ** 2)
        theta = 90 - np.degrees(np.arccos(position[:, 2] / r))  # polar angle
        phi = np.degrees(np.arctan2(position[:, 1], position[:, 0]))  # azimuthal angle
    return r, theta, phi


def calcCenter(run, halo, snap, part_type, maxTemp=1e5, maxDist=4):
    from hestia.particles import retrieve_particles
    from hestia.gas import calc_temperature
    from hestia.geometry import calc_distanceDisk

    particles = retrieve_particles(run, halo, snap, part_type=part_type, verbose=False)
    if part_type == 'PartType0':
        particles['Temperature'] = calc_temperature(particles['InternalEnergy'], particles['ElectronAbundance'], 0.76)
        cold_mask = particles['Temperature'] < maxTemp
        particles = {key: val[cold_mask] for key, val in particles.items()}
    elif part_type == 'PartType4':
        stellar_mask = particles['GFM_StellarFormationTime'] > 0
        particles = {key: val[stellar_mask] for key, val in particles.items()}
    else:
        print('Error: invalid part_type for calcCenter()!')
        exit(1)

    particles['Distance'] = calc_distanceDisk(particles)
    disk_mask = particles['Distance'] < maxDist  # kpc
    particles = {key: val[disk_mask] for key, val in particles.items()}

    center = [np.average(particles['position'][:, 0], weights=particles['Masses']),
              np.average(particles['position'][:, 1], weights=particles['Masses']),
              np.average(particles['position'][:, 2], weights=particles['Masses'])]
    return np.array(center)


def construct_NH0(run, subject, snap, frame, pixels, verbose=True):
    from hestia.geometry import get_redshift, transform_haloFrame
    from hestia.astrometry import vrai_frame
    from hestia.particles import retrieve_particles
    from hestia.halos import get_halo_params
    from hestia.image import sph_columnH0_projection

    # -------------------------------------
    database = 'stars'  # to define the center of the LMC; stellar velocity field
    fov_lim = 360  # deg, total span of field of view in degrees
    # -------------------------------------

    z_ = f'{get_redshift(run, snap):.3f}'
    h = 0.677

    verbose and print(f'\nen train de travailler au snapshot * {snap}, z = {z_} * ...')

    # if this is the 09_18/MS-analog
    if subject == 'stream' or subject == 'halo_08' or subject == 'smc':
        # handles the satellite, i.e. SMC-analog
        smc_params = get_halo_params(run, 'smc', snap)
        smc = {
            'ParticleIDs': np.ones(1),  # dummy particle id
            'Coordinates': smc_params['halo_pos'] / h,
            'Velocities': smc_params['halo_vel'],
            'Masses': smc_params['M_halo'] / h
        }
        smc = transform_haloFrame(run, 'halo_08', snap, smc, verbose=True)

    cells = retrieve_particles(run, subject, snap, 'PartType0', verbose=verbose)

    # checks for valid coordinate frame
    if frame != 'faux' and frame != 'radec' and frame != 'mag':
        print(f'Error: {frame} is an invalid sky frame; line {inspect.currentframe().f_lineno}')
        exit(1)

    if frame == 'faux':
        # ---------------------------------
        z_observer = 55  # kpc
        lat_max = 30  # deg
        # ---------------------------------
        if 'position' not in cells.keys():
            cells['position'] = cells['Halo_Coordinates']
        # ---------------------------------

        # Shift to observer-centered coordinates
        cells['position'][:, 2] -= z_observer
        cells['faux'] = np.zeros(shape=cells['position'].shape)  # initiate new column with same shape as coordinates
        _, cells['faux'][:, 1], cells['faux'][:, 0] = spherical(cells['position'])

        smc['position'][2] -= z_observer
        smc['faux'] = np.zeros(shape=smc['position'].shape)  # initiate new column with same shape as coordinates
        _, smc['faux'][1], smc['faux'][0] = spherical(smc['position'])

        verbose and print(f'\t\tobserver located at {z_observer} kpc above disk plane.')
        bounds = np.array([[-fov_lim / 2, fov_lim / 2], [-90, lat_max]])

    else:  # if frame != 'faux'
        smc = vrai_frame(run, smc, snap, frame, tracer=database, verbose=True)
        # checks to see if SMC-analog is in field of view
        smc_fov = ((smc[frame][:, 1] >= -fov_lim / 2) & (smc[frame][:, 1] <= 0)
                   & (smc[frame][:, 2] >= -fov_lim) & (smc[frame][:, 2] <= fov_lim))
        if smc_fov.sum() == 0:
            print('\tWarning : SMC-analog not in field of view! Is this intentional')

        cells = vrai_frame(run, cells, snap, frame, tracer=database)
        if frame == 'radec':
            bounds = np.array([[-fov_lim, fov_lim], [-fov_lim / 2, 0]])
        elif frame == 'mag':
            bounds = np.array([[-120, 60], [-60, 60]])

    cells_fov = ((cells[frame][:, 0] >= bounds[0, 0]) & (cells[frame][:, 0] <= bounds[0, 1])
                 & (cells[frame][:, 1] >= bounds[1, 0]) & (cells[frame][:, 1] <= bounds[1, 1]))
    verbose and print(f'\t\tnumber of cells in field of view = {cells_fov.sum()}')
    # Estimate smoothing lengths from mass and density
    volume = cells['Masses'] / cells['Density']  # kpc^3
    hsml = (3 / (4 * np.pi) * volume) ** (1 / 3)  # kpc

    nH_map, lon_edges, lat_edges = sph_columnH0_projection(cells[frame][:, 0][cells_fov],
                                                           cells[frame][:, 1][cells_fov],
                                                           hsml[cells_fov], cells['Masses'][cells_fov],
                                                           cells['GFM_Metals'][:, 0][cells_fov],
                                                           cells['NeutralHydrogenAbundance'][cells_fov],
                                                           bounds=bounds, nbins=pixels)

    data_to_save = {'nH_map': nH_map, 'lon_edges': lon_edges, 'lat_edges': lat_edges,
                    'redshift': z_, 'r_smc': smc[frame]}

    verbose and print(f'termine avec le snapshot {snap}.')

    output_base = '/z/rschisholm'
    output_path = f'/halos/{run}/{subject}/observables/NH0/{run}.{subject}.snap{snap}.NH0.{frame}.npz'

    np.savez_compressed(output_base + output_path, **data_to_save)
    return output_path


def construct_PDF(run, subject, snapshot, sigma_snap, frame, pixels=400, reDerive_lmcSnapshot=False, verbose=True):
    from hestia.geometry import get_redshift, get_lookbackTimes
    from hestia.astrometry import vrai_frame, Measurements
    from hestia.particles import retrieve_particles, get_softeningLength
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter

    # -------------------------------------
    fov_deg = 10  # total span of fov in degrees
    n_samples = 2000  # number of time stamps for interpolated functions
    tracer = 'disk'
    # -------------------------------------

    z_ = f'{get_redshift(run, snapshot):.3e}'
    m = Measurements()

    verbose and print(f'\nen train de travailler au snapshot * {snapshot}, z = {z_} * ...')

    if subject == 'halo_08':  # if the LMC-SMC analog system
        if reDerive_lmcSnapshot:
            from hestia.astrometry import compute_LMCanalog_snapshot
            prospective_snapshots = [220, 307]
            verbose and print(f'\tcomputing appropriate snapshot for {frame} LMC frame')
            snapshot = compute_LMCanalog_snapshot(prospective_snapshots)
        else:  # else use snapshot 255
            snapshot = snapshot
        verbose and print(f'\tusing snapshot {snapshot}.')
    else:  # if interested in any other system (default to present-day, adjusted for sampling size)
        snapshot = 307 - sigma_snap if snapshot is None else snapshot

    # Note: eps_BH varies with z, however in this routine it is static, with the assumption that within
    # the snapshot bounds, it does not vary significantly
    eps_BH = get_softeningLength(run, snapshot, part_type='PartType5')  # eps_BH(z) ~ 0.2 kpc
    verbose and print(f'\t\t... +- {sigma_snap} snapshots,\n'
                      f'\t\tobserver: {m[f"LMC/{tracer}/distance"]}, eps_BH: {1e3 * eps_BH} pc\n'
                      f'\t\tfov: {fov_deg} x {fov_deg} deg^2, num of interpolation samples: {n_samples}')

    #  in vrai frame, e_i = (east, north, line of sight); in faux frame, e_i = (x, y, z)_halo
    e_1, e_2, e_3 = np.array([]), np.array([]), np.array([])

    redshifts, times = get_lookbackTimes(run, range(snapshot - sigma_snap, snapshot + sigma_snap))

    verbose and print('\tretrieving black holes')
    from hestia.geometry import calc_distanceDisk

    useUncertainty = False
    N_samples = 100

    if useUncertainty:
        inclinations = np.random.normal(m['LMC/disk/inclination'], m['LMC/disk/sigma_i'], size=N_samples)
        nodes = np.random.normal(m['LMC/disk/nodes'], m['LMC/disk/sigma_nodes'], size=N_samples)
        for j in range(N_samples):
            e_1, e_2, e_3 = np.array([]), np.array([]), np.array([])
            verbose and print(f'--------------------------------\n'
                              f'j : {j}, i : {inclinations[j]}, nodes : {nodes[j]}\n'
                              f'--------------------------------')
            for snap in range(snapshot - sigma_snap, snapshot + sigma_snap):
                bh = retrieve_particles(run, subject, snap, 'PartType5', verbose=False)
                bh['Distances'] = calc_distanceDisk(bh)
                dist_mask = bh['Distances'] < 10  # kpc
                bh = {key: val[dist_mask] for key, val in bh.items()}

                bh = vrai_frame(run, bh, snap, frame, tracer=tracer, halo=subject,
                                inclination=inclinations[j], nodes=nodes[j])
                e_1 = np.append(e_1, bh[frame][:, 0])  # los, in kpc
                e_2 = np.append(e_2, np.degrees(bh[frame][:, 1]))  # dec, in deg
                e_3 = np.append(e_3, np.degrees(bh[frame][:, 2]))  # ra, in deg

            print(f'times.shape : {times.shape}')
            print(f'e_1.shape : {e_1.shape}')
            f_1 = interp1d(times, e_1, kind="cubic")  # cubic spline for smooth path
            f_2 = interp1d(times, e_2, kind="cubic")
            f_3 = interp1d(times, e_3, kind='cubic')

            t_fine = np.linspace(times.min(), times.max(), n_samples)
            e1_fine, e2_fine, e3_fine = f_1(t_fine), f_2(t_fine), f_3(t_fine)
            verbose and print('\tcomputed interpolation of coordinates')
            verbose and print(f'\t\tmean(' + ('LOS, dec, ra' if frame == 'radec' else 'x, y, z')
                              + f') : ({np.mean(e_1)}, {np.mean(e_2)}, {np.mean(e_3)})')

            if j == 0:
                coordinates = np.array([e1_fine, e2_fine, e3_fine]).T
            else:
                coordinates = np.vstack((coordinates, np.array([e1_fine, e2_fine, e3_fine]).T))
            print(f'coordinates.shape : {coordinates.shape}')

    else:
        for snap in range(snapshot - sigma_snap, snapshot + sigma_snap):
            bh = retrieve_particles(run, subject, snap, 'PartType5', verbose=True)
            bh['Distances'] = calc_distanceDisk(bh)
            dist_mask = bh['Distances'] < 10  # kpc
            bh = {key: val[dist_mask] for key, val in bh.items()}

            bh = vrai_frame(run, bh, snap, frame, tracer=tracer, halo=subject)
            e_1 = np.append(e_1, bh[frame][:, 0])  # los, in kpc
            e_2 = np.append(e_2, np.degrees(bh[frame][:, 1]))  # dec, in deg
            e_3 = np.append(e_3, np.degrees(bh[frame][:, 2]))  # ra, in deg

        f_1 = interp1d(times, e_1, kind="cubic")  # cubic spline for smooth path
        f_2 = interp1d(times, e_2, kind="cubic")
        f_3 = interp1d(times, e_3, kind='cubic')

        t_fine = np.linspace(times.min(), times.max(), n_samples)
        e1_fine, e2_fine, e3_fine = f_1(t_fine), f_2(t_fine), f_3(t_fine)
        verbose and print('\tcomputed interpolation of coordinates')
        verbose and print(f'\t\tmean(' + ('LOS, dec, ra' if frame == 'radec' else 'x, y, z')
                          + f') : ({np.mean(e_1)}, {np.mean(e_2)}, {np.mean(e_3)})')

        coordinates = np.array([e1_fine, e2_fine, e3_fine]).T
    bar = bh['bar']

    # Field of view
    if frame == 'radec':
        ra, dec = m[f'LMC/{tracer}/ra'], m[f'LMC/{tracer}/dec']
        bounds = ((ra - fov_deg / 2, ra + fov_deg / 2), (dec - fov_deg / 2, dec + fov_deg / 2))  # degrees
    elif frame == 'mag':
        bounds = ((-fov_deg / 2, fov_deg / 2), (-fov_deg / 2, fov_deg / 2))

    in_fov = ((coordinates[:, 2] >= bounds[0][0]) & (coordinates[:, 2] <= bounds[0][1])
              & (coordinates[:, 1] >= bounds[1][0]) & (coordinates[:, 1] <= bounds[1][1]))
    lon, lat, los = coordinates[:, 2][in_fov], coordinates[:, 1][in_fov], coordinates[:, 0][in_fov]
    verbose and print(f'\t\tnumber of particles in field of view = {in_fov.sum()}')
    # lon, lat, los, bounds, in_fov = projectOntoSky(coordinates, r_observer, frame, fov_deg=fov_deg, verbose=verbose)
    # verbose and print('\tprojected positions onto sky plane')

    lon_e = np.linspace(bounds[0][0], bounds[0][1], pixels + 1)  # lon edges
    lat_e = np.linspace(bounds[1][0], bounds[1][1], pixels + 1)  # lat edges
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])  # lon center
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])  # lat center
    dlon = lon_e[1] - lon_e[0]  # pixel (lon) scale in deg
    dlat = lat_e[1] - lat_e[0]  # pixel (lat) scale in deg

    dt = np.diff(t_fine, prepend=t_fine[0])  # delta_t for fine cadence
    print(f'\t\tdelta_t = {1e3 * dt[1]} Myr')
    f_PDF = np.zeros((pixels, pixels))

    for i in range(len(lon)):
        sigma_deg = np.degrees(eps_BH / los[i])  # angular size for softening length
        # converts to pixel units
        sigma_pix_lon = sigma_deg / dlon
        sigma_pix_lat = sigma_deg / dlat
        # finds the nearest pixel
        ix = np.argmin(np.abs(lon_c - lon[i]))
        iy = np.argmin(np.abs(lat_c - lat[i]))
        # adds weight into a small patch around (ix, iy)
        patch = np.zeros_like(f_PDF)
        # patch[iy, ix] = dt[i]
        patch[iy, ix] = dt[i % n_samples]
        # gaussian blur the patch
        blurred = gaussian_filter(patch, sigma=(sigma_pix_lat, sigma_pix_lon), mode='constant')
        f_PDF += blurred

    verbose and print(f'\tgaussian blurred accordingly with eps_BH\n'
                      f'\t\tmax angular sigma ~ {sigma_deg:.3f} deg')
    verbose and print(f'termine avec le snapshot {snapshot}.')

    output_base = '/z/rschisholm'
    output_path = f'/halos/{run}/{subject}/observables/bhPDF/'
    output_name = f'{run}.{subject}.snap{snapshot}-{sigma_snap}.bhPDF.{frame}.npz'

    if not os.path.exists(output_base + output_path):
        os.makedirs(output_base + output_path)
        print('\toutput directory written')

    np.savez(output_base + output_path + output_name, f_PDF=f_PDF, lon_edges=lon_e, lat_edges=lat_e, bar=bar,
             center_bh=np.array([np.mean(e_1), np.mean(e_2), np.mean(e_3)]))
    return output_path + output_name


def construct_SSD(run, subject, snap, frame, pixels, verbose=True):
    from hestia.geometry import get_redshift, transform_haloFrame
    from hestia.astrometry import vrai_frame, Measurements
    from hestia.particles import retrieve_particles
    from hestia.halos import get_halo_params
    from hestia.image import create_spherical_brightness_projection

    # -------------------------------------
    database = 'stars'  # to define the center of the LMC; stellar velocity field
    fov_lim = 360  # deg, total span of field of view in degrees
    central_region = True
    # -------------------------------------

    z_ = f'{get_redshift(run, snap):.3f}'
    h = 0.677

    verbose and print(f'\nen train de travailler au snapshot * {snap}, z = {z_} * ...')
    m = Measurements()
    # if this is the 09_18/MS-analog
    if subject == 'halo_08' or subject == 'smc':
        # handles the satellite, i.e. SMC-analog
        smc_params = get_halo_params(run, 'smc', snap)
        smc = {
            'ParticleIDs': np.ones(1),  # dummy particle id
            'Coordinates': smc_params['halo_pos'] / h,
            'Velocities': smc_params['halo_vel'],
            'Masses': smc_params['M_halo'] / h
        }
        smc = transform_haloFrame(run, 'halo_08', snap, smc, verbose=True)
    elif subject == 'stream':
        print(f'Error: "{subject}" is an invalid subject for SSD; line {inspect.currentframe().f_lineno}')
        exit(1)

    stars = retrieve_particles(run, subject, snap, 'PartType4', verbose=verbose)
    stellar_mask = stars['GFM_StellarFormationTime'] > 0
    stars = {key: val[stellar_mask] for key, val in stars.items()}

    # checks for valid coordinate frame
    if frame != 'faux' and frame != 'radec' and frame != 'mag':
        print(f'Error: {frame} is an invalid sky frame; line {inspect.currentframe().f_lineno}')
        exit(1)

    if frame == 'faux':
        # ---------------------------------
        z_observer = 55  # kpc
        lat_max = 30  # deg
        # ---------------------------------
        if 'position' not in stars.keys():
            stars['position'] = stars['Halo_Coordinates']
        # ---------------------------------

        # Shift to observer-centered coordinates
        stars['position'][:, 2] -= z_observer
        stars['faux'] = np.zeros(shape=stars['position'].shape)  # initiate new column with same shape as coordinates
        stars['faux'][:, 2], stars['faux'][:, 1], stars['faux'][:, 0] = spherical(stars['position'])

        smc['position'][2] -= z_observer
        smc['faux'] = np.zeros(shape=smc['position'].shape)  # initiate new column with same shape as coordinates
        smc['faux'][2], smc['faux'][1], smc['faux'][0] = spherical(smc['position'])

        verbose and print(f'\t\tobserver located at {z_observer} kpc above disk plane.')
        bounds = np.array([[-fov_lim / 2, fov_lim / 2], [-90, lat_max]])

        stars_fov = ((stars[frame][:, 0] >= bounds[0, 0]) & (stars[frame][:, 0] <= bounds[0, 1])
                     & (stars[frame][:, 1] >= bounds[1, 0]) & (stars[frame][:, 1] <= bounds[1, 1]))

    else:  # if frame != 'faux'
        print('\tbelow is smc vrai_frame')
        smc = vrai_frame(run, smc, snap, frame, tracer=database, verbose=True)
        # checks to see if SMC-analog is in field of view
        smc_fov = ((smc[frame][:, 1] >= -fov_lim / 2) & (smc[frame][:, 1] <= 0)
                   & (smc[frame][:, 2] >= -fov_lim) & (smc[frame][:, 2] <= fov_lim))
        if smc_fov.sum() == 0:
            print('\tWarning : SMC-analog not in field of view! Is this intentional')

        print('\tbelow is stars vrai_frame')
        stars = vrai_frame(run, stars, snap, frame, tracer=database, verbose=True)
        if frame == 'radec':
            if not central_region:
                bounds = np.array([[-fov_lim / 2, fov_lim / 2], [-fov_lim / 4, 0]])
                stars_fov = ((stars[frame][:, 2] >= bounds[0, 0]) & (stars[frame][:, 2] <= bounds[0, 1])
                         & (stars[frame][:, 1] >= bounds[1, 0]) & (stars[frame][:, 1] <= bounds[1, 1]))
            else:
                ra, dec = m[f'{database}/LMC/ra'], m[f'{database}/LMC/dec']
                bounds = np.array([[ra - 4 / 2, ra + 4 / 2], [dec - 4 / 2, dec + 4 / 2]])  # degrees
                stars_fov = ((stars[frame][:, 2] >= bounds[0, 0]) & (stars[frame][:, 2] <= bounds[0, 1])
                         & (stars[frame][:, 1] >= bounds[1, 0]) & (stars[frame][:, 1] <= bounds[1, 1]))
        elif frame == 'mag':
            bounds = np.array([[-120, 60], [-60, 60]])

    verbose and print(f'\t\tnumber of stars in field of view = {stars_fov.sum()}')

    idx_lon = 2 if frame == 'faux' else 0
    # L_map, lon_edges, lat_edges = star_surface_density_projection(stars[frame][:, idx_lon][stars_fov],
    #                                                               stars[frame][:, 1][stars_fov],
    #                                                               stars['Masses'][stars_fov],
    #                                                               bounds=tuple(bounds), nbins=pixels)

    mag = stars['GFM_StellarPhotometrics'][:, 2] + 5 * np.log10(stars[frame][:, 2]) - 5

    L_map, lon_edges, lat_edges = create_spherical_brightness_projection(stars[frame][:, 0][stars_fov],
                                                                         stars[frame][:, 1][stars_fov],
                                                                         mag[stars_fov],
                                                                         bins=(360, 90),
                                                                         background=25,
                                                                         bounds=tuple(bounds))

    data_to_save = {'L_map': L_map, 'lon_edges': lon_edges, 'lat_edges': lat_edges,
                    'redshift': z_, 'r_smc': smc[frame]}

    verbose and print(f'termine avec le snapshot {snap}.')

    output_base = '/z/rschisholm'
    output_path = f'/halos/{run}/{subject}/observables/SSD/'
    output_name = f'{run}.{subject}.snap{snap}.SSD.{frame}.npz'

    if not os.path.exists(output_base + output_path):
        os.mkdir(output_base + output_path)
        print('\toutput directory written')

    np.savez_compressed(output_base + output_path + output_name, **data_to_save)
    return output_path + output_name


def main(cluster):
    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    # necessary arguments
    parser.add_argument('map', nargs='?', default='nh0', help='parameter map to be constructed')
    parser.add_argument('run', nargs='?', default='09_18', help='simulation run')
    parser.add_argument('subject', nargs='?', default='stream', help='subject to be processed')
    parser.add_argument('snap', nargs='?', default=124, help='snapshot to be processed')

    # arguments to specify coordinate system
    parser.add_argument('--faux', dest='faux', action='store_true',
                        help='keep in static hal frame')
    parser.add_argument('--radec', dest='radec', action='store_true',
                        help='transform to (ra, dec) frame ; based on irl positions of LMC and SMC')
    parser.add_argument('--galactic', dest='galactic', action='store_true',
                        help='transform to galactic coordiante frame (l, b); based on irl positions of LMC and SMC')
    parser.set_defaults(faux=False, radec=False, galactic=False)

    # arguments to specify particle type (defaults set for various maps)
    parser.add_argument('--bh', dest='bh', action='store_true', help='use central bh particle(s)')
    parser.add_argument('--gas', dest='gas', action='store_true', help='use gas cells')
    parser.add_argument('--stars', dest='stars', action='store_true', help='use star particles')
    parser.set_defaults(bh=False, gas=False, stars=False)

    # Optional arguments
    parser.add_argument("--sigma_snap", type=int, default=20, help='for pdf, sigma_snap')
    parser.add_argument("--pixels", type=int, default=400, help='side length of image in pixels')
    parser.add_argument('--v', dest='verbose', action='store_true', help='verbose print statements')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    # if args.map == 'pdf':
    # partType = 'PartType0' if args.gas else (partType = 'par')

    if args.radec:
        frame = 'radec'
    elif args.galactic:
        frame = 'galactic'
    else:  # if no input is given
        frame = 'faux'

    print('--------------------------------------------------------------------------------\n'
          + f'en train de creer du graphique * {args.map} * ...\n'
          + f'sim_run -- {args.run}\n'
          + f'subject -- {args.subject}\n'
          + f'snapshot -- {args.snap}\n'
          + (f'sigma_snap -- {args.sigma_snap}\n' if args.map == 'pdf' else '')
          + f'frame -- {frame} \n'
          + f'pixels -- {args.pixels}\n'
          + '--------------------------------------------------------------------------------')

    if args.map == 'nh0':  # H0 column densities
        output_path = construct_NH0(args.run, args.subject, int(args.snap), frame, pixels=args.pixels)
    elif args.map == 'pdf':  # prob. density function of a specified particle (or center of a collection of particles)
        output_path = construct_PDF(args.run, args.subject, int(args.snap), args.sigma_snap, frame, pixels=args.pixels)
    elif args.map == 'ssd':  # surface distribution of a specified particle type
        output_path = construct_SSD(args.run, args.subject, int(args.snap), frame, pixels=args.pixels)

    print(f'le dossier de sortie a ete ecrit, trouvez le chemin d\'access ci-dessous:\n\n{output_path}\n')


def plotting():
    from local.plots import dispatch_plot

    # PARAMETERS TO CHANGE
    # ------------------------------------
    run = '09_18_lastgigyear'
    subject = 'halo_41'
    parameter = 'bhPDF'
    snap = 284
    frame = 'radec'
    chisholm2025_plot = False
    # ------------------------------------
    sigma_snap = 12
    # ------------------------------------

    home = Path.home()
    current_file = Path(__file__)
    project_root = current_file.parent.parent  # adjust as needed
    basePath = home / project_root / 'halos' / run / subject / 'observables' / parameter

    if chisholm2025_plot:
        from local.chisholm2025 import streamPlot
        streamPlot()

    elif parameter == 'NH0':
        dispatch_plot('observables', f'{parameter}', frame=frame,
                      input_path=f'{basePath}/{run}.{subject}.snap{snap}.NH0.{frame}.npz',
                      output_path=f'{basePath}/{run}.{subject}.snap{snap}.NH0.{frame}.png')

    elif parameter == 'bhPDF':
        dispatch_plot('observables', f'{parameter}',
                      input_path=f'{basePath}/{run}.{subject}.snap{snap}-{sigma_snap}.bhPDF.{frame}.npz',
                      output_path=f'{basePath}/{run}.{subject}.snap{snap}-{sigma_snap}.bhPDF.{frame}.png')

    elif parameter == 'SSD':
        dispatch_plot('observables', f'{parameter}', frame=frame,
                      input_path=f'{basePath}/{run}.{subject}.snap{snap}.SSD.{frame}.npz',
                      output_path=f'{basePath}/{run}.{subject}.snap{snap}.SSD.{frame}.png')

    else:
        print(f'Error: <<{parameter}>> is an invalid parameter!')
        exit(1)


if __name__ == "__main__":
    import socket

    machine = socket.gethostname()

    if 'aip.de' in machine:  # aip cluster
        main('erebos')
    elif machine == 'scylla':  # scylla cluster
        main('scylla')
    else:  # local machine
        plotting()
