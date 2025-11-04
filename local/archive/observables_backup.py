import numpy as np
import argparse


def projectOntoSky(coordinates, r_observer, hsml, masses, h_frac, f_H0,
                   fov_deg=360, num_pixels=400, lat_max=90, verbose=True):
    """
    projects sph cells onto the sky as seen by an observer
    :param coordinates: matrix indicating the 3-dim cartesian position (in kpc) of the gas cells
    :param hsml: smoothing lengths (in kpc)
    :param masses: cells masses (in M_solar)
    :param h_frac: X_H
    :param f_H0: neutral hydrogen abundance
    :param r_observer: 3-dim cartesian position of observer (in kpc)
    :param fov_deg: field of view (in degrees)
    :param num_pixels: number of image pixels (i.e. number of bins)
    :param lat_max: maximum latitude of image
    :param verbose: verbose print statements
    :return: nH_map, lon_edges, lat_edges: projected column density map (N_H0) and angular bin edges
    """
    from hestia import sph_columnH0_projection

    # Shift to observer-centered coordinates
    x_rel = coordinates[:, 0] - r_observer[0]
    y_rel = coordinates[:, 1] - r_observer[1]
    z_rel = coordinates[:, 2] - r_observer[2]
    # Spherical coordinates
    r = np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)
    theta = np.arccos(z_rel / r)  # polar angle
    phi = np.arctan2(y_rel, x_rel)  # azimuthal angle
    # Convert to degrees
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    # Define angular bounds centered on view direction
    lat_center = -45
    # lat_min = lat_center - fov_deg / 2
    lat_min = -90
    # lat_max = lat_center + fov_deg / 2
    lon_min = -fov_deg / 2
    lon_max = +fov_deg / 2
    # Select particles in field of view
    lat = 90 - theta_deg  # latitude on the sky (degrees)
    lon = phi_deg  # longitude on the sky (degrees)
    in_fov = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

    # Project only selected particles
    x_proj = lon[in_fov]
    y_proj = lat[in_fov]
    verbose and print(f'\t\tobserver located at {r_observer}.')
    verbose and print(f'\t\tnumber of particles in field of view: {in_fov.sum()}.')

    # Project using SPH kernel in angular space
    bounds = ((lon_min, lon_max), (lat_min, lat_max))  # degrees
    return sph_columnH0_projection(
        x_proj, y_proj,
        hsml[in_fov],
        masses[in_fov],
        h_frac[in_fov],
        f_H0[in_fov],
        bounds=bounds,
        nbins=num_pixels
    )


def cartesian_to_projected_spherical(snap, satellite, r_observer=(0, 0, 55)):
    from hestia import transform_haloFrame

    satellite = transform_haloFrame('09_18', 'halo_08', snap, satellite)
    # pos_rot, vel_rot = transform_haloFrame_singleParticle(snap, pos_xyz, vel_xyz)

    # Translate vector to observer frame
    vec = np.array(satellite['position']) - np.array(r_observer)
    # Normalize to unit vector
    r = np.linalg.norm(vec)
    if r == 0:
        raise ValueError('Point is at the observer location.')
    unit_vec = vec / r

    # Get spherical angles
    x, y, z = unit_vec
    lat_rad = np.arcsin(z)  # latitude
    lon_rad = np.arctan2(y, x)  # longitude

    # Convert to degrees
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    # Wrap longitude to [-180, 180]
    if lon_deg > 180:
        lon_deg -= 360
    elif lon_deg < -180:
        lon_deg += 360
    print('\t\tcalculated positions')

    # v_lon, v_lat = project_velocity_tangential(pos_rot, vel_rot, observer=observer)
    vec = np.array(satellite['Velocities']) - np.array(r_observer)
    r = np.linalg.norm(vec)
    unit_vec = vec / r
    x, y, z = unit_vec
    lat_rad = np.arcsin(z)  # latitude
    lon_rad = np.arctan2(y, x)  # longitude
    # Convert to degrees
    v_lat = np.degrees(lat_rad)
    v_lon = np.degrees(lon_rad)

    # Wrap longitude to [-180, 180]
    if v_lon > 180:
        v_lon -= 360
    elif v_lat < -180:
        v_lat += 360
    print('\t\tcalculated velocities')

    return np.array([lon_deg, lat_deg]), np.array([v_lon, v_lat])


def project_velocity_tangential(pos, vel, observer):
    """
    Project a 3D velocity vector into spherical tangential components
    (v_phi, v_theta) from an observer's point of view.

    Parameters:
        pos : array_like
            3D Cartesian position of the particle [x, y, z]
        vel : array_like
            3D Cartesian velocity of the particle [vx, vy, vz]
        observer : array_like, optional
            3D Cartesian position of the observer

    Returns:
        v_phi : float
            Tangential velocity in the longitude direction (degrees increasing)
        v_theta : float
            Tangential velocity in the latitude direction (degrees increasing)
    """
    # Convert to numpy arrays
    pos = np.array(pos)
    vel = np.array(vel)
    obs = np.array(observer)

    # Vector from observer to particle
    r_vec = pos - obs
    r_hat = r_vec / np.linalg.norm(r_vec)

    # Compute spherical angles
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    theta = np.arccos(z / r)  # polar angle from +z axis
    phi = np.arctan2(y, x)  # azimuth angle from +x axis in xy plane

    # Spherical basis vectors (same as used for projecting positions)
    e_r = r_hat
    e_theta = (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta))
    e_phi = (-np.sin(phi), np.cos(phi), 0.0)

    # Project velocity into components
    v_theta = np.dot(vel, np.array(e_theta))
    v_phi = np.dot(vel, np.array(e_phi))
    'Calculated velocities!'
    return v_phi, v_theta


def minimizingFunction(snaps, lookback_times, smc_pos, smc_vel):
    SMC_pos = (3.40, -15.87, 17.46)  # kpc
    SMC_vel = (-27.31, -101.73, 46.54)  # kpc

    data = {'snapshot': snaps,
            'lookback_times': lookback_times,
            'smc_pos': smc_pos,
            'smc_vel': smc_vel,
            'f_pos': np.linalg.norm(smc_pos) - np.linalg.norm(np.array(SMC_pos)),
            'f_vel': np.linalg.norm(smc_vel) - np.linalg.norm(np.array(SMC_vel))}

    order = np.argsort(data['f_pos'])
    for k in data:
        data[k] = data[k][order]

    for k in data:
        print(data[k][:20])


def construct_bhPDF(run, subject, snap, frame, pixels=400, reDerive_lmcSnapshot=False, verbose=True):
    from hestia import get_redshift, get_lookbackTimes
    from hestia.astrometry import vrai_frame
    from hestia import retrieve_particles, get_softeningLength
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter

    # -------------------------------------
    snapSigma = 20  # the half-length of snapshot bounds centered on the principal snapshot to sample for PDF
    r_observer = (0, 0, -49.59)  # in kpc
    fov_deg = 4  # total span of fov in degrees
    n_samples = 2000  # number of time stamps for interpolated functions
    # -------------------------------------

    z_ = f'{get_redshift(run, snap):.3e}'
    h = 0.677

    verbose and print(f'\nen train de travailler au snapshot * {snap}, z = {z_} * ...')

    if reDerive_lmcSnapshot:
        from hestia import compute_LMCanalog_snapshot
        prospective_snapshots = [220, 307]
        verbose and print(f'\tcomputing appropriate snapshot for {frame} LMC frame')
        snap = compute_LMCanalog_snapshot(prospective_snapshots)
    else:  # else use snapshot 255
        snap = 255
    verbose and print(f'\tusing snapshot {snap}.')

    # Note: eps_BH varies with z, however in this routine it is static, with the assumption that within
    # the snapshot bounds, it does not vary signifcantly
    eps_BH = get_softeningLength(run, snap, part_type='PartType5')  # eps_BH(z) ~ 0.2 kpc
    verbose and print(f'\t\t... +- {snapSigma} snapshots,\n'
                      f'\t\t r_observer = {r_observer},\n'
                      f'\t\t fov = {fov_deg} x {fov_deg} deg^2,\n'
                      f'\t\tnumber of interpolation samples = {n_samples},\n'
                      f'\t\teps_BH = {1e3 * eps_BH} pc ;')

    #  in vrai frame, e_i = (east, north, line of sight); in faux frame, e_i = (x, y, z)_halo
    e_1, e_2, e_3 = np.array([]), np.array([]), np.array([])

    redshifts, times = get_lookbackTimes(run, range(snap - snapSigma, snap + snapSigma))
    for snap in range(snap - snapSigma, snap + snapSigma):
        bh = retrieve_particles(run, subject, snap, 'PartType5', verbose=verbose)

        if frame == 'vrai':
            bh = vrai_frame(bh, snap)
            e_1 = np.append(e_1, bh['LMC_Coordinates'][:, 0])  # east, in kpc
            e_2 = np.append(e_2, bh['LMC_Coordinates'][:, 1])  # north, in kpc
            e_3 = np.append(e_3, bh['LMC_Coordinates'][:, 2])  # line of sight, in kpc
        else:  # if synthetic frame has been explicitly requested
            e_1 = np.append(e_1, bh['position'][:, 0])  # x, in kpc
            e_2 = np.append(e_2, bh['position'][:, 1])  # y, in kpc
            e_3 = np.append(e_3, bh['position'][:, 2])  # z, in kpc

    f_1 = interp1d(times, e_1, kind="cubic")  # cubic spline for smooth path
    f_2 = interp1d(times, e_2, kind="cubic")
    f_3 = interp1d(times, e_3, kind='cubic')

    t_fine = np.linspace(times.min(), times.max(), n_samples)
    e1_fine, e2_fine, e3_fine = f_1(t_fine), f_2(t_fine), f_3(t_fine)
    verbose and print('\tcomputed interpolation of coordinates')
    verbose and print(f'\t\tmean(' + ('E, N, LOS' if frame == 'LMC' else 'x, y, z')
                      + f') = ({np.mean(e_1)}, {np.mean(e_2)}, {np.mean(e_3)})')

    e1_fine -= r_observer[0]
    e2_fine -= r_observer[1]
    e3_fine -= r_observer[2]
    los = np.sqrt(e1_fine ** 2 + e2_fine ** 2 + e3_fine ** 2)
    lat = np.arccos(e2_fine / los)  # polar angle
    lon = np.arctan2(e1_fine, e3_fine)  # azimuthal angle

    lat = 90 - np.degrees(lat)   # latitude
    lon = np.degrees(lon)          # longitude
    print(f'\t\tmean(lat, lon) = ({np.average(lat)}, {np.average(lon)})')
    # Time spacing weights
    dt = np.diff(t_fine, prepend=t_fine[0])
    print(f'\t\tdelta_t = {1e3 * dt} Myr')
    # Field of view
    lat_max = fov_deg / 2
    lon_min, lon_max = -fov_deg / 2, + fov_deg / 2
    lat_min = -lat_max
    in_fov = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

    time_map = np.zeros((pixels, pixels))
    # bin edges
    lon_edges = np.linspace(lon_min, lon_max, pixels+1)
    lat_edges = np.linspace(lat_min, lat_max, pixels+1)
    lon_centers = 0.5*(lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5*(lat_edges[:-1] + lat_edges[1:])

    # Pixel scale in deg
    dlon = lon_edges[1] - lon_edges[0]
    dlat = lat_edges[1] - lat_edges[0]

    for i in np.where(in_fov)[0]:
        # Angular kernel sigma (deg) from softening length
        sigma_deg = np.degrees(eps_BH / los[i])

        # Convert to pixel units
        sigma_pix_lon = sigma_deg / dlon
        sigma_pix_lat = sigma_deg / dlat

        # Find nearest pixel
        ix = np.argmin(np.abs(lon_centers - lon[i]))
        iy = np.argmin(np.abs(lat_centers - lat[i]))

        # Add weight into a small patch around (ix, iy)
        patch = np.zeros_like(time_map)
        patch[iy, ix] = dt[i]

        # Gaussian blur the patch
        blurred = gaussian_filter(patch, sigma=(sigma_pix_lat, sigma_pix_lon), mode='constant')
        time_map += blurred

    if verbose:
        print(f"Observer: {r_observer}, Softening: {eps_BH} kpc")
        print(f"Max angular sigma ~ {sigma_deg:.3f} deg")

    """    
    # Grid in the x-y plane
    grid_size = 200
    extent = 2.0  # kpc across in each dimension (adjust to your halo)
    x_grid = np.linspace(-extent, extent, grid_size)
    y_grid = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Gaussian kernel for uncertainty (softening length as sigma)
    sigma = eps_BH
    heatmap = np.zeros_like(X)
    # For each sampled BH position, add Gaussian-weighted contribution
    for xi, yi in zip(x_fine, y_fine):
        rv = multivariate_normal([xi, yi], [[sigma ** 2, 0], [0, sigma ** 2]])
        heatmap += rv.pdf(np.dstack((X, Y)))

    # Normalize to represent time spent (integrates to 1)
    heatmap /= heatmap.sum()
    """

    # extent = 10
    # rnge = ((-1 * extent, extent), (-1 * extent, extent))
    # from hestia.gas import calc_temperature

    # try:
    #     stars['temperature'] = calc_temperature(stars['InternalEnergy'], stars['ElectronAbundance'], 0.76)
    # except KeyError:
    #     pass
    # image, x_e, y_e = np.histogram2d(stars['LMC_Coordinates'][:, 0], stars['LMC_Coordinates'][:, 1],
    #                                  # weights=1/stars['temperature'],
    #                                  bins=pixels, range=np.array(rnge))

    output_base = '/z/rschisholm'
    output_path = '/halos/09_18_lastgigyear/halo_08/observables/bhPDF/output.npz'
    np.savez(output_base + output_path, image=time_map, lon_edges=lon_edges, lat_edges=lat_edges)
    return output_path


def construct_NH0(run, subject, snap, pixels, verbose=True):
    from hestia import get_redshift
    from hestia import retrieve_particles
    from hestia import get_halo_params

    halo_id = None
    z_ = f'{get_redshift(run, snap):.3e}'
    h = 0.677

    verbose and print(f'\nen train de travailler au snapshot * {snap}, z = {z_} * ...')

    # Retrieves particles either directly from full snapshot files
    # or processed halo snapshot files, via processHalo.py
    cells = retrieve_particles(run, subject, snap, float(z_), 'PartType0', verbose=verbose)

    if subject == 'stream':
        # --------------------------------------
        r_observer = (0, 0, 55)  # position of observer in halo reference frame (in kpc)
        # --------------------------------------

        # Estimate smoothing lengths from mass and density
        volume = cells['Masses'] / cells['Density']  # kpc^3
        hsml = (3 / (4 * np.pi) * volume) ** (1 / 3)  # kpc
        nH_map, lon_edges, lat_edges = projectOntoSky(cells['position'],
                                                      hsml=hsml, masses=cells['Masses'],
                                                      h_frac=cells['GFM_Metals'][:, 0],
                                                      f_H0=cells['NeutralHydrogenAbundance'],
                                                      r_observer=r_observer, num_pixels=pixels)
        # handles the satellite, i.e. SMC-analog
        smc_params = get_halo_params('09_18', 'smc', snap)
        smc = {
            'ParticleIDs': np.ones(1),  # dummy particle id
            'Coordinates': smc_params['halo_pos'] / h,
            'Velocities': smc_params['halo_vel'],
            'Masses': smc_params['M_halo'] / h
        }
        r_smc, v_smc = cartesian_to_projected_spherical(snap, smc, r_observer=r_observer)

        data_to_save = {'nH_map': nH_map, 'lon_edges': lon_edges, 'lat_edges': lat_edges,
                        'redshift': z_, 'r_smc': r_smc, 'v_smc': v_smc}

        """
        if snap == 119:
            smc_xyz = np.array([46489.3459, 49353.8503, 50022.3398]) / h  # in kpc, taken manually from AHF.dat file
            smc_vel = np.array([-355.17, 627.59, -78.98])  # in km/s, taken manually from AHF.dat file
        elif snap == 123:
            smc_xyz = np.array([46371.4624, 49613.5356, 49966.3674]) / h
            smc_vel = np.array([-259.21, 324.78, -201.75])
        elif snap == 124:
            smc_xyz = np.array([46341.0743, 49644.1782, 49945.0037]) / h
            smc_vel = np.array([-387.53, 392.64, -247.20])
        elif snap == 125:
            smc_xyz = np.array([46287.1002, 49709.4511, 49915.7042]) / h
            smc_vel = np.array([-380.96, 537.98, -175.10])
        elif snap == 127:
            smc_xyz = np.array([46213.5841, 49839.6366, 49891.8326]) / h  # in kpc, taken manually from AHF.dat file
            smc_vel = np.array([-249.60, 572.15, -30.83])  # in km/s, taken manually from AHF.dat file
        else:
            print('Error: missing smc_xyz and smc_vel entries!')
            exit(1)
        """

    else:
        print('Error: invalid subject!')
        exit(1)

    verbose and print(f'termine avec le snapshot {snap}.')

    output_base = '/z/rschisholm'
    output_path = f'/halos/{run}/{subject}/syntheticObservation/{run}_{subject}_{param}_snap{snap}.npz'

    np.savez_compressed(output_base + output_path, **data_to_save)
    return output_path


def main():
    # PARAMETERS TO CHANGE
    # ------------------------------------
    run = '09_18'
    subject = 'stream'  # chosen halo frame of reference, or 'stream' for MS-analog
    snap = 119
    bins = 200
    # ------------------------------------

    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    # Positional or optional argument for simulation run
    parser.add_argument('run', nargs='?', default=run,
                        help='simulation run')

    # Positional or optional argument for halo to be processed
    parser.add_argument('subject', nargs='?', default=subject,
                        help='subject to be processed')

    # Positional or optional argument for particle type to be processed
    parser.add_argument('snap', nargs='?', default=snap,
                        help='snapshot to be processed')

    parser.add_argument('--N_H0', dest='N_H0', action='store_true',
                        help='construct neutral H column densities')
    parser.set_defaults(N_H0=False)

    parser.add_argument('--bh', dest='bh', action='store_true',
                        help='construct central blackhole probability density field')
    parser.set_defaults(N_H0=False)

    parser.add_argument('--vrai', dset='vrai', action='store_true',
                        help='transform to real life LMC frame')

    parser.add_argument('--faux', dset='faux', action='store_true',
                        help='transform to synthetic LMC frame')

    # Optional arguments
    parser.add_argument("--pixels", type=int, default=bins, help='side length of image in pixels')
    parser.add_argument('--v', dest='verbose', action='store_true', help='verbose print statements')
    parser.set_defaults(sph=False)

    args = parser.parse_args()

    parameter = 'N_H0' if args.N_H0 else ('bh pdf' if args.bh else None)
    if not args.vrai and not args.faux:  # if no input is given
        frame = 'vrai' if args.bh else 'faux'  # set frame to vrai if bhPDF is requested
    else:
        frame = 'vari' if args.vrai else ('faux' if args.faux else None)

    print('--------------------------------------------------------------------------------\n'
          + f'Creating {parameter} map ...\n'
          + f'sim_run -- {args.run}\n'
          + f'subject -- {args.subject}\n'
          + f'snapshot -- {args.snap}\n'
          + f'frame -- {frame} LMC'
          + f'pixels -- {args.pixels}'
          + '--------------------------------------------------------------------------------')

    if args.N_H0:
        output_path = construct_NH0(args.run, args.subject, int(args.snap), pixels=args.pixels)
    elif args.bh:
        output_path = construct_bhPDF(args.run, args.subject, int(args.snap), frame, pixels=args.pixels)

    print('Finished writing image-map data file,\n' + output_path)


def plotting():
    from scripts.local.plots import dispatch_plot

    # PARAMETERS TO CHANGE
    # ------------------------------------
    run = '09_18_lastgigyear'
    subject = 'halo_08'  # chosen halo frame of reference, or 'stream' for MS-analog
    parameter = 'bhPDF'
    snap = 255
    chisholm2025_plot = False
    # ------------------------------------

    if chisholm2025_plot:
        from scripts.local.chisholm2025 import streamPlot
        streamPlot()

    elif parameter == 'bhPDF':
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig = plt.figure(figsize=(6, 6))
        data = np.load('/Users/ursa/smorgasbord/observables/bhPDF/output.npz')
        plt.imshow(data['image'].T,
                   extent=[data['lon_edges'][0], data['lon_edges'][-1],
                           data['lat_edges'][0], data['lat_edges'][-1]],
                   origin='lower', norm=LogNorm(vmin=1e-5, vmax=1e-3))
        plt.show()

    else:
        basePath = f'/Users/dear-prudence/smorgasbord/syntheticObservations/{parameter}{subject}/'

        dispatch_plot('syntheticObservation', f'{parameter}{subject}',
                      input_path=f'{basePath}{run}_{subject}_{parameter}_snap{snap}.npz',
                      output_path=f'{basePath}{run}_{subject}_{parameter}_snap{snap}.png')


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------

if __name__ == "__main__":
    if machine == 'geras':
        main()
    elif machine == 'dear-prudence':
        plotting()
