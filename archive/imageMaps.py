import sys
import numpy as np
from scripts.hestia import append_particles, filter_particles
from scripts.hestia import get_halo_params
from scripts.hestia import calc_temperature
from scripts.hestia import transform_haloFrame, rid_h_units


def create_weighted_histogram(part_type, param, x, y, weights, masses, background, bins, bounds=None):
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=bins, range=bounds, weights=masses)
    # Compute the sum of densities in each bin
    sum_hist, _, _ = np.histogram2d(x, y, bins=bins, range=bounds, weights=weights * masses)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_hist = np.divide(sum_hist, hist, where=(hist != 0))

    threshold = 1
    avg_hist[hist < threshold] = background
    """
    avg_hist[hist < threshold] = 1
    for i in range(bins[0]):
        for j in range(bins[0]):
            if hist[i, j] < threshold:
                sample_radius = 4  # radius of kernel to average over
                # Define valid neighbor ranges, ensuring we stay inside the array
                i_min, i_max = max(0, i - sample_radius), min(bins[0], i + sample_radius + 1)
                j_min, j_max = max(0, j - sample_radius), min(bins[0], j + sample_radius + 1)

                # Extract the surrounding subarray
                neighborhood = avg_hist[i_min:i_max, j_min:j_max].flatten()
                # Remove the center bin itself if it exists in the extracted neighborhood
                neighborhood = np.delete(neighborhood, np.where((neighborhood == avg_hist[i, j]))[0])
                # Replace the bin with the mean of the valid neighbors
                avg_hist[i, j] = np.mean(neighborhood) if neighborhood.size > 0 else avg_hist[i, j]
    """
    if part_type == 'PartType0':
        if param == 'column_H0':
            return np.log10(sum_hist), x_e, y_e
        else:
            return avg_hist, x_e, y_e
    # returns the total (mass) per bin, instead of average mass per particle
    elif part_type == 'PartType4':
        vol_per_bin = float(bounds[0, 1] - bounds[0, 0] / bins[0])
        return sum_hist / vol_per_bin, x_e, y_e


def filter_unphysical_Z(data):
    for key in data.keys():
        data[key] = np.array(data[key])  # Convert to numpy array for easier indexing
    # Get the indices where metallicities are negative
    original_length = len(data['GFM_Metallicity'])
    indices_to_keep = np.where(data['GFM_Metallicity'][:] > 0)[0]
    print('Done with unphysical metallicity filtering, number of thrown-out particles: '
          + str(original_length - len(indices_to_keep)))
    for key in data.keys():
        data[key] = data[key][indices_to_keep]
    return data


def filter_stars(particles):
    # stars have SFT > 0, wind particles have SFT < 0
    stellar_mask = particles['GFM_StellarFormationTime'] > 0
    return {key: val[stellar_mask] for key, val in particles.items()}


def add_temperature(particles):
    temp_column = calc_temperature(u=np.array(particles['InternalEnergy']), e_abundance=np.array(particles['ElectronAbundance']),
                                   x_h=np.array(particles['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    particles['Temperature'] = temp_column
    return particles


def add_velocity(particles):
    v_xyz = np.zeros(particles['Halo_Velocities'].shape[0])
    for j in range(particles['Halo_Velocities'].shape[0]):
        v_xyz[j] = np.sqrt(particles['Halo_Velocities'][j, 0] ** 2
                           + particles['Halo_Velocities'][j, 1] ** 2
                           + particles['Halo_Velocities'][j, 2] ** 2)
    particles['Vel_Mag'] = v_xyz
    return particles


def add_orbital_velocities(particles, run, halo, snap):
    from scripts.hestia import get_rotation_curve
    starting_idx = 307 if run == '09_18_lastgigyear' else 127

    """Compute rotation curves only if they haven't been computed before."""
    if not hasattr(add_orbital_velocities, "rotation_curves"):
        # Simulate computationally expensive array creation
        add_orbital_velocities.rotation_curves = get_rotation_curve(run, halo, snaps)
    print('rotation_curves = ' + str(add_orbital_velocities.rotation_curves))

    # Compute radial distance r and angle theta
    r = np.sqrt(particles['Halo_Coordinates'][:, 0] ** 2 + particles['Halo_Coordinates'][:, 1] ** 2)
    theta = np.arctan2(particles['Halo_Coordinates'][:, 1], particles['Halo_Coordinates'][:, 0])
    # Compute polar velocities
    v_r = particles['Halo_Velocities'][:, 0] * np.cos(theta) + particles['Halo_Velocities'][:, 1] * np.sin(theta)
    v_phi = -particles['Halo_Velocities'][:, 0] * np.sin(theta) + particles['Halo_Velocities'][:, 1] * np.cos(theta)

    # Interpolate bulk rotation velocity at each particle's radius
    v_bulk = np.interp(r, add_orbital_velocities.rotation_curves[starting_idx - snap, :, 0],
                       add_orbital_velocities.rotation_curves[starting_idx - snap, :, 1])
    # Subtract bulk rotation from v_phi
    v_phi_corr = v_phi - v_bulk
    # Convert back to Cartesian coordinates
    particles['Orbital_Velocities'] = np.zeros(particles['Halo_Velocities'].shape)
    particles['Orbital_Velocities'][:, 0] = v_r * np.cos(theta) - v_phi_corr * np.sin(theta)
    particles['Orbital_Velocities'][:, 1] = v_r * np.sin(theta) + v_phi_corr * np.cos(theta)
    particles['V_phi_adjusted'] = v_phi_corr
    return particles


def rhoH_to_nH(particles, mu=0.59):
    rhoH = particles['Density'] * particles['GFM_Metals'][:, 0]
    # 1.00784 for mostly H0, 0.59 for mostly H1
    M_solar_to_g = 1.989e33  # Solar mass in grams
    kpc_to_cm = 3.086e21  # 1 kpc in cm
    amu_to_g = 1.66053906660e-24  # Atomic mass unit in grams
    # Safeguard against extremely large inputs
    rhoH = np.asarray(rhoH, dtype=np.float64)  # Ensure it's a float array or scalar
    if np.any(rhoH <= 0):
        raise ValueError("Hydrogen mass density must be positive and non-zero.")
    # Convert hydrogen mass density to g/cm^3
    rhoH_cgs = (rhoH * M_solar_to_g) / (kpc_to_cm ** 3)
    # Convert mass density to number density
    m_h = mu * amu_to_g  # Mass of a hydrogen atom/molecule in grams
    nH = rhoH_cgs / m_h
    particles['n_H'] = nH
    return particles


def add_nH(particles, snap, type_H):
    from scripts.hestia import calc_numberDensity, calc_temperature, calc_fH0_cloudy

    X_H = 0.76
    mu = 0.59 if type_H == ('H1' or 'HII') else 1.00784

    if 'NeutralHydrogenAbundance' in particles.keys():
        particles['f_H'] = particles['GFM_Metals'][:, 0]
        particles['n_H'] = calc_numberDensity(particles['Density'] * particles['f_H'], mu=mu)
        particles['f_H0'] = particles['NeutralHydrogenAbundance']
        particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                    e_abundance=np.array(particles['ElectronAbundance']),
                                                    x_h=particles['GFM_Metals'][:, 0])
    else:
        particles['f_H'] = np.zeros(particles['Density'].shape) + X_H
        particles['n_H'] = calc_numberDensity(particles['Density'] * X_H, mu=mu)
        particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                    e_abundance=np.array(particles['ElectronAbundance']), x_h=X_H)
        particles['f_H0'] = calc_fH0_cloudy(particles['Temperature'])

    print('mean(particles[temperature] = ' + str(np.mean(particles['Temperature'])))
    print('mean(particles[f_H]) = ' + str(np.mean(particles['f_H'])))
    print('mean(particles[n_H]) = ' + str(np.mean(particles['n_H'])))
    print('mean(particles[f_H0]) = ' + str(np.mean(particles['f_H0'])))
    print('mean(particles[n_e] = ' + str(np.mean(particles['ElectronAbundance'] - 1)))

    return particles


def param_processing(part_type, param, particles, run, halo, snap):
    X_H = 0.76
    if part_type == 'PartType0':

        if param == 'massDen':
            fini = particles
            weights = particles['Density']
            masses = particles['Masses']
            background = 30  # mass density of IGM

        elif param == 'temperature':
            fini = particles
            weights = calc_temperature(u=np.array(particles['InternalEnergy']),
                                       e_abundance=np.array(particles['ElectronAbundance']),
                                       x_h=X_H)
            # masses = particles['Masses'] * particles['GFM_Metals'][:, 0] * particles['NeutralHydrogenAbundance'] \
            #     if halo == 'stream' else particles['Masses']
            masses = particles['Masses']
            background = 1e5  # temperature of IGM

        elif param == 'velMag':
            fini = add_velocity(particles)
            weights = fini['Vel_Mag']
            masses = particles['Masses']
            background = 0

        elif param == 'v_phi':
            fini = add_orbital_velocities(particles, run, halo, snap)
            # weights = np.sqrt(fini['Orbital_Velocities'][:, 0] ** 2 + fini['Orbital_Velocities'][:, 1] ** 2)
            weights = fini['V_phi_adjusted']
            masses = particles['Masses']
            background = 0

        elif param == 'metallicity':
            fini = filter_unphysical_Z(particles)
            weights = np.log10(fini['GFM_Metallicity'] / fini['GFM_Metals'][:, 0])
            masses = particles['Masses']
            background = -6,  # very metal poor primordial gas

        elif param == 'num_H0':
            fini = add_nH(particles, snap, type_H='H0')
            weights = fini['n_H'] * fini['f_H0']
            masses = fini['Masses'] * X_H * particles['f_H0']
            background = 1e-10,  # number density of H0 in IGM

        elif param == 'num_H1':
            fini = add_nH(particles, snap, type_H='H1')
            weights = fini['n_H'] * (1 - fini['f_H0'])
            masses = particles['Masses'] * X_H * (1 - particles['f_H0'])
            background = 1e-7,  # number density of H1 in IGM

        elif param == 'column_H0':
            fini = rhoH_to_nH(particles, mu=1.00784)
            weights = fini['n_H'] * fini['NeutralHydrogenAbundance']
            # approximate length of each Voroni cell in cm (assuming shape is a cuboid)
            masses = 3.086e21 * (particles['Masses'] / particles['Density']) ** 1/3
            background = 1

        elif param == 'E_diss':
            fini = particles
            weights = fini['EnergyDissipation']
            # only for HI
            masses = particles['Masses'] * particles['GFM_Metals'][:, 0] * particles['NeutralHydrogenAbundance']
            background = 0

        elif param == 'cooling_rate':
            fini = particles
            weights = fini['GFM_CoolingRate']
            # for all H
            masses = particles['Masses'] * particles['GFM_Metals'][:, 0]
            background = 0

        elif param == 'agn_radiation':
            fini = particles
            weights = fini['GFM_AGNRadiation']
            # only for HI
            masses = particles['Masses'] * particles['GFM_Metals'][:, 0] * particles['NeutralHydrogenAbundance']
            background = 1e-10

        else:
            print('Error: \"' + param + '\" is an invalid parameter!')
            exit(0)

    elif part_type == 'PartType4':
        if param == 'massDen':
            fini = filter_stars(particles)
            weights = fini['Masses']
            masses = particles['Masses']
            background = 1e-10  # arbitrarily low stellar density for IGM
        elif param == 'velocity':
            fini = add_velocity(filter_stars(particles))
            weights = fini['Vel_Mag']
            masses = particles['Masses']
            background = 0
        else:
            print('Error: \"' + param + '\" is an invalid parameter!')
            exit(0)

    elif part_type == 'PartType1':
        if param == 'massDen':
            fini = particles
            weights = fini['Masses']
            masses = fini['Masses']
            background = 1e-10  # arbitrarily low stellar density for IGM

    else:
        print('Error: \"' + part_type + '\" is an invalid particle type!')
        exit(0)

    return fini, weights, masses, background


def retrieve_particles(run, halo, snap, z, part_type, using_isolated_halo=False, previous_halo_id=None):
    import h5py

    h = 0.677
    alpha = 1 / h if halo == 'stream' else 1e-3
    snap_ = '0' + str(snap) if snap < 100 else str(snap)

    halo_params = get_halo_params(run, halo, snap, previous_halo_id=previous_halo_id)
    halo_id, pos_h, vel_h, l_h, r_vir_h = (halo_params['halo_id_zi'], halo_params['halo_pos'],
                                           halo_params['halo_vel'], halo_params['halo_l'], halo_params['R_vir'])

    lb, ub = (pos_h - 4 * r_vir_h) * alpha, (pos_h + 4 * r_vir_h) * alpha

    if using_isolated_halo is True:
        file_path = ('/z/rschisholm/storage/snapshots_' + halo
                     + ('/snapshots_lmc_traditional' if run == '09_18' else '') + '/snapshot_' + snap_ + '.hdf5')
        with h5py.File(file_path, 'r') as file:
            keys = file[part_type].keys()
        all_particles = {name: None for name in keys}
        all_particles = append_particles(part_type, file_path, key_names=keys, existing_arrays=all_particles)

    else:
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run + '/output_2x2.5Mpc/snapdir_'
                     + snap_ + '/snapshot_' + snap_ + '.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        with h5py.File(base_path + '0' + file_extension, 'r') as file:
            keys = file[part_type].keys()
        all_particles = {name: None for name in keys}
        for file_path in file_paths:
            all_particles = append_particles(part_type, file_path, key_names=keys,
                                             existing_arrays=all_particles)

    processed_particles = transform_haloFrame(run, halo, snap,
                                              rid_h_units(
                                         filter_particles(all_particles, lb, ub), z, part_type),
                                              previous_halo_id=previous_halo_id) \
        if halo != 'stream' else transform_haloFrame(run, halo, snap, filter_particles(all_particles, lb, ub),
                                                     previous_halo_id=previous_halo_id)
    return processed_particles, halo_id, r_vir_h / h


def make_snap(part_type, param, particles_full_depth, weights_full_depth, masses_full_depth,
              background, bins, axis, size):
    h = 0.677

    cartesian = np.array([0, 1, 2])
    # Get the indices not equal to the specified axis
    axs = cartesian[cartesian != axis]

    bounds = np.array([[-1 * size[axs[0]] / 2, size[axs[0]] / 2], [-1 * size[axs[1]] / 2, size[axs[1]] / 2]])

    column_mask = np.abs(particles_full_depth['Halo_Coordinates'][:, axis]) < (size[axis] / 2)
    particles = {key: val[column_mask] for key, val in particles_full_depth.items()}
    weights = weights_full_depth[column_mask]
    masses = masses_full_depth[column_mask]

    return create_weighted_histogram(part_type, param,
                                     particles['Halo_Coordinates'][:, axs[0]], particles['Halo_Coordinates'][:, axs[1]],
                                     weights=weights, masses=masses, background=background,
                                     bins=bins, bounds=bounds)


def package_data(run, param, dims, n_bins, particle_type, halo, bool_isolated_halo):
    from scripts.hestia import get_lookbackTimes, get_redshift

    print('---------------------------------------------------\n'
          + 'Creating image-map...\n'
          + 'sim_run -- ' + run + '\n'
          + 'halo -- ' + halo + '\n'
          + 'part_type -- ' + particle_type + '\n'
          + 'parameter -- ' + param + '\n'
          + 'dims, bins -- ' + str(dims) + ', ' + str(n_bins) + '\n'
          + '---------------------------------------------------')

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4'}
    part_type = part_to_type[particle_type]

    bins = [n_bins, n_bins]  # for the 2-dim histograms
    redshifts, lookback_times = get_lookbackTimes(run, snaps)

    # Dictionary to hold results for each axis/dimension
    all_image = {'y-z': None, 'x-z': None, 'x-y': None,
                 'y_e': None, 'z_e': None, 'x_e': None}

    halo_id = None
    virial_radii = np.array([])
    for snap in range(snaps[1], snaps[0], -1):
        z = float(get_redshift(run, snap))
        print('------------ $z = {}$ ------------'.format(z))

        # l_b and u_b are in ckpc (converted from _h units)
        particles, halo_id, R_vir = retrieve_particles(run, halo, snap, z, part_type,
                                                       using_isolated_halo=bool_isolated_halo,
                                                       previous_halo_id=halo_id)
        print('Particles retrieved!')
        particles_fini, weights, masses, background = param_processing(part_type, param, particles, run, halo, snap)
        print('Particles processed!')
        virial_radii = np.append(virial_radii, R_vir)

        print('len(processed_particles) = ' + str(len(particles['ParticleIDs'])))
        print('len(particles_fini) = ' + str(len(particles_fini['ParticleIDs'])))

        print('mean(Coordinates) = ' + str([np.mean([particles['Halo_Coordinates'][:, 0]]),
                                            np.mean([particles['Halo_Coordinates'][:, 1]]),
                                            np.mean([particles['Halo_Coordinates'][:, 2]])]))

        print('weights[0] = ' + str(weights[0]))
        print('masses[0] = ' + str(masses[0]))
        print('background = ' + str(background))

        # Loop through the axes
        for axis, name_i, edge_i in zip(range(3), ['y-z', 'x-z', 'x-y'], ['y_e', 'z_e', 'x_e']):

            # creates the prism to restrict particles being plotted by
            S = np.array([dims[1], dims[1], dims[1]])
            S[axis] = dims[0]

            current_image, i_edge, j_edge = make_snap(part_type, param, particles_fini, weights, masses, background,
                                                      bins, axis, S)

            # Initialize arrays if this is the first snapshot

            # np.dstack is the reason you have to take the transpose in the plotting routine, consider np.vstack

            if snap == snaps[1]:
                all_image[name_i] = current_image
                # store the edges, for the x-z case, store the z_edges (second axis) instead
                all_image[edge_i] = j_edge if axis == 1 else i_edge
            else:
                # Append new snapshot data
                all_image[name_i] = np.dstack((all_image[name_i], current_image))
                all_image[edge_i] = j_edge if axis == 1 else i_edge  # edges only need to be stored once

        # Now all_image['x-y'], all_image['y-z'], image['x-z'], etc., hold the stacked data
        print('Snapshot ' + str(snap) + ': check')

    # Combine all dictionaries into one
    data_to_save = all_image.copy()  # Start with all_planes
    data_to_save['redshifts'] = redshifts  # timestamps
    data_to_save['lookback_times'] = lookback_times
    data_to_save['column_width'] = round(dims[1] / float(n_bins), 3)  # in ckpc
    data_to_save['column_depth'] = dims[0]  # in ckpc
    data_to_save['image_size'] = dims[1]  # in ckpc
    data_to_save['virial_radii'] = virial_radii

    # Save data
    output_path = ('/z/rschisholm/storage/images/' + param + '/'
                   + run + '_' + halo + '_' + particle_type + '_' + param + '_'
                   + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc.npz')
    np.savez(output_path, **data_to_save)

    print('Done!\n----------------------------------------------')
    print('scp -P 2222 rschisholm@geras.aip.de:' + output_path +
          ' /Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
          + run + '_' + halo + '_' + particle_type + '_' + param + '_'
          + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc.npz')
    print('----------------------------------------------')
    sys.exit(0)


def plotting(typeplot, run, halo, sspt, param, dims, particle_type, planes=None):
    from scripts.local.archive.plots_old import plot_imageMap, plot_imageMap_frames
    from scripts.local.archive.plots_old import plot_chisholm2025_fig1

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
                  + run + '_' + halo + '_' + particle_type + '_' + param + '_'
                  + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc.npz')

    if typeplot == 'image':
        output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
                       + run + '_' + halo + '_' + particle_type + '_' + param + '_'
                       + 'snap' + str(sspt) + '_' + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc' + '.pdf')
        plot_imageMap(param, sspt, input_path, output_path, scale=dims, plane=planes[0])

    elif typeplot == 'panels':
        output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
                       + run + '_' + halo + '_' + particle_type + '_' + param + '_'
                       + 'panels' + str(sspt[0]) + '-' + str(sspt[-1]) + '_'
                       + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc' + '.pdf')
        plot_chisholm2025_fig1(param, sspt, input_path, output_path, scale=dims, plane=planes[0])

    elif typeplot == 'frames':
        output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
                       + str(dims[0]) + 'x' + str(dims[1]) + '_frames/')
        # optional argument "axis" where default is face-on, edge-on
        plot_imageMap_frames(param, input_path, output_path, scale=dims, planes=planes)


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
parameter = 'temperature'
dims_ = [300, 600]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
bins_ = 400
snaps = [68, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
halo_ = 'lmc'  # chosen halo frame of reference, or 'stream' for MS-analog
particle_type_ = 'gas'
# frame_ = 'halo'  # will transform to halo coords if "frame_" is set to 'halo', else will transform to specified halo
isolated_halo = False
# ------------------------------------
type_plot = 'frames'
planes_ = ['x-y', 'x-z']
# snapshots = [95, 104, 111, 116, 161, 307]
# snapshots = [95, 104, 111, 116, 121, 127]
snapshots = 127
# ------------------------------------

if machine == 'geras':
    package_data(simulation_run, parameter, dims_, bins_, particle_type_, halo_, isolated_halo)

elif machine == 'dear-prudence':
    plotting(type_plot, simulation_run, halo_, snapshots, parameter, dims_, particle_type_, planes=planes_)
