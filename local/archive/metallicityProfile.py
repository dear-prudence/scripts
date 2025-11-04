"""This script will extract particles from a specified snapshot and plot their relative mass density in
temperature-coordinate space, the so-called temperature profile of a specified halo"""


def add_temperature(data):
    temp_column = calc_temperature(u=np.array(data['InternalEnergy']), e_abundance=np.array(data['ElectronAbundance']),
                                   x_h=np.array(data['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    data['Temperature'] = temp_column
    return data


def histo(distances, temperatures, densities, metallicities, num_bins, size, aspect,
          volume_weight=True, z_weight=False, h_max=None):
    log_temperature_range = [4, 7]
    hist, x_edges, y_edges = np.histogram2d(distances, np.log10(temperatures),
                                            range=np.array([size, log_temperature_range]),
                                            bins=[num_bins, num_bins / aspect],
                                            weights=densities)
    if volume_weight is True:
        volumes = spherical_volume_array(x_edges, num_bins / aspect)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Compute the average temperature in each bin
            h_vol = np.divide(hist, volumes, where=(volumes != 0))
    else:
        h_vol = hist

    if z_weight is True:
        h_n = z_param_weight(h_vol, distances, temperatures, metallicities, densities, num_bins,
                             extent=np.array([size, log_temperature_range]), aspect=aspect)
    else:
        h_n = hist
    return h_n, x_edges, y_edges


def z_param_weight(hist, x, y, z, densities, bins, extent, aspect):
    z_solar = 0.0127
    threshold = 1
    # Compute the sum of metallicities in each bin
    sum_z, _, _ = np.histogram2d(x, np.log10(y), range=extent,
                                 bins=[bins, bins / aspect], weights=densities * z / z_solar)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        hist_z = np.divide(sum_z, hist, where=(hist != 0))
    hist_z[hist < threshold] = np.NaN
    return hist_z


def z_gradient(x, z, bins, extent):
    print('len x: ' + str(x))
    # Create a number density histogram
    hist, x_e = np.histogram(x, bins=bins, range=extent)
    print('hist: ' + str(hist))
    # Compute the sum of metallicities in each bin
    print('z-array: ' + str(z))
    j = 0
    for zed in z:
        if zed is np.NaN:
            print('There is a NaN at ' + str(zed) + '!')
        else:
            j += 1
    print('Num no NaNs: ' + str(j))
    sum_z, _ = np.histogram(x, bins=x_e, weights=np.array(z))
    print('sum_z: ' + str(sum_z))
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_z = np.divide(sum_z, hist, where=(hist != 0))
        print('avg_z: ' + str(avg_z))
    return avg_z, x_e


def filter_particles_by_param(data, parameter, inq, threshold, usage=False):
    if usage is True:
        print('Filtering particles by ' + str(parameter) + '...')
        for key in data.keys():
            data[key] = np.array(data[key])  # Convert to numpy array for easier indexing
        if inq == '<=':
            indices_to_keep = np.where(data[parameter] <= threshold)[0]
        elif inq == '<':
            indices_to_keep = np.where(data[parameter] < threshold)[0]
        elif inq == '>':
            indices_to_keep = np.where(data[parameter] > threshold)[0]
        else:
            print('Error: Invalid inq value!')
            exit(1)
        print('Done with ' + str(parameter) + ' filtering.')
        for key in data.keys():
            data[key] = data[key][indices_to_keep]
    return data


def spherical_volume_array(radii, ny):
    # Calculate the volumes for each distance
    volumes = (4 / 3.0) * np.pi * radii ** 3
    # Create an empty 2D array to hold the volumes
    volumes_array = np.empty((len(radii) - 1, ny))
    # Fill the 2D array with the volumes
    for i, volume in enumerate(volumes[:-1]):
        volumes_array[i] = volume
    return volumes_array


def filter_unphysical_metallicities(data):
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


def retrieve_particles(snap, halo, key_names, isolate):
    print('Processing Snapshot ' + snap + '...')
    halo_id, halo_pos, _, _ = get_halo_params(halo, snap)

    if isolate is True:
        # for when extracting the isolated halo snapshots, snapshot_xxx.halo.hdf5
        file_path = ('/z/rschisholm/storage/snapshots_' + halo
                     + '/snapshot_' + snap + '/snapshot_' + snap + '.' + halo + '.hdf5')
        particles_h = append_particles('PartType0', file_path, key_names=key_names)
    else:
        # for when extracting the entire snapshot (all particles from all halos)
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_'
                     + str(snap) + '/snapshot_' + str(snap) + '.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        all_particles = {name: None for name in key_names}
        for file_path in file_paths:
            all_particles = append_particles('PartType0', file_path, key_names=key_names,
                                             existing_arrays=all_particles)
        # Initial filtering to cut down on unnecessary distance calculations
        init_filtering_box = np.array([200, 200, 200])  # in ckpc/h
        l_b = halo_pos - (init_filtering_box / 2)  # set minimum bounds
        u_b = halo_pos + (init_filtering_box / 2)  # set maximum bounds
        particles_h = filter_particles(all_particles, min_b=l_b * 1e-3, max_b=u_b * 1e-3)
    return particles_h


def create_plot(plot_type, snap, halo, bins, size, aspect, params=None, h_max=None):
    h0_threshold = 1e-6
    z_threshold = 1.0
    z_solar = 0.0127  # primordial metallicity of the Sun
    # -----------------------
    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    # initiating parameters array which turns on and off certain functionalities
    if params is None:
        params = [True, True, False, False, False]
    isolate, volume_weight, z_weight, h0_filter, z_filter = params
    # initiate list of columns to extract from snapshot files
    key_names = ['Coordinates', 'InternalEnergy', 'ElectronAbundance', 'GFM_Metals', 'Density',
                 'StarFormationRate', 'NeutralHydrogenAbundance', 'GFM_Metallicity']
    # extracts the particles from the snapshot
    particles_h = retrieve_particles(snap_, halo, key_names, isolate)
    # transforms particles from h-units to co-moving units
    particles = cosmo_transform(particles_h, 'ckpc/h', 'ckpc')
    # Calculate distances to the center of the disk for each particle
    particles_w_d = distance_to_disk(particles, snap=snap_, frame='co-moving')  # in kpc

    if plot_type == 'histogram':
        # Add a temperature column to the particles
        particles_w_dt = add_temperature(particles_w_d)
        # If the H0_filter is marked True, will filter particles via neutral hydrogen abundance
        # and then subsequently do the same for filtering via metallicity
        particles_fini = filter_particles_by_param(
            filter_particles_by_param(particles_w_dt, 'NeutralHydrogenAbundance', '<=',
                                      threshold=h0_threshold, usage=h0_filter),
            'GFM_Metallicity', '<=', threshold=z_threshold * z_solar, usage=z_filter)
        return histo(particles_fini['Distances'], particles_fini['Temperature'],
                     particles_fini['Density'], particles_fini['GFM_Metallicity'],
                     num_bins=bins, size=size, aspect=aspect,
                     volume_weight=volume_weight, z_weight=z_weight, h_max=h_max)

    elif plot_type == 'lineplot':
        particles_fini = filter_unphysical_metallicities(filter_particles_by_param(
            filter_particles_by_param(particles_w_d, 'NeutralHydrogenAbundance', '<=',
                                      threshold=h0_threshold, usage=h0_filter),
            'GFM_Metallicity', '<=', threshold=z_threshold * z_solar, usage=z_filter))
        # some of the GFM_Metallicities are negative (unphysical), temporarily added a np.abs() function to correct,
        # need to go back and investigate more later
        return z_gradient(particles_fini['Distances'],
                          np.log10(particles_fini['GFM_Metals'][:, 4] / particles_fini['GFM_Metals'][:, 0]) + 12,
                          bins, size)

    else:
        print('Error: Invalid plot type!')
        exit(1)


# ------------------------------------
mode = 'dear-prudence'
# ------------------------------------

if mode == 'geras':
    import sys
    from hestia import calc_temperature
    from hestia import get_halo_params
    from hestia import time_edges
    from hestia import distance_to_disk

    if __name__ == "__main__":
        # -----------------------
        simulation_run = '09_18'
        Halo = 'lmc'
        snaps = [67, 127]
        spatial_size = [0, 250]  # in ckpc
        n_bins_per_kpc = 1
        aspect_ratio = 2  # aspect ratio of the image, s.t. each phase space bin is square
        plot_type = 'lineplot'  # 'histogram' for the temperature profile, 'lineplot' for the metalicity gradient
        isolated = False
        H0_filtering = False
        Z_filtering = False
        volume_weighing = False
        Z_weighing = True
        # -----------------------
        parameters = [isolated, volume_weighing, Z_weighing, H0_filtering, Z_filtering]

        if plot_type == 'histogram':
            n_bins = np.array(spatial_size[1] - spatial_size[0]) * n_bins_per_kpc
            if isinstance(snaps, int):
                H, d_edges, logT_edges = create_plot(plot_type, snaps, Halo, bins=n_bins,
                                                     size=np.array(spatial_size),
                                                     aspect=aspect_ratio, params=parameters)
                time_edges = time_edges(sim=simulation_run, snaps=[snaps])
            elif isinstance(snaps, list):
                for i in range(snaps[1], snaps[0], -1):
                    if i == snaps[1]:
                        H_i, d_edges, logT_edges = create_plot(plot_type, i, Halo, bins=n_bins,
                                                               size=np.array(spatial_size),
                                                               aspect=aspect_ratio, params=parameters)
                        h_m = np.max(H_i)
                        H = H_i
                    else:
                        # noinspection PyUnboundLocalVariable
                        H_i, d_edges, logT_edges = create_plot(plot_type, i, Halo, bins=n_bins,
                                                               size=np.array(spatial_size),
                                                               aspect=aspect_ratio, params=parameters,
                                                               h_max=h_m)
                        # Append the new snapshot
                        H = np.dstack((H, H_i))

                time_edges = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1))
            # noinspection PyUnboundLocalVariable
            np.savez('/z/rschisholm/storage/analytical_plots/metallicityProfiles/' + str(simulation_run)
                     + ('_snap' + str(snaps) if isinstance(snaps, int) else '_snapsMultiple')
                     + '_gas_temperatureProfile_' + Halo + ('_isolated' if isolated is True else '')
                     + '_radius' + str(spatial_size[1]) + 'kpc'
                     + ('_aspect1' if aspect_ratio == 1 else '')
                     + ('_volume' if volume_weighing is True else '')
                     + ('_metallicityWeight' if Z_weighing is True else '')
                     + ('_corona' if H0_filtering is True and Z_filtering is True else '')
                     + ('_filtered' if volume_weighing or H0_filtering or Z_filtering is True else '')
                     + '.npz',
                     data=H, x_e=d_edges, y_e=logT_edges, time=time_edges)

        elif plot_type == 'lineplot':
            n_bins = n_bins_per_kpc * spatial_size[1]

            for i in range(snaps[1], snaps[0], -1):
                H_i, d_edges = create_plot(plot_type, i, Halo, bins=n_bins,
                                           size=np.array(spatial_size), aspect=aspect_ratio,
                                           params=parameters)
                if i == snaps[1]:
                    H = H_i
                else:
                    # Append the new snapshot
                    H = np.dstack((H, H_i))

            time_edges = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1))
            np.savez('/z/rschisholm/storage/analytical_plots/metallicityProfiles/' + str(simulation_run)
                     + ('_snap' + str(snaps) if isinstance(snaps, int) else '_snapsMultiple')
                     + '_gas_metallicityGradient_' + Halo + ('_isolated' if isolated is True else '')
                     + '_radius' + str(spatial_size[1]) + 'kpc'
                     + ('_H0' if H0_filtering is True else '')
                     + ('_Z' if Z_filtering is True else '')
                     + ('_filtered' if volume_weighing or H0_filtering or Z_filtering is True else '')
                     + '_O.npz',
                     data=H, x_e=d_edges, time=time_edges)

        # Indicate that the script has completed its task
        print('Done!')
        # Terminate the script
        sys.exit(0)

elif mode == 'dear-prudence':
    from scripts.local.archive.plots_old import plot_metallicity_profile, plot_metal_gradient

    plot_type = 'lineplot'
    cmap = 'BuPu'

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/metallicityProfiles_lmc/'
                  '09_18_snapsMultiple_gas_metallicityGradient_lmc_radius250kpc_O.npz')
    output_path = '/Users/ursa/Desktop/smorgasbord/metallicityProfiles_lmc/metallicityGradient_O.png'

    # snaps = 108
    # snaps = [70, 94, 89, 90, 92]
    # snaps = [94, 106, 114, 119, 122]
    snaps = [77, 85, 94, 98, 105]
    # snap 94 (z = 0.526): "primordial LMC"
    # snap 108 (z = 0.258): fully-formed, near-peak mass LMC
    # snap 114 (z = 0.165): LMC peak mass, M_vir ~ 3.52e+11
    # snap 119 (z = 0.099): after major ejection of gas
    # snap 122 (z = 0.060): LMC passes R_vir of the MW
    # snap 127 (z = 0.0): present-day (M_vir ~ 1.85e+11, M_gas ~ 1.26e+10)
    # dear-prudence.plot_temperature_profile(len(snaps), input_path, output_path, snaps, beta)
    if plot_type == 'histogram':
        plot_metallicity_profile(input_path, output_path, snaps=snaps)
    elif plot_type == 'lineplot':
        plot_metal_gradient(input_path, output_path, snaps=snaps)

