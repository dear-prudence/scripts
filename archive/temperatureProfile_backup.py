"""This script will extract particles from a specified snapshot and plot their relative mass density in
temperature-coordinate space, the so-called temperature profile of a specified halo"""
import sys
from archive.hestia import time_edges, center_halo
from archive.hestia import calc_temperature
from archive.hestia import transform_haloFrame
from archive.hestia import get_halo_params


def add_temperature(particles):
    temp_column = calc_temperature(u=np.array(particles['InternalEnergy']), e_abundance=np.array(particles['ElectronAbundance']),
                                   x_h=np.array(particles['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    particles['Temperature'] = temp_column
    return particles


def add_distances(particles):
    # Initialize an array to store distances
    distances = np.zeros(particles['Coordinates'].shape[0])
    # Compute distances for each particle
    for i in range(particles['Halo_Coordinates'].shape[0]):
        distances[i] = np.linalg.norm(particles['Halo_Coordinates'][i])
    # Add the new column to the data dictionary
    particles['Distances'] = distances
    return particles


def create_histogram(distances, temperatures, densities, num_bins, size, h_max=None):
    log_temperature_range = [4, 7]
    aspect_ratio = 2
    print('Max distances: ' + str(np.max(distances)))
    print('size: ' + str(size))
    hist, x_e, y_e = np.histogram2d(distances, np.log10(temperatures),
                                    range=np.array([size, log_temperature_range]),
                                    bins=[num_bins, num_bins / aspect_ratio],
                                    weights=densities)
    h_n = hist
    print('max(hist): ' + str(np.max(h_n)))
    print('h_max: ' + str(h_max))
    print('-----------------------------')
    h_normed = h_n / np.max(h_n) if h_max is None else h_n / h_max
    return h_normed, x_e, y_e, np.max(h_n)


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


def retrieve_particles(snap_i, z, sim_run, size, halo, bool_isolated_halo=False):
    from archive.hestia import halo_dictionary
    h = 0.677
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    halo_id = halo_dictionary(sim_run, halo)
    sze = np.array([size[1], size[1], size[1]]) * h  # in kpc_h units

    l_b, u_b = center_halo(run=sim_run, halo_id=halo_id, snap=snap, size=sze) * 1e-3  # these are in _h units!
    # l_b and u_b carve a cube with side length 2x specified in "size", to account for the gaps in spatial
    # distribution of particles after the halo coordinate transformation

    key_names = ['Coordinates', 'Density', 'ElectronAbundance', 'GFM_Metallicity', 'GFM_Metals',
                 'InternalEnergy', 'Masses', 'NeutralHydrogenAbundance', 'StarFormationRate', 'Velocities']

    if bool_isolated_halo is True:
        base_path = ('/z/rschisholm/storage/snapshots_' + halo + '/snapshot_' + snap + '.hdf5')
        file_paths = [base_path]
    else:
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim_run + '/output_2x2.5Mpc/snapdir_'
                     + snap + '/snapshot_' + snap + '.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
    all_particles = {name: None for name in key_names}
    print('Processing Snapshot ' + snap + '...')
    for file_path in file_paths:
        all_particles = append_particles('PartType0', file_path, key_names=key_names,
                                         existing_arrays=all_particles)

    # Filter particles based on the bounding box
    if bool_isolated_halo is True:
        filtered_particles = cosmo_transform(all_particles, 'ckpc/h', 'ckpc', part_type='PartType0')
    else:
        filtered_particles = cosmo_transform(filter_particles(all_particles, l_b, u_b),
                                             'ckpc/h', 'ckpc', z, part_type='PartType0')

    rotated_particles = transform_haloFrame(sim_run, halo_id, snap, filtered_particles)
    return rotated_particles, l_b / h, u_b / h


def create_plot(particles, bins, size, r_vir, params=None, threshold_values=None, h_max=None):
    h0_threshold, z_threshold, max_rho_threshold, min_rho_threshold = threshold_values
    h = 0.677
    z_solar = 0.0127  # primordial metallicity of the Sun
    # -----------------------

    # initiating parameters array which turns on and off certain functionalities
    if params is None:
        params = [False, False, False]
    isolated, h0_filter, z_filter, rho_filter = params

    particles_unfiltered = add_temperature(add_distances(particles))

    print('Avg dist = ' + str(np.average(particles['Distances'])))
    print('Max dist = ' + str(np.max(particles['Distances'])))

    # If the H0_filter is marked True, will filter particles via neutral hydrogen abundance
    # and then subsequently do the same for filtering via metallicity
    particles_filtered_H = filter_particles_by_param(particles_unfiltered, 'NeutralHydrogenAbundance',
                                                     '<=', threshold=h0_threshold, usage=h0_filter)
    particles_filtered_z = filter_particles_by_param(particles_filtered_H, 'GFM_Metallicity',
                                                     '<=', threshold=z_threshold * z_solar, usage=z_filter)
    particles_filtered_max_rho = filter_particles_by_param(particles_filtered_z, 'Density',
                                                           '<', threshold=max_rho_threshold, usage=rho_filter)
    particles_fini = filter_particles_by_param(particles_filtered_max_rho, 'Density',
                                               '>', threshold=min_rho_threshold, usage=rho_filter)

    print('Len particles_fini = ' + str(len(particles_fini['Masses'])))

    mass_enclosed_radius = r_vir / h  # in kpc
    radius_mask = particles['Distances'] <= mass_enclosed_radius
    total_mass = np.sum(particles_fini['Masses'][radius_mask] * 1e10)

    avg_temp = np.average(particles_fini['Temperature'][radius_mask])

    average_nH0 = np.average(particles_fini['NeutralHydrogenAbundance'])

    return (create_histogram(particles_fini['Distances'], particles_fini['Temperature'], particles_fini['Density'],
                             num_bins=bins, size=size, h_max=h_max),
            total_mass, avg_temp, average_nH0)


def package_data(sim_run, halo, snaps, size, bins_per_kpc,
                 bool_isolated_halo, bool_h0_filtering, bool_z_filtering, bool_rho_filtering,
                 h0_threshold, z_threshold, rho_threshold):
    n_bins = np.array(size[1] - size[0]) * bins_per_kpc
    parameters = [bool_isolated_halo, bool_h0_filtering, bool_z_filtering, bool_rho_filtering]
    thresholds = [h0_threshold, z_threshold, rho_threshold[0], rho_threshold[1]]

    time_e = time_edges(sim=sim_run, snaps=np.arange(snaps[1], snaps[0], step=-1))

    for i in range(snaps[1], snaps[0], -1):

        particles, _, _ = retrieve_particles(i, time_e[127 - i, 0], sim_run, size, halo, bool_isolated_halo)

        _, _, _, _, r_vir = get_halo_params(sim_run, halo, i)

        # in general, this condition is for redshift z = 0.0
        if i == snaps[1]:
            [H_i, d_e, logT_e, h0_max], mass_i, temp_i, nH0 = create_plot(particles, n_bins, size, r_vir, parameters,
                                                                          thresholds)
            h_m = h0_max
            H = H_i
            masses = mass_i
            temps = temp_i
            nH0s = nH0
        else:
            # noinspection PyUnboundLocalVariable
            [H_i, d_e, logT_e, _], mass_i, temp_i, nH0 = create_plot(particles, n_bins, size, r_vir, parameters,
                                                                     thresholds, h_max=h_m)
            # Append the new snapshot
            H = np.dstack((H, H_i))
            masses = np.append(masses, mass_i)
            temps = np.append(temps, temp_i)
            nH0s = np.append(nH0s, nH0)

    # noinspection PyUnboundLocalVariable
    np.savez('/z/rschisholm/storage/analytical_plots/temperatureProfiles/'
             + sim_run + '_gas_temperatureProfile_' + halo
             + ('_isolated' if bool_isolated_halo is True else '')
             + ('_H0' if bool_h0_filtering is True else '')
             + ('_Z' if bool_z_filtering is True else '')
             + ('_rho' if bool_rho_filtering is True else '')
             + ('_filtered' if bool_h0_filtering or bool_z_filtering is True else '') + '.npz',
             profiles=H, d_e=d_e, logT_e=logT_e, time=time_e, masses=masses, temps=temps, n_H0=nH0s,
             h0_threshold=H0_threshold, z_threshold=Z_threshold, rho_threshold=rho_threshold)


def plotting(sim_run, halo, snap, bool_isolated_halo, bool_h0_filtering, bool_z_filtering, bool_rho_filtering):
    from scripts.util.archive.plots_old import plot_temperature_profile

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/temperatureProfiles/' + halo + '/'
                  + sim_run + '_gas_temperatureProfile_' + halo
                  + ('_isolated' if bool_isolated_halo is True else '')
                  + ('_H0' if bool_h0_filtering is True else '')
                  + ('_Z' if bool_z_filtering is True else '')
                  + ('_rho' if bool_rho_filtering is True else '')
                  + ('_filtered' if bool_h0_filtering or bool_z_filtering is True else '') + '.npz')

    output_path = ('/Users/dear-prudence/Desktop/smorgasbord/temperatureProfiles/' + halo + '/'
                   + sim_run + '_gas_temperatureProfile_' + halo + '_snapshot' + str(snap) + '.png')

    plot_temperature_profile(input_path, output_path, snap)


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
halo_ = 'halo_21'
size_ = [0, 250]  # in ckpc
n_bins_per_kpc = 2
snaps_ = [125, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
isolated_halo = False
H0_filtering = True
H0_threshold = 10 ** -5
Z_filtering = False
Z_threshold = 1.0
density_filtering = True
rho_b = 6.2317  # M_solar/kpc
density_threshold = np.array([50, 5]) * rho_b  # [max, min] density cutoffs
# [50, 5] * rho_b corresponds to 1.47e-6 - 1.47e-5 cm^-3
# [1000, 10] * rho_b corresponds to 1e-5.5 - 1e-3.5 cm^-3
# ------------------------------------
snap_ = 127
# ------------------------------------

if machine == 'geras':
    if __name__ == "__main__":
        package_data(simulation_run, halo_, snaps_, size_, n_bins_per_kpc,
                     isolated_halo, H0_filtering, Z_filtering, density_filtering,
                     H0_threshold, Z_threshold, density_threshold)
        # Terminate the script
        sys.exit(0)

elif machine == 'dear-prudence':
    plotting(simulation_run, halo_, snap_, isolated_halo, H0_filtering, Z_filtering, density_filtering)

# Indicate that the script has completed its task
print('Done!')
