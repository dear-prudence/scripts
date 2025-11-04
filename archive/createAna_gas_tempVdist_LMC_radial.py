import sys
from scripts.hestia import calc_temperature
from scripts.hestia import center_lmc
from scripts.hestia import time_edges
from scripts.hestia import distance_to_disk

def add_temperature(data):
    temp_column = calc_temperature(u=np.array(data['InternalEnergy']), e_abundance=np.array(data['ElectronAbundance']),
                                   x_h=np.array(data['GFM_Metals'][:, 0]))
    # Add the new column to the data dictionary
    data['Temperatures'] = temp_column
    return data


def onedim_weighted_histo(distances, temperatures, masses, num_bins, size):
    # Compute the histogram of distances
    hist, edges = np.histogram(distances, bins=num_bins, range=(size[0], size[1]))
    # Compute the sum of temperatures in each distance bin
    temp_sum, _ = np.histogram(distances, bins=edges, range=(size[0], size[1]), weights=temperatures)
    # Compute the count of particles in each distance bin
    counts, _ = np.histogram(distances, range=(size[0], size[1]), bins=edges, weights=masses / np.mean(masses))
    # Compute the average temperature in each distance bin
    avg_temps = np.divide(temp_sum, counts, where=(counts != 0))
    return avg_temps, edges


def twodim_weighted_histo(distances, temperatures, masses, num_bins, size):
    hist, x_edges, y_edges = np.histogram2d(distances * 1e3, np.log10(temperatures),
                                            range=np.array([[size[0], size[1]], [4, 7]]),
                                            bins=[num_bins, num_bins / 2],
                                            weights=masses / np.mean(masses))
    # weights=1 / np.square(distances / (size * 10 ** -3)))
    volumes = spherical_volume_array(x_edges, num_bins / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        h_n = np.divide(hist, volumes, where=(volumes != 0))
    # Normalizes the histogram to 100
    scale_factor = 100 / np.max(h_n)
    h_normed = scale_factor * h_n
    return h_normed, x_edges, y_edges


def spherical_volume_array(radii, ny):
    # Calculate the volumes for each distance
    volumes = (4 / 3.0) * np.pi * radii ** 3
    # Create an empty 2D array to hold the volumes
    volumes_array = np.empty((len(radii) - 1, ny))
    # Fill the 2D array with the volumes
    for i, volume in enumerate(volumes[:-1]):
        volumes_array[i] = volume
    return volumes_array


# In this script, I am taking the approximation that the 09_18 LMC's disk in oriented in the y-z plane,
# such that its normal vector lies in the x-plane
def create_plot(type_, snap, run, bins, size):
    # bins object must be a two element list
    # axes object follows the formal [x_axis, y_axis] (i.e. if plotting z vs y, axes = [1, 2])
    key_names = ['Coordinates', 'InternalEnergy', 'ElectronAbundance', 'GFM_Metals', 'Masses']
    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_'
                 + str(snap) + '/snapshot_' + str(snap) + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_particles = {name: None for name in key_names}
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + str(snap) + '...')
    for file_path in file_paths:
        all_particles = append_particles('PartType0', file_path, key_names=key_names, existing_arrays=all_particles)
    # Initial filtering to cut down on unnecessary distance calculations
    l_b, u_b = center_lmc(run=run, lmc_halo_id='10', snap=snap,
                                size=[600, 600, 600]) * 1e-3  # convert from kpc to Mpc
    init_filtered_particles = cosmo_transform(filter_particles(all_particles, min_b=l_b, max_b=u_b),
                                              'ckpc/h', 'ckpc')
    # Calculate distances to the center of the disk for each particle
    particles_w_d = distance_to_disk(init_filtered_particles, run=run, lmc_halo_id='10', snap=snap, frame='proper')
    # ^^^ in Mpc
    print(len(particles_w_d['Distances']))
    # Filter particles based on the sphere of interest
    filtered_particles = filter_particles_sphere(particles_w_d, rad=size[1] / 1e3)
    print(len(filtered_particles['Distances']))
    print(np.mean(filtered_particles['Distances']))
    # Add a temperature column to the particles
    filtered_particles_w_t = add_temperature(filtered_particles)
    if type_ == 'one_dim_histo':
        return onedim_weighted_histo(distances=filtered_particles_w_t['Distances'],
                                     temperatures=np.log10(filtered_particles_w_t['Temperatures']),
                                     masses=filtered_particles_w_t['Masses'], num_bins=bins,
                                     size=size)
    elif type_ == 'scatter':
        return filtered_particles_w_t['Distances'], filtered_particles_w_t['Temperatures']
    elif type_ == 'grid':
        return twodim_weighted_histo(filtered_particles_w_t['Distances'], filtered_particles_w_t['Temperatures'],
                                     filtered_particles_w_t['Masses'],
                                     num_bins=bins, size=size)
    else:
        print('Invalid plot type!')
        exit()


if __name__ == "__main__":
    # -------------------
    simulation_run = '09_18'
    # for this script, box needs to be a cube
    spatial_size = np.array([5, 200])  # in kpc
    n_bins = (spatial_size[1] - spatial_size[0]) * 3  # the factor is the number of bins per kpc
    snapshot = int(input('Snapshot? '))
    type_plot = 'histo'
    # h = 0.677
    # co_moving_size = np.array(spatial_size) / h
    # -------------------
    if type_plot == 'histo':
        # avgT_curve, bin_edges = create_plot('one_dim_histo', snapshot, simulation_run, n_bins,
        #                                     size=spatial_size * 1e-3)
        avgT_curve = np.array([])
        bin_edges = np.array([])
        #H, d_edges, logT_edges = create_plot('grid', snapshot, simulation_run, n_bins,
        #                                     size=cosmo_transform(spatial_size, 'ckpc', 'ckpc/h'))
        H, d_edges, logT_edges = create_plot('grid', snapshot, simulation_run, n_bins,
                                             size=spatial_size)
        np.savez('/z/rschisholm/storage/analytical_plots/' + str(simulation_run) + '_snap' + str(snapshot)
                 + '_gas_tempVdistHist_LMC_radius' + str(spatial_size[1]) + 'kpc_wMasses.npz',
                 Tcurve=avgT_curve, bin_edges=bin_edges, data=H, d_edges=d_edges, logT_edges=logT_edges,
                 time=time_edges('09_18', snaps=[snapshot]))
    elif type_plot == 'scatter':
        d_arr, T_arr = create_plot(type_plot, snapshot, simulation_run, n_bins, size=spatial_size * 1e-3)
        np.savez('/z/rschisholm/storage/analytical_plots/' + str(simulation_run) + '_snap' + str(snapshot)
                 + '_gas_tempVdistScatter_LMC_radius' + str(spatial_size[1]) + 'kpc.npz', x_data=d_arr, y_data=T_arr)
    else:
        print('Invalid plot type!')
        exit()
    # Indicate that the script has completed its task
    print('Done!')
    # Terminate the script
    sys.exit(0)
