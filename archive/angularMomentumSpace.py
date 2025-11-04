import numpy as np
from scripts.hestia import cosmo_transform


def stream_filtering(particles):
    for key in particles.keys():
        particles[key] = np.array(particles[key])  # Convert to numpy array for easier indexing

    radii = np.zeros(particles['Halo_Coordinates'].shape[0])
    for i in range(particles['Halo_Coordinates'].shape[0]):
        radii[i] = np.linalg.norm(particles['Halo_Coordinates'][i])
    particles['Radii'] = radii

    metals = np.log10(particles['GFM_Metallicity'] / particles['GFM_Metals'][:, 0])
    particles['Z'] = metals

    indices_to_keep = np.where(
        # (1e3 <= particles['Masses']) & (particles['Masses'] <= 1e5) &
        # (-3 <= particles['Z']) & (-1 > particles['Z']) &
        (18 <= particles['Radii']) &
        (-100 < particles['Halo_Coordinates'][:, 0]) &
        (150 > particles['Halo_Coordinates'][:, 0]) &
        (-50 < particles['Halo_Coordinates'][:, 1]) &
        (50 > particles['Halo_Coordinates'][:, 1]) &
        (-50 < particles['Halo_Coordinates'][:, 2]) &
        (50 > particles['Halo_Coordinates'][:, 2])
    )[0]
    for key in particles.keys():
        particles[key] = particles[key][indices_to_keep]
    return particles


def retrieve_particles(snap_i, z, run, halo, part_type, size, follow_gal=None, frame=None):
    from scripts.hestia import transform_haloFrame
    from scripts.hestia import center_halo
    from scripts.hestia import append_particles, filter_particles
    from scripts.hestia import halo_dictionary

    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    halo_id = halo_dictionary(run, halo)

    l_b, u_b = center_halo(run=run, halo_id=halo_id, snap=snap, size=size) * 1e-3  # these are in _h units!

    key_names = ['Coordinates', 'Masses', 'Velocities', 'GFM_Metallicity', 'GFM_Metals']
    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + str(run) + '/output_2x2.5Mpc/snapdir_'
                 + snap + '/snapshot_' + snap + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_particles = {name: None for name in key_names}
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + snap + '...')
    for file_path in file_paths:
        all_particles = append_particles(part_type, file_path, key_names=key_names,
                                         existing_arrays=all_particles)
    # Filter particles based on the bounding box
    filtered_particles = cosmo_transform(filter_particles(all_particles, l_b, u_b),
                                         'ckpc/h', 'ckpc', z, part_type=part_type)
    if frame == 'halo':
        rotated_particles = transform_haloFrame(run, halo_id, snap, filtered_particles)
    else:
        rotated_particles = filtered_particles

    return rotated_particles, l_b / h, u_b / h


def make_snap(particles, bins, n_dims, axis, extent):

    if n_dims == 1:
        return np.histogram(particles['Angular_Momenta'][:, axis], weights=particles['Masses'] * 1e10,
                            bins=bins, density=True, range=extent)
    elif n_dims == 2:
        cartesian = np.array([0, 1, 2])
        # Get the indices not equal to the specified axis
        remaining_axes = cartesian[cartesian != axis]
        # Rotate to match vector calculus order
        axe = np.roll(remaining_axes, -axis)

        a, m, n = np.histogram2d(particles['Angular_Momenta'][:, axe[0]], particles['Angular_Momenta'][:, axe[1]],
                                 weights=particles['Masses'] * 1e10, bins=bins,
                                 range=np.array([extent, extent]))

        # Normalize to get density
        bin_area = np.outer(np.diff(m), np.diff(n))
        b = a / (a.sum() * bin_area)
        return b, m, n

    else:
        print('Error: invalid number of dimensions!')
        exit(1)


def package_data(sim_run, halo, particle, snaps, dims, n_bins, extent, isolate_stream):
    from scripts.hestia import time_edges

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4'}
    part_type = part_to_type[particle]

    S = {'Coordinates': np.array([dims, dims, dims])}  # creates the cube to restrict particles by

    bins = [n_bins, n_bins]  # for the 2-dim histograms

    time_e = time_edges(sim=simulation_run, snaps=np.arange(snaps[1], snaps[0], step=-1))

    # Dictionary to hold results for each axis/dimension
    all_planes = {'lx': None, 'ly': None, 'lz': None,
                  'ly_lz': None, 'lz_lx': None, 'lx_ly': None,
                  'x_e': None, 'y_e': None, 'z_e': None}

    # Loop through the snapshots in reverse order
    for i in range(snaps[1], snaps[0], -1):
        # Transform S coordinates
        z_value = float(time_e[127 - i][0])
        S_h = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=z_value)
        print('-------- $z = {}$ --------'.format(z_value))

        # l_b and u_b are in ckpc (converted from _h units)
        particles_, l_b, u_b = retrieve_particles(i, z_value, sim_run, halo, part_type, np.array(S_h['Coordinates']),
                                                  follow_gal=halo, frame=frame_)

        if isolate_stream is True:
            particles = stream_filtering(particles_)
        else:
            particles = particles_

        # Loop through the axes for both 1D and 2D dimensions
        for axis, name_1d, name_2d, edge_i in zip(range(3), ['lx', 'ly', 'lz'],
                                                  ['ly_lz', 'lz_lx', 'lx_ly'], ['x_e', 'y_e', 'z_e']):
            current_l_1d, edges_1d = make_snap(particles, n_bins, n_dims=1, axis=axis, extent=extent)
            current_l_2d, _, _ = make_snap(particles, bins, n_dims=2, axis=axis, extent=extent)

            # Initialize arrays if this is the first snapshot
            if i == snaps[1]:
                all_planes[name_1d] = current_l_1d
                all_planes[name_2d] = current_l_2d
                all_planes[edge_i] = edges_1d  # Store the edges
            else:
                # Append new snapshot data
                all_planes[name_1d] = np.dstack((all_planes[name_1d], current_l_1d))
                all_planes[name_2d] = np.dstack((all_planes[name_2d], current_l_2d))
                all_planes[edge_i] = edges_1d  # Edges only need to be stored once

        # Now all_planes['x'], all_planes['y'], all_planes['z'], etc., hold the stacked data
        print('Snapshot ' + str(i) + ': check')

    # Save data
    # Combine all dictionaries into one
    data_to_save = all_planes.copy()  # Start with all_planes
    data_to_save['time'] = time_e  # Add any other variables, like time_e
    # Save the combined dictionary
    np.savez(('/z/rschisholm/storage/analytical_plots/angularMomenta/'
              + sim_run + '_' + halo + '_' + particle + '_angularMomentaMap'
              + ('_filtered' if isolate_stream is True else '') + '.npz'), **data_to_save)

    # Indicate that the script has completed its task
    print('Done!')


def plotting(sim_run, halo, particle, snap, param_space, l_edge):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/isolate_stream/'
                  + sim_run + '_' + halo + '_' + particle + '_angularMomentaMap_filtered.npz')
    output_path = '/Users/ursa/Desktop/smorgasbord/isolate_stream/angularMomenta_map_hist_dim1.png'

    c_map = 'GnBu'
    background_color = plt.get_cmap(c_map)(0)

    data = np.load(input_path)
    dates = data['time']

    d_1 = ['lx', 'ly', 'lz']
    d_2 = ['ly_lz', 'lz_lx', 'lx_ly']
    e_i = ['x_e', 'y_e', 'z_e']

    if param_space in d_1:
        plt.figure(figsize=(10, 6))
        domain = data[e_i[d_1.index(param_space)]][:-1]
        plt.step(domain, data[param_space][0, :, 127 - snap], where='mid')

        key_to_label = {'lx': 'L_x', 'ly': 'L_y', 'lz': 'L_z'}
        plt.xlabel('${}$'.format(key_to_label[param_space]))
        plt.show()

    elif param_space in d_2:
        extent = (-1 * l_edge, l_edge, -1 * l_edge, l_edge)
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        fig.tight_layout()

        ax.set_facecolor(background_color)
        ax.imshow(data[param_space][:, :, 127 - snap].T, origin='upper',
                  extent=extent, cmap=c_map,
                  norm=LogNorm(vmin=1, vmax=np.max(data[param_space][:, :, 127 - snap])))

        key_to_label = {'ly_lz': ['L_y', 'L_z'], 'lz_lx': ['L_z', 'L_x'], 'lx_ly': ['L_x', 'L_y']}
        plt.xlabel('${}$'.format(key_to_label[param_space][0]))
        plt.ylabel('${}$'.format(key_to_label[param_space][1]))
        plt.show()

    print('Done!')


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
halo_ = 'halo_08'
particle_ = 'gas'
snaps_ = [120, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
dims_ = 300  # in c-kpc, restricts to particles within a cube of side length dims_
bins_ = 200
L_edge = 1e9  # range of the 1-dim histograms, range = (-L_edge, +L_edge)
frame_ = 'halo'
isolate_stream_ = True  # if special parameters should be used, helping isolate the stream-analog
# ------------------------------------
selected_paramSpace = 'ly'
snap_ = 127
# ------------------------------------

if machine == 'geras':
    h = 0.677
    package_data(simulation_run, halo_, particle_, snaps_, dims_, bins_, extent=(-1 * L_edge, L_edge),
                 isolate_stream=isolate_stream_)

elif machine == 'dear-prudence':
    plotting(simulation_run, halo_, particle_, snap_, selected_paramSpace, L_edge)
