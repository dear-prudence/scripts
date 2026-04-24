"""This script will extract particles from a specified snapshot and plot their relative mass density in
temperature-coordinate space, the so-called temperature profile of a specified halo"""
from archive.hestia import time_edges
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


def rhoH_to_nH(rhoH, f_neutral=1.0, mu=1.00784):
    """
    Convert hydrogen mass density from M_solar/kpc^3 to hydrogen number density in cm^-3.

    Parameters:
    rho_h (float): Hydrogen mass density in M_solar/kpc^3.
    f_neutral (float): Fraction of hydrogen that is neutral (default is 1.0, i.e., fully neutral).
    mu (float): Mean atomic/molecular mass of hydrogen in atomic mass units (amu).
                Defaults to 1.00784 for neutral hydrogen.

    Returns:
    float: Hydrogen number density in cm^-3.
    """
    # Constants
    M_solar_to_g = 1.989e33  # Solar mass in grams
    kpc_to_cm = 3.086e21  # 1 kpc in cm
    amu_to_g = 1.66053906660e-24  # Atomic mass unit in grams

    # mu = 4 * m_p / (1 + 3 * x_h + 4 * x_h * e_abundance)

    # Safeguard against extremely large inputs
    rhoH = np.asarray(rhoH, dtype=np.float64)  # Ensure it's a float array or scalar
    if np.any(rhoH <= 0):
        raise ValueError("Hydrogen mass density must be positive and non-zero.")

    # Convert hydrogen mass density to g/cm^3
    rhoH_cgs = (rhoH * M_solar_to_g) / (kpc_to_cm ** 3)

    # Convert mass density to number density
    m_h = mu * amu_to_g  # Mass of a hydrogen atom/molecule in grams
    # n_h0 = (rhoH_cgs / m_h) * f_neutral  # Apply neutral fraction
    nH = (rhoH_cgs / m_h)

    return nH


def fit_2d_histogram(hist_2d, x_edges, y_edges, func, p0=None):
    from scipy.optimize import curve_fit
    """
    Fit a 1D function to 2D histogram data by converting the 2D histogram into coordinate pairs
    with weights proportional to brightness (histogram values).

    Parameters:
    hist_2d (2D array): The 2D histogram (image-like data).
    x_edges (1D array): The bin edges for the x-axis.
    y_edges (1D array): The bin edges for the y-axis.
    func (callable): The function to fit, of the form f(x, a, b, c, ...).
    p0 (array-like, optional): Initial guesses for the fitting parameters.

    Returns:
    popt (array): Optimal parameters for the fitting function.
    pcov (2D array): Covariance matrix of the fitting parameters.
    """
    # Compute bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create a grid of x and y values
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

    # Flatten the grids and histogram data
    x_data = x_grid.ravel()
    y_data = y_grid.ravel()
    weights = hist_2d.ravel()

    # Remove zero-weight (empty) bins
    mask = weights > 0
    x_data = x_data[mask]
    y_data = y_data[mask]
    weights = weights[mask]

    # Repeat data points according to weights (integer approximation)
    x_data_repeated = np.repeat(x_data, weights.astype(int))
    y_data_repeated = np.repeat(y_data, weights.astype(int))

    print(x_data_repeated.shape)
    # Perform curve fitting
    popt, _ = curve_fit(func, x_data_repeated, y_data_repeated, p0=p0, maxfev=1500)
    return popt, x_data_repeated, y_data_repeated


def beta_profile(r, alpha, beta):
    # alpha = log(n_0 * r_c^3beta)
    # n_r = n0 * (1 + r_reduced ** 2) ** (-3 * beta / 2)
    # log_n_r = log_n0 - 1.5 * beta * np.log10(1 + (r / r_c) ** 2)
    # approx ...
    log_n_r = alpha - 3 * beta * np.log10(r)
    return log_n_r


def retrieve_particles(snap_i, z, sim, halo):
    from archive.hestia import halo_dictionary
    from archive.hestia import rid_h_units

    using_isolated_halo = True

    h = 0.677
    if snap_i < 100:
        snap = '0' + str(snap_i)
    else:
        snap = str(snap_i)

    halo_id = halo_dictionary(sim, halo)
    _, pos_h, _, _, r_vir_h = get_halo_params(sim, halo, snap_i)

    lb_h, ub_h = pos_h - 2 * r_vir_h, pos_h + 2 * r_vir_h

    print('Processing Snapshot ' + snap + '...')

    if using_isolated_halo is True:
        file_path = '/z/rschisholm/storage/snapshots_lmc/snapshot_' + snap + '.hdf5'
        with h5py.File(file_path, 'r') as file:
            keys = file['PartType0'].keys()
        all_particles = {name: None for name in keys}
        all_particles = append_particles('PartType0', file_path, key_names=keys,
                                         existing_arrays=all_particles)

    else:
        base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + sim + '/output_2x2.5Mpc/snapdir_'
                     + snap + '/snapshot_' + snap + '.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        with h5py.File(base_path + '0' + file_extension, 'r') as file:
            keys = file['PartType0'].keys()
        all_particles = {name: None for name in keys}
        for file_path in file_paths:
            all_particles = append_particles('PartType0', file_path, key_names=keys,
                                             existing_arrays=all_particles)

    processed_particles = add_temperature(
        add_distances(
            transform_haloFrame(sim, halo_id, snap,
                                rid_h_units(
                           filter_particles(all_particles, lb_h / 1e3, ub_h / 1e3), z, 'PartType0'))))

    print('len(processed_particles) = ' + str(len(processed_particles['ParticleIDs'])))

    # carves a sphere of radius 2R_vir
    enclosed_radius = 2 * r_vir_h / h  # in kpc
    radius_mask = processed_particles['Distances'] <= enclosed_radius
    filtered_particles = {name: None for name in processed_particles.keys()}
    for key in processed_particles.keys():
        filtered_particles[key] = processed_particles[key][radius_mask]

    filtered_particles['n_H'] = rhoH_to_nH(filtered_particles['Density'] * filtered_particles['GFM_Metals'][:, 0])

    print('len(filtered_particles) = ' + str(len(filtered_particles['ParticleIDs'])))

    print('average(filtered_particles[n_H]) = ' + str(np.average(filtered_particles['n_H'])))
    print('min(filtered_particles[n_H]) = ' + str(min(filtered_particles['n_H'])))
    print('max(filtered_particles[n_H]) = ' + str(max(filtered_particles['n_H'])))

    print('average(filtered_particles[temp]) = ' + str(np.average(filtered_particles['Temperature'])))
    print('min(filtered_particles[temp]) = ' + str(min(filtered_particles['Temperature'])))
    print('max(filtered_particles[temp]) = ' + str(max(filtered_particles['Temperature'])))

    return filtered_particles, r_vir_h / h


def create_plot(mode, particles, bins):
    # -----------------------------------
    log_temperature_range = [3.5, 6.5]
    log_nH_range = [-8, -2]
    radius_range = [0, 240]
    # -----------------------------------

    if mode == 'radial':
        hist, x_e, y_e = np.histogram2d(particles['Distances'], np.log10(particles['n_H']),
                                        bins=bins, range=np.array([radius_range, log_nH_range]),
                                        weights=particles['Masses'] * particles['GFM_Metals'][:, 0])
    elif mode == 'temperature':
        hist, x_e, y_e = np.histogram2d(np.log10(particles['n_H']), np.log10(particles['Temperature']),
                                        bins=bins, range=np.array([log_nH_range, log_temperature_range]),
                                        weights=particles['Masses'] * particles['GFM_Metals'][:, 0])
    else:
        print('Error: Invalid mode inputted!')
        exit(1)

    h_normed = hist / np.max(hist)
    return h_normed, x_e, y_e


def package_data(mode, sim, halo, snaps):
    n_bins = 400

    time_e = time_edges(sim=sim, snaps=np.arange(snaps[1], snaps[0], step=-1))

    for snap_i in range(snaps[1], snaps[0], -1):

        particles, R_vir = retrieve_particles(snap_i, time_e[127 - snap_i, 0], sim, halo)

        # _, _, _, _, r_vir_h = get_halo_params(sim_run, halo, i)

        # T, r = T_profile(halo, i)
        # (unfiltered_particles, bins, r_vir_h, h_max=None)

        # in general, this condition is for redshift z = 0.0
        if snap_i == snaps[1]:
            H_i, x_e, y_e = create_plot(mode, particles, n_bins)
            H = H_i
            R_virs = R_vir
        else:
            # noinspection PyUnboundLocalVariable
            H_i, _, _ = create_plot(mode, particles, n_bins)
            # Append the new snapshot
            H = np.dstack((H, H_i))
            R_virs = np.append(R_virs, R_vir)

    # noinspection PyUnboundLocalVariable
    if mode == 'radial':
        output_path = ('/z/rschisholm/storage/analytical_plots/densityProfiles/'
                       + sim + '_gas_nH-R_profile_' + halo + '.npz')
    elif mode == 'temperature':
        output_path = ('/z/rschisholm/storage/analytical_plots/densityProfiles/'
                       + sim + '_gas_T-nH_profile_' + halo + '.npz')
    else:
        print('Error: Invalid mode inputted!')
        exit(1)

    np.savez(output_path, profiles=H, x_e=x_e, y_e=y_e, time=time_e,
             R_vir=R_virs)

    # Indicate that the script has completed its task
    print('Done!\n----------------------------------------------')
    print('scp -P 2222 rschisholm@geras.aip.de:' + output_path +
          ' /Users/dear-prudence/Desktop/smorgasbord/densityProfiles/' + sim + '_gas_'
          + ('nH-R' if mode == 'radial' else 'T-nH') + '_profile_' + halo + '.npz')


def plotting(mode, sim, halo, snap):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/densityProfiles/' + sim + '_gas_'
                  + ('nH-R' if mode == 'radial' else 'T-nH') + '_profile_' + halo + '.npz')

    output_path = ('/Users/dear-prudence/Desktop/smorgasbord/densityProfiles/' + sim + '_gas_'
                   + ('nH-R' if mode == 'radial' else 'T-nH') + '_profile_' + halo + '_snapshot' + str(snap) + '.png')

    c_map = 'GnBu'
    background_color = plt.get_cmap(c_map)(0)

    data = np.load(input_path)
    profile = data['profiles'][:, :, 127 - snap]
    x_e, y_e = data['x_e'], data['y_e']
    lookback_times = data['time'][:, 0]
    redshifts = data['time'][:, 1]
    R_vir = data['R_vir'][127 - snap]

    if mode == 'radial':
        # Fit the data
        popt, x_data, y_data = fit_2d_histogram(profile * 1e5, x_e, y_e, beta_profile, p0=[-2, 0.71])
        print("Optimal parameters:", popt)

    fig = plt.figure(figsize=(14, 8))
    fig.tight_layout()
    plt.gca().set_facecolor(background_color)
    plt.imshow(np.rot90(profile), origin='upper', extent=(x_e[0], x_e[-1], y_e[0], y_e[-1]),
               aspect=(2 if mode == 'temperature' else 40), cmap=c_map,
               norm=LogNorm(vmin=1e-2, vmax=1))
    # plt.scatter(x_data, y_data, alpha=0.1)

    # Create extra white space to the right of the right subplot
    plt.grid(visible=True, ls='-', alpha=0.4)
    if mode == 'radial':
        plt.xlabel(r'$R$' + ' kpc')
        plt.ylabel(r'$\log n_H$ ' + r'$cm^{-3}$')

        plt.plot(x_e, beta_profile(x_e, popt[0], popt[1]), color='k')

        plt.plot([R_vir, R_vir], [-2, -8], linestyle='dotted', color='k', alpha=0.2)
        plt.text(R_vir + 5, -2.2, s=r'$R_{vir}$', fontsize='small')
    else:
        plt.xlabel(r'$\log n_H$ ' + r'$cm^{-3}$')
        plt.ylabel(r'$\log(T)$')

    plt.ylim([y_e[0], y_e[-1]])
    # plt.xticks(np.linspace(x_e[0], x_e[-1], int(np.round(x_e[-1] / 25 + 1, 0)), endpoint=True))
    # cax = plt.axes((0.92, 0.12, 0.02, 0.75))  # [left, bottom, width, height]
    # plt.colorbar(cax=cax, label='Relative density of particles')

    plt.savefig(output_path, dpi=240)
    plt.show()
    plt.close()


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
mode_ = 'radial'  # 'radial' for nH-R space, 'temperature' for T-nH space
simulation_run = '09_18'
halo_ = 'lmc'
snaps_ = [125, 127]
# ------------------------------------
snap_ = 127
# ------------------------------------

if machine == 'geras':
    # (sim_run, halo, snaps, size, bins_per_kpc, bool_rho_filtering, rho_threshold
    package_data(mode_, simulation_run, halo_, snaps_)

elif machine == 'dear-prudence':
    plotting(mode_, simulation_run, halo_, snap_)
