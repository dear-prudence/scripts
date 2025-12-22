# This script contains routines relating to the image produced from snapshot particle data
import numpy as np
import scipy.ndimage


def twoD_Gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()  # flatten for fitting


# This routine reduces the graininess of an image produced by bins that have too few particles, and thus
# have outlier temperatures (in the future, perhaps using number density as a cutoff is advantageous)
def reduce_outliers(arr, lo_threshold, up_threshold):
    new_arr = np.copy(arr)
    # Get the shape of the array
    rows, cols = arr.shape
    # Iterate over each element in the array
    for i in range(rows):
        for j in range(cols):
            # Check if the current value is an outlier
            if arr[i, j] > up_threshold or arr[i, j] < lo_threshold:
                # Define the indices of neighboring elements
                neighbors = []
                if i > 0:
                    neighbors.append(arr[i - 1, j])  # Above
                if i < rows - 1:
                    neighbors.append(arr[i + 1, j])  # Below
                if j > 0:
                    neighbors.append(arr[i, j - 1])  # Left
                if j < cols - 1:
                    neighbors.append(arr[i, j + 1])  # Right

                # Replace the outlier with the average of neighboring values
                new_arr[i, j] = np.mean(neighbors)
    return new_arr


# This routine essentially runs a moving average across the image, to reduce noise but produces a
# lower resolution (fuzzier) image
def kernel_smooth(histogram_data, kernel_size):
    # Create a 2D kernel for the moving average
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)
    # Convolve the histogram data with the kernel
    smoothed_data = scipy.ndimage.convolve(histogram_data, kernel, mode='constant', cval=0.0)
    return smoothed_data


def create_projection_histogram(part_type, param, x, y, weights, masses, background, bins, bounds=None, verbose=False):
    """
    Creates a 2-dim scalar image of a parameter, weighted by mass and returned in surface density units.
    :param part_type: particle type (e.g. PartType0, PartType1)
    :param param: physical quantity to be plotted (e.g. density, temperature)
    :param x, y: 1-dim arrays of particle positions in x, y in kpc
    :param weights: 1-dim array of physical quantity to be plotted (e.g. temperature, density)
    :param masses: 1-dim array of cell masses in M_solar
    :param background: value to assign pixels without a sufficient number of particles/cells
    :param bins: array of length 2 denoting number of bins in x, y directions
    :param bounds: range of the image in kpc
    :param verbose: verbose print statements
    """

    # computes total mass in each bin
    total_mass, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=bounds, weights=masses)
    # computes total weighted value in each bin
    weighted_sum, _, _ = np.histogram2d(x, y, bins=bins, range=bounds, weights=weights * masses)
    # computes bin area in kpc^2
    dx = (x_edges[1] - x_edges[0])
    dy = (y_edges[1] - y_edges[0])
    bin_area = dx * dy  # kpc^2

    # surface density or average of parameter
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_value = np.divide(weighted_sum, total_mass, where=(total_mass != 0))  # e.g., density, temperature
        # surface_density = total_mass / bin_area  # M_solar / kpc^2

        if param == 'surfaceBrightness':
            surface_density = -2.5 * np.log10(weighted_sum / bin_area)  # mag / kpc^2
        else:
            surface_density = total_mass / bin_area  # M_solar / kpc^2

    # apply background fill to low-mass bins
    threshold = 1  # M_solar, or adjust based on mass resolution
    verbose and print(f'\t\t\tapplying background to pixels with < {threshold} M_solar')
    avg_value[total_mass < threshold] = background
    surface_density[total_mass < threshold] = background

    if part_type == 'PartType0':
        verbose and print(f'\t\t\tprojected image for gas cells, mean(value) = {np.average(avg_value)}')
        return avg_value, x_edges, y_edges
    elif part_type == 'PartType4' or part_type == 'PartType1':
        verbose and print(f'\t\t\tprojected surface image for {part_type}, mean(value) = {np.average(surface_density)}')
        return surface_density, x_edges, y_edges


def create_spherical_brightness_projection(lon, lat, mags, bins=(360, 90), background=np.nan, bounds=None,
                                           verbose=False):
    """
    Projects stellar particles onto a longitude–latitude grid and computes surface brightness (mag / arcsec^2).

    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude of particles in degrees.
    mags : array-like
        Apparent magnitudes per particle (band-independent).
    bins : tuple of ints
        Number of bins in longitude and latitude directions.
    background : float
        Value to fill bins with no flux.
    bounds : tuple of tuples or None
        ((lon_min, lon_max), (lat_min, lat_max)) in degrees.
        Default: full-sky (-180,180), (-90,90)
    """

    if bounds is None:
        bounds = ((-180, 180), (-90, 90))

    # Convert magnitudes to linear fluxes
    fluxes = 10 ** (-0.4 * mags)

    # Sum flux per bin (flux-weighted histogram)
    flux_sum, lon_edges, lat_edges = np.histogram2d(lon, lat, bins=bins, range=bounds, weights=fluxes)

    # Bin size
    dlon = (lon_edges[1] - lon_edges[0])
    dlat = (lat_edges[1] - lat_edges[0])

    # Compute bin area in arcsec^2
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    cos_lat = np.cos(np.deg2rad(lat_centers))[:, None]
    dOmega = np.deg2rad(dlon) * np.deg2rad(dlat) * cos_lat  # sr
    bin_area_arcsec2 = dOmega * (206265 ** 2)  # arcsec^2

    # Broadcast to match shape of flux_sum
    bin_area_arcsec2 = np.broadcast_to(bin_area_arcsec2.T, flux_sum.shape)

    # Surface brightness (mag / arcsec^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        surf_brightness = -2.5 * np.log10(flux_sum / bin_area_arcsec2)
        surf_brightness[np.isnan(surf_brightness)] = background
        surf_brightness[flux_sum <= 0] = background

    if verbose:
        mean_brightness = np.nanmean(surf_brightness)
        print(f"\tProjected mean surface brightness = {mean_brightness:.2f} mag/arcsec^2")

    return surf_brightness, lon_edges, lat_edges


def sph_kernel_projection(x, y, hsml, weights, masses, bounds, n_bins, verbose):
    """
    :param x, y: 1-dim arrays of cell positions in kpc
    :param hsml: 1-dim array of smoothing lengths in kpc
    :param weights: 1-dim array of physical quantity to be plotted (e.g. temperature, density)
    :param masses: 1-dim array of cell masses in M_solar
    :param bounds: tuple of ((xmin, xmax), (ymin, ymax)) in kpc
    :param n_bins: integer number of bins along one axis (produces square image)
    :param verbose: verbose print statements
    :return: 2D numpy array of size (nbins, nbins) with projected values in physical units
    """
    from scipy.spatial import cKDTree

    verbose and print("\t\t\tinitializing empty image and normalization arrays...")

    # img stores weighted sums of the projected quantity
    img = np.zeros((n_bins, n_bins), dtype=np.float64)
    # norm stores the normalization factors
    norm = np.zeros_like(img)

    verbose and print("\t\t\tsetting up pixel size and grid centers...")
    # creates grid and defines the physical size of each pixel
    (xmin, xmax), (ymin, ymax) = bounds
    dx = (xmax - xmin) / n_bins
    dy = (ymax - ymin) / n_bins
    x_edges = np.linspace(xmin, xmax, n_bins + 1)
    y_edges = np.linspace(ymin, ymax, n_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    # finds the center of each bin and builds a 2-dim meshgrid at those centers
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing='ij')

    verbose and print("\t\t\tflattening 2D grid to 1D list of coordinates...")
    grid_positions = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    verbose and print("\t\t\tBuilding KD-tree of grid pixels...")
    # builds a KDTree to find which pixels fall within the hsml of each particle
    grid_tree = cKDTree(grid_positions)

    verbose and print("\t\t\tStarting particle loop...")
    for i in range(len(x)):
        if not (xmin <= x[i] <= xmax and ymin <= y[i] <= ymax):
            # if particle is outside field of view
            continue
        pos = np.array([x[i], y[i]])
        h = hsml[i]
        # finds pixels that fall within 2 * hsml of the particle
        idxs = grid_tree.query_ball_point(pos, 2 * h)
        if not idxs:
            # if no grid points are affected by particle
            continue

        for idx in idxs:
            # loops over each particle affected by subject particle
            gx, gy = grid_positions[idx]
            r = np.sqrt((x[i] - gx) ** 2 + (y[i] - gy) ** 2)  # distance between particles
            if r >= 2 * h:
                continue

            q = r / h  # normalized distance between particles
            sigma = 10 / (7 * np.pi * h ** 2)  # sph kernel normalization
            # applies cubic spline kernel function in 2-dim
            if q <= 1:
                w = sigma * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
            elif q <= 2:
                w = sigma * 0.25 * (2 - q) ** 3
            else:
                w = 0.0

            px = idx // n_bins
            py = idx % n_bins
            # increases image value at the corresponding pixel by weight * mass * kernel
            img[px, py] += weights[i] * masses[i] * w
            # increases normalization at the corresponding pixel by mass * kernel
            norm[px, py] += masses[i] * w
            # print(f"\t\t\t  -> Pixel ({px}, {py}): Added weighted value {weights[i] * masses[i] * w:.3e}")

    verbose and print("\t\t\tnormalizing image by mass contribution...")

    with np.errstate(divide='ignore', invalid='ignore'):
        img = np.divide(img, norm, where=norm != 0)
    verbose and print("\t\t\tfinished SPH kernel projection.")

    return img, x_edges, y_edges


def sph_columnH0_projection(x, y, hsml, masses, h_frac, neutral_frac, bounds, nbins, verbose=True):
    from scipy.spatial import cKDTree
    """
    DEBUG version of SPH-style kernel projection of hydrogen number density [cm^-3].
    Prints internal states for debugging purposes.
    """
    msun_to_g = 1.98847e33  # grams
    m_H = 1.6735e-24  # grams
    kpc_to_cm = 3.08567758e21

    masses = masses.astype(np.float64)
    h_frac = h_frac.astype(np.float64)
    neutral_frac = neutral_frac.astype(np.float64)

    verbose and print("\t\t\tmass range:", masses.min(), masses.max())
    verbose and print("\t\t\th_frac range:", h_frac.min(), h_frac.max())
    verbose and print("\t\t\tneutral_frac range:", neutral_frac.min(), neutral_frac.max())

    n_particles = len(x)
    verbose and print(f"\t\t\tNumber of particles: {n_particles}")

    # Derived quantity: hydrogen atom count weight
    nH_weight = (h_frac * neutral_frac * masses * msun_to_g) / m_H
    verbose and print("\t\t\tnH_weight calculated.")

    verbose and print("\t\t\tnH_weight stats: min =", np.min(nH_weight), "max =", np.max(nH_weight))
    verbose and print("\t\t\tAny NaNs:", np.any(np.isnan(nH_weight)))
    verbose and print("\t\t\tAny infs:", np.any(np.isinf(nH_weight)))

    # Check for overflows or NaNs
    bad_mask = ~np.isfinite(nH_weight)
    if np.any(bad_mask):
        print("\t\tWarning: non-finite values in nH_weight, setting to 0")
        nH_weight[bad_mask] = 0.0

    img = np.zeros((nbins, nbins), dtype=np.float64)
    norm = np.zeros_like(img)

    (xmin, xmax), (ymin, ymax) = bounds
    dx = (xmax - xmin) / nbins
    dy = (ymax - ymin) / nbins
    verbose and print(f"\t\t\tPixel size: dx = {dx}, dy = {dy}")

    x_edges = np.linspace(xmin, xmax, nbins + 1)
    y_edges = np.linspace(ymin, ymax, nbins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_positions = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    grid_tree = cKDTree(grid_positions)
    verbose and print("\t\t\tKDTree built.")

    contributing_particles = 0
    total_contributions = 0

    for i in range(n_particles):
        if not (xmin <= x[i] <= xmax and ymin <= y[i] <= ymax):
            continue

        contributing_particles += 1
        pos = np.array([x[i], y[i]])
        h = hsml[i]
        if h == 0:
            print(f"\t\tWarning: hsml is zero for particle {i}")
            continue

        idxs = grid_tree.query_ball_point(pos, 2 * h)
        if not idxs:
            continue

        for idx in idxs:
            gx, gy = grid_positions[idx]
            r = np.sqrt((x[i] - gx) ** 2 + (y[i] - gy) ** 2)
            if r >= 2 * h:
                continue

            q = r / h
            sigma = 10 / (7 * np.pi * h ** 2)
            if q <= 1:
                w = sigma * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
            elif q <= 2:
                w = sigma * 0.25 * (2 - q) ** 3
            else:
                w = 0.0

            px = idx // nbins
            py = idx % nbins

            img[px, py] += nH_weight[i] * w
            norm[px, py] += w
            total_contributions += 1

    verbose and print(f"\t\t\tParticles contributing to image: {contributing_particles}")
    verbose and print(f"\t\t\tTotal kernel contributions made: {total_contributions}")

    area_cm2 = (dx * kpc_to_cm) * (dy * kpc_to_cm)
    verbose and print(f"\t\t\tPixel area in cm^2: {area_cm2:.3e}")

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = area_cm2 * norm
        verbose and print("\t\t\tMax img:", np.nanmax(img))
        verbose and print("\t\t\tMin denom:", np.nanmin(denom[denom > 0]))
        verbose and print("\t\t\tAny denom==0:", np.any(denom == 0))
        verbose and print("\t\t\tAny denom<1e-30:", np.any(denom < 1e-30))

        nH_map = np.zeros_like(img)
        valid = denom > 0
        nH_map[valid] = img[valid] / denom[valid]

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     nH_map = np.divide(img, area_cm2 * norm, where=norm != 0)
    nH_map[nH_map == 0] = 1  # or a small value like 1e-10 if you prefer

    if np.all(nH_map == 0):
        print("\t\tWARNING: Final nH_map is all zeros!")

    return nH_map, x_edges, y_edges


def star_surface_density_projection(x, y, masses, bounds, nbins, verbose=True):
    """
    Projects star particles into a 2D surface density map [Msun/kpc^2].
    Parameters
    ----------
    x, y : array_like
        Star particle positions [kpc].
    masses : array_like
        Stellar particle masses [Msun].
    bounds : tuple
        ((xmin, xmax), (ymin, ymax)) bounds of the projection in kpc.
    nbins : int
        Number of pixels along each axis.
    verbose : bool
        If True, print debug information.

    Returns
    -------
    surf_map : 2D ndarray
        Surface density map in Msun/kpc^2.
    x_edges, y_edges : 1D ndarrays
        Bin edges along x and y.
    """

    import numpy as np

    msun_to_g = 1.98847e33   # grams, if you later want to convert
    kpc_to_cm = 3.08567758e21

    masses = np.asarray(masses, dtype=np.float64)

    n_particles = len(x)
    verbose and print(f"\t\tNumber of star particles: {n_particles}")
    verbose and print(f"\t\tMass range: {masses.min()} - {masses.max()} Msun")

    # Bin edges
    (xmin, xmax), (ymin, ymax) = bounds
    x_edges = np.linspace(xmin, xmax, nbins + 1)
    y_edges = np.linspace(ymin, ymax, nbins + 1)

    # Histogram in 2D, weighted by mass
    H, x_edges, y_edges = np.histogram2d(
        x, y, bins=[x_edges, y_edges], weights=masses
    )

    # Pixel area in kpc^2
    dx = (xmax - xmin) / nbins
    dy = (ymax - ymin) / nbins
    pixel_area = dx * dy

    verbose and print(f"\t\tPixel size: dx={dx:.3f} kpc, dy={dy:.3f} kpc")
    verbose and print(f"\t\tPixel area: {pixel_area:.3f} kpc^2")

    # Convert to surface density (Msun / kpc^2)
    surf_map = H / pixel_area

    return surf_map, x_edges, y_edges


def projectOntoSky(x_cart, v_cart, r_observer, frame, fov_deg=40, lat_max=None, verbose=True):
    """
    projects sph cells onto the sky as seen by an observer
    :param x_cart: matrix indicating the 3-dim cartesian position (in kpc) of the gas cells
    :param r_observer: 3-dim cartesian position of observer (in kpc)
    :param fov_deg: field of view (in degrees)
    :param lat_max: maximum latitude of image
    :param verbose: verbose print statements
    :return: e1_proj, e2_proj, los, in_fov
    """
    print(f'v_cart.ndim : {v_cart.ndim}')
    if x_cart.ndim == 1:
        x_cart = x_cart[np.newaxis, :]
    if v_cart.ndim == 1:
        v_cart = v_cart[np.newaxis, :]

    # shift to observer-centered coordinates
    E = x_cart[:, 0] - r_observer[0]
    N = x_cart[:, 1] - r_observer[1]
    L = x_cart[:, 2] - r_observer[2]

    if frame == 'vrai':
        print('\t\t\tcomputing spherical coordinates in vrai frame')
        # los = np.sqrt(E ** 2 + N ** 2 + L ** 2)
        # lat = 90 - np.degrees(np.arccos(N / los))
        # lon = np.degrees(np.arctan(E / L))
        # b, l = np.radians(lat), np.radians(lon)

        # radii
        r = np.sqrt(E ** 2 + N ** 2 + L ** 2)
        rho = np.sqrt(E ** 2 + L ** 2)

        # spherical coordinates
        b = np.degrees(np.arctan2(N, rho))  # latitude (rad), range [-pi/2, pi/2]
        l = np.degrees(np.arctan2(E, L))  # longitude (rad), range [-pi, pi]
        x_sph = np.stack([r, b, l], axis=1)

        # build rows of M for each particle
        R_row1 = np.stack([E / r, N / r, L / r], axis=1)
        R_row2 = np.stack([-E * N / (r * rho), rho / r, -N * L / (r * rho)], axis=1)
        R_row3 = np.stack([-L / rho, np.zeros_like(r), E / rho], axis=1)
        # stack into (N,3,3)
        M = np.stack([R_row1, R_row2, R_row3], axis=1)

        # apply R to all velocities: einsum over n (particle index)
        print(f'M.shape : {M.shape}')
        print(f'vcart.shape : {v_cart.shape}')
        v_sph = np.einsum("nij,nj->ni", M, v_cart)

        # def sin(x):
        #     return np.sin(x)

        # def cos(x):
        #     return np.cos(x)

        # M = (
        #     (cos(b) * sin(l), sin(b), cos(b) * cos(l)),
        #     (-sin(b) * cos(l), cos(b), -sin(b) * sin(l)),
        #     (-sin(l), 0, cos(l))
        # )

        # v_sph = np.array(M) @ np.array(v_cart)
        # v_r, v_lat, v_lon = v_sph
    else:
        exit(1)
        # print('\t\t\tcomputing spherical coordinates in faux frame')
        # los = np.sqrt(e1_rel ** 2 + e2_rel ** 2 + e3_rel ** 2)
        # lat = 90 - np.degrees(np.arccos(e3_rel / los))
        # lon = np.degrees(np.arctan2(e2_rel / e1_rel))

    verbose and print(f'\t\tmean(l, b, r)_clouds : ({np.average(l)}, {np.average(b)}, {np.average(r)} kpc)')

    # Field of view
    lat_max = fov_deg / 2 if lat_max is None else lat_max
    lon_min, lon_max = -fov_deg / 2, + fov_deg / 2
    lat_min = -lat_max if frame == 'vrai' else -90
    in_fov = (b >= lat_min) & (b <= lat_max) & (l >= lon_min) & (l <= lon_max)

    # e1_proj = l[in_fov]
    # e2_proj = b[in_fov]
    verbose and print(f'\t\tnumber of particles in field of view = {in_fov.sum()}')
    bounds = ((lon_min, lon_max), (lat_min, lat_max))  # degrees
    return x_sph, v_sph, bounds, in_fov
