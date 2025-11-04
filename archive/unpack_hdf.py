import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


def lookback_time(z):
    # Define a cosmology (you can customize it as needed)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # Calculate the lookback time
    return cosmo.lookback_time(z).to(u.Gyr).value


def create_2d_histogram(x, y, ranges, bins=12):
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=ranges)
    return histogram, x_edges, y_edges

def plot_2d_histogram(histogram, x_edges, y_edges, redshift):
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the size by changing the figsize parameter
    # Create a background with a specific color (color at the minimum of the Viridis colormap)
    c_map ='viridis'
    age = -1 * round(lookback_time(redshift), 2)
    background_color = plt.get_cmap(c_map)(0)
    ax.set_facecolor(background_color)
    im = ax.imshow(histogram.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap=c_map, aspect='auto', norm=LogNorm(vmin=1))
    plt.colorbar(im, label='Number Density')
    plt.xlabel('X (kpc)')
    plt.ylabel('Y (kpc)')
    # plt.title('LG (Run 09_18), $z=' + str(redshift) + '$, $t = ' + str(age) + '$ Gyr')
    # draw_circles(plt.gca())
    # plt.savefig('/Users/dear-prudence/Desktop/LG_simData_09_18_' + str(redshift) + '_gas.png', dpi=240)
    plt.show()


def manual_range():
    x_lower = -10
    y_lower = 0
    size = 10
    return [[x_lower, x_lower + size], [y_lower, y_lower + size]]


def calc_T(u, e_abundance, he_mass_fraction):
    gamma = 5 / 3
    print(gamma)
    k_b = 1.380649 * 10 ** -23
    m_p = 1.67262192 * 10 ** -27
    y_he = he_mass_fraction / (4 * (1 - he_mass_fraction))

    # Assuming e_abundance and he_mass_fraction are arrays
    mu = m_p * ((1 + 4 * y_he) / (1 + y_he + e_abundance))

    return (mu / k_b) * (gamma - 1) * (u * 10 ** 6)


snapshot = '301'
z = '0.000'  # needs to be in form x.xxx

file_path = '/Volumes/enceladus/LG_simulations/09_18/snapshot_127_hdf/snapshot_127.0.hdf5'
with h5py.File(file_path, 'r') as f:
    dset = f['PartType0']
    coords = np.array(dset['Coordinates'])
    metal = np.array(dset['GFM_Metals'])
    metallicity = np.array(dset['GFM_Metallicity'])

    temp_column = calc_T(u=np.array(dset['InternalEnergy']), e_abundance=np.array(dset['ElectronAbundance']),
                                    he_mass_fraction=np.array(dset['GFM_Metals'][:, 1]))
    print(np.mean(temp_column))



'''# Now, resulting_array contains the combined coordinates and masses
x_data = coords[:, 0]
y_data = coords[:, 2]
print(len(x_data))

# Define bounds
# ra = follow_mw(length=2, snap=snapshot, filename='/Volumes/enceladus/LG_simulations/09_18/MW_history_09_18'
#                                                 '/HESTIA_100Mpc_8192_09_18.127_halo_127000000000003.dat')
ra = manual_range()
histogram, x_edges, y_edges = create_2d_histogram(x_data, y_data, ra, bins=200)
plot_2d_histogram(histogram, x_edges, y_edges, redshift=z)'''




