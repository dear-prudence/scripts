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
                   cmap=c_map, aspect='auto', norm=LogNorm(vmin=10))
    plt.colorbar(im, label='Number Density')
    plt.xlabel('X (Mpc)')
    plt.ylabel('Y (Mpc)')
    plt.title('LG (Run 09_18), $z=' + str(redshift) + '$, $t = ' + str(age) + '$ Gyr')
    # draw_circles(plt.gca())
    # plt.savefig('/Users/dear-prudence/Desktop/LG_simData_09_18_' + str(redshift) + '_gas.png', dpi=240)
    plt.show()


def draw_circles(ax):
    m31_position = (48815.20492896, 46704.5145657)
    mw_position = (48715.76343264, 47062.33186381)
    radius = 0.1  # Adjust the radius as needed

    # Create a circle
    circle_m31 = plt.Circle(m31_position, radius,
                            color='none', ec='black', linestyle='dashed', linewidth=1, label='M31')
    circle_mw = plt.Circle(mw_position, radius,
                           color='none', ec='black', linestyle='dashed', linewidth=1, label='MW')

    # Add the circle to the plot
    ax.add_patch(circle_m31)
    ax.add_patch(circle_mw)
    # Add text annotation next to the circle
    ax.annotate('M31', xy=(m31_position[0] + radius, m31_position[1] + radius),
                color='black', fontsize=10, va='center')
    ax.annotate('MW', xy=(mw_position[0] + radius, mw_position[1] + radius),
                color='black', fontsize=10, va='center')


def follow_mw(length, snap, filename):
    # NEED TO CHANGE THIS IF PLOTTING SOMETHING OTHER THAN Y VS X
    row = 127 - int(snap)
    mw = np.loadtxt(filename)
    coordinates_mw = np.array([mw[row, 6], mw[row, 7], mw[row, 8]]) * (10 ** -3)
    lower_x = round(coordinates_mw[0] - (length / 2), 2)
    lower_y = round(coordinates_mw[1] - (length / 2), 2)
    return [[lower_x, lower_x + length], [lower_y, lower_y + length]]


def manual_range():
    x_lower = 49.00
    y_lower = 49.70
    size = 0.2
    return [[x_lower, x_lower + size], [y_lower, y_lower + size]]


def append_coordinates_hdf(file_path, existing_array=None):
    with h5py.File(file_path, 'r') as f:
        dset = f['PartType0']
        coords = np.array(dset['Coordinates'])

    return np.append(existing_array, coords, axis=0) if existing_array is not None else coords


def get_hdf(base_path, file_extension):
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    print(file_paths)
    # Initialize the resulting array
    resulting_array = None
    # Loop through the file paths and append coordinates
    for file_path in file_paths:
        resulting_array = append_coordinates_hdf(file_path, existing_array=resulting_array)
    return resulting_array


snapshot = '127'
z = '0.000'  # needs to be in form x.xxx

# Choose which file type to load
file_type = 'hdf'  # 'hdf' for the hdf particle files, 'ascii' for the processed particle coordinates
if file_type == 'hdf':
    data = get_hdf('/Volumes/enceladus/LG_simulations/09_18/snapshot_127_hdf/snapshot_127.',
                   '.hdf5')
elif file_type == 'ascii':
    data = np.loadtxt('/Volumes/enceladus/LG_simulations/09_18/coords/09_18_' + str(snapshot) + '_coords.dat',
                      skiprows=1)
else:
    data = []
    print('Error: incorrect file type input')

# Now, resulting_array contains the combined coordinates and masses
y_data = data[:, 1]
z_data = data[:, 2]

# Define bounds
# ra = follow_mw(length=2, snap=snapshot, filename='/Volumes/enceladus/LG_simulations/09_18/MW_history_09_18'
#                                                 '/HESTIA_100Mpc_8192_09_18.127_halo_127000000000003.dat')
ra = manual_range()
histogram, x_edges, y_edges = create_2d_histogram(y_data, z_data, ra, bins=500)
plot_2d_histogram(histogram, x_edges, y_edges, redshift=z)




