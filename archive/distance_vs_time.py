import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


def lookback_time(z):
    # Define a cosmology (you can customize it as needed)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # Calculate the lookback time
    return cosmo.lookback_time(z).to(u.Gyr).value

def calculate_distances(coords1, coords2):
    """
    Calculate distances between corresponding points in two arrays of coordinates.

    Parameters:
    - coords1: Array of coordinates for the first set of points (e.g., [[x1, y1, z1], [x2, y2, z2], ...]).
    - coords2: Array of coordinates for the second set of points (same format as coords1).

    Returns:
    - distances: Array of distances between corresponding points.
    """
    distances = []
    for point1, point2 in zip(coords1, coords2):
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        distances.append(distance)
    return np.array(distances)


def calculate_aspect(x, y):
    x_ra = max(x) - min(x)
    y_ra = max(y) - min(y)
    return x_ra / y_ra


# Example usage:
redshift = 2.0
result = lookback_time(redshift)
print(float(result))


lmc = np.loadtxt('/Volumes/enceladus/LG_simulations/09_18/LMC_history_09_18/HESTIA_100Mpc_8192_09_18'
                   '.127_halo_127000000000010.dat')
mw = np.loadtxt('/Volumes/enceladus/LG_simulations/09_18/MW_history_09_18/HESTIA_100Mpc_8192_09_18'
                   '.127_halo_127000000000003.dat')

f_z = 20  # furthest lookback time in terms of snapshots
redshift = np.array(lmc[0:f_z, 0])
time = np.array(lookback_time(redshift)) * -1
coordinates_lmc = np.array([lmc[0:f_z, 6], lmc[0:f_z, 7], lmc[0:f_z, 8]])
coordinates_mw = np.array([mw[0:f_z, 6], mw[0:f_z, 7], mw[0:f_z, 8]])
print(mw[0, 65])  # MW mass?

distances = calculate_distances(coordinates_mw.T, coordinates_lmc.T)
print(distances[0])
plt.figure(figsize=(10, 6))
plt.plot(time, distances, linestyle='-')
plt.title('Distance b/w MW and LMC (Run 09_18)')
plt.xlabel('$t$ (Gyr)')
plt.ylabel('$d$ (kpc)')
plt.grid(True, alpha=0.5)

# Set the aspect ratio
plt.gca().set_aspect(calculate_aspect(abs(time), distances), adjustable='box')
plt.savefig('/Volumes/enceladus/LG_simulations/figs/d_vs_t_MW_LMC_zoom.png', dpi=240)
plt.show()

