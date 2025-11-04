import sys
import numpy as np
from scripts.hestia import append_particles
from scripts.hestia import cosmo_transform
from scripts.hestia import time_edges
from scripts.hestia import center_lmc


def create_weighted_histogram(x, y, densities, bins, bounds=None):
    threshold = 1
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=bins, range=bounds)
    # Compute the sum of densities in each bin
    sum_dens, _, _ = np.histogram2d(x, y, bins=(x_e, y_e), range=bounds, weights=densities)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_density = np.divide(sum_dens, hist, where=(hist != 0))
    avg_density[hist < threshold] = 0
    return avg_density, x_e, y_e


def make_snap(snapshot, z, run, bins, size, axs, follow_gal=None):
    # bins object must be a two element list
    # axes object follows the formal [x_axis, y_axis] (i.e. if plotting z vs y, axes = [1, 2])
    if snapshot < 100:
        snapshot = '0' + str(snapshot)
    else:
        snapshot = str(snapshot)

    key_names = ['Coordinates', 'Density']
    file_path = '/z/rschisholm/storage/snapshots_lmc/snapshot_' + snapshot + '/snapshot_' + snapshot + '.lmc.hdf5'
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + snapshot + '...')
    lmc_particles_h = append_particles('PartType0', file_path, key_names=key_names)
    if follow_gal == 'lmc':
        l_b, u_b = center_lmc(run=run, lmc_halo_id='10', snap=snap, size=size) * 1e-3  # convert from kpc to Mpc
    else:
        print('Error: Invalid reference galaxy! routine is WIP.')
        exit(1)
    bounds = cosmo_transform({'Coordinates': np.array([[l_b[axes[0]], u_b[axes[0]]], [l_b[axes[1]], u_b[axes[1]]]])},
                             'ckpc/h', 'ckpc', z)['Coordinates']
    # Filter particles based on the bounding box
    lmc_particles = cosmo_transform(lmc_particles_h, 'ckpc/h', 'ckpc', z)
    return create_weighted_histogram(lmc_particles['Coordinates'][:, axs[0]],
                                     lmc_particles['Coordinates'][:, axs[1]],
                                     densities=lmc_particles['Density'], bins=bins, bounds=bounds)


if __name__ == "__main__":
    # -------------------
    simulation_run = '09_18'
    # for this script, box needs to be a cube
    dims = [50, 400]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
    axes = [0, 1]
    n_bins = [800, 800]
    snap = 127
    # -------------------
    spatial_size = [dims[0] for i in range(3)]
    for ax in axes:
        spatial_size[ax] = dims[1]
    S = {'Coordinates': np.array(spatial_size)}
    # module that writes the panel
    time_edges = time_edges(sim=simulation_run, snaps=np.array([snap]))
    S_co = cosmo_transform(S.copy(), 'ckpc', 'ckpc/h', z=time_edges[0][0])
    print('time: z = ' + str(time_edges[0][0]))
    print('S_co coordinates: ' + str(S_co['Coordinates']))
    current_snapshot, x_edges, y_edges = make_snap(snap, float(time_edges[0][0]), simulation_run, bins=n_bins,
                                                   size=np.array(S_co['Coordinates']),
                                                   axs=axes, follow_gal='lmc')
    # Save data
    np.savez('/z/rschisholm/storage/snapshots_lmc/snapshot_' + str(snap) + '/' + str(simulation_run)
             + '_gas_massDen_lmc_snap' + str(snap) + '_'
             + str(spatial_size[0]) + 'x' + str(spatial_size[1]) + 'x' + str(spatial_size[2])
             + 'ckpc_bin' + str(round(spatial_size[axes[0]] / float(n_bins[0]), 2)) + 'ckpc_'
             + str(axes[0]) + '_' + str(axes[1]) + '.npz',
             data=current_snapshot, x_edges=x_edges, y_edges=y_edges, time=time_edges)

    # Indicate that the script has completed its task
    print('Done!')

    # Terminate the script
    sys.exit(0)
