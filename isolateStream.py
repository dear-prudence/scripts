import h5py
import argparse
import numpy as np


def add_gasHeaders(cells):
    cells['Z'] = np.log10(cells['GFM_Metallicity'] / cells['GFM_Metals'][:, 0])
    cells['Lx'] = cells['Angular_Momenta'][:, 0]
    cells['Ly'] = cells['Angular_Momenta'][:, 1]
    cells['Lz'] = cells['Angular_Momenta'][:, 2]

    return cells


def calc_streamProperties(cells, snap, redshift):
    from hestia.gas import calc_temperature, calc_numberDensity
    from hestia.geometry import calc_distanceDisk
    cells['Temperature'] = calc_temperature(cells['InternalEnergy'], cells['ElectronAbundance'],
                                            cells['GFM_Metals'][:, 0])
    cells['n_H'] = calc_numberDensity(cells['Density'] * cells['GFM_Metals'][:, 0])
    cells['Distances'] = calc_distanceDisk(cells)

    # in case there are any drifters
    h = 0.677
    virial_radius = 104.74 / h  # in kpc, at z = 0.099 (snapshot 119)
    virial_mask = np.where(cells['Distances'] < virial_radius)

    sterilized_cells = {name: None for name in cells.keys()}
    for key in sterilized_cells.keys():
        sterilized_cells[key] = cells[key][virial_mask]

    M_H0 = np.sum(sterilized_cells['Masses'] * sterilized_cells['GFM_Metals'][:, 0]
                  * sterilized_cells['NeutralHydrogenAbundance'])

    mean_n_H0 = np.average(np.log10(sterilized_cells['n_H']), weights=sterilized_cells['NeutralHydrogenAbundance'])
    # Variance formula for weighted samples
    variance = np.average((np.log10(sterilized_cells['n_H']) - mean_n_H0) ** 2,
                          weights=sterilized_cells['NeutralHydrogenAbundance'])
    sigma_n_H0 = np.sqrt(variance)

    temperature = np.average(np.log10(sterilized_cells['Temperature']),
                             weights=sterilized_cells['NeutralHydrogenAbundance'])
    # Variance formula for weighted samples
    variance = np.average((np.log10(sterilized_cells['Temperature']) - temperature) ** 2,
                          weights=sterilized_cells['NeutralHydrogenAbundance'])
    sigma_T = np.sqrt(variance)

    return np.array([int(snap), float(redshift), np.log10(M_H0), mean_n_H0, sigma_n_H0, temperature, sigma_T])


def get_filtered_indices(cells, cutoffs):
    mask = np.ones(len(next(iter(cells.values()))), dtype=bool)  # start with all True
    for key, (lower, upper) in cutoffs.items():
        if key not in cells:
            raise KeyError(f"'{key}' not found in data.")
        # Apply range cut
        current_mask = (cells[key] >= lower) & (cells[key] <= upper)
        mask &= current_mask  # logical AND
    return np.where(mask)[0]  # return indices of particles to keep


def filter_gaseousStream(all_cells, cutoffs):
    # get the indices to keep
    indices_to_keep = get_filtered_indices(all_cells, cutoffs)
    # attribute indices to cell ids
    cell_ids = all_cells['ParticleIDs'][indices_to_keep]
    # Create a mask to filter particles with IDs in array_ids
    mask = np.in1d(all_cells['ParticleIDs'].astype(np.int64), np.array(cell_ids, dtype=np.int64),
                   assume_unique=True)
    filtered_cells = {name: None for name in all_cells.keys()}
    for key in filtered_cells.keys():
        filtered_cells[key] = all_cells[key][mask]

    print(f'len(all_cells) = {len(all_cells["ParticleIDs"])}')
    print(f'len(filtered_cells) = {len(filtered_cells["ParticleIDs"])}')

    return filtered_cells


def create_stream_hdf5(snap, cutoffs):
    from hestia.particles import retrieve_particles
    from hestia.geometry import get_redshift

    output_base_path = '/z/rschisholm/halos/09_18/stream/snapshot_files/'

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    z_ = get_redshift('09_18', snap)

    all_cells = retrieve_particles('09_18', 'halo_08', snap, float(z_), 'PartType0', padding=True)
    all_cells = add_gasHeaders(all_cells)
    stream_cells = filter_gaseousStream(all_cells, cutoffs)

    # Create a new HDF5 file and write the filtered particles to it
    output_path = output_base_path + 'snapshot_' + snap_ + '.hdf5'
    with h5py.File(output_path, 'w') as outfile:
        # Write the filtered particles dataset
        for key, data in stream_cells.items():
            outfile.create_dataset('PartType0/' + key, data=data)

    print(f'Snapshot written as \'' + output_path + '\'' + '\n' + '--------------------')

    return calc_streamProperties(stream_cells, snap, z_)


def main():
    # --------------------------------------
    snaps = [95, 127]
    cutoffs = {
        'Lx': (1e7, 1e8),
        'Ly': (1e8, 1e9),
        'Lz': (1e8, 1e9),
        'Z': (-2.5, -2.0)
    }
    # --------------------------------------
    column_names = ['Snapshot', 'z', 'M_H0', 'mean_n_H0', 'sigma_n_H0', 'T_avg', 'sigma_T']
    # --------------------------------------

    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    # Optional arguments for snapshot range
    parser.add_argument("--start", type=int, default=snaps[0], help='starting snapshot')
    parser.add_argument("--end", type=int, default=snaps[1], help='ending snapshot')

    args = parser.parse_args()

    print('Writing snapshot files for 09_18 gaseous stream ...')
    print(f'running from snapshot {args.start} to {args.end}, with cutoffs ...')
    for key in cutoffs:
        print(f'\t{key}: {cutoffs[key]}')

    # Define your column names and data
    datFile_path = '/z/rschisholm/halos/09_18/stream/snapshot_files/streamProperties.dat'

    # Write to a .dat file
    with open(datFile_path, 'w') as f:
        # Write the header (commented out with '#')
        f.write('# ' + ' '.join(column_names) + '\n')

    for snap in range(args.end, args.start, -1):
        datFile_data = create_stream_hdf5(snap, cutoffs)

        # Write to a .dat file
        with open(datFile_path, 'a') as f:
            f.write(' '.join(f'{val:.6e}' if isinstance(val, float) else str(val) for val in datFile_data) + '\n')

    print('Finished writing snapshot files for 09_18.halo_08.stream!')


# -------------------------------
verbose = True
# -------------------------------

if __name__ == "__main__":
    main()
