import h5py
import argparse
import numpy as np
from archive.hestia import append_particles, convert_to_supported_dtype


def format_val(val):
    if isinstance(val, int):
        return f'{val:>20d}'  # Right-aligned integer
    elif isinstance(val, float):
        return f'{val:>14.6e}'  # Right-aligned float in scientific notation
    else:
        return str(val)  # Fallback for other types


# efficiently extracts particles from AHF_particles
def extract_particle_ids(file_path, target_halo_id, verbose=True):
    with open(file_path, 'r') as f:
        halo_count = 0
        while True:
            header = f.readline()
            if not header:
                if verbose:
                    print("Reached end of file without finding the target halo.")
                break  # End of file

            parts = header.strip().split()
            if len(parts) != 2:
                if verbose:
                    print(f"Skipping malformed header at halo index {halo_count}: {header.strip()}")
                continue  # skip malformed lines

            num_particles, halo_id = int(parts[0]), str(parts[1])

            # 1000 in max number of halos checking
            if verbose and halo_count % 1000 == 0:
                print(f"Checked {halo_count} halos so far...")

            if halo_id == target_halo_id:
                if verbose:
                    print(f'Found halo {halo_id} at index {halo_count}, with {num_particles} particles.')

                # Collect the particle IDs (may span multiple lines)
                particle_ids = []
                while len(particle_ids) < num_particles:
                    line = f.readline()
                    if not line:
                        raise ValueError("Unexpected end of file.")
                    particle_ids.extend(line.strip().split())

                if verbose:
                    print(f'Successfully extracted {len(particle_ids)} particle IDs.')
                return particle_ids  # Found and returned

            else:
                if verbose:
                    print(f'Skipping {num_particles} lines belonging to halo {halo_id}.')
                # Skip particle lines for this halo
                for _ in range(num_particles):
                    f.readline()
                # print(f'Skipped {skipped} lines.')

            halo_count += 1

    raise ValueError(f'Halo ID {target_halo_id} not found.')


def isolate_halo_padding(particles, run, halo, snap, cushioning_factor=2.):
    from archive.hestia import get_halo_params
    from archive.hestia import filter_particles

    # cushioning_factor determines the scale factor multiplies by R_vir to serve as halo particle bounds (deafult=4)

    halo_params = get_halo_params(run, halo, snap)
    lb_h, ub_h = (halo_params['halo_pos'] - cushioning_factor * halo_params['R_vir'],
                  halo_params['halo_pos'] + cushioning_factor * halo_params['R_vir'])
    filtered_particles = filter_particles(particles, lb_h / 1e3, ub_h / 1e3)
    if verbose:
        print(f'Filtered particles with padding, cushioning factor = {cushioning_factor}; '
              f'{len(particles["ParticleIDs"])} --> {len(filtered_particles["ParticleIDs"])} particles.')
    return filtered_particles


def isolate_halo_AHF(all_particles, run, halo, snap):
    from archive.hestia import halo_dictionary
    from archive.hestia.geometry import get_redshift

    particles = isolate_halo_padding(all_particles, run, halo, snap)

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    halo_id_z0 = halo_dictionary(run, halo)
    filename = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc'
                f'/HESTIA_100Mpc_8192_{run}.127_halo_{halo_id_z0}.dat')
    row = 127 - int(snap)
    halo_data = np.loadtxt(filename)
    halo_id_zi = str(int(halo_data[row, 1]))
    redshift = get_redshift(run, snap)

    # input_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run + '/AHF_output_2x2.5Mpc/'
    #               + 'HESTIA_100Mpc_8192_' + run + '.' + snap_ + '.z' + redshift + '.AHF_particles')
    input_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/AHF_output_2x2.5Mpc/'
                  + f'HESTIA_100Mpc_8192_{run}.{snap_}.z{redshift}.AHF_particles')
    particles_assignments = extract_particle_ids(input_path, halo_id_zi)
    print(f'{len(particles_assignments)} particles found for halo {halo_id_zi}.')

    # halo_particles = {pid: all_particles['ParticleIDs'] for pid in particles_assignments
    #                   if pid in all_particles['ParticleIDs']}

    original_length = len(particles['ParticleIDs'])
    indices_to_keep = np.where(particles['ParticleIDs'] in particles_assignments)[0]
    print('Done with particle assignments, number of thrown-out particles: '
          + str(original_length - len(indices_to_keep)))
    filtered_particles = {name: None for name in particles.keys()}
    for key in particles.keys():
        filtered_particles[key] = particles[key][indices_to_keep]

    print(f'len(particle_ids) = {len(filtered_particles["ParticleIDs"])}')

    return filtered_particles


def create_halo_hdf5(run, halo, snap, output_base_path, verbose=True):
    from archive.hestia.geometry import transform_haloFrame, rid_h_units, get_redshift

    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    z = get_redshift(run, snap)

    verbose and print(f'\nen train de travailler au snapshot * {snap}, z = {z} * ...')

    # This module retrieves the keys for all the particle types
    key_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/'
                f'output' + ('_2x2.5Mpc' if run != '09_18_lastgigyear' else '') + '/snapdir_127/snapshot_127.0.hdf5')
    with h5py.File(key_path, 'r') as k:
        part0_keys = list(k['PartType0'].keys())
        part1_keys = list(k['PartType1'].keys())
        part4_keys = list(k['PartType4'].keys())
        part5_keys = list(k['PartType5'].keys())

    base_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}'
                 f'/output' + ('_2x2.5Mpc' if run != '09_18_lastgigyear' else '') +
                 f'/snapdir_{snap_}/snapshot_{snap_}.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_parts0 = {name: None for name in part0_keys}
    all_parts1 = {name: None for name in part1_keys}
    all_parts4 = {name: None for name in part4_keys}
    all_parts5 = {name: None for name in part5_keys}

    # Loop through the file paths and append coordinates
    print('\tcurrently extracting particles ... ')
    for file_path in file_paths:
        all_parts0 = append_particles('PartType0', file_path,
                                      key_names=part0_keys, existing_arrays=all_parts0)
        all_parts1 = append_particles('PartType1', file_path,
                                      key_names=part1_keys, existing_arrays=all_parts1)
        all_parts4 = append_particles('PartType4', file_path,
                                      key_names=part4_keys, existing_arrays=all_parts4)
        try:
            all_parts5 = append_particles('PartType5', file_path,
                                          key_names=part5_keys, existing_arrays=all_parts5)
        except KeyError:  # if no bhs
            pass
    if verbose:
        print(f'\textracted {len(all_parts0["ParticleIDs"])} gas cells,\n'
              f'\t\t {len(all_parts1["ParticleIDs"])} dm particles,\n'
              f'\t\t {len(all_parts4["ParticleIDs"])} star/wind particles,\n'
              f'\t\t {len(all_parts5["ParticleIDs"])} bh particles.\n')

    filtered_particles0 = isolate_halo_padding(all_parts0, run, halo, snap)
    filtered_particles1 = isolate_halo_padding(all_parts1, run, halo, snap)
    filtered_particles4 = isolate_halo_padding(all_parts4, run, halo, snap)
    filtered_particles5 = isolate_halo_padding(all_parts5, run, halo, snap, cushioning_factor=0.5)

    # Convert filtered particles to supported dtypes
    filtered_particles0 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles0.items()}
    filtered_particles1 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles1.items()}
    filtered_particles4 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles4.items()}
    filtered_particles5 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles5.items()}

    # Transform coordinates to halo frame
    processed_particles0 = transform_haloFrame(run, halo, snap,
                                               rid_h_units(filtered_particles0, z, part_type='PartType0'))
    processed_particles1 = transform_haloFrame(run, halo, snap,
                                               rid_h_units(filtered_particles1, z, part_type='PartType1'))
    processed_particles4 = transform_haloFrame(run, halo, snap,
                                               rid_h_units(filtered_particles4, z, part_type='PartType4'))
    processed_particles5 = transform_haloFrame(run, halo, snap,
                                               rid_h_units(filtered_particles5, z, part_type='PartType5'))

    if verbose:
        print(f'\tprocessed {len(processed_particles0["ParticleIDs"])} gas cells,\n'
              f'\t\t {len(processed_particles1["ParticleIDs"])} dm particles,\n'
              f'\t\t {len(processed_particles4["ParticleIDs"])} star/wind particles,\n'
              f'\t\t {len(processed_particles5["ParticleIDs"])} bh particles.\n')
        print(
            f'\t\t~\t mean(position), \tmean(veloctiy), \tmean(mass) ~\n'
            f'\t\tgas:\t{np.average(np.linalg.norm(processed_particles0["position"][0]))}'
            f'\t{np.average(np.linalg.norm(processed_particles0["velocity"][0]))}'
            f'\t{np.average(processed_particles0["Masses"][0]):.3e}\n'
            f'\t\tdm:\t{np.average(np.linalg.norm(processed_particles1["position"][0]))}'
            f'\t{np.average(np.linalg.norm(processed_particles1["velocity"][0]))}'
            f'\t{np.average(processed_particles1["Masses"][0]):.3e}\n'
            f'\t\tstars:\t{np.average(np.linalg.norm(processed_particles4["position"][0]))}'
            f'\t{np.average(np.linalg.norm(processed_particles4["velocity"][0]))}'
            f'\t{np.average(processed_particles4["Masses"][0]):.3e}\n'
            f'\t\tbh:\t{np.average(np.linalg.norm(processed_particles5["position"][0]))}'
            f'\t{np.average(np.linalg.norm(processed_particles5["velocity"][0]))}'
            f'\t{np.average(processed_particles5["Masses"][0]):.3e}\n'
        )

    # Create a new HDF5 file and write the filtered particles to it
    output_path = output_base_path + 'snapshot_' + snap_ + '.hdf5'
    with h5py.File(output_path, 'w') as outfile:
        # Write the filtered particles dataset
        for key, data in processed_particles0.items():
            outfile.create_dataset('PartType0/' + key, data=data)
        for key, data in processed_particles1.items():
            outfile.create_dataset('PartType1/' + key, data=data)
        for key, data in processed_particles4.items():
            outfile.create_dataset('PartType4/' + key, data=data)
        for key, data in processed_particles5.items():
            outfile.create_dataset('PartType5/' + key, data=data)

    verbose and print(f'termine avec le snapshot {snap},\n'
                      f'ca a ecrit comme \'{output_path}\'\n'
                      f'-----------------------------------------')


def reconstruct_AHF(halo, output_base_path):
    from archive.hestia.geometry import get_redshift
    from archive.hestia import halo_dictionary
    import pandas as pd

    basePath = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/AHF_output/'

    # copied directly since np.load() does not load in commented headers
    header_fields = [
        "ID(1)", "hostHalo(2)", "numSubStruct(3)", "Mvir(4)", "npart(5)", "Xc(6)", "Yc(7)", "Zc(8)", "VXc(9)",
        "VYc(10)", "VZc(11)", "Rvir(12)", "Rmax(13)", "r2(14)", "mbp_offset(15)", "com_offset(16)", "Vmax(17)",
        "v_esc(18)", "sigV(19)", "lambda(20)", "lambdaE(21)", "Lx(22)", "Ly(23)", "Lz(24)", "b(25)", "c(26)",
        "Eax(27)", "Eay(28)", "Eaz(29)", "Ebx(30)", "Eby(31)", "Ebz(32)", "Ecx(33)", "Ecy(34)", "Ecz(35)", "ovdens(36)",
        "nbins(37)", "fMhires(38)", "Ekin(39)", "Epot(40)", "SurfP(41)", "Phi0(42)", "cNFW(43)", "n_gas(44)",
        "M_gas(45)", "lambda_gas(46)", "lambdaE_gas(47)", "Lx_gas(48)", "Ly_gas(49)", "Lz_gas(50)", "b_gas(51)",
        "c_gas(52)", "Eax_gas(53)", "Eay_gas(54)", "Eaz_gas(55)", "Ebx_gas(56)", "Eby_gas(57)", "Ebz_gas(58)",
        "Ecx_gas(59)", "Ecy_gas(60)", "Ecz_gas(61)", "Ekin_gas(62)", "Epot_gas(63)", "n_star(64)", "M_star(65)",
        "lambda_star(66)", "lambdaE_star(67)", "Lx_star(68)", "Ly_star(69)", "Lz_star(70)", "b_star(71)", "c_star(72)",
        "Eax_star(73)", "Eay_star(74)", "Eaz_star(75)", "Ebx_star(76)", "Eby_star(77)", "Ebz_star(78)", "Ecx_star(79)",
        "Ecy_star(80)", "Ecz_star(81)", "Ekin_star(82)", "Epot_star(83)", "mean_z_gas(84)", "mean_z_star(85)",
        "n_star_excised(86)", "M_star_excised(87)", "mean_z_star_excised(88)"
    ]

    # Open the file for writing
    output_filePath = (output_base_path
                       + f'HESTIA_100Mpc_8192_09_18_lastgigyear.127_halo_'
                         f'{halo_dictionary("09_18_lastgigyear", halo)}.dat')
    with open(output_filePath, 'w') as f:
        # Write header as a single commented line
        f.write('# ' + ' '.join(header_fields) + '\n')

        for snap in range(307, 118, -1):
            redshift = f'{get_redshift("09_18_lastgigyear", snap):.3f}'

            try:  # if this is not the first instance of the loop
                haloId = str(parent_haloID)
            except NameError:
                haloId = halo_dictionary('09_18_lastgigyear', halo)  # if this is the first instance of the loop

            print(snap)
            dat_inputPath = f'{basePath}HESTIA_100Mpc_8192_09_18_lastgigyear.{snap}.z{redshift}.AHF_halos'
            # Define the full dtype: first column is int, rest are float
            n_cols = len(header_fields)  # or however many columns your .dat file has
            dtype = [('f0', 'i8')] + [(f'f{i}', 'f8') for i in range(1, n_cols)]
            full_data = np.genfromtxt(dat_inputPath, dtype=dtype, comments='#')
            dat_row = (redshift,) + tuple(full_data[int(haloId[4:]) - 1])
            print(f'dat_row = {dat_row[:7]},...') if verbose else None

            # Format the row as a space-separated string and write
            f.write('\t'.join(format_val(val) for val in dat_row) + '\n')

            # use merger tree to trace halo backwards by one snapshot
            mtree_inputPath = f'{basePath}HESTIA_100Mpc_8192_09_18_lastgigyear.{snap}.z{redshift}.AHF_mtree_idx'
            mtree = pd.read_csv(mtree_inputPath, delim_whitespace=True, header=None, skiprows=0, engine="python",
                                dtype={0: str})
            idx_halo = mtree.loc[mtree[0].astype(str) == haloId]
            row_halo = idx_halo.index[0]
            parent_haloID = mtree.iloc[row_halo, 1]
            print(f'haloID, parent_haloID = {haloId}, {parent_haloID}') if verbose else None


def main():
    # --------------------------------------
    fileType = 'hdf5'  # 'hdf5' for snapshot files, or 'dat' for reconstructing AHF halo.dat files for 09_18_lastgigyear
    run = '09_18'
    halo = 'halo_08'
    snaps = [67, 127]
    # --------------------------------------

    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    parser.add_argument('fileType', nargs='?', default=fileType, help='.hdf5 or .dat')
    parser.add_argument('run', nargs='?', default=run, help='simulation run')
    parser.add_argument('halo', nargs='?', default=halo, help='halo to be processed')
    # Optional arguments for padding or not (padding indicates including additional particles in the nearby cosmic
    # environment to preserve the util cosmography, mainly used for larger images and such)

    # runs into error when not using padding with 09_18_lastgigyear
    parser.add_argument('--padding', dest='padding', action='store_true', help="Add padding")
    parser.set_defaults(padding=False)
    # Optional arguments for snapshot range
    parser.add_argument("--start", type=int, default=snaps[0], help='starting snapshot')
    parser.add_argument("--end", type=int, default=snaps[1], help='ending snapshot')

    args = parser.parse_args()

    # checks to make sure the specified halo is in 09_18 if AHF halo.dat file reconstruction is requested
    if args.fileType == 'dat' and args.run != '09_18':
        print('Error: AHF halo.dat file reconstruction only valid for halos in run 09_18!')
        exit(1)

    print('---------------------------------------------------\n'
          + f'Writing .{args.fileType} file(s) for \n'
          + f'sim_run -- {args.run}\n'
          + f'halo -- {args.halo}\n'
          + (f'running from snapshot {args.start} to {args.end}\n' if args.fileType == 'hdf5' else '')
          + (('sans padding.\n' if not args.padding else '') if args.fileType == 'hdf5' else '')
          + '---------------------------------------------------')

    if args.fileType == 'hdf5':
        base_path = '/store/erebos/rschisholm/halos/' + args.run + '/' + args.halo + '/'
        for snap in range(args.end, args.start, -1):
            create_halo_hdf5(args.run, args.halo, snap, output_base_path=base_path)

    elif args.fileType == 'dat':
        base_path = '/z/rschisholm/halos/' + args.run + '/' + args.halo + '/'
        reconstruct_AHF(args.halo, output_base_path=base_path)

    else:
        print('Error: invalid file type, please try ".hdf5" or ".dat"!')
        exit(1)

    print('Finished writing snapshot files for ' + args.run + ', ' + args.halo + '!')


# -------------------------------
verbose = True
# -------------------------------

if __name__ == "__main__":
    main()
