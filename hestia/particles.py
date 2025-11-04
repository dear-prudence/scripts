# This script includes many routines relating to data manipulation and logistics of the snapshot files from
# the galaxy simulations
from __future__ import division
import h5py
import numpy as np
import os


# This routine will append the snapshot hdf files (snapshotXXX.0.hdf,...,snapshotXXX.7.hdf) containing particle
# information into a dictionary of numpy arrays
def append_particles(part_type, filename, key_names, existing_arrays=None):
    if existing_arrays is None:
        existing_arrays = {name: None for name in key_names}

    with h5py.File(filename, 'r') as file:
        f = file[part_type]
        for key in key_names:
            if key in f:
                data = np.array(f[key])
                existing_arrays[key] = np.append(existing_arrays[key], data, axis=0) \
                    if existing_arrays[key] is not None else data
            else:
                print('Error: ' + str(key) + ' is not a key in hdf file!')
    return existing_arrays


# The routine will filter out any particles outside a given box of interest (to reduce unnecessary calculations)
def filter_particles(particles, min_b, max_b):
    for key in particles.keys():
        particles[key] = np.array(particles[key])  # Convert to numpy array for easier indexing
    indices_to_keep = np.where(
        (min_b[0] <= particles['Coordinates'][:, 0]) & (particles['Coordinates'][:, 0] <= max_b[0]) &
        (min_b[1] <= particles['Coordinates'][:, 1]) & (particles['Coordinates'][:, 1] <= max_b[1]) &
        (min_b[2] <= particles['Coordinates'][:, 2]) & (particles['Coordinates'][:, 2] <= max_b[2])
    )[0]
    for key in particles.keys():
        particles[key] = particles[key][indices_to_keep]
    return particles


def retrieve_particles(run, halo, snap, part_type, padding=None, verbose=True):
    import h5py
    from .halos import get_halo_params
    from .geometry import transform_haloFrame, rid_h_units, get_redshift
    z_ = round(get_redshift(run, snap), 3)

    partType_dict = {'PartType0': 'gas cells', 'PartType1': 'dm particles',
                     'PartType4': 'star particles', 'PartType5': 'central bh'}
    snap_ = '0' + str(snap) if snap < 100 else str(snap)
    verbose and print(f'\tretrieving {partType_dict[part_type]} for {run}/{halo}, snapshot {snap};')

    basePath = '/store/erebos/rschisholm/halos/'

    if halo == 'stream':
        filePath = f'09_18/stream/snapshot_files/snapshot_{snap_}.hdf5'
    else:
        filePath = f'{run}/{halo}/snapshot_{snap_}.hdf5'

    halo_params = get_halo_params(run, halo, snap)
    halo_id, pos_h, vel_h, l_h, r_vir_h = (halo_params['halo_id_zi'], halo_params['halo_pos'],
                                           halo_params['halo_vel'], halo_params['halo_l'], halo_params['R_vir'])

    if os.path.exists(f'{basePath}{filePath}') and padding is None:
        verbose and print('\t\tos.path exists')
        # if a processed halo directory already exists
        with h5py.File(basePath + filePath, 'r') as file:
            keys = file[part_type].keys()
            all_particles = {name: None for name in keys}
            all_particles = append_particles(part_type, basePath + filePath,
                                             key_names=keys, existing_arrays=all_particles)
        processed_particles = all_particles

        if part_type == 'PartType5' and 'position' not in processed_particles.keys():
            # older versions of processHalo.py did not handle bhs
            lb, ub = ((pos_h - r_vir_h / 10),
                      (pos_h + r_vir_h / 10))
            processed_particles = transform_haloFrame(run, halo, snap, rid_h_units(
                filter_particles(all_particles, 1e-3 * lb, 1e-3 * ub), z_, part_type))

    else:
        if padding is None:
            verbose and print('\t\tos.path does not exist, extracting particles from simulation snapshot files')
        else:
            verbose and print(f'\t\trequested padding : {padding} R_vir, '
                              f'extracting particles from simulation snapshot files')
        # if no processed halo directory exists, extract particles from full simulation output
        base_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/output'
                     + ('_2x2.5Mpc' if run != '09_18_lastgigyear' else '') + f'/snapdir_{snap_}/snapshot_{snap_}.')
        file_extension = '.hdf5'
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        with h5py.File(base_path + '0' + file_extension, 'r') as file:
            keys = file[part_type].keys()
            all_particles = {name: None for name in keys}
            for filePath in file_paths:
                all_particles = append_particles(part_type, filePath, key_names=keys,
                                                 existing_arrays=all_particles)

        if part_type != 'PartType5':
            lb, ub = ((pos_h - padding * r_vir_h),
                      (pos_h + padding * r_vir_h))
            processed_particles = transform_haloFrame(run, halo, snap, rid_h_units(
                filter_particles(all_particles, 1e-3 * lb, 1e-3 * ub), z_, part_type))
        else:
            # handles bh particles separately
            alpha = 1 / 10  # bounds_factor_virialRadius, in kpc
            # -----------------------------
            processed_particles = filter_particles(all_particles,
                                                   1e-3 * (pos_h - r_vir_h * alpha),
                                                   1e-3 * (pos_h + r_vir_h * alpha))

    verbose and print(f'\t\tnumber of particles/cells extracted : {len(processed_particles["ParticleIDs"])}')
    return processed_particles


def convert_to_supported_dtype(data):
    """Convert data to a dtype supported by HDF5 if necessary."""
    if isinstance(data, np.ndarray) and data.dtype == 'O':
        # Convert object arrays to string or numeric arrays
        if all(isinstance(item, (int, float)) for item in data):
            return np.array(data, dtype=np.float64)
        else:
            # Convert to string array (fixed-length)
            return np.array(data, dtype='S')
    return data


def get_softeningLength(run, snap, part_type):
    # not valid for gas (PartType0), uses adaptive softening lengths
    from .geometry import get_redshift
    param_filePath = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{run}/'
                      f'output' + ('_2x2.5Mpc' if run != '09_18_lastgigyear' else '') + '/parameters-usedvalues')
    z = get_redshift(run, snap)
    a = 1 / (1 + z)
    h = 0.677

    with open(param_filePath, 'r') as f:
        contents = f.read()
        # obtains softening particle type reference for desired part_type
        for line in contents.splitlines():
            try:
                if str(line[:24]) == f'SofteningTypeOfPartType{part_type[-1]}':
                    part_index = int(line[-1])
            except IndexError:
                pass

        for line in contents.splitlines():
            try:
                if str(line[:22]) == f'SofteningComovingType{part_index}':
                    epsilon_comoving = float(line[-8:]) * 1e3 / h  # in ckpc
                if str(line[:21]) == f'SofteningMaxPhysType{part_index}':
                    epsilon_phys = float(line[-8:]) * 1e3 / h  # in pkpc
            except IndexError:
                pass
            except NameError:
                print(f'Error: could not find particle reference type #SofteningTypeOf{part_type}!')
                exit(1)
    try:
        epsilon_minPhys = np.min(np.array([a * epsilon_comoving, epsilon_phys]))
    except NameError:
        print(f'Error: could not find softening lengths in {param_filePath}!')
        exit(1)

    return epsilon_minPhys / a  # in ckpc


def isolate_halo(particles, run, halo, snap, previous_halo_id=None):
    from .halos import get_halo_params

    halo_id, _, halo_pos_h, _, _, r_vir_h = get_halo_params(run, halo, snap, previous_halo_id=previous_halo_id)
    lb_h, ub_h = halo_pos_h - 4 * r_vir_h, halo_pos_h + 4 * r_vir_h
    return filter_particles(particles, lb_h / 1e3, ub_h / 1e3), halo_id


def create_halo_hdf5(run, halo, snap, output_path):
    """
    Opens an HDF5 file, filters the particle data, and writes a new HDF5 file
    with only the filtered particles.

    Parameters:
    input_file (str): Path to the input HDF5 file.
    output_file (str): Path to the output HDF5 file.
    """

    snap_ = '0' + str(snap) if snap < 100 else str(snap)

    # This module retrieves the keys for all the particle types
    key_path = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run + '/output_2x2.5Mpc/snapdir_127/snapshot_127.0.hdf5'
    with h5py.File(key_path, 'r') as k:
        part0_keys = list(k['PartType0'].keys())
        part1_keys = list(k['PartType1'].keys())
        part4_keys = list(k['PartType4'].keys())
        part5_keys = list(k['PartType5'].keys())

    base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/' + run + '/output_2x2.5Mpc/snapdir_'
                 + snap_ + '/snapshot_' + snap_ + '.')
    file_extension = '.hdf5'
    # Generate file paths using a loop
    file_paths = [base_path + str(x) + file_extension for x in range(8)]
    # Initialize the resulting array
    all_parts0 = {name: None for name in part0_keys}
    all_parts1 = {name: None for name in part1_keys}
    all_parts4 = {name: None for name in part4_keys}
    all_parts5 = {name: None for name in part5_keys}
    # Loop through the file paths and append coordinates
    print('Processing Snapshot ' + snap_ + '...')
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
        except KeyError:
            pass
    print('Length of un-filtered gas particles: ' + str(len(all_parts0['ParticleIDs'])))

    filtered_particles0, _ = isolate_halo(all_parts0, run, halo, snap)
    filtered_particles1, _ = isolate_halo(all_parts1, run, halo, snap)
    filtered_particles4, _ = isolate_halo(all_parts4, run, halo, snap)
    filtered_particles5, _ = isolate_halo(all_parts5, run, halo, snap)

    # Convert filtered particles to supported dtypes
    filtered_particles0 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles0.items()}
    filtered_particles1 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles1.items()}
    filtered_particles4 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles4.items()}
    filtered_particles5 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles5.items()}

    # Create a new HDF5 file and write the filtered particles to it
    with h5py.File(output_path, 'w') as outfile:
        # Write the filtered particles dataset
        for key, data in filtered_particles0.items():
            outfile.create_dataset('PartType0/' + key, data=data)
        for key, data in filtered_particles1.items():
            outfile.create_dataset('PartType1/' + key, data=data)
        for key, data in filtered_particles4.items():
            outfile.create_dataset('PartType4/' + key, data=data)
        for key, data in filtered_particles5.items():
            outfile.create_dataset('PartType5/' + key, data=data)

    print('Done!')


def create_halo_hdf5_lastgigyear(halo, snap, output_path, previous_halo_id=None):
    """
    Opens an HDF5 file, filters the particle data, and writes a new HDF5 file
    with only the filtered particles.

    Parameters:
    input_file (str): Path to the input HDF5 file.
    output_file (str): Path to the output HDF5 file.
    """

    if snap < 118:
        create_halo_hdf5('09_18', halo, snap, output_path)
    else:
        snap_ = '0' + str(snap) if snap < 100 else str(snap)

        # This module retrieves the keys for all the particle types
        key_path = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/output/snapdir_307/snapshot_307.0.hdf5'
        with h5py.File(key_path, 'r') as k:
            part0_keys = list(k['PartType0'].keys())
            part1_keys = list(k['PartType1'].keys())
            part4_keys = list(k['PartType4'].keys())
            part5_keys = list(k['PartType5'].keys())

        if snap == 308:
            base_path = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_127/snapshot_127.'
        else:
            base_path = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18_lastgigyear/output/snapdir_'
                         + snap_ + '/snapshot_' + snap_ + '.')
        file_extension = '.hdf5'

        # Generate file paths using a loop
        file_paths = [base_path + str(x) + file_extension for x in range(8)]
        # Initialize the resulting array
        all_parts0 = {name: None for name in part0_keys}
        all_parts1 = {name: None for name in part1_keys}
        all_parts4 = {name: None for name in part4_keys}
        all_parts5 = {name: None for name in part5_keys}
        # Loop through the file paths and append coordinates
        print('Processing Snapshot ' + snap_ + '...')
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
            except KeyError:
                pass
        print('Length of un-filtered gas particles: ' + str(len(all_parts0['ParticleIDs'])))

        filtered_particles0, halo_id = isolate_halo(all_parts0, '09_18_lastgigyear', halo, snap,
                                                    previous_halo_id=previous_halo_id)
        filtered_particles1, _ = isolate_halo(all_parts1, '09_18_lastgigyear', halo, snap,
                                              previous_halo_id=previous_halo_id)
        filtered_particles4, _ = isolate_halo(all_parts4, '09_18_lastgigyear', halo, snap,
                                              previous_halo_id=previous_halo_id)
        filtered_particles5, _ = isolate_halo(all_parts5, '09_18_lastgigyear', halo, snap,
                                              previous_halo_id=previous_halo_id)

        # Convert filtered particles to supported dtypes
        filtered_particles0 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles0.items()}
        filtered_particles1 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles1.items()}
        filtered_particles4 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles4.items()}
        filtered_particles5 = {key: convert_to_supported_dtype(data) for key, data in filtered_particles5.items()}

        # Create a new HDF5 file and write the filtered particles to it
        with h5py.File(output_path, 'w') as outfile:
            # Write the filtered particles dataset
            for key, data in filtered_particles0.items():
                outfile.create_dataset('PartType0/' + key, data=data)
            for key, data in filtered_particles1.items():
                outfile.create_dataset('PartType1/' + key, data=data)
            for key, data in filtered_particles4.items():
                outfile.create_dataset('PartType4/' + key, data=data)
            for key, data in filtered_particles5.items():
                outfile.create_dataset('PartType5/' + key, data=data)
        print('Done!')

    return halo_id
