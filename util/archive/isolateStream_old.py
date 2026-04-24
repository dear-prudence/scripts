import sys
import numpy as np
from hestia import append_particles
from hestia import calc_temperature


def snap_to_redshift(snap):
    filename = ('/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/AHF_output_2x2.5Mpc/'
                'HESTIA_100Mpc_8192_09_18.127_halo_127000000000001.dat')
    redshift = '{:.{}f}'.format(np.loadtxt(filename)[127 - snap, 0], 3)
    return redshift  # returns the redshift of a given snapshot in format x.xxx


def filter_unphysical_metallicities(data):
    for key in data.keys():
        data[key] = np.array(data[key])  # Convert to numpy array for easier indexing
    # Get the indices where metallicities are negative
    original_length = len(data['GFM_Metallicity'])
    indices_to_keep = np.where(data['GFM_Metallicity'][:] > 0)[0]
    print('Done with unphysical metallicity filtering, number of thrown-out particles: '
          + str(original_length - len(indices_to_keep)))
    for key in data.keys():
        data[key] = data[key][indices_to_keep]
    return data


def param_processing(particles):
    h = 0.677

    particles['Temperature'] = calc_temperature(u=np.array(particles['InternalEnergy']),
                                                e_abundance=np.array(particles['ElectronAbundance']),
                                                x_h=np.array(particles['GFM_Metals'][:, 0]))
    # ------------------------------------------
    f_H0 = particles['GFM_Metals'][:, 0] * particles['NeutralHydrogenAbundance']
    h0_mass = np.sum(particles['Masses'] * f_H0)
    avg_nH = np.average(particles['n_H'])
    avg_temp = np.average(particles['Temperature'] * f_H0)
    sum_agn = np.sum(particles['GFM_AGNRadiation'] * f_H0)

    print('len(stream_particles) = ' + str(len(particles['ParticleIDs'])))
    # ------------------------------------------

    return particles, h0_mass, avg_nH, avg_temp, sum_agn


def package_data(snaps, primary_snap):
    from hestia import time_edges
    from hestia import create_stream_snapshotFiles
    import h5py

    time_e = time_edges(sim='09_18', snaps=np.arange(snaps[1], snaps[0], step=-1))

    output_base = '/z/rschisholm/storage/snapshots_stream/'

    # ---------------------------------------------------------
    if primary_snap is not None:
        output_path = output_base + 'snapshot_' + str(primary_snap) + '.hdf5'
        stream_ids = create_stream_snapshotFiles(primary_snap, output_path)
        np.savez(output_path + 'streamParticleIDs.npz', ParticleIDs=stream_ids)
    # ---------------------------------------------------------

    # This module retrieves the keys for all the particle types
    key_path = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_127/snapshot_127.0.hdf5'
    with h5py.File(key_path, 'r') as k:
        part0_keys = list(k['PartType0'].keys())

    for snap in range(snaps[1], snaps[0], -1):
        snap_ = '0' + str(snap) if snap < 100 else str(snap)

        output_path = output_base + 'snapshot_' + snap_ + '.hdf5'
        _ = create_stream_snapshotFiles(snap, output_path, particle_ids=stream_ids if primary_snap is not None else None)

        stream_particles = {name: None for name in part0_keys + ['n_H']}

        stream_particles = append_particles('PartType0', output_path, key_names=part0_keys + ['n_H'],
                                            existing_arrays=stream_particles)

        _, sum_mass, avg_n_H0, avg_temp, sum_agn = param_processing(stream_particles)

        print('M_H0 = ' + str(sum_mass) + ',\tn_H0 = ' + str(avg_n_H0)
              + ',\tT_stream = ' + str(avg_temp) + ',\tsum_AGN = ' + str(sum_agn))

    # Indicate that the script has completed its task
    print('Done!')
    # Terminate the script
    sys.exit(0)


# ------------------------------------
machine = 'geras'
# ------------------------------------
primary_snapshot = None  # if None then each snapshot is processed as primary
snaps_ = [101, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
# ------------------------------------

if machine == 'geras':
    package_data(snaps_, primary_snapshot)

elif machine == 'dear-prudence':
    pass
