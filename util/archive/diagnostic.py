import h5py
import numpy as np


def hdf_info(file, m):
    print(file)
    with h5py.File(file, 'r') as f:
        keys = f.keys()
        if m == 'hestia':
            for key in keys:
                sub_keys = list(f[key].keys())
                print('<< ' + str(key) + ' >>')
                print(sub_keys)
        elif m == 'util':
            for key in keys:
                sub_keys = list(f[key].keys())
                print('<< ' + str(key) + ' >>')
                print(sub_keys)
                if 'ParticleIDs' in sub_keys:
                    print('# of particles: ' + str(len(list(f[key]['ParticleIDs']))))
                else:
                    pass
        print('------------------------------------')
        print(list(f['PartType5']['ParticleIDs']))
        print(list(f['PartType5']['Masses']))


mode = 'hestia'
filename = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/snapdir_127/snapshot_127.2.hdf5'
hdf_info(filename, mode)

