import h5py
import numpy as np
import argparse


def list_hdf5_keys(filename):
    def recurse(name, obj):
        indent = '  ' * name.count('/')
        if isinstance(obj, h5py.Group):
            # Check if group contains any datasets with nonzero size
            datasets_in_group = [
                dset for dset in obj.values() if isinstance(dset, h5py.Dataset)
            ]
            if not datasets_in_group:
                print(f"{indent}{name}/  [Group] EMPTY (no datasets)")
            elif all(dset.size == 0 for dset in datasets_in_group):
                print(f"{indent}{name}/  [Group] EMPTY (datasets exist but have size 0)")
            else:
                print(f"{indent}{name}/  [Group]")
        elif isinstance(obj, h5py.Dataset):
            size = obj.size
            if size == 0:
                print(f"{indent}{name}  [Dataset] EMPTY (shape={obj.shape}, dtype={obj.dtype})")
            else:
                print(f"{indent}{name}  [Dataset] shape={obj.shape}, dtype={obj.dtype}, size={size}")

    with h5py.File(filename, 'r') as f:
        print(f"Listing contents of: {filename}")
        f.visititems(recurse)


parser = argparse.ArgumentParser()
parser.add_argument('run', default='09_18')
parser.add_argument('snap', nargs='?', default=127)
args = parser.parse_args()

input_path = (f'/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{args.run}/'
              f'output' + ('_2x2.5Mpc' if args.run != '09_18_lastgigyear' else '')
              + f'/snapdir_{args.snap}/snapshot_{args.snap}.0.hdf5')
list_hdf5_keys(input_path)
