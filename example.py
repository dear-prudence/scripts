import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm


# what I do on the cluster
input_path = '/Users/ursa/dear-prudence/halos/MW_gal/output/snapshot_025.hdf5'
f = h5py.File(input_path, 'r')
gas = f['PartType2']
print(gas.keys())

hist, x_edges, y_edges = np.histogram2d(gas['Coordinates'][:, 0], gas['Coordinates'][:, 1],
                                        weights=np.array(gas['Masses']) * 1e10,
                                        bins=500,
                                        range=np.array([[-50, 50], [-50, 50]]))
# save as .npz file

# what I do locally
fig = plt.figure(figsize=(6, 6))
plt.imshow(hist.T, extent=(float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])), origin='lower',
           norm=LogNorm(vmin=1e5, vmax=1e8), cmap='Blues')
plt.show()

