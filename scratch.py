from hestia.geometry import transform_haloFrame
from hestia.halos import get_halo_params
import numpy as np
h = 0.677

# snapshots = [307, 247, 184, 119]
snapshots = [127, 124, 121, 118]
for snap in snapshots:
    halo_01 = get_halo_params('09_18', 'halo_454', snap)
    halo_01['Coordinates'] = halo_01['halo_pos'][np.newaxis, :] / h
    halo_01['Velocities'] = np.ones((1, 3))
    halo_01['Masses'] = np.ones(1)
    halo_01['ParticleIDs'] = np.ones(1)

    pos = transform_haloFrame('09_18', 'halo_38', snap, halo_01, verbose=True)['position']
    print(f'halo_01 position : {pos} kpc')