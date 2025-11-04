import numpy as np


def get_barCenter(run, halo, snap, disk_cutoff=10, verbose=False):
    from .particles import retrieve_particles

    verbose and print('\t\t\tcomputing center of stellar bar...')
    stars = retrieve_particles(run, halo, snap, 'PartType4', verbose=False)

    # stars have SFT > 0, wind particles have SFT < 0
    stellar_mask = stars['GFM_StellarFormationTime'] > 0
    stars = {key: val[stellar_mask] for key, val in stars.items()}

    stars['norm'] = np.linalg.norm(stars['position'], axis=1)  # norm from halo center
    mask = stars['norm'] < disk_cutoff  # in kpc
    nuclearStars = {k: v[mask] for k, v in stars.items()}
    verbose and print(f'\t\t\tnum of nuclear stars : {len(nuclearStars["ParticleIDs"])}')

    return np.array([
        np.average(nuclearStars['position'][:, 0], weights=nuclearStars['Masses']),
        np.average(nuclearStars['position'][:, 1], weights=nuclearStars['Masses']),
        np.average(nuclearStars['position'][:, 2], weights=nuclearStars['Masses'])
    ])