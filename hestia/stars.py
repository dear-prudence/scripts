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


def compute_sigmaV(run, halo, snap, radius_cutoff, verbose=False):
    from .particles import retrieve_particles

    verbose and print('\t\t\tcomputing stellar velocity dispersion...')
    stars = retrieve_particles(run, halo, snap, 'PartType4', verbose=False)

    # stars have SFT > 0, wind particles have SFT < 0
    stellar_mask = stars['GFM_StellarFormationTime'] > 0
    stars = {key: val[stellar_mask] for key, val in stars.items()}

    stars['norm'] = np.linalg.norm(stars['position'], axis=1)  # norm from halo center
    mask = stars['norm'] < radius_cutoff  # in kpc
    nuclearStars = {k: v[mask] for k, v in stars.items()}
    verbose and print(f'\t\t\tnum of nuclear stars : {len(nuclearStars["ParticleIDs"])}')

    v_phi = ((nuclearStars['position'][:, 0] * nuclearStars['velocity'][:, 1] - nuclearStars['velocity'][:, 0] * nuclearStars['position'][:, 1])
               / np.sqrt(nuclearStars["position"][:, 0] ** 2 + nuclearStars["position"][:, 1] ** 2))

    print(f'specific ang mom l_z: {np.mean(nuclearStars["angularMomentum"][:, 2] / nuclearStars["Masses"])},'
          f'{np.std(nuclearStars["angularMomentum"][:, 2] / nuclearStars["Masses"])}, '
          f'{np.min(nuclearStars["angularMomentum"][:, 2] / nuclearStars["Masses"])}, '
          f'{np.max(nuclearStars["angularMomentum"][:, 2] / nuclearStars["Masses"])}')

    print(f'|v|_mean, |v|_sigma : {np.mean(np.linalg.norm(nuclearStars["velocity"], axis=1))}, '
          f'{np.std(np.linalg.norm(nuclearStars["velocity"], axis=1))}\n'
          f'vPhi_mean, vPhi_sigma : {np.mean(v_phi)} ,{np.std(v_phi)}')
