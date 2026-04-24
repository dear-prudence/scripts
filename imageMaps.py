import os
import inspect
import argparse
import numpy as np
import astropy.units as u
from util.hestia import Particles, Halo


class GasProcessor:
    def __init__(self, cells):
        self.cells = cells

    def process_massDen(self):  # mass density of the gas cells in M_solar/kpc^3
        self.cells.weights = self.cells.Density
        self.cells.background = 1e-10
        return self.cells

    def process_H0(self):
        if self.cells.run == '09_18_lastgigyear':
            print(f'Error: \'H0\' is an invalid parameter for \'09_18_lastgigyear\'; '
                  f'line {inspect.currentframe().f_lineno}')
            exit(1)
        f_H0 = self.cells.NeutralHydrogenAbundance
        self.cells.weights = self.cells.nH * f_H0
        self.cells.background = 1e-10
        return self.cells

    def process_H1(self):
        if self.cells.run == '09_18_lastgigyear':
            print(f'Error: \'H1\' is an invalid parameter for \'09_18_lastgigyear\'; '
                  f'line {inspect.currentframe().f_lineno}')
            exit(1)
        f_H1 = 1 - self.cells.NeutralHydrogenAbundance
        self.cells.weights = self.cells.nH * f_H1
        self.cells.background = 1e-10
        return self.cells

    def process_temperature(self):
        self.cells.weights = self.cells.temperature
        self.cells.background = 1e5  # approximate temperature of the IGM
        return self.cells

    def process_metallicity(self):
        self.cells = self.cells.filter(self.cells.GFM_Metallicity > 0)
        self.cells.weights = self.cells.GFM_Metallicity  # metal fraction
        self.cells.background = 1e-10  # primordial metal-free gas
        return self.cells


class DarkMatterProcessor:
    def __init__(self, dms):
        self.dms = dms

    def process_numDen(self):  # number density of the gas cells in kpc^-3
        self.dms.weights = np.ones(self.dms.ParticleIDs.shape)
        self.dms.background = 1e-10  # arbitrarily diffuse background


class StarsProcessor:
    def __init__(self, stars):
        # stars have SFT > 0, wind particles have SFT < 0
        self.stars = stars.filter(stars.GFM_StellarFormationTime > 0)
        self.unity = u.dimensionless_unscaled

    def process_surfaceDen(self):  # surface mass density of the star particles in M_solar/kpc^2
        self.stars.weights = np.ones(self.stars.Masses.shape) * self.unity
        self.stars.background = 1e-10  # arbitrarily diffuse background
        return self.stars

    def process_surfaceBrightness(self):  # surface brightness of the star particles for given band in mag/kpc^2
        band_dict = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
        # these bands are Buser's 'X' filter, where 'X' = {U, B3, V} (Vega magnitudes),
        # then IR K filter + Palomar 200 IR detectors + atmosphere.57 (Vega),
        # then SDSS Camera 'X' Response Function, airmass = 1.3 (June 2001), where 'X'= {g, r, i, z} (AB magnitudes).
        # ----------------------------------------------
        band = 'V'
        # ----------------------------------------------
        self.stars.weights = np.power(10, -0.4 * self.stars.GFM_StellarPhotometrics[:, band_dict[band]]) * self.unity
        self.stars.Masses = np.ones(self.stars.Masses.shape) * u.M_sun
        self.stars.background = 100  # arbitrarily dark background
        return self.stars

    def process_metallicity(self):
        self.stars = self.stars.filter(self.stars.GFM_Metallicity > 0)
        # Z / H
        self.stars.weights = self.stars.GFM_Metallicity / self.stars.GFM_Metals[:, 0] * self.unity
        self.stars.background = 1e-10  # primordial metal-free gas
        return self.stars

    def process_Fe_H(self):
        if self.stars.run == '09_18_lastgigyear':
            print(f'Error: \'Fe_H\' is an invalid parameter for \'09_18_lastgigyear\'; '
                  f'line {inspect.currentframe().f_lineno}')
            exit(1)
        self.stars = self.stars.filter(self.stars.GFM_Metallicity > 0)
        # Fe / H
        self.stars.weights = self.stars.GFM_Metals[:, 8] / self.stars.GFM_Metals[:, 0] * self.unity
        self.stars.background = 1e-10  # primordial metal-free gas
        return self.stars

    def process_alpha_Fe(self):
        if self.stars.run == '09_18_lastgigyear':
            print(f'Error: \'alpha_Fe\' is an invalid parameter for \'09_18_lastgigyear\'; '
                  f'line {inspect.currentframe().f_lineno}')
            exit(1)
        self.stars = self.stars.filter(self.stars.GFM_Metallicity > 0)
        # alpha / Fe
        alpha = (self.stars.GFM_Metals[:, 4]
                 + self.stars.GFM_Metals[:, 5]
                 + self.stars.GFM_Metals[:, 6]
                 + self.stars.GFM_Metals[:, 7])
        self.stars.weights = alpha / self.stars.GFM_Metals[:, 8] * self.unity
        self.stars.background = 1e-10  # primordial metal-free gas
        return self.stars

    def process_potential(self):
        self.stars.weights = self.stars.Potential - np.min(self.stars.Potential)
        self.stars.background = 0  # zero potential

    # def process_mbp(self):
    #     from hestia.stars import get_mbp
    #     mbp_ids = get_mbp('09_18_lastgigyear', 'halo_08', snap=119, numParts=100, verbose=True)
    #     mbp_mask = np.isin(self.stars['ParticleIDs'], mbp_ids)
    #     mbp_stars = {k: v[mbp_mask] for k, v in self.stars.items()}
    #     print(f'\t\t{len(mbp_stars["ParticleIDs"])} mbps located in this snapshot\n'
    #           f'\t\t\tmean(norm) : {np.average(np.linalg.norm(mbp_stars["position"].T, axis=0)):2f} kpc')
    #     return Processing(
    #         mbp_stars,
    #         mbp_stars['Masses'],
    #         mbp_stars['Masses'],
    #         1e-10  # arbitrarily low density
    #     )


# noinspection PyUnboundLocalVariable
def param_processing(part_type, param, particles, verbose=True):
    verbose and print(f'\t\tdispatching "{part_type}/{param}" to corresponding routine ...')

    if part_type == 'PartType0':  # gas
        gas_proc = GasProcessor(particles)
        dispatcher = {
            'massDen': gas_proc.process_massDen,
            'num_H0': gas_proc.process_H0,
            'num_H1': gas_proc.process_H1,
            'temperature': gas_proc.process_temperature,
            'metallicity': gas_proc.process_metallicity,
        }
    elif part_type == 'PartType1':  # dm
        dm_proc = DarkMatterProcessor(particles)
        dispatcher = {
            'numDen': dm_proc.process_numDen,
        }
    elif part_type == 'PartType4':  # stars
        stars_proc = StarsProcessor(particles)
        dispatcher = {
            'surfaceDen': stars_proc.process_surfaceDen,
            'surfaceBrightness': stars_proc.process_surfaceBrightness,
            'metallicity': stars_proc.process_metallicity,
            'Fe_H': stars_proc.process_Fe_H,
            'alpha_Fe': stars_proc.process_alpha_Fe,
            'potential': stars_proc.process_potential,
            # 'mbp': stars_proc.process_mbp,
        }

    # return <-- particles, weights, masses, background = dispatcher[param]()
    return dispatcher[param]()


def make_snap(P, param, bins, axis, S, v):
    from util.hestia import sphProjection

    # Get the indices not equal to the specified axis
    cartesian = np.array([0, 1, 2])
    axs = cartesian[cartesian != axis]
    bounds = (-S[axs[0]].value / 2, S[axs[0]].value / 2, -S[axs[1]].value / 2, S[axs[1]].value / 2)  # in kpc

    v and print(f'\t\t truncating particles, column depth : {S[axis]}')
    column_mask = np.abs(P.position[:, axis]) < (S[axis] / 2)
    specks = P.filter(column_mask)

    v and print(f'\t\t calling hestia.image.create_projection_histogram()')
    return sphProjection(specks, param, axs, bins, bounds, v=v)


def package_data(run, halo, snaps, particle_type, param, dims, pixels, padding, bool_sph, v):
    from archive.hestia.geometry import get_lookbackTimes, get_redshift

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4', 'tout': 'tout'}
    part_type = part_to_type[particle_type]

    bins = [pixels, pixels]  # for the 2-dim histograms
    redshifts, lookback_times = get_lookbackTimes(run, np.array(range(snaps[1], snaps[0], -1)))

    # Dictionary to hold results for each axis/dimension
    all_image = {'y-z': None, 'x-z': None, 'x-y': None,
                 'y_e': None, 'z_e': None, 'x_e': None}

    for snap in range(snaps[1], snaps[0], -1):
        z_ = get_redshift(run, snap)
        v and print(f'\nen train de travailler au snapshot * {snap}, z = {z_} * ...')

        P = param_processing(part_type, param, Particles(run, halo, snap, part_type, verbose=v), verbose=v)
        v and print(f'\tprocessed {P.len} particles/cells with the following properties:'
                    + f'\n\t\t mean({param}.weights) : {np.average(P.weights)}'
                    + f'\n\t\t background : {P.background}')

        # Loop through the axes
        for axis, name_i, edge_i in zip(range(3), ['y-z', 'x-z', 'x-y'], ['y_e', 'z_e', 'x_e']):
            S = np.array([dims[1], dims[1], dims[1]]) * u.kpc
            S[axis] = dims[0] * u.kpc  # creates the prism to restrict particles being plotted by

            v and print(f'\tprojecting particles/cells onto {name_i}--plane ...')
            image, i_e, j_e = make_snap(P, param, bins, axis, S, v=v)

            # Initialize arrays if this is the first snapshot
            if snap == snaps[1]:
                all_image[name_i] = image.reshape((pixels, pixels, 1))
                # store the edges, for the x-z case, store the z_edges (second axis) instead
                all_image[edge_i] = j_e if axis == 1 else i_e

            else:
                # Append new snapshot data
                all_image[name_i] = np.dstack((all_image[name_i], image.reshape((pixels, pixels, 1))))
                all_image[edge_i] = j_e if axis == 1 else i_e  # edges only need to be stored once

        # Now all_image['x-y'], all_image['y-z'], image['x-z'], etc., hold the stacked data
        v and print(f'\t\t planar shape of image array : {all_image["x-y"].shape}')
        v and print(f'\tsnapshot {snap} : check.')

    # Combine all dictionaries into one
    data_to_save = all_image.copy()

    data_to_save['redshifts'] = redshifts  # timestamps
    data_to_save['lookback_times'] = lookback_times
    data_to_save['column_width'] = round(dims[1] / float(pixels), 3)  # in ckpc
    data_to_save['column_depth'] = dims[0]  # in ckpc
    data_to_save['image_size'] = dims[1]  # in ckpc
    data_to_save['virial_radii'] = np.array([Halo(run, halo, snap).R.value for snap in range(snaps[1], snaps[0], -1)])

    # Save data
    output_base = '/z/rschisholm'
    output_path = f'/halos/{run}/{halo}/images/{particle_type}/{param}/'
    output_name = f'{run}.{halo}.{particle_type}.{param}.{dims[0]}x{dims[1]}kpc.npz'

    try:
        os.mkdir(output_base + output_path)
        print('\toutput directory written')
    except FileExistsError:
        pass

    np.savez_compressed(output_base + output_path + output_name, **data_to_save)
    return output_path + output_name


def main(cluster):
    # input arguments and parser
    parser = argparse.ArgumentParser(
        description='Run image map script for a galaxy and snapshot range.'
    )

    # --- required positional args ---
    positional_args = [
        ('run', 'indicated simulation run, e.g. 09_18'),
        ('halo', 'halo to be processed, e.g. halo_08'),
        ('particle_type', 'particle type to be processed, e.g. gas, stars, dm, tout'),
        ('parameter', 'parameter to be processed, e.g. massDen'),
    ]

    # --- optional args ---
    optional_args = [
        ('--length', dict(type=int, default=400, help='side length of image in kpc')),
        ("--depth", dict(type=int, default=100, help='column depth of image in kpc')),
        ("--pixels", dict(type=int, default=400, help='side length of image in pixels')),
        ("--start", dict(type=int, default=97, help='starting snapshot')),
        ("--end", dict(type=int, default=127, help='ending snapshot')),
        ('--padding', dict(type=int, default=None, help='No padding?')),
    ]

    # --- boolean flags ---
    bool_args = [
        ('--sph', dict(dest='sph', action='store_true', help='sph kernel projection?')),
        ('--v', dict(dest='verbose', action='store_true', help='verbose print statements')),
    ]

    for name, helptext in positional_args:
        parser.add_argument(name, help=helptext)
    for name, kwargs in optional_args:
        parser.add_argument(name, **kwargs)
    for name, kwargs in bool_args:
        parser.add_argument(name, **kwargs)
    parser.set_defaults(h0=False, bar=False, bh=False, sph=False, v=False)
    args = parser.parse_args()

    print('--------------------------------------------------------------------------------\n'
          + f'en train de creer du graphique ...\n'
          + f'sim_run -- {args.run}\n'
          + f'halo -- {args.halo}\n'
          + f'part_type -- {args.particle_type}\n'
          + f'parameter -- {args.parameter}\n'
          + f'depth, length, pixels -- {args.depth}, {args.length}, {args.pixels}\n'
          + f'start, end snapshot -- {args.start}, {args.end}\n'
          + (f'padding -- {args.padding}\n' if args.padding is not None else '')
          + ('with sph-projection.\n' if args.sph else '')
          + 'verbose print statements -- ' + ('True\n' if args.verbose else 'False\n')
          + '--------------------------------------------------------------------------------')

    if cluster == 'erebos':
        output_path = package_data(args.run, args.halo, (args.start, args.end), args.particle_type, args.parameter,
                                   dims=(args.depth, args.length), pixels=args.pixels, padding=args.padding,
                                   bool_sph=args.sph, v=args.verbose)
    elif cluster == 'scylla':
        output_path = package_data(args.run, args.halo, (args.start, args.end), args.particle_type, args.parameter,
                                   dims=(args.depth, args.length), pixels=args.pixels, padding=args.padding,
                                   bool_sph=args.sph, bool_bh=args.bh, v=args.verbose)
    else:
        print(f'Error: {cluster} is an invalid cluster given (e.g. erebos, scylla, etc...); '
              f'line {inspect.currentframe().f_lineno}')
        exit(1)

    print(f'le dossier de sortie a ete ecrit, trouvez le chemin d\'access ci-dessous:\n\n{output_path}\n')


def plotting():
    """
    plotting dispatcher (local) ;
    choose b/w one of several plot types :
        {
    frames: will write a double-projection frame (iterated over all specified snapshots)
    potential: will plot a specified (semi-)analytic potential from galpy
    }
    """
    from util.imaging import (dispatch_plot)

    # ----------------------------------------------------
    plot_type = 'frames'
    # ----------------------------------------------------
    # for discrete particles ...
    planes = ['x-y', 'x-z']
    run = '09_18'
    halo = 'halo_38'  # chosen halo frame of reference, or 'stream' for MS-analog
    particle_type = 'stars'
    parameter = 'Fe_H'
    dims = [5, 20]  # in c-kpc, 0-entry is smaller dim, 1-entry is larger image dim
    # ----------------------------------------------------
    # for semi-analytical functions ...
    # parameter = 'potential'
    # pot = 'lmc-smc-mw'
    # plane = 'x-y'
    # pixels = 200
    # ----------------------------------------------------

    snapshot = 238

    if plot_type == 'frames':
        input_path = (f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}'
                      f'/{run}.{halo}.{particle_type}.{parameter}.{dims[0]}x{dims[1]}kpc.npz')
        output_path = (f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}/'
                       f'{dims[0]}x{dims[1]}_frames/')
        dispatch_plot('imageMaps', plot_type, input_path, output_path, partType=particle_type,
                      run=run, parameter=parameter, planes=planes)

    elif plot_type == 'special':
        input_path = (f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}'
                      f'/{run}.{halo}.{particle_type}.{parameter}.{dims[0]}x{dims[1]}kpc.npz')
        output_path = f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}'
        dispatch_plot('imageMaps', plot_type, input_path, output_path, parameter=parameter,
                      partType=particle_type, snapshot=snapshot)

    elif plot_type == 'tempOffset':
        input_path = (f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}'
                      f'/{run}.{halo}.{particle_type}.{parameter}.{dims[0]}x{dims[1]}kpc.npz')
        output_path = f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}'
        dispatch_plot('imageMaps', plot_type, input_path, output_path, parameter=parameter,
                      partType=particle_type)

    elif plot_type == 'dynamics':
        from scripts.util.archive.mainDEPRECEATED import imageMap
        imageMap(parameter, pot, plane, pixels)

    elif plot_type == 'chisholm2025':
        from util.publications import imageMap
        imageMap()

    else:
        exit(1)


if __name__ == "__main__":
    import socket

    machine = socket.gethostname()

    if 'aip.de' in machine:  # aip cluster
        main('erebos')
    elif machine == 'scylla':  # scylla cluster
        main('scylla')
    else:  # util machine
        plotting()
