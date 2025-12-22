import numpy as np
import argparse
import inspect
import os


class GasProcessor:
    def __init__(self, cells):
        self.cells = cells

    def process_massDen(self):  # mass density of the gas cells in M_solar/kpc^3
        return {
            'particles': self.cells,
            'weights': self.cells['Density'],
            'masses': self.cells['Masses'],
            'background': 1e-10  # arbitrarily diffuse background
        }

    def process_H0(self):
        from hestia.gas import calc_numberDensity
        return {
            'particles': self.cells,
            'weights': calc_numberDensity(self.cells['Density'] * self.cells['GFM_Metals'][:, 0]
                                          * self.cells['NeutralHydrogenAbundance']),
            'masses': self.cells['Masses'] * self.cells['GFM_Metals'][:, 0],  # H0 mass
            'background': 1e-10  # arbitrarily diffuse background
        }

    def process_H1(self):
        from hestia.gas import calc_numberDensity
        return {
            'particles': self.cells,
            'weights': calc_numberDensity(self.cells['Density'] * self.cells['GFM_Metals'][:, 0]
                                          * (1 - self.cells['NeutralHydrogenAbundance'])),
            'masses': self.cells['Masses'] * self.cells['GFM_Metals'][:, 0],  # H0 mass
            'background': 1e-10  # arbitrarily diffuse background
        }

    def process_columnH0(self):
        return {
            'particles': self.cells,
            'weights': np.ones(self.cells['ParticleIDs'].shape),
            'masses': self.cells['ParticleIDs'].shape,
            'background': 1
        }

    def process_temperature(self):
        from hestia.gas import calc_temperature
        return {
            'particles': self.cells,
            'weights': calc_temperature(self.cells['InternalEnergy'], self.cells['ElectronAbundance'],
                                        x_h=(self.cells['GFM_Metals'][:, 0] if 'GFM_Metals' in self.cells.keys()
                                             else 0.76)),
            'masses': self.cells['Masses'],
            'background': 1e5  # approximate temperature of the IGM
        }

    def process_metallicity(self):
        particles = self.filter_unphysical_Z()
        return {
            'particles': particles,
            'weights': np.log10(particles['GFM_Metallicity'] / particles['GFM_Metals'][:, 0]),
            'masses': particles['Masses'],
            'background': 1e-10  # arbitrarily metal-poor gas
        }

    def process_v(self):
        coordinate_dict = {'x': 0, 'y': 1, 'z': 2}
        # ----------------------------------------------
        desired_direction = 'x'
        # ----------------------------------------------
        return {
            'particles': self.cells,
            'weights': self.cells['Halo_Velocities'][:, coordinate_dict[desired_direction]],
            'masses': self.cells['Masses'],
            'background': 0  # v_z = 0
        }

    def process_L(self):
        coordinate_dict = {'x': 0, 'y': 1, 'z': 2}
        # ----------------------------------------------
        desired_direction = 'x'
        # ----------------------------------------------
        return {
            'particles': self.cells,
            'weights': self.cells['Angular_Momenta'][:, coordinate_dict[desired_direction]],
            'masses': self.cells['Masses'],
            'background': 0  # L_i = 0
        }

    def filter_unphysical_Z(self):
        mask = self.cells['GFM_Metallicity'] > 0
        return {k: v[mask] for k, v in self.cells.items()}


class DarkMatterProcessor:
    def __init__(self, dms):
        self.dms = dms

    def process_massDen(self):  # mass density of the gas cells in M_solar/kpc^3
        return {
            'particles': self.dms,
            'weights': np.ones(self.dms['ParticleIDs'].shape),
            # * 1.5 * 10 ** 5,  # taken from the hestia paper, \S 3.2.2
            'masses': self.dms['Masses'],
            'background': 1e-10  # arbitrarily diffuse background
        }


class StarsProcessor:
    def __init__(self, stars):
        self.stars = stars

    def filter_stars(self):
        # stars have SFT > 0, wind particles have SFT < 0
        stellar_mask = self.stars['GFM_StellarFormationTime'] > 0
        return {key: val[stellar_mask] for key, val in self.stars.items()}

    def process_massDen(self):  # surface mass density of the star particles in M_solar/kpc^2
        filtered_stars = self.filter_stars()
        return {
            'particles': filtered_stars,
            'weights': filtered_stars['Masses'],
            'masses': filtered_stars['Masses'],
            'background': 1e-10  # arbitrarily diffuse background
        }

    def process_surfaceBrightness(self):  # surface brightness of the star particles for given band in mag/kpc^2
        band_dict = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
        # these bands are Buser's 'X' filter, where 'X' = {U, B3, V} (Vega magnitudes),
        # then IR K filter + Palomar 200 IR detectors + atmosphere.57 (Vega),
        # then SDSS Camera 'X' Response Function, airmass = 1.3 (June 2001), where 'X'= {g, r, i, z} (AB magnitudes).
        filtered_stars = self.filter_stars()
        # ----------------------------------------------
        desired_band = 'V'
        # ----------------------------------------------
        return {
            'particles': filtered_stars,
            # 2 for V-band Vega, 5 for r-band SDSS
            # calculates luminosity L (up to factor L_0)
            'weights': np.power(10, -0.4 * filtered_stars['GFM_StellarPhotometrics'][:, band_dict[desired_band]]),
            'masses': np.ones(filtered_stars['Masses'].shape),
            'background': 100  # arbitrarily dark background
        }

    def process_metallicity(self):
        stars = self.filter_stars()
        mask = stars['GFM_Metallicity'] > 0
        filtered_stars = {k: v[mask] for k, v in stars.items()}
        return {
            'particles': filtered_stars,
            'weights': np.log10(filtered_stars['GFM_Metallicity'] / filtered_stars['GFM_Metals'][:, 0]),
            'masses': filtered_stars['Masses'],
            'background': 1e-10  # arbitrarily metal-poor gas
        }

    def process_potential(self):
        stars = self.filter_stars()
        mask = stars['GFM_Metallicity'] > 0
        filtered_stars = {k: v[mask] for k, v in stars.items()}
        return {
            'particles': filtered_stars,
            'weights': filtered_stars['Potential'] - np.min(filtered_stars['Potential']),
            'masses': filtered_stars['Masses'],
            'background': 0  # arbitrarily metal-poor gas
        }


# noinspection PyUnboundLocalVariable
def param_processing(part_type, param, particles, verbose=True):
    verbose and print(f'\t\tdispatching "{part_type}/{param}" to corresponding routine ...')

    if part_type == 'PartType0':  # gas
        gas_proc = GasProcessor(particles)
        dispatcher = {
            'massDen': gas_proc.process_massDen,
            'num_H0': gas_proc.process_H0,
            'num_H1': gas_proc.process_H1,
            'column_H0': gas_proc.process_columnH0,
            'temperature': gas_proc.process_temperature,
            'metallicity': gas_proc.process_metallicity,
            'velocity': gas_proc.process_v,
            'L': gas_proc.process_L,
        }
    elif part_type == 'PartType1':  # dm
        dm_proc = DarkMatterProcessor(particles)
        dispatcher = {
            'massDen': dm_proc.process_massDen,
        }
    elif part_type == 'PartType4':  # stars
        stars_proc = StarsProcessor(particles)
        dispatcher = {
            'massDen': stars_proc.process_massDen,
            'surfaceBrightness': stars_proc.process_surfaceBrightness,
            'metallicity': stars_proc.process_metallicity,
            'potential': stars_proc.process_potential,
        }

    # return <-- particles, weights, masses, background = dispatcher[param]()  # now it runs
    return dispatcher[param]()


def make_snap(part_type, particles_full_depth, param, weights_full_depth, masses_full_depth, background,
              bins, axis, size, bool_sph, verbose):
    from hestia.image import create_projection_histogram, sph_kernel_projection, sph_columnH0_projection

    # Get the indices not equal to the specified axis
    cartesian = np.array([0, 1, 2])
    axs = cartesian[cartesian != axis]

    bounds = np.array([[-1 * size[axs[0]] / 2, size[axs[0]] / 2], [-1 * size[axs[1]] / 2, size[axs[1]] / 2]])

    # ----------------------------------------------
    # to be compatible with depreceated keys (will remove eventually)
    if 'Halo_Coordinates' in particles_full_depth.keys():
        position = 'Halo_Coordinates'
    else:
        position = 'position'
    # ----------------------------------------------

    if param == 'column_H0':
        verbose and print(f'\t\t parameter = {param} detected, calling hestia.image.column_H0()')
        particles = particles_full_depth

        # Estimate smoothing lengths from mass and density
        mass = particles["Masses"]  # in Msun
        density = particles["Density"]  # in Msun / kpc^3
        volume = mass / density  # kpc^3
        hsml = (3.0 / (4.0 * np.pi) * volume) ** (1.0 / 3.0)  # kpc

        return sph_columnH0_projection(particles[position][:, axs[0]],
                                       particles[position][:, axs[1]],
                                       hsml=hsml, masses=particles['Masses'],
                                       h_frac=particles['GFM_Metals'][:, 0],
                                       neutral_frac=particles['NeutralHydrogenAbundance'], bounds=bounds,
                                       nbins=bins[0], verbose=verbose)
    else:
        verbose and print(f'\t\t truncating particles, column depth = {size[axis]} kpc')
        column_mask = np.abs(particles_full_depth[position][:, axis]) < (size[axis] / 2)
        particles = {key: val[column_mask] for key, val in particles_full_depth.items()}
        weights = weights_full_depth[column_mask]
        masses = masses_full_depth[column_mask]

        if bool_sph:
            # Estimate smoothing lengths from mass and density
            mass = particles["Masses"]  # in Msun
            density = particles["Density"]  # in Msun / kpc^3
            volume = mass / density  # kpc^3
            hsml = (3.0 / (4.0 * np.pi) * volume) ** (1.0 / 3.0)  # kpc
            verbose and print(f'\t\t sph argument detected, mean(hsml) = {np.average(hsml)} kpc'
                              f'\n\t\t calling hestia.image.sph_kernel_projection()')
            return sph_kernel_projection(particles[position][:, axs[0]],
                                         particles[position][:, axs[1]],
                                         hsml=hsml, weights=weights, masses=masses,
                                         bounds=bounds, n_bins=bins[0], verbose=verbose)
        else:
            verbose and print(f'\t\t calling hestia.image.create_projection_histogram()')
            return create_projection_histogram(part_type, param,
                                               particles[position][:, axs[0]],
                                               particles[position][:, axs[1]],
                                               weights=weights, masses=masses, background=background,
                                               bins=bins, bounds=bounds, verbose=verbose)


def package_data(run, halo, snaps, particle_type, param, dims, pixels, padding,
                 bool_h0, bool_bar, bool_bh, bool_sph, verbose):
    from hestia.geometry import get_lookbackTimes, get_redshift
    from hestia.particles import retrieve_particles
    from hestia.halos import get_halo_params, get_centralBH

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4', 'tout': 'tout'}
    part_type = part_to_type[particle_type]

    bins = [pixels, pixels]  # for the 2-dim histograms
    redshifts, lookback_times = get_lookbackTimes(run, np.array(range(snaps[1], snaps[0], -1)))

    # Dictionary to hold results for each axis/dimension
    all_image = {'y-z': None, 'x-z': None, 'x-y': None,
                 'y_e': None, 'z_e': None, 'x_e': None}

    # -----------------------
    sph_snapshots = [96, 110, 118, 119, 127]  # for image map plot in chisholm+2025
    # -----------------------

    virial_radii, center_h0, center_bar, central_bh = np.array([]), np.ones(3), np.ones(3), np.ones(3)

    for snap in range(snaps[1], snaps[0], -1):
        z_ = get_redshift(run, snap)

        verbose and print(f'\nen train de travailler au snapshot * {snap}, z = {z_} * ...')
        particles = retrieve_particles(run, halo, snap, part_type, padding=padding, verbose=verbose)

        # Processes the particles using Processor modules located above
        exported_dict = param_processing(part_type, param, particles, verbose=verbose)
        particles, weights, masses, background = (exported_dict['particles'], exported_dict['weights'],
                                                  exported_dict['masses'], exported_dict['background'])
        verbose and print(f'\tprocessed {len(particles["ParticleIDs"])} particles/cells with the following properties:'
                          + f'\n\t\t mean({param}.weights) = {np.average(weights)}'
                          + f'\n\t\t mean({particle_type}.masses) = {np.average(masses)}'
                          + f'\n\t\t background = {background}')

        if halo != 'stream':  # if processing a halo ...
            halo_params = get_halo_params(run, halo, snap)
            virial_radii = np.append(virial_radii, halo_params['R_vir'])

        # Loop through the axes
        for axis, name_i, edge_i in zip(range(3), ['y-z', 'x-z', 'x-y'], ['y_e', 'z_e', 'x_e']):
            # creates the prism to restrict particles being plotted by
            S = np.array([dims[1], dims[1], dims[1]])
            S[axis] = dims[0]

            # -----------------------
            if bool_sph or (snap in sph_snapshots and name_i == 'x-z' and particle_type == 'PartType0'):
                bool_sph = True
            else:
                bool_sph = False
            # -----------------------

            verbose and print(f'\tprojecting particles/cells onto {name_i}--plane ...')
            current_image, i_edge, j_edge = make_snap(part_type, particles, param, weights, masses, background,
                                                      bins, axis, S, bool_sph, verbose=verbose)

            # Initialize arrays if this is the first snapshot
            if snap == snaps[1]:
                all_image[name_i] = current_image.reshape((pixels, pixels, 1))
                # store the edges, for the x-z case, store the z_edges (second axis) instead
                all_image[edge_i] = j_edge if axis == 1 else i_edge

            else:
                # Append new snapshot data
                all_image[name_i] = np.dstack((all_image[name_i], current_image.reshape((pixels, pixels, 1))))
                all_image[edge_i] = j_edge if axis == 1 else i_edge  # edges only need to be stored once

        # as a first order approx., simply calculating weighted average position of nuclear particles
        if bool_h0:
            from hestia.gas import get_h0Center
            center_h0_i = get_h0Center(run, halo, snap, disk_cutoff=5, verbose=True)
            center_h0 = np.vstack((center_h0, center_h0_i))
        if bool_bar:
            from hestia.stars import get_barCenter
            center_bar_i = get_barCenter(run, halo, snap, disk_cutoff=5, verbose=True)
            center_bar = np.vstack((center_bar, center_bar_i))
        # indexes zeroth element since it directly returns an array of shape (1, 3)
        if bool_bh:
            central_bh_i, _ = get_centralBH(run, halo, snap)
            central_bh = np.vstack((central_bh, central_bh_i['position'][0]))

        # Now all_image['x-y'], all_image['y-z'], image['x-z'], etc., hold the stacked data
        verbose and print(f'\t\t planar shape of image array : {all_image["x-y"].shape}')
        verbose and print(f'\tsnapshot {snap} : check.')

    # Combine all dictionaries into one
    data_to_save = all_image.copy()  # Start with all_planes

    # removes the first row initiated by np.ones()
    if bool_h0:
        data_to_save['center_h0'] = center_h0[1:]
    if bool_bar:
        data_to_save['center_bar'] = center_bar[1:]
    if bool_bh:
        data_to_save['central_BH'] = central_bh[1:]

    data_to_save['redshifts'] = redshifts  # timestamps
    data_to_save['lookback_times'] = lookback_times
    data_to_save['column_width'] = round(dims[1] / float(pixels), 3)  # in ckpc
    data_to_save['column_depth'] = dims[0]  # in ckpc
    data_to_save['image_size'] = dims[1]  # in ckpc
    if halo != 'stream':
        data_to_save['virial_radii'] = virial_radii

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
        ("--h0", dict(dest='h0', action='store_true', help="loc of HI center?")),
        ("--bar", dict(dest='bar', action='store_true', help="loc of bar?")),
        ("--bh", dict(dest='bh', action='store_true', help="Central BH?")),
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
          + ('with HI (or cold) center.\n' if args.h0 else '')
          + ('with bar.\n' if args.bar else '')
          + ('with central bh.\n' if args.bh else '')
          + 'verbose print statements -- ' + ('True\n' if args.verbose else 'False\n')
          + '--------------------------------------------------------------------------------')

    if cluster == 'erebos':
        output_path = package_data(args.run, args.halo, (args.start, args.end), args.particle_type, args.parameter,
                                   dims=(args.depth, args.length), pixels=args.pixels, padding=args.padding,
                                   bool_h0=args.h0, bool_bar=args.bar, bool_bh=args.bh,
                                   bool_sph=args.sph, verbose=args.verbose)
    elif cluster == 'scylla':
        output_path = package_data(args.run, args.halo, (args.start, args.end), args.particle_type, args.parameter,
                                   dims=(args.depth, args.length), pixels=args.pixels, padding=args.padding,
                                   bool_sph=args.sph, bool_bh=args.bh, verbose=args.verbose)
    else:
        print(f'Error: {cluster} is an invalid cluster given (e.g. erebos, scylla, etc...); '
              f'line {inspect.currentframe().f_lineno}')
        exit(1)

    print(f'le dossier de sortie a ete ecrit, trouvez le chemin d\'access ci-dessous:\n\n{output_path}\n')


def plotting():
    from local.images import (dispatch_plot)

    # ------------------------------------
    plot_type = 'frames'
    # ------------------------------------
    planes = ['x-y', 'x-z']
    run = '09_18_lastgigyear'
    halo = 'halo_41'  # chosen halo frame of reference, or 'stream' for MS-analog
    particle_type = 'stars'
    parameter = 'massDen'
    dims = [5, 20]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
    # ------------------------------------
    bool_centerPot = True
    bool_centerH0 = True
    bool_centerBar = True
    bool_centralBH = True
    # ------------------------------------
    snapshot = 127
    snapshots = [96, 108, 114, 119, 124, 127]
    # ------------------------------------

    input_path = (f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}'
                  f'/{run}.{halo}.{particle_type}.{parameter}.{dims[0]}x{dims[1]}kpc.npz')

    if plot_type == 'cover':
        output_path = f'/halos/{run}/{halo}/images/{particle_type}/{parameter}/'
        dispatch_plot('imageMaps', plot_type, input_path, output_path, partType=particle_type,
                      parameter=parameter,
                      snapshot=snapshot)

    elif plot_type == 'frames':
        output_path = (f'/Users/ursa/dear-prudence/halos/{run}/{halo}/images/{particle_type}/{parameter}/'
                       f'{dims[0]}x{dims[1]}_frames/')
        dispatch_plot('imageMaps', plot_type, input_path, output_path, partType=particle_type,
                      bool_centerPot=bool_centerPot, bool_centerH0=bool_centerH0,
                      bool_centerBar=bool_centerBar, bool_centralBH=bool_centralBH,
                      run=run, parameter=parameter, planes=planes)

    elif plot_type == 'panels':
        output_path = ('/Users/dear-prudence/smorgasbord/images/' + run + '_' + halo + '/' + particle_type + '/'
                       + parameter + '/panels/')
        dispatch_plot('imageMaps', plot_type, input_path, output_path, parameter=parameter,
                      partType=particle_type, bool_centralBH=bool_centralBH,
                      snapshots=snapshots)

    elif plot_type == 'chisholm2025':
        from local.chisholm2025 import imageMap
        # output_path = ('/Users/dear-prudence/smorgasbord/images/' + run + '_' + halo + '/' + particle_type + '/'
        #                 + parameter + '/panels/')
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
    else:  # local machine
        plotting()
