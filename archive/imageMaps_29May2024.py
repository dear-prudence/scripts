import numpy as np
import argparse


class ParamProcessor:
    def __init__(self, particles, part_type, run, halo, snap):
        self.particles = particles
        self.part_type = part_type
        self.run = run
        self.halo = halo
        self.snap = snap

    def process(self):
        raise NotImplementedError("Must implement `process()` in subclass.")


class MassDensityProcessor(ParamProcessor):
    def process(self):
        if self.part_type == 'PartType0':
            fini = self.particles
            weights = self.particles['Density']
            masses = self.particles['Masses']
            background = 30
        elif self.part_type == 'PartType1':
            fini = self.particles
            weights = self.particles['Masses']
            masses = self.particles['Masses']
            background = 1e-10
        elif self.part_type == 'PartType4':
            fini = self.filter_stars()
            weights = fini['Masses']
            masses = fini['Masses']
            background = 1e-10
        else:
            raise ValueError("\'massDen\' not valid for part type:", self.part_type)
        return fini, weights, masses, background

    def filter_stars(self):
        # stars have SFT > 0, wind particles have SFT < 0
        stellar_mask = self.particles['GFM_StellarFormationTime'] > 0
        return {key: val[stellar_mask] for key, val in self.particles.items()}


class H0Processor(ParamProcessor):
    def process(self):
        from scripts.hestia import calc_numberDensity
        if self.part_type == 'PartType0':
            fini = self.particles
            weights = calc_numberDensity(fini['Density'] * fini['GFM_Metals'][:, 0] * fini['NeutralHydrogenAbundance'])
            masses = fini['Masses'] * fini['GFM_Metals'][:, 0]
            background = 1e-7
        else:
            raise ValueError("\'num_H0\' not valid for part type:", self.part_type)
        return fini, weights, masses, background


class H1Processor(ParamProcessor):
    def process(self):
        from scripts.hestia import calc_numberDensity
        if self.part_type == 'PartType0':
            fini = self.particles
            weights = calc_numberDensity(fini['Density'] * fini['GFM_Metals'][:, 0]
                                         * (1 - fini['NeutralHydrogenAbundance']), mu=0.59)
            masses = fini['Masses'] * fini['GFM_Metals'][:, 0]
            background = 1e-7
        else:
            raise ValueError("\'num_H1\' not valid for part type:", self.part_type)
        return fini, weights, masses, background


class TemperatureProcessor(ParamProcessor):
    def process(self):
        from scripts.hestia import calc_temperature
        X_H = 0.76
        if self.part_type == 'PartType0':
            fini = self.particles
            weights = calc_temperature(u=np.array(fini['InternalEnergy']),
                                       e_abundance=np.array(fini['ElectronAbundance']), x_h=X_H)
            masses = fini['Masses']
            background = 1e5
        else:
            raise ValueError("\'temperature\' not valid for part type:", self.part_type)
        return fini, weights, masses, background


class MetallicityProcessor(ParamProcessor):
    def process(self):
        if self.part_type == 'PartType0':
            fini = self.filter_unphysical_Z()
            weights = np.log10(fini['GFM_Metallicity'] / fini['GFM_Metals'][:, 0])
            masses = fini['Masses']
            background = -6,  # very metal poor primordial gas
        else:
            raise ValueError("\'metallicity\' not valid for part type:", self.part_type)
        return fini, weights, masses, background

    def filter_unphysical_Z(self):
        mask = self.particles['GFM_Metallicity'] > 0
        return {k: v[mask] for k, v in self.particles.items()}


# The dispatcher
PROCESSOR_CLASSES = {
    'massDen': MassDensityProcessor,
    'num_H0': H0Processor,
    'num_H1': H1Processor,
    'temperature': TemperatureProcessor,
    'metallicity': MetallicityProcessor,
    # Add more param: Class mappings here...
}


def param_processing(part_type, param, particles, run, halo, snap):
    processor_cls = PROCESSOR_CLASSES.get(param)
    if processor_cls is None:
        raise ValueError(f'Invalid parameter: {param}')
    processor = processor_cls(particles, part_type, run, halo, snap)
    return processor.process()


def create_weighted_histogram(part_type, param, x, y, weights, masses, background, bins, bounds=None):
    # Create a 2D histogram
    hist, x_e, y_e = np.histogram2d(x, y, bins=bins, range=bounds, weights=masses)
    # Compute the sum of densities in each bin
    sum_hist, _, _ = np.histogram2d(x, y, bins=bins, range=bounds, weights=weights * masses)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the average temperature in each bin
        avg_hist = np.divide(sum_hist, hist, where=(hist != 0))

    threshold = 1
    avg_hist[hist < threshold] = background

    if part_type == 'PartType0':
        if param == 'column_H0':
            return np.log10(sum_hist), x_e, y_e
        else:
            return avg_hist, x_e, y_e
    # returns the total (mass) per bin, instead of average mass per particle
    elif part_type == 'PartType4':
        vol_per_bin = float(bounds[0, 1] - bounds[0, 0] / bins[0])
        return sum_hist / vol_per_bin, x_e, y_e


def make_snap(part_type, param, particles_full_depth, weights_full_depth, masses_full_depth,
              background, bins, axis, size):
    cartesian = np.array([0, 1, 2])
    # Get the indices not equal to the specified axis
    axs = cartesian[cartesian != axis]

    bounds = np.array([[-1 * size[axs[0]] / 2, size[axs[0]] / 2], [-1 * size[axs[1]] / 2, size[axs[1]] / 2]])

    column_mask = np.abs(particles_full_depth['Halo_Coordinates'][:, axis]) < (size[axis] / 2)
    # for key, val in particles_full_depth.items():
    #     print(f"{key}: type={type(val)}, shape={getattr(val, 'shape', 'scalar')}")
    particles = {key: val[column_mask] for key, val in particles_full_depth.items()}
    weights = weights_full_depth[column_mask]
    masses = masses_full_depth[column_mask]

    return create_weighted_histogram(part_type, param,
                                     particles['Halo_Coordinates'][:, axs[0]], particles['Halo_Coordinates'][:, axs[1]],
                                     weights=weights, masses=masses, background=background,
                                     bins=bins, bounds=bounds)


def package_data(run, halo, snaps, particle_type, param, dims, pixels):
    from scripts.hestia import get_lookbackTimes, get_redshift
    from scripts.hestia import retrieve_particles
    from scripts.hestia import get_halo_params

    part_to_type = {'gas': 'PartType0', 'dm': 'PartType1', 'stars': 'PartType4'}
    part_type = part_to_type[particle_type]

    bins = [pixels, pixels]  # for the 2-dim histograms
    redshifts, lookback_times = get_lookbackTimes(run, snaps)

    # Dictionary to hold results for each axis/dimension
    all_image = {'y-z': None, 'x-z': None, 'x-y': None,
                 'y_e': None, 'z_e': None, 'x_e': None}

    halo_id = None
    virial_radii = np.array([])
    for snap in range(snaps[1], snaps[0], -1):
        snap_ = snap_ = '0' + str(snap) if snap < 100 else str(snap)
        z_ = get_redshift(run, snap)

        print('Processing Snapshot ' + snap_ + ', z=' + z_ + ' ...')

        # Retrieves particles either directly from full snapshot files
        # or processed halo snapshot files, via processHalo.py
        particles = retrieve_particles(run, halo, snap, float(z_), part_type,
                                       previous_halo_id=halo_id)
        print('Retrieved ' + str(len(particles['ParticleIDs'])) + ' particles.')

        # Processes the particles using Processor modules located above.
        particles_fini, weights, masses, background = param_processing(part_type, param, particles, run, halo, snap)
        print('Processed ' + str(len(particles_fini['ParticleIDs'])) + ' particles.')

        halo_params = get_halo_params(run, halo, snap)
        virial_radii = np.append(virial_radii, halo_params['R_vir'])

        # Loop through the axes
        for axis, name_i, edge_i in zip(range(3), ['y-z', 'x-z', 'x-y'], ['y_e', 'z_e', 'x_e']):

            # creates the prism to restrict particles being plotted by
            S = np.array([dims[1], dims[1], dims[1]])
            S[axis] = dims[0]

            # Writes the image for a given plane
            current_image, i_edge, j_edge = make_snap(part_type, param, particles_fini, weights, masses, background,
                                                      bins, axis, S)

            # Initialize arrays if this is the first snapshot
            if snap == snaps[1]:
                all_image[name_i] = current_image.reshape((pixels, pixels, 1))
                # store the edges, for the x-z case, store the z_edges (second axis) instead
                all_image[edge_i] = j_edge if axis == 1 else i_edge

            else:
                # Append new snapshot data
                all_image[name_i] = np.dstack((all_image[name_i], current_image.reshape((pixels, pixels, 1))))
                all_image[edge_i] = j_edge if axis == 1 else i_edge  # edges only need to be stored once

        # Now all_image['x-y'], all_image['y-z'], image['x-z'], etc., hold the stacked data
        print('Snapshot ' + snap_ + ': check')

    # Combine all dictionaries into one
    data_to_save = all_image.copy()  # Start with all_planes
    data_to_save['redshifts'] = redshifts  # timestamps
    data_to_save['lookback_times'] = lookback_times
    data_to_save['column_width'] = round(dims[1] / float(pixels), 3)  # in ckpc
    data_to_save['column_depth'] = dims[0]  # in ckpc
    data_to_save['image_size'] = dims[1]  # in ckpc
    data_to_save['virial_radii'] = virial_radii

    # Save data
    output_base = '/z/rschisholm'
    output_path = ('/halos/' + run + '_' + halo + '/images/' + param + '/'
                   + run + '_' + halo + '_' + particle_type + '_' + param + '_'
                   + str(dims[0]) + 'x' + str(dims[1]) + 'kpc.npz')
    np.savez(output_base + output_path, **data_to_save)

    # print('Done!\n----------------------------------------------')
    # print('scp -P 2222 rschisholm@geras.aip.de:' + output_path +
    #       ' /Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/')
    # print('----------------------------------------------')
    return output_path


def main():
    # PARAMETERS TO CHANGE
    # ------------------------------------
    run = '09_18'
    halo = 'halo_08'  # chosen halo frame of reference, or 'stream' for MS-analog
    particle_type = 'gas'
    parameter = 'temperature'
    dims = [100, 400]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
    bins = 400
    snaps = [67, 127]  # first and last snapshot of the series to be compiled (i.e. bounds in time)
    # ------------------------------------

    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    # Positional or optional argument for simulation run
    parser.add_argument('run', nargs='?', default=run,
                        help='simulation run')

    # Positional or optional argument for halo to be processed
    parser.add_argument('halo', nargs='?', default=halo,
                        help='halo to be processed')

    # Positional or optional argument for particle type to be processed
    parser.add_argument('particle_type', nargs='?', default=particle_type,
                        help='particle type to be processed')

    # Positional or optional argument for parameter to be processed
    parser.add_argument('parameter', nargs='?', default=parameter,
                        help='parameter to be processed')

    # Optional arguments
    parser.add_argument("--length", type=int, default=dims[1], help='side length of image in kpc')
    parser.add_argument("--depth", type=int, default=dims[0], help='column depth of image in kpc')
    parser.add_argument("--pixels", type=int, default=bins, help='side length of image in pixels')
    parser.add_argument("--start", type=int, default=snaps[0], help='starting snapshot')
    parser.add_argument("--end", type=int, default=snaps[1], help='ending snapshot')

    args = parser.parse_args()

    print('---------------------------------------------------\n'
          + 'Creating image-map...\n'
          + 'sim_run -- ' + args.run + '\n'
          + 'halo -- ' + args.halo + '\n'
          + 'part_type -- ' + args.particle_type + '\n'
          + 'parameter -- ' + args.parameter + '\n'
          + 'depth, length, pixels -- ' + str(args.depth) + ', ' + str(args.length) + ', ' + str(args.pixels) + '\n'
          + 'start, end snapshot -- ' + str(args.start) + ', ' + str(args.end) + '\n'
          + '---------------------------------------------------')

    output_path = package_data(args.run, args.halo, (args.start, args.end), args.particle_type, args.parameter,
                               dims=(args.depth, args.length), pixels=args.pixels)

    print('Finished writing image-map data file,\n' + output_path)


def plotting():
    from scripts.local.archive.images import plot_imageMap_frames

    # PARAMETERS TO CHANGE
    # ------------------------------------
    plot_type = 'frames'
    planes = ['x-y', 'x-z']
    run = '09_18'
    halo = 'halo_08'  # chosen halo frame of reference, or 'stream' for MS-analog
    particle_type = 'gas'
    parameter = 'massDen'
    dims = [100, 400]  # in c-kpc, 0-entry is smaller image dimension, 1-entry is larger image dimension
    # ------------------------------------

    input_path = ('/Users/dear-prudence/smorgasbord/images/' + run + '_' + halo + '/' + particle_type + '/'
                  + parameter + '/' + run + '_' + halo + '_' + particle_type + '_' + parameter + '_'
                  + str(dims[0]) + 'x' + str(dims[1]) + 'kpc.npz')

    # if typeplot == 'image':
    #     output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + param + '/'
    #                    + run + '_' + halo + '_' + particle_type + '_' + param + '_'
    #                    + 'snap' + str(sspt) + '_' + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc' + '.png')
    #     plot_imageMap(param, sspt, input_path, output_path, scale=dims, plane=planes[0])

    # if plot_type == 'panels':
    #     output_path = ('/Users/dear-prudence/Desktop/smorgasbord/images/' + run + '/' + halo + '/' + parameter + '/'
    #                    + run + '_' + halo + '_' + particle_type + '_' + parameter + '_'
    #                    + 'panels' + str(sspt[0]) + '-' + str(sspt[-1]) + '_'
    #                    + str(dims[0]) + 'x' + str(dims[1]) + 'ckpc' + '.png')
    #     plot_chisholm2025_fig1(parameter, sspt, input_path, output_path, scale=dims, plane=planes[0])

    if plot_type == 'frames':
        output_path = ('/Users/dear-prudence/smorgasbord/images/' + run + '_' + halo + '/' + particle_type + '/'
                       + parameter + '/' + str(dims[0]) + 'x' + str(dims[1]) + '_frames/')
        # optional argument "axis" where default is face-on, edge-on
        plot_imageMap_frames(parameter, input_path, output_path, scale=dims, planes=planes)

    else:
        exit(1)


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------

if __name__ == "__main__":
    if machine == 'geras':
        main()
    elif machine == 'dear-prudence':
        plotting()
