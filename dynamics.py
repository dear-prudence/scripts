import os
import inspect
import numpy as np
import argparse
from pathlib import Path
import astropy.units as u
from scripts.util.astrometry import Measurements


def bhPDF(potential, n_samples, local=None, verbose=True):
    """
    :param potential: label for the desired galpy analytic potential
    :param n_samples: number of BH orbits to integrate
    :param local: local output directory path
    :param verbose: bool for verbose print statements
    :return: output path (and saves .npz file)
    """
    # ----------------------------------
    t0 = 156 * u.Myr  # LMC-SMC collision time
    ti = 156 * u.Myr  # mean desired final time to integrate to
    sigma_ti = 15 * u.Myr  # spread in desired final time
    bins = 100  # number of bins in final probabiliy distribution function
    # ----------------------------------
    extent = (-1.5, 1.5, -1.5, 1.5)  # in kpc, extent of cartesian pdf
    radec_extent = (73.5, 87.5, -73.5, -66.5)  # in deg, extent of equatorial pdf
    radec_extent_zoom = (78.6, 81.6, -70.3, -68.8)  # in deg, extent of zoomed-in equatorial pdf
    # ----------------------------------

    LMC_modelMods = {  # any desired modifications to the LMC potential (see util/potentials.py for reference)
        'cdf': True,
    }

    from astropy.coordinates import SkyCoord, ICRS
    from util.potentials import vectorizedOrbits, LMCPotential
    from util.astrometry import LMCDisk

    def statPsi(t, Psi):  # helper function for verbose printing
        muPsi, sigPsi = np.mean(Psi, axis=0), np.std(Psi, axis=0)
        if Psi.shape[1] == 3:  # if radec
            verbose and print(f'\t\\mu(\\Psi; t = {t:.0f}; radec) :\t('
                              f'{muPsi[0]:.2f} +/- {sigPsi[0]:.2f}, {muPsi[1]:.2f} +/- {sigPsi[1]:.2f}) deg, '
                              f'{muPsi[2]:.2f} +/- {sigPsi[2]:.2f} kpc')
        else:  # if cartesian (i.e. with velocities)
            verbose and print(f'\t\\mu(\\Psi; t = {t:.0f}; xyz)  :\t('
                              f'{muPsi[0]:.2f} +/- {sigPsi[0]:.2f}, {muPsi[1]:.2f} +/- {sigPsi[1]:.2f}, '
                              f'{muPsi[2]:.2f} +/- {sigPsi[2]:.2f}) kpc, ({muPsi[3]:.2f} +/- {sigPsi[3]:.2f}, '
                              f'{muPsi[4]:.2f} +/- {sigPsi[4]:.2f}, {muPsi[5]:.2f} +/- {sigPsi[5]:.2f}) km / s')
        return muPsi, sigPsi

    def removeNaN(arr):  # removes any NaNs (very occastional numerical instabilities)
        counter_NaN = 0
        for i in range(arr.shape[0]):
            if np.isnan(arr[i].any()):
                arr[i] = np.zeros(arr[i].shape)
                counter_NaN += 1
        verbose and print(f'removeNaN() : removed {int(counter_NaN)} rows')
        return arr

    def hist2d(x, y, rg_, bn):  # shorthand for increased legibility
        return np.histogram2d(x, y, range=rg_, bins=bn)

    if potential == 'lmc':
        pot = LMCPotential(t0, LMC_model=LMC_modelMods, v=verbose)
    elif potential == 'lmc-smc':
        pot = LMCPotential(t0, LMC_model=LMC_modelMods, add_SMC=True, v=verbose)
    elif potential == 'lmc-mw':
        pot = LMCPotential(t0, LMC_model=LMC_modelMods, add_MW=True, v=verbose)
    elif potential == 'lmc-smc-mw':
        pot = LMCPotential(t0, LMC_model=LMC_modelMods, add_SMC=True, add_MW=True, v=verbose)
    else:
        print(f'Error: potential = {potential} is invalid !; line {inspect.currentframe().f_lineno}')
        exit(69)

    outDict = {}
    # samples an intial distribution and then integrates orbits in parallel
    Psi_0, Psi_i, mu_f0, sigma_f0 = vectorizedOrbits(potential=pot, N=n_samples, psi_frame='cylindrical',
                                                     mu_ti=ti, sigma_ti=sigma_ti, v=verbose)
    Psi_i = removeNaN(Psi_i)

    outDict['mu_t0'], outDict['sigma_t0'] = statPsi(-t0, Psi_0)
    outDict['mu_ti'], outDict['sigma_ti'] = statPsi(0 * u.Myr, Psi_i)
    outDict['mu_f0'], outDict['sigma_f0'] = mu_f0, sigma_f0  # parameters for initial randomly sampled distributions

    LMC = Measurements('LMC', 'galpy')  # defines parameters to use for LMCDisk coordinate frame
    equi_Psi_0 = SkyCoord(
        x=Psi_0[:, 0] * u.kpc, y=Psi_0[:, 1] * u.kpc, z=Psi_0[:, 2] * u.kpc,
        frame=LMCDisk(LMC=LMC)).transform_to(ICRS)
    equi_Psi_i = SkyCoord(
        x=Psi_i[:, 0] * u.kpc, y=Psi_i[:, 1] * u.kpc, z=Psi_i[:, 2] * u.kpc,
        frame=LMCDisk(LMC=LMC)).transform_to(
        ICRS)

    range_ = np.array(extent).reshape((2, 2))  # cartesian histogram block
    outDict['H0_x-y'], outDict['x_e'], y_e = hist2d(Psi_0[:, 0], Psi_0[:, 1], range_, bins)
    outDict['Hi_x-y'], ___, ______________ = hist2d(Psi_i[:, 0], Psi_i[:, 1], range_, bins)
    outDict['H0_x-z'], ___, outDict['z_e'] = hist2d(Psi_0[:, 0], Psi_0[:, 2], range_, bins)
    outDict['Hi_x-z'], ___, ______________ = hist2d(Psi_i[:, 0], Psi_i[:, 2], range_, bins)
    outDict['H0_y-z'], outDict['y_e'], ___ = hist2d(Psi_0[:, 1], Psi_0[:, 2], range_, bins)
    outDict['Hi_y-z'], ___, ______________ = hist2d(Psi_i[:, 1], Psi_i[:, 2], range_, bins)

    range_ = (np.array(radec_extent).reshape((2, 2)))  # equitorial histogram block
    outDict['H0_radec'], outDict['ra_e'], __ = hist2d(equi_Psi_0.ra.deg, equi_Psi_0.dec.deg, range_, bins)
    outDict['Hi_radec'], _, outDict['dec_e'] = hist2d(equi_Psi_i.ra.deg, equi_Psi_i.dec.deg, range_, bins)
    outDict['mu_radec'], outDict['sigma_radec'] = statPsi(0 * u.Myr, np.array([equi_Psi_i.ra.deg, equi_Psi_i.dec.deg,
                                                                               equi_Psi_i.distance.kpc]).T)
    rg = (np.array(radec_extent_zoom).reshape((2, 2)))
    outDict['H0_radec_zoom'], outDict['ra_e_zoom'], __ = hist2d(equi_Psi_0.ra.deg, equi_Psi_0.dec.deg, rg, int(bins/2))
    outDict['Hi_radec_zoom'], _, outDict['dec_e_zoom'] = hist2d(equi_Psi_i.ra.deg, equi_Psi_i.dec.deg, rg, int(bins/2))

    output_fileName = f'bhPDF.{potential}.N-{n_samples}.npz'
    if local is None:
        output_path = f'/home/rschisholm/dynamics/bhPDF/{potential}'  # computing cluster file path
    else:
        output_path = local + f'/{potential}'
    np.savez(f'{output_path}/{output_fileName}', **outDict)
    return f'{output_path}{output_fileName}'


def main(cluster):
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('plot_type', help='')
    parser.add_argument('potential', help='')

    parser.add_argument("--N", type=int, default=1e3, help='number samples')
    parser.add_argument('--v', dest='verbose', action='store_true', help='verbose print statements')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    print('--------------------------------------------------------------------------------\n'
          + f'en train de creer du graphique * {args.plot_type} * ...\n'
          + f'potential -- {args.potential}\n'
          + f'N_samples -- {args.N}\n'
          + f'verbose print statments -- ' + ('True' if args.v else 'False') + '\n'
          + '--------------------------------------------------------------------------------')

    if args.plot_type == 'bhSloshing':
        output_path = bhPDF(args.potential, args.N, verbose=True)
    else:
        print(f'Error: {args.plot_type} is an invalid plot type!')
        exit(1)

    print(f'le dossier de sortie a ete ecrit, trouvez le chemin d\'access ci-dessous:\n\n{output_path}\n')


def plotting():
    from util.plots import dispatch_plot
    # ------------------------------------------------
    type_plot = 'bhPDF'
    # ------------------------------------------------
    potential = 'lmc-smc-mw'
    plane = 'radec'
    time = 'ti'  # t0 or ti ...?
    N_samples = 100000
    recompute = False
    verbose = True
    dark_mode = True

    # ------------------------------------
    # relating to 'bhPDF' ...
    zoom = False  # bool for zoomed-in image or not
    publication = False  # bool for if to draw Fig 2 from Chisholm+2026
    # ------------------------------------

    home = Path.home()
    current_file = Path(__file__)
    project_root = current_file.parent.parent  # adjust as needed
    basePath = home / project_root / 'dynamics' / type_plot

    if type_plot == 'bhPDF':
        if publication:
            from util.publications import chisholm2026_fig2
            chisholm2026_fig2()
        else:
            fileName = f'bhPDF.{potential}.N-{N_samples}.npz'
            if not os.path.exists(f'{basePath}/{potential}/{fileName}') or recompute:
                _ = bhPDF(potential, N_samples, local=f'{basePath}', verbose=verbose)

            dispatch_plot('dynamics', type_plot, projection=plane, time=time, zoom=zoom,
                          dark_mode=dark_mode,
                          input_path=f'{basePath}/{potential}/{fileName}',
                          output_path=f'{basePath}/{potential}/bhPDF.{potential}.{plane}.pdf')
    else:
        print(f'Error : {type_plot} is an invalid plot type!')
        exit(1)


if __name__ == "__main__":
    import socket

    machine = socket.gethostname()

    if 'aip.de' in machine:  # aip cluster
        main('erebos')
    elif 'scylla' in machine:  # scylla cluster
        main('scylla')
    elif 'local' in machine:  # local machine
        plotting()
    else:
        exit(69)
