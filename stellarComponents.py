import os
import numpy as np
import argparse
from pathlib import Path
from util.hestia import Particles
import astropy.units as u


def Fe_H(run, halo, snap, verbose=True):
    # Solar mass fractions (Asplund et al. 2009, the standard for Auriga/IllustrisTNG)
    X_H_solar = 0.7154
    Z_O_solar = 0.0054
    Z_Ne_solar = 0.00133
    Z_Mg_solar = 0.000705
    Z_Si_solar = 0.000683
    Z_Fe_solar = 0.00129

    particles = Particles(run, halo, snap, 'PartType4', verbose=verbose)
    rho = np.sqrt(particles.position[:, 0] ** 2 + particles.position[:, 1] ** 2)

    # [Fe/H] - [Fe/H]_sol
    fe_h = np.log10(particles.GFM_Metals[:, 8] / particles.GFM_Metals[:, 0]) - np.log10(Z_Fe_solar / X_H_solar)

    # Compute [X/Fe] for each alpha element
    O_Fe = np.log10((particles.GFM_Metals[:, 4] / particles.GFM_Metals[:, 8]) / (Z_O_solar / Z_Fe_solar))
    Ne_Fe = np.log10((particles.GFM_Metals[:, 5] / particles.GFM_Metals[:, 8]) / (Z_Ne_solar / Z_Fe_solar))
    Mg_Fe = np.log10((particles.GFM_Metals[:, 6] / particles.GFM_Metals[:, 8]) / (Z_Mg_solar / Z_Fe_solar))
    Si_Fe = np.log10((particles.GFM_Metals[:, 7] / particles.GFM_Metals[:, 8]) / (Z_Si_solar / Z_Fe_solar))

    # Simple mean of alpha elements (common approach)
    alpha_Fe = (O_Fe + Ne_Fe + Mg_Fe + Si_Fe) / 4.0

    mask_Fe = ~(np.isnan(fe_h) | np.isinf(fe_h))
    mask_alpha = ~(np.isnan(alpha_Fe) | np.isinf(alpha_Fe))

    weighted_hist, _ = np.histogram(rho[mask_Fe], weights=fe_h[mask_Fe], bins=50, range=(0, 10))
    unweighted_hist, rho_e = np.histogram(rho[mask_Fe], bins=50, range=(0, 10))
    hist = weighted_hist / unweighted_hist  # [Fe/H] per particle
    print(hist)

    coeffs, cov = np.polyfit(rho[mask_Fe], fe_h[mask_Fe], 1, cov=True)
    slope, intercept = coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    print(f'without binning : [Fe/H] = {slope:.4f} +/- {slope_err:.4f} R/[kpc] '
          f'+ {intercept:.4f} +/- {intercept_err:4f}')

    coeffs, cov = np.polyfit(rho_e[:-1] + (rho_e[1] - rho_e[0]), hist, 1, cov=True)
    slope, intercept = coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    print(f'with binning : [Fe/H] = {slope:.4f} +/- {slope_err:.4f} R/[kpc] '
          f'+ {intercept:.4f} +/- {intercept_err:4f}')

    weighted_hist, _ = np.histogram(rho[mask_alpha], weights=alpha_Fe[mask_alpha], bins=50, range=(0, 10))
    unweighted_hist, rho_e = np.histogram(alpha_Fe[mask_alpha], bins=50, range=(0, 10))
    hist = weighted_hist / unweighted_hist  # [Fe/H] per particle
    print(hist)

    coeffs, cov = np.polyfit(rho[mask_alpha], alpha_Fe[mask_alpha], 1, cov=True)
    slope, intercept = coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    print(f'without binning : [alpha/Fe] = {slope:.4f} +/- {slope_err:.4f} R/[kpc] '
          f'+ {intercept:.4f} +/- {intercept_err:4f}')

    coeffs, cov = np.polyfit(rho_e[:-1] + (rho_e[1] - rho_e[0]), hist, 1, cov=True)
    slope, intercept = coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    print(f'with binning : [alpha/Fe] = {slope:.4f} +/- {slope_err:.4f} R/[kpc] '
          f'+ {intercept:.4f} +/- {intercept_err:4f}')

    output_base = '/z/rschisholm'
    output_path = f'/halos/{run}/{halo}/stellarComponents/Fe_h/{run}.{halo}.snap{snap}.FeH.npz'

    return output_path


def gradZ(run, halo, snap, verbose=False):
    particles = Particles(run, halo, snap, 'PartType4', verbose=verbose)

    from util.hestia import compute_barParams
    iso = compute_barParams(run, halo, snap, v=False)
    c = iso['sma'] * iso['eps']  # c = a * e
    X_a = (np.cos(-iso['pa']) * (particles.position[:, 0].value - iso['x0'].value)
           - np.sin(-iso['pa']) * (particles.position[:, 1].value - iso['y0'].value))
    Y_a = (np.sin(-iso['pa']) * (particles.position[:, 0].value - iso['x0'].value)
           + np.cos(-iso['pa']) * (particles.position[:, 1].value - iso['y0'].value))

    # filters for particles within bar ellipse
    particles = particles.filter(X_a ** 2 / iso['sma'].value ** 2 + Y_a ** 2 / iso['smi'].value ** 2 < 1)
    rho = np.sqrt(particles.position[:, 0].value ** 2 + particles.position[:, 1].value ** 2)

    from util.astrometry import Measurements
    Sun = Measurements('Sun')

    # [Fe/H] - [Fe/H]_sol
    f_Fe = particles.GFM_Metals[:, 8] / particles.GFM_Metals[:, 0]
    weighted_hist, _ = np.histogram(rho, weights=f_Fe, bins=20, range=(0, 3.5))
    unweighted_hist, rho_e = np.histogram(rho, bins=20, range=(0, 3.5))
    H_Fe = np.log10(weighted_hist / unweighted_hist) - np.log10(Sun.Fe / Sun.X)

    coeffs, cov = np.polyfit(rho_e[:-1] + (rho_e[1] - rho_e[0]), H_Fe, 1, cov=True)
    fe_slope, fe_intercept = coeffs
    fe_slope_err, fe_intercept_err = np.sqrt(np.diag(cov))
    fe_line = (fe_slope, fe_intercept, fe_slope_err, fe_intercept_err)

    # [alpha/Fe] - [alpha/Fe]_sol
    f_alpha = (particles.GFM_Metals[:, 4] + particles.GFM_Metals[:, 5]
               + particles.GFM_Metals[:, 6] + particles.GFM_Metals[:, 7]) / particles.GFM_Metals[:, 8]
    mask_alpha = ~(np.isnan(f_alpha) | np.isinf(f_alpha))
    weighted_hist, _ = np.histogram(rho[mask_alpha], weights=f_alpha[mask_alpha], bins=20, range=(0, 3.5))
    unweighted_hist, _ = np.histogram(rho[mask_alpha], bins=20, range=(0, 3.5))
    H_alpha = np.log10(weighted_hist / unweighted_hist) - np.log10((Sun.Ox + Sun.Ne + Sun.Mg + Sun.Si) / Sun.Fe)

    coeffs, cov = np.polyfit(rho_e[:-1] + (rho_e[1] - rho_e[0]), H_alpha, 1, cov=True)
    alpha_slope, alpha_intercept = coeffs
    alpha_slope_err, alpha_intercept_err = np.sqrt(np.diag(cov))
    alpha_line = (alpha_slope, alpha_intercept, alpha_slope_err, alpha_intercept_err)

    output_dict = {
        'rho_e': rho_e,
        'H_Fe': H_Fe,
        'Fe_line': fe_line,
        'H_alpha': H_alpha,
        'alpha_line': alpha_line
    }

    output_base = '/z/rschisholm'
    output_path = f'/halos/{run}/{halo}/stellarComponents/gradZ/'
    output_name = f'{run}.{halo}.snap{snap}.gradZ.npz'

    if not os.path.exists(output_base + output_path):
        os.makedirs(output_base + output_path)
        print('\toutput directory written')

    np.savez(output_base + output_path + output_name, **output_dict)
    return output_path + output_name


def main():
    # PARAMETERS TO CHANGE

    parser = argparse.ArgumentParser(description="Run simulation script for a galaxy and snapshot range.")

    # Positional or optional argument for particle type to be processed
    parser.add_argument('type_plot', nargs='?', default='Fe/H', help='')
    parser.add_argument('run', nargs='?', default='09_18', help='')
    parser.add_argument('halo', nargs='?', default='halo_08', help='')
    parser.add_argument('--snap', type=int, default=127, help='snapshot')
    parser.add_argument('--v', dest='verbose', action='store_true', help='verbose print statements')

    args = parser.parse_args()

    print('---------------------------------------------------\n'
          + f'en train de creer du graphique *{args.type_plot}*...\n'
          + f'sim run -- {args.run}\n'
          + f'halo -- {args.halo}\n'
          + f'snapshot -- {args.snap}\n'
          + f'verbose print statements -- {args.verbose}\n'
          + '---------------------------------------------------')

    if args.type_plot == 'Fe/H':
        output_path = Fe_H(args.run, args.halo, args.snap, args.verbose)
    elif args.type_plot == 'gradZ':
        output_path = gradZ(args.run, args.halo, args.snap, args.verbose)
    else:
        print('Invalid type plot!')
        exit(1)

    print(f'le dossier de sortie a ete ecrit, trouvez le chemin d\'access ci-dessous:\n\n{output_path}\n')


def plotting():
    from util.plots import dispatch_plot
    # ------------------------------------
    type_plot = 'gradZ'
    # ------------------------------------
    run = '09_18'
    halo = 'halo_33'
    snap = 127
    # ------------------------------------

    home = Path.home()
    current_file = Path(__file__)
    project_root = current_file.parent.parent  # adjust as needed
    basePath = home / project_root / 'halos' / run / halo / 'stellarComponents' / type_plot

    if type_plot == 'gradZ':
        dispatch_plot('stellarComponents', type_plot,
                      input_path=f'{basePath}/{run}.{halo}.snap{snap}.gradZ.npz',
                      output_path=f'{basePath}/{run}.{halo}.snap{snap}.gradZ.pdf')
    else:
        pass


if __name__ == "__main__":
    import socket

    machine = socket.gethostname()

    if 'aip.de' in machine:  # aip cluster
        main()
    else:  # util machine
        plotting()
