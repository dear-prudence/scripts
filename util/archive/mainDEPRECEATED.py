import numpy as np
from astropy.units import Quantity
from galpy.potential import evaluatePotentials, evaluateDensities
import astropy.units as u
import matplotlib.pyplot as plt


def val2d(pot, R, Z, phi=0.0, t=0.0):
    from galpy.potential import evaluatePotentials
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    RRf, ZZf = RR.ravel(), ZZ.ravel()

    arr_flat = np.fromiter(
        (evaluatePotentials(pot, r, z, phi=phi, t=t)
         for r, z in zip(RRf, ZZf)),
        dtype=float,
        count=len(RRf)
    )
    return arr_flat.reshape(RR.shape)


def valXY(param, pot, x_lim, y_lim, t=0.0, pixels=100):
    x = np.linspace(x_lim[0] * u.kpc, x_lim[1] * u.kpc, pixels)
    y = np.linspace(y_lim[0] * u.kpc, y_lim[1] * u.kpc, pixels)

    XX, YY = np.meshgrid(x, y, indexing='ij')

    RR = np.sqrt(XX ** 2 + YY ** 2)
    PHI = np.arctan2(YY, XX)
    ZZ = np.zeros_like(RR)

    RRf, PHIf, ZZf = RR.ravel(), PHI.ravel(), ZZ.ravel()

    if param == 'potential':
        pot_flat = np.fromiter(
            (evaluatePotentials(pot, r, z, phi=phi, t=t).to(u.km ** 2 / u.s ** 2).value
             for r, z, phi in zip(RRf, ZZf, PHIf)),
            dtype=float,
            count=len(RRf)
        )
    elif param == 'density':
        pot_flat = np.fromiter(
            (evaluateDensities(pot, r, z, phi=phi, t=t, forcepoisson=True).to(u.M_sun / u.pc ** 3).value
             for r, z, phi in zip(RRf, ZZf, PHIf)),
            dtype=float,
            count=len(RRf)
        )

    pot_grid = pot_flat.reshape(RR.shape)

    return pot_grid.reshape(RR.shape)


def valXZ(param, pot, x_lim, z_lim, t=0.0, pixels=100):
    x = np.linspace(x_lim[0] * u.kpc, x_lim[1] * u.kpc, pixels)
    z = np.linspace(z_lim[0] * u.kpc, z_lim[1] * u.kpc, pixels)

    XX, ZZ = np.meshgrid(x, z, indexing='ij')

    # Convert Cartesian (x, z) with y=0 into cylindrical coords
    RR = np.abs(XX)
    PHI = np.where(XX >= 0, 0.0, np.pi) * u.rad

    RRf = RR.ravel()
    PHIf = PHI.ravel()
    ZZf = ZZ.ravel()

    if param == 'potential':
        pot_flat = np.fromiter(
            (evaluatePotentials(pot, r, z, phi=phi, t=t)
             .to(u.km ** 2 / u.s ** 2).value
             for r, z, phi in zip(RRf, ZZf, PHIf)),
            dtype=float,
            count=len(RRf)
        )

    elif param == 'density':
        pot_flat = np.fromiter(
            (evaluateDensities(pot, r, z, phi=phi, t=t, forcepoisson=True)
             .to(u.M_sun / u.pc ** 3).value
             for r, z, phi in zip(RRf, ZZf, PHIf)),
            dtype=float,
            count=len(RRf)
        )

    pot_grid = pot_flat.reshape(RR.shape)

    return pot_grid


def imageMap(param, nom_pot, plane, t0=240*u.Myr, ti=250*u.Myr, pixels=200):
    from .potentials import LMCPotential, SMCPotential, LMC_SMC_MWPotential, orbit
    # --------------------------------------
    bool_orbit = False
    # t0 = 240 * u.Myr  # present day (dynamical time)
    # ti = 250 * u.Myr  # where in time are we ?
    for_talk = True

    i_lim = [-2.9999, 2.9999]
    j_lim = [-2.9999, 2.9999]
    # --------------------------------------
    from scripts.util.utils import add_ticks, add_colorbar
    fig, ax = plt.subplots(figsize=(5, 5))

    if nom_pot == 'lmc':
        pot = LMCPotential(t0)
    elif nom_pot == 'lmc-smc':
        pot = SMCPotential(t0)
    elif nom_pot == 'lmc-smc-mw':
        pot = LMC_SMC_MWPotential(t0)
    else:
        print(f'Error: {nom_pot} is an invalid potential name!')
        exit(1)

    if for_talk:
        plt.rcParams.update({  # "grid.linestyle": "--",  # Dashed grid lines
            'xtick.color': 'white',
            'ytick.color': 'white',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'text.color': 'white',
        })
        # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
        # axs = axs.flatten()
        fig = plt.figure(figsize=(6, 6))
        fig.set_facecolor((33 / 255, 33 / 255, 33 / 255))
        # Adjust the layout *here* — before plotting anything
        # fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
        gs = fig.add_gridspec(2, 2, hspace=0.02, wspace=0.02)
        fig.tight_layout()
        axes = gs.subplots(sharex=True, sharey=True)
        axs = axes.flatten()
        times = np.array([0, 75, 150, 220]) * u.Myr
        for ax, t in zip(axs, times):
            img = valXY(param, pot, i_lim, j_lim, t=t, pixels=pixels)
            extent = (i_lim[0], i_lim[1], j_lim[0], j_lim[1])
            c = ax.imshow(img.T, origin='lower', cmap='Spectral', extent=extent,
            #                vmin=-51000, vmax=-37000)
                          vmin=img.min(), vmax=img.max())
            add_ticks(
                ax, [1, 0.2], i_lim, [1, 0.2], j_lim,
                x_label='', y_label=''
            )
    else:
        if plane == 'x-y':
            img = valXY(param, pot, i_lim, j_lim, t=ti, pixels=pixels)
            add_ticks(
                ax, [1, 0.2], i_lim, [1, 0.2], j_lim,
                x_label='x' + r' $[$' + 'kpc' + r'$]$', y_label='y' + r' $[$' + 'kpc' + r'$]$'
            )
        elif plane == 'x-z':
            img = valXZ(param, pot, i_lim, j_lim, t=t0, pixels=pixels)
            add_ticks(
                ax, [2, 1], i_lim, [2, 1], j_lim,
                x_label='x' + r' $[$' + 'kpc' + r'$]$', y_label='z' + r' $[$' + 'kpc' + r'$]$'
            )
        else:
            print(f'Error: {plane} is an invalid plane!')
            exit(1)
        extent = (i_lim[0], i_lim[1], j_lim[0], j_lim[1])

        if param == 'potential':
            c = plt.imshow(img.T, origin='lower', cmap='Spectral', extent=extent,
                           vmin=img.min(), vmax=img.max())
        elif param == 'density':
            c = plt.imshow(np.log10(img.T), origin='lower', cmap='Spectral', extent=extent,
                           vmin=-3, vmax=0)
        else:
            print(f'Error: {param} is an invalid parameter!')
            exit(1)

    add_colorbar(fig, c, 'potential (km/s)' + r'$^2$')

    if bool_orbit:
        x, y = orbit(pot, t0, plane)
        ax.plot(x, y)

    plt.savefig(f'/Users/ursa/dear-prudence/dynamics/test.pdf', dpi=240, bbox_inches='tight')
    plt.show()


def run_monte_carlo_vectorized(potential, n_samples, sigma, mean=None, tmax=240.0, nt=100):
    from galpy.orbit import Orbit
    import matplotlib.pyplot as plt

    if not isinstance(tmax, Quantity):
        tmax = tmax * u.Myr
    ts = np.linspace(0.0, tmax, nt)
    # print(f'\t\\vec{{r}} : [R, phi, z, vR, vphi, vz]')
    # sigma = [0.5, 0.5, 0.5, 20, 20, 20]

    if mean is None:
        means = np.zeros(6)
    cov = np.diag(np.array(sigma) ** 2)
    init_states = np.random.multivariate_normal(means, cov, size=int(n_samples))
    print(f'\te.g. \\vec{{r}}_1(t=-t0) : {init_states[0]}')

    # reorder to galpy format: [R, vR, vT, z, vz, phi]
    vxvv = [
        np.sqrt(init_states[:, 0] ** 2 + init_states[:, 1] ** 2) * u.kpc,
        ((init_states[:, 0] * init_states[:, 3] + init_states[:, 1] * init_states[:, 4]) /
         np.sqrt(init_states[:, 0] ** 2 + init_states[:, 1] ** 2)) * u.km / u.s,
        ((init_states[:, 0] * init_states[:, 4] - init_states[:, 1] * init_states[:, 3]) /
         np.sqrt(init_states[:, 0] ** 2 + init_states[:, 1] ** 2)) * u.km / u.s,
        init_states[:, 2] * u.kpc,
        init_states[:, 5] * u.km / u.s,
        np.arctan2(init_states[:, 1], init_states[:, 0]) * u.rad
    ]
    orbits = Orbit(vxvv)
    # import os
    # ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    orbits.integrate(ts, potential, numcores=3)
    print(type(orbits.R(ts)))

    final_states = np.column_stack([
        orbits.R(ts[-1]).to(u.kpc).value,
        orbits.phi(ts[-1]).to(u.rad).value,
        orbits.z(ts[-1]).to(u.kpc).value,
        orbits.vR(ts[-1]).to(u.km / u.s).value,
        orbits.vT(ts[-1]).to(u.km / u.s).value,
        orbits.vz(ts[-1]).to(u.km / u.s).value,
    ])
    # final_states = np.column_stack([
    #     orbits.R(ts[-1]),
    #     orbits.phi(ts[-1]),
    #     orbits.z(ts[-1]),
    #     orbits.vR(ts[-1]),
    #     orbits.vT(ts[-1]),
    #     orbits.vz(ts[-1]),
    # ])
    print(f'\te.g. \\vec{{r}}_1(t=0) : {final_states[0]}')

    return init_states, final_states
