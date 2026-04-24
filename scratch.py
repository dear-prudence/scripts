import numpy as np
import astropy.units as u
from astropy.coordinates import Galactocentric


class Complex:
    """e.g. docstring """

    def __init__(self, R, I):
        self.r = R
        self.i = I

    @property
    def mag(self): return np.sqrt(self.r ** 2 + self.i ** 2)

    @property
    def angle(self): return np.degrees(np.arctan2(self.i, self.r))

    def square(self):
        return Complex(self.r ** 2 - self.i ** 2, 2 * self.r * self.i)

    def __mul__(self, other):
        return Complex(
            self.r * other.r - self.i * other.i,
            self.r * other.i + self.i * other.r
        )


def C():
    x = Complex(3, 4)
    y = Complex(2, 3)
    z = x * y
    print(x.r)
    print(f'{z.r} + {z.i}i,\t|z| = {z.mag:f}')
    x.mag()


def gal():
    import numpy as np
    from galpy.potential import MWPotential2014
    from galpy.potential import evaluatePotentials as evalPot
    from galpy.orbit import Orbit
    import matplotlib.pyplot as plt
    E, Lz = -1.25, 0.6
    o1 = Orbit([0.8, 0, Lz / 0.8, 0, np.sqrt(2 * (E - evalPot(MWPotential2014, 0.8, 0) - (Lz / 0.8) ** 2 / 2)), 0])
    ts = np.linspace(0, 100, 2001)
    o1.integrate(ts, MWPotential2014)
    o1.plot(xrange=[0.3, 1], yrange=[-0.2, 0.2], c='k')
    plt.show()

def coordinateTransformTest():
    from astropy.coordinates import SkyCoord, ICRS
    from scripts.util.archive.astrometry import LMCTangent, LMCDisk, Measurements

    SMC = Measurements('SMC', 'stars')
    MW = Measurements('MW', 'SagA*')
    LMC = Measurements('LMC', 'disk')
    rgb = Measurements('LMC', 'RGBstars')
    RGB = SkyCoord(
        ra=rgb.ra * u.deg,
        dec=rgb.dec * u.deg,
        distance=rgb.distance * u.kpc,
        # pm_ra_cosdec=SMC.mu_alpha * u.mas / u.yr,
        # pm_dec=SMC.mu_delta * u.mas / u.yr,
        # radial_velocity=SMC.vlsr * u.km / u.s,
        frame="icrs",
    )
    smc = SkyCoord(
        ra=SMC.ra * u.deg,
        dec=SMC.dec * u.deg,
        distance=SMC.distance * u.kpc,
        pm_ra_cosdec=SMC.mu_alpha * u.mas / u.yr,
        pm_dec=SMC.mu_delta * u.mas / u.yr,
        radial_velocity=SMC.vlsr * u.km / u.s,
        frame="icrs",
    )
    mw = SkyCoord(
        ra=MW.ra * u.deg,
        dec=MW.dec * u.deg,
        distance=MW.distance * u.kpc,
        pm_ra_cosdec=MW.mu_alpha * u.mas / u.yr,
        pm_dec=MW.mu_delta * u.mas / u.yr,
        radial_velocity=MW.vlsr * u.km / u.s,
        frame="icrs",
    )

    print(f'mw icrs: {mw}')
    mw_lmc = mw.transform_to(LMCTangent(LMC=LMC))
    print(f'mw tangent: {mw_lmc}')
    mw_disk = mw_lmc.transform_to(LMCDisk(LMC=LMC))
    print(f'mw disk : {mw_disk}')
    x, y, z = mw_disk.cartesian.xyz
    vx, vy, vz = mw_disk.cartesian.differentials['s'].d_xyz.to(u.km / u.s)
    mw_mw = SkyCoord(
        x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz,
        frame=LMCDisk(LMC=LMC),
        representation_type='cartesian', differential_type='cartesian')
    mw_mw = mw_mw.transform_to(Galactocentric)
    print(f'mw mw : {mw_mw}')

    print(f'smc icrs: {smc}')
    star_lmc = smc.transform_to(LMCTangent(LMC=LMC))
    print(f'smc tangent: {star_lmc}')
    star_disk = star_lmc.transform_to(LMCDisk(LMC=LMC))
    print(f'smc disk : {star_disk}')
    x, y, z = star_disk.cartesian.xyz
    try:
        vx, vy, vz = star_disk.cartesian.differentials['s'].d_xyz.to(u.km / u.s)
        star_disk = SkyCoord(
            x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz,
            frame=LMCDisk(LMC=LMC),
            representation_type='cartesian', differential_type='cartesian')
    except KeyError:
        star_disk = SkyCoord(
            x=x, y=y, z=z,
            frame=LMCDisk(LMC=LMC),
            representation_type='cartesian')
    star_lmc = star_disk.transform_to(LMCTangent(LMC=LMC))
    print(f'smc tangent: {star_lmc}')
    smc = star_lmc.transform_to(ICRS())
    print(f'smc icrs : {smc}')

    # disk : <SkyCoord (LMCDisk: LMC=<util.astrometry.Measurements object at 0x12a8e3310>): (x, y, z) in kpc
    #     (-5.67490593, -23.12545402, -1.71835071)
    #  (v_x, v_y, v_z) in km / s
    #     (-29.43502637, -82.18562972, -36.93060767)>

def galpy_test():
    from util.imaging import plot_scalarField
    plot_scalarField('lmc', 'Phi', 'x-y', extent=(-2, 2, -2, 2),
                     t0=150 * u.Myr, ti=250 * u.Myr, pixels=200,
                     bool_dark_mode=True, bool_orbit=False, verbose=True)

# from util.hestia import compute_barParams
# compute_barParams('09_18', 'halo_38', 127)

def logNormalTest():
    import matplotlib.pyplot as plt
    mean_psi = [0.5, np.pi, 0, 0, 30, 0]
    sigma_psi = [0.4, np.pi / 2, 0.1, 20, 20, 20]
    # sigma ~ 20 km /s from https://iopscience.iop.org/article/10.1086/317023/pdf
    # mean_psi = [0., np.pi, 0, 0, 30, 0] if mean_psi is None else mean_psi
    # sigma_psi = [0.5, np.pi / 2, 0.2, 15, 15, 15] if sigma_psi is None else sigma_psi

    rng = np.random.default_rng(seed=None)
    # Convert (mean, std) in linear space to log-space parameters
    sigma_ln = np.sqrt(np.log(1 + (sigma_psi[0] / mean_psi[0]) ** 2))
    mu_ln = np.log(mean_psi[0]) - sigma_ln ** 2 / 2
    x = np.linspace(0, 2, 1000)
    f = 1 / (x * sigma_ln * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x)-mu_ln) ** 2 / (2 * sigma_ln ** 2))

    fig = plt.figure(figsize=(5, 5))
    plt.plot(x, f)
    plt.show()


def compute_pm():
    from astropy.coordinates import SkyCoord, ICRS
    from util.astrometry import LMCDisk, Measurements

    LMC = Measurements('LMC', 'galpy')

    bh = SkyCoord(
        x=-0.055 * u.kpc,
        y=0.083 * u.kpc,
        z=-0.037 * u.kpc,
        v_x=-1.134 * u.km / u.s,
        v_y=-0.897 * u.km / u.s,
        v_z=0.293 * u.km / u.s,
        frame=LMCDisk(LMC=LMC),
        representation_type='cartesian', differential_type='cartesian').transform_to(ICRS)
    print('(mu_alpha*, mu_delta, vlsr) : ('
          f'{bh.pm_ra_cosdec:.3f}, '
          f'{bh.pm_dec:.3f}, '
          f'{bh.radial_velocity:.1f})')
    print('(Delta mu_alpha*, Delta mu_delta, Delta vlsr) : ('
          f'{bh.pm_ra_cosdec - LMC.mu_alpha * u.mas / u.yr:.3f}, '
          f'{bh.pm_dec - LMC.mu_delta * u.mas / u.yr:.3f}, '
          f'{bh.radial_velocity - LMC.vlsr * u.km / u.s:.1f})')


def fitGaussian():
    from scipy.optimize import curve_fit

    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = xy
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate coordinates into the Gaussian's principal frame
        xr = cos_t * (x - x0) + sin_t * (y - y0)
        yr = -sin_t * (x - x0) + cos_t * (y - y0)
        return amplitude * np.exp(
            -(xr ** 2 / (2 * sigma_x ** 2) + yr ** 2 / (2 * sigma_y ** 2))
        )

    basePath = f'/Users/ursa/dear-prudence/dynamics/bhPDF/lmc-smc-mw/'
    fileName = f'bhPDF.lmc-smc-mw.N-100000.npz'
    data = np.load(basePath + fileName)
    lon_e = data['ra_e_zoom']
    lat_e = data['dec_e_zoom']
    f_PDF = data['Hi_radec_zoom']
    x_c = (lon_e[:-1] + lon_e[1:]) / 2
    y_c = (lat_e[:-1] + lat_e[1:]) / 2
    X, Y = np.meshgrid(x_c, y_c)

    lower_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    initial_guess = (1, 80, -70, 0.5, 0.5, 0)  # amplitude, xo, yo, sigma_x, sigma_y, offset
    popt, _ = curve_fit(gaussian_2d, (X.ravel(), Y.ravel()), f_PDF.T.ravel(), p0=initial_guess,
                        bounds=(lower_bounds, upper_bounds), maxfev=9999)
    print(f'2-dim gaussian fit returned parameters -- \n'
          f'\tA : {popt[0]:.2e}\n'
          f'\tmu_ra : {popt[1]:.3f}\n'
          f'\tmu_dec : {popt[2]:.3f}\n'
          f'\tsigma_ra : {popt[3]:.3f}\n'
          f'\tsigma_dec : {popt[4]:.3f}\n'
          f'\tposition angle : {(popt[5] % (2 * np.pi) * u.rad).to(u.deg):.3f}\n')


# from util.publications import chisholm2026_fig1
# chisholm2026_fig1()
coordinateTransformTest()
