import inspect
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from galpy.potential import Potential, HernquistPotential, NFWPotential, RotateAndTiltWrapperPotential, MWPotential


def LMCPotential(t0=150 * u.Myr, LMC_model=None, add_SMC=False, SMC_model=None,
                 add_MW=False, MW_model=None, v=False):
    """
    :param t0: lookback time of SMC collision, in Gyr
    :param LMC_model: dictionary of LMC model parameters to use (defaults below)
    :param v: verbose (bool)
    :return: LMC potential object
    """
    from galpy.potential import MiyamotoNagaiPotential
    from .astrometry import Measurements

    LMC_dict = {
        # dm halo, modeled with a Hernquist profile
        'M_halo': 1.5e11 * u.M_sun,  # mass of the halo
        'a_halo': 15 * u.kpc,  # halo scale length

        # stellar + gas disk, modeled with a Miyamoto-Nagai profile
        'M_disk': 3.5e9 * u.M_sun,  # mass of the disk
        'a_disk': 1.5 * u.kpc,  # disk scale length
        'b_disk': 0.5 * u.kpc,  # disk scale height

        # bar, modeled with a variety of profiles (e.g. ferrer, softenedNeedle,...)
        'type_bar': 'perfectEllipsoid',  # type of profile to model bar
        'f_bar': 0.25,  # f_bar: fractional bar mass relative to disk mass, dimensionless

        # stellar + gas disk wobble parameters, modeled with custom ShiftedPotental wrapper
        'theta_wobble': 27.8 * u.deg,  # angle of wobble's "orbit", from vertical axis?
        'tau_wobble': 250 * u.Myr,  # period of the disk wobble, from pardy+16 retrograde collision
        'p': 1.0 * u.kpc,  # max displacement of wobbled disk from origin (modeled with an ellipse in the x-y plane)
        'q': 0.25 * u.kpc,  # min displacement of wobbled disk from origin, both from pardy+16

        # Chandesekar dynamical friction
        'cdf': True,  # bool to indicate if chandesekar dyanmical friction is to be implmented
        'GMs': 1e5 * u.M_sun,  # BH mass
        'rhm': 0 * u.kpc,  # BH effective radius
        'minr': 1 * u.pc,  # minimum radius to apply cdf
        'maxr': 2 * u.kpc  # maxmimum radius to apply cdf
    }

    LMC_model = {} if LMC_model is None else LMC_model
    for key in LMC_dict:
        LMC_dict[key] = LMC_model[key] if key in LMC_model else LMC_dict[key]

    LMC = Measurements('LMC', 'bar')
    ro = 8 * u.kpc
    vo = 220 * u.km / u.s
    G = 4.30091e-6 * u.kpc * u.km ** 2 / u.s ** 2 / u.M_sun

    # convert quantities into internal galpy units
    Omega_galpy = (LMC.patternSpeed * u.km / u.s / u.kpc * ro / vo).value
    t0_galpy = (t0 / ro * vo).to(1).value
    pa = -Omega_galpy * t0_galpy  # sets position angle s.t. theta(t = t0) = 0

    # Centered bar
    if LMC_dict['type_bar'] == 'ferrer':  # nice but horrendously expensive,
        # also wierd about physical units, also density is wierd
        from galpy.potential import FerrersPotential
        bar = FerrersPotential(amp=(LMC_dict['f_bar'] * LMC_dict['M_disk'].value * G / (vo ** 2 * ro)).value,
                               omegab=(LMC.patternSpeed * (u.km / u.s / u.kpc) * ro / vo).value,
                               a=LMC.R_bar / 8,
                               b=LMC.axisRatio * LMC.R_bar / 8,
                               c=LMC.axisRatio * LMC.R_bar / 8,
                               pa=pa, ro=8, vo=220)
    elif LMC_dict['type_bar'] == 'softenedNeedle':  # no R2deriv so 'density' impossible (w/o custom implementation)
        from galpy.potential import SoftenedNeedleBarPotential
        bar = SoftenedNeedleBarPotential(amp=LMC_dict['f_bar'] * LMC_dict['M_disk'],
                                         a=LMC.R_bar * u.kpc,
                                         b=0,
                                         c=LMC.R_bar * LMC.axisRatio * u.kpc,
                                         omegab=LMC.patternSpeed * (u.km / u.s / u.kpc),
                                         pa=-LMC.patternSpeed * (u.km / u.s / u.kpc) * t0 * u.rad)
    elif LMC_dict['type_bar'] == 'perfectEllipsoid':  # ok shape, relatively inexpensive, I think best option
        from galpy.potential import PerfectEllipsoidPotential, SolidBodyRotationWrapperPotential
        bar = SolidBodyRotationWrapperPotential(
            pot=PerfectEllipsoidPotential(amp=LMC_dict['f_bar'] * LMC_dict['M_disk'],
                                          a=LMC.R_bar * u.kpc,
                                          b=LMC.axisRatio,
                                          c=LMC.axisRatio),
            omega=LMC.patternSpeed * (u.km / u.s / u.kpc), pa=-LMC.patternSpeed * (u.km / u.s / u.kpc) * t0 * u.rad)
    elif LMC_dict['type_bar'] == 'powerEllipsoid':  # feels arbitrary, also relatively expensive
        from galpy.potential import TwoPowerTriaxialPotential
        bar = TwoPowerTriaxialPotential(amp=LMC_dict['f_bar'] * LMC_dict['M_disk'],
                                        alpha=0.5,
                                        beta=5,
                                        a=LMC.R_bar * u.kpc,
                                        b=LMC.axisRatio,
                                        c=LMC.axisRatio,
                                        pa=0)
    else:
        print(f'Error: {LMC_dict["type_bar"]} is an invalid bar type !; line {inspect.currentframe().f_lineno}')
        exit(1)

    # centered halo
    halo = HernquistPotential(amp=2 * LMC_dict['M_halo'], a=LMC_dict['a_halo'])

    # centered disk (pre-wobbled)
    disk = MiyamotoNagaiPotential(amp=LMC_dict['M_disk'], a=LMC_dict['a_disk'], b=LMC_dict['b_disk'])

    # "wobbling" the disk (asymmetric scatter of stellar disk from retrograde collision with SMC,
    # see pardy+16 and yozin+14 for reference
    wobbled_disk = ShiftedPotential(disk,
                                    p=LMC_dict['p'], q=LMC_dict['q'],
                                    omega=(1 / LMC_dict['tau_wobble'].to(u.Gyr).value) * 2 * np.pi / u.Gyr,
                                    theta=LMC_dict['theta_wobble'].to(u.rad) - np.pi * u.rad)

    if v:  # compute the force due to the wobbling disk at the origin as a sanity check
        unwobbled_disk = ShiftedPotential(disk, p=0 * u.kpc, q=0 * u.kpc, theta=LMC_dict['theta_wobble'],
                                          omega=(1 / LMC_dict['tau_wobble'].to(u.Gyr).value) * 2 * np.pi / u.Gyr)
        f_u = (vo ** 2 / ro).to(u.km / (u.s * u.Myr)).value
        Fx, Fy, Fz = wobbled_disk._cartesian_forces(R=1e-4, z=0, phi=0, t=t0_galpy)
        print(f'\t\t\tcheck, does this make sense?\n'
              f'\t\t\t\t\\vec g_disk (\\vec r = 0) : ({Fx * f_u:.3f}, {Fy * f_u:.3f}, {Fz * f_u:.3f}) km/s/Myr\n'
              f'\t\t\t\t|Phi|_wobbledDisk (\\vec r = 0) : {wobbled_disk._evaluate(R=1e-4, z=0, t=t0_galpy)}\n'
              f'\t\t\t\t|Phi|_unwobbledDisk (\\vec r = 0) : {unwobbled_disk._evaluate(R=1e-4, z=0, t=t0_galpy)}')

    if LMC_dict['cdf']:
        v and print('... with dynamical friction ... ')
        from galpy.potential import ChandrasekharDynamicalFrictionForce
        # we are assuming that the disk is un-wobbled so cdf results will be slightly off
        # cdf = ChandrasekharDynamicalFrictionForce(GMs=LMC_dict['GMs'], rhm=LMC_dict['rhm'],
        #                                           minr=LMC_dict['minr'], maxr=LMC_dict['maxr'],
        #                                           dens=(bar + halo + disk))
        # lmc = [bar, wobbled_disk, halo, cdf]
        cdf = ChandrasekharDynamicalFrictionForce(GMs=LMC_dict['M_disk'], rhm=LMC_dict['a_disk'],
                                                  minr=LMC_dict['minr'], maxr=10 * u.kpc,
                                                  dens=halo)
        lmc = [halo, cdf]

    else:
        v and print('... without dynamical friction ... ')
        # Combined potential
        lmc = [halo]

    if add_SMC:
        lmc += smcPotential(t0, SMC_model)
    if add_MW:
        lmc += mwPotential(t0, MW_model)

    return lmc


def smcPotential(t0, SMC_model=None):
    SMC_dict = {
        'M_halo': 1.9e10 * u.M_sun,  # mass of the SMC halo (Hernquist profile), from Lucchini+21
        'a_halo': 2.5 * u.kpc,  # halo scale length
        'dynamic': True,  # integrates SMC orbit back in time and then dynamically moves SMC accordingly
    }

    SMC_model = {} if SMC_model is None else SMC_model
    for key in SMC_dict:
        SMC_dict[key] = SMC_model[key] if key in SMC_model else SMC_dict[key]

    from .astrometry import LMCDisk, Measurements
    SMC = Measurements('SMC', 'stars')
    SMC_center = SkyCoord(ra=SMC.ra * u.deg, dec=SMC.dec * u.deg, distance=SMC.distance * u.kpc,
                          pm_ra_cosdec=SMC.mu_alpha * u.mas / u.yr, pm_dec=SMC.mu_delta * u.mas / u.yr,
                          radial_velocity=SMC.vlsr * u.km / u.s,
                          frame="icrs").transform_to(LMCDisk)
    SMC_center = SMC_center.transform_to(LMCDisk())  # (x,y,z) ~ (-5.675, -23.125, -1.718) kpc

    # SMC halo
    smc_halo = HernquistPotential(amp=2 * SMC_dict['M_halo'], a=SMC_dict['a_halo'])
    if SMC_dict['dynamic']:
        from galpy.potential import MovingObjectPotential, MiyamotoNagaiPotential
        from galpy.orbit import Orbit
        from .astrometry import LMCDisk, Measurements

        SMC = Measurements('SMC', 'stars')
        smc = SkyCoord(ra=SMC.ra * u.deg, dec=SMC.dec * u.deg, distance=SMC.distance * u.kpc,
                       pm_ra_cosdec=SMC.mu_alpha * u.mas / u.yr, pm_dec=SMC.mu_delta * u.mas / u.yr,
                       radial_velocity=SMC.vlsr * u.km / u.s,
                       frame="icrs").transform_to(LMCDisk)
        r, v = smc.cartesian.xyz, (smc.velocity.d_x, smc.velocity.d_y, smc.velocity.d_z)
        o = Orbit([
            np.sqrt(r[0] ** 2 + r[1] ** 2),  # r
            ((r[0] * v[0] + r[1] * v[1]) / np.sqrt(r[0] ** 2 + r[1] ** 2)),  # v_r
            ((r[0] * v[1] - r[1] * v[0]) / np.sqrt(r[0] ** 2 + r[1] ** 2)),  # v_phi
            r[2],  # z
            v[2],  # v_z
            np.arctan2(r[1], r[0])])  # phi

        # toy LMC model (since cdf cannot handle non-axisymmetric potentials)
        halo = HernquistPotential(amp=2 * 1.5e11 * u.M_sun, a=15 * u.kpc)
        disk = MiyamotoNagaiPotential(amp=3.5e9 * u.M_sun, a=1.5 * u.kpc, b=0.5 * u.kpc)  # centered disk (un-wobbled)
        from galpy.potential import ChandrasekharDynamicalFrictionForce
        cdf = ChandrasekharDynamicalFrictionForce(GMs=SMC_dict['M_halo'], rhm=SMC_dict['a_halo'],
                                                  dens=(halo + disk))

        ts = np.linspace(0, -t0.to(u.Myr).value, 100) * u.Myr
        o.integrate(ts, halo + disk + cdf)

        # -------------------------------------
        compute_thetaV_smc = False  # compute the direction of the SMC's velocity vector at time of collision
        if compute_thetaV_smc:
            x_dot = (o.vR(ts)[-1].value * np.cos(o.phi(ts)[-1]).value
                     - o.R(ts)[-1].value * o.vT(ts)[-1].value * np.sin(o.phi(ts)[-1]).value)
            y_dot = (o.vR(ts)[-1].value * np.sin(o.phi(ts)[-1]).value
                     + o.R(ts)[-1].value * o.vT(ts)[-1].value * np.cos(o.phi(ts)[-1]).value)
            thetaV = np.arctan2(y_dot, x_dot)
        # -------------------------------------

        p = Orbit([o.R(ts)[-1], o.vR(ts)[-1], o.vT(ts)[-1], o.z(ts)[-1], o.vz(ts)[-1], o.phi(ts)[-1]])
        ts = np.linspace(0, t0.to(u.Myr).value, 100) * u.Myr
        p.integrate(ts, halo + disk + cdf)

        smc_shifted = MovingObjectPotential(orbit=p, pot=smc_halo)
    else:
        smc_shifted = RotateAndTiltWrapperPotential(pot=smc_halo, offset=-SMC_center.cartesian.xyz)

    return smc_shifted


def mwPotential(t0, MW_model=None):
    MW_dict = {
        'M_200': 1.1e12 * u.M_sun,  # mass of the MW halo (NFW profile), from Lucchini+25
        'conc': 10  # concentration parameter
    }

    MW_model = {} if MW_model is None else MW_model
    for key in MW_dict:
        MW_dict[key] = MW_model[key] if key in MW_model else MW_dict[key]

    from .astrometry import LMCDisk, Measurements
    MW = Measurements('MW', 'SagA*')
    MW_center = SkyCoord(ra=MW.ra * u.deg, dec=MW.dec * u.deg,
                         distance=MW.distance * u.kpc, frame="icrs")
    MW_center = MW_center.transform_to(LMCDisk())  # (x,y,z) ~ (3.521, 14.675, -46.956) kpc

    # mw_halo = NFWPotential(mvir=MW_dict['M_200'] / (1e12 * u.M_sun), conc=MW_dict['conc'], overdens=200)
    mw_halo = HernquistPotential(amp=2 * 1.1e12 * u.M_sun, a=20 * u.kpc)
    mw_shifted = RotateAndTiltWrapperPotential(pot=mw_halo, offset=-MW_center.cartesian.xyz)
    return mw_shifted


class ShiftedPotential(Potential):
    def __init__(self, base_pot, p=0., q=0., omega=0., theta=0.):
        """
        :param base_pot: galpy potenital object
        :param p: max amplitude (natural units or u.kpc)
        :param q: min amplitude, a = (p + q) / 2, b = sqrt(p * q)
        :param omega: inverse period of the shifted potential
        :param theta: position angle of the ellipse, from the vertical axis
        :return shifted galpy potential object
        """
        Potential.__init__(self, amp=base_pot._amp, ro=base_pot._ro, vo=base_pot._vo)
        self.base = base_pot
        self.hasC = False

        ro = self._ro * u.kpc
        vo = self._vo * u.km / u.s
        t_unit = (ro / vo).to(u.Gyr)

        # converts input parameters to natural units
        self.p = (p / ro).decompose().value if isinstance(p, u.Quantity) else p
        self.q = (q / ro).decompose().value if isinstance(q, u.Quantity) else q
        self.omega = (omega * t_unit).decompose().value if isinstance(omega, u.Quantity) else omega
        self.theta = theta.decompose().value if isinstance(theta, u.Quantity) else theta

    def _shift_cartesian(self, R, z, phi, t):
        # converts input parameters (max and min displacement) into semi-major and semi-minor axis lengths
        a = (self.p + self.q) / 2
        b = np.sqrt(self.p * self.q)
        c = (self.p - self.q) / 2
        # computes time-dependent cartesian offset
        x0 = (a * np.cos(self.omega * t) * np.cos(-self.theta)
              + b * np.sin(self.omega * t) * np.sin(-self.theta)
              - c * np.cos(-self.theta))
        y0 = (-a * np.cos(self.omega * t) * np.sin(-self.theta)
              + b * np.sin(self.omega * t) * np.cos(-self.theta)
              + c * np.sin(-self.theta))
        z0 = 0.
        # computes original cartesian coordinates
        x, y = R * np.cos(phi), R * np.sin(phi)
        # applies cartesian offset
        xs, ys, zs = x - x0, y - y0, z - z0
        # converts back to cylindrical coordiantes
        Rs, phis = np.sqrt(xs ** 2 + ys ** 2), np.arctan2(ys, xs)
        return xs, ys, zs, Rs, phis

    def _evaluate(self, R, z, phi=0., t=0.):
        _, _, zs, Rs, phis = self._shift_cartesian(R, z, phi, t)
        return self.base._evaluate(Rs, zs, phi=phis, t=t)

    def _cartesian_forces(self, R, z, phi, t):
        # returns (Fx, Fy, Fz) in the original frame
        xs, ys, zs, Rs, phis = self._shift_cartesian(R, z, phi, t)
        # computs forces in cylindrical coordinates
        FRs = self.base._Rforce(Rs, zs, phi=phis, t=t)
        Fzs = self.base._zforce(Rs, zs, phi=phis, t=t)
        Fphis = self.base._phiforce(Rs, zs, phi=phis, t=t) if hasattr(self.base, "_phiforce") else 0.0
        Rs = np.maximum(Rs, 1e-10)  # accounts for numerical drift
        # trasnforms from cylindrical to cartesian
        Fx = FRs * np.cos(phis) - Fphis * np.sin(phis) / Rs
        Fy = FRs * np.sin(phis) + Fphis * np.cos(phis) / Rs
        Fz = Fzs
        return Fx, Fy, Fz

    def _Rforce(self, R, z, phi=0., t=0.):
        Fx, Fy, _ = self._cartesian_forces(R, z, phi, t)
        return Fx * np.cos(phi) + Fy * np.sin(phi)

    def _phiforce(self, R, z, phi=0., t=0.):
        Fx, Fy, _ = self._cartesian_forces(R, z, phi, t)
        return -Fx * np.sin(phi) + Fy * np.cos(phi)

    def _zforce(self, R, z, phi=0., t=0.):
        _, _, Fz = self._cartesian_forces(R, z, phi, t)
        return Fz


class ShiftedPotentialSecondVersion(Potential):
    def __init__(self, base_pot, p=0., q=0., omega=0., theta=0.):
        """
        :param base_pot: galpy potenital object
        :param p: max amplitude (natural units or u.kpc)
        :param q: min amplitude, a = (p + q) / 2, b = sqrt(p * q)
        :param omega: inverse period of the shifted potential
        :param theta: position angle of the ellipse, from the vertical axis
        :return shifted galpy potential object
        """
        Potential.__init__(self, amp=base_pot._amp, ro=base_pot._ro, vo=base_pot._vo)
        self.base = base_pot
        self.hasC = False
        self.hasC_dens = False
        self.hasC_dxdv = False

        ro = self._ro * u.kpc
        vo = self._vo * u.km / u.s
        t_unit = (ro / vo).to(u.Gyr)

        # converts input parameters to natural units
        self.p = (p / ro).decompose().value if isinstance(p, u.Quantity) else p
        self.q = (q / ro).decompose().value if isinstance(q, u.Quantity) else q
        self.omega = (omega * t_unit).decompose().value if isinstance(omega, u.Quantity) else omega
        self.theta = theta.decompose().value if isinstance(theta, u.Quantity) else theta

    def _shift_cartesian(self, R, z, phi, t):
        # converts input parameters (max and min displacement) into semi-major and semi-minor axis lengths
        a = (self.p + self.q) / 2
        b = np.sqrt(self.p * self.q)
        c = (self.p - self.q) / 2
        # computes time-dependent cartesian offset
        x0 = (a * np.cos(self.omega * t) * np.cos(-self.theta)
              + b * np.sin(self.omega * t) * np.sin(-self.theta)
              - c * np.cos(-self.theta))
        y0 = (-a * np.cos(self.omega * t) * np.sin(-self.theta)
              + b * np.sin(self.omega * t) * np.cos(-self.theta)
              + c * np.sin(-self.theta))
        z0 = 0.
        # computes original cartesian coordinates
        x, y = R * np.cos(phi), R * np.sin(phi)
        # applies cartesian offset
        xs, ys, zs = x - x0, y - y0, z - z0
        # converts back to cylindrical coordiantes
        Rs = np.sqrt(xs ** 2 + ys ** 2 + 1e-8)  # accounts for numerical drift
        # Rs = np.sqrt(xs ** 2 + ys ** 2)
        # mask = Rs > 1e-10
        # Rs = np.maximum(Rs, 1e-8)
        # phis = np.zeros_like(Rs)
        # phis[mask] = np.arctan2(ys[mask], xs[mask])
        return xs, ys, zs, Rs

    def _evaluate(self, R, z, phi=0., t=0.):
        _, _, zs, Rs = self._shift_cartesian(R, z, phi, t)
        return self.base._evaluate(Rs, zs, t=t)

    def _cartesian_forces(self, R, z, phi, t):
        # returns (Fx, Fy, Fz) in the original frame
        xs, ys, zs, Rs = self._shift_cartesian(R, z, phi, t)
        # computs forces in cylindrical coordinates
        FRs = self.base._Rforce(Rs, zs, t=t)
        Fzs = self.base._zforce(Rs, zs, t=t)

        # if hasattr(self.base, "_phiforce"):
        #     Fphis = self.base._phiforce(Rs, zs, phi=phis, t=t)
        #     # transforms from cylindrical to cartesian
        #     Fx = FRs * np.cos(phis) - Fphis * np.sin(phis) / Rs
        #     Fy = FRs * np.sin(phis) + Fphis * np.cos(phis) / Rs
        #     Fz = Fzs
        # else:
        # project radial force directly in Cartesian
        Fx = FRs * xs / Rs
        Fy = FRs * ys / Rs
        Fz = Fzs

        return Fx, Fy, Fz

    # def _Rforce(self, R, z, phi=0., t=0.):
    #     Fx, Fy, _ = self._cartesian_forces(R, z, phi, t)
    #     return Fx * np.cos(phi) + Fy * np.sin(phi)
    def _Rforce(self, R, z, phi=0., t=0.):
        xs, ys, zs, Rs = self._shift_cartesian(R, z, phi, t)
        Fx, Fy, _ = self._cartesian_forces(R, z, phi, t)
        return (Fx * xs + Fy * ys) / Rs  # project along shifted R

    # def _phiforce(self, R, z, phi=0., t=0.):
    #     Fx, Fy, _ = self._cartesian_forces(R, z, phi, t)
    #     return -Fx * np.sin(phi) + Fy * np.cos(phi)
    def _phiforce(self, R, z, phi=0., t=0.):
        xs, ys, zs, Rs = self._shift_cartesian(R, z, phi, t)
        Fx, Fy, _ = self._cartesian_forces(R, z, phi, t)
        return (Fx * ys - Fy * xs) / Rs  # project along shifted phi

    def _zforce(self, R, z, phi=0., t=0.):
        _, _, Fz = self._cartesian_forces(R, z, phi, t)
        return Fz

    def _R2deriv(self, R, z, phi=0., t=0.):
        dR = 1e-5
        return -(
                self._Rforce(R + dR, z, phi=phi, t=t)
                - self._Rforce(R - dR, z, phi=phi, t=t)
        ) / (2.0 * dR)

    def _z2deriv(self, R, z, phi=0., t=0.):
        dz = 1e-5
        return (
                self._evaluate(R, z + dz, phi=phi, t=t)
                - 2.0 * self._evaluate(R, z, phi=phi, t=t)
                + self._evaluate(R, z - dz, phi=phi, t=t)
        ) / dz ** 2


def orbit(pot, t0, obj=None, plane='x-y'):
    from galpy.orbit import Orbit

    if obj is None:
        o = Orbit(
            [0.5 * u.kpc,  # r
             20 * u.km / u.s,  # v_r
             90 * u.km / u.s,  # v_phi
             0 * u.kpc,  # z
             0.0 * u.km / u.s,  # v_z
             (27 - 90) * u.deg])  # phi
        ts = np.linspace(0, t0.to(u.Myr).value, 100) * u.Myr

    elif obj == 'smc':
        from astropy.coordinates import SkyCoord
        from .astrometry import LMCDisk, Measurements

        SMC = Measurements('SMC', 'stars')
        smc = SkyCoord(ra=SMC.ra * u.deg, dec=SMC.dec * u.deg, distance=SMC.distance * u.kpc,
                       pm_ra_cosdec=SMC.mu_alpha * u.mas / u.yr, pm_dec=SMC.mu_delta * u.mas / u.yr,
                       radial_velocity=SMC.vlsr * u.km / u.s,
                       frame="icrs").transform_to(LMCDisk)
        r, v = smc.cartesian.xyz, (smc.velocity.d_x, smc.velocity.d_y, smc.velocity.d_z)
        o = Orbit([
            np.sqrt(r[0] ** 2 + r[1] ** 2),  # r
            ((r[0] * v[0] + r[1] * v[1]) / np.sqrt(r[0] ** 2 + r[1] ** 2)),  # v_r
            ((r[0] * v[1] - r[1] * v[0]) / np.sqrt(r[0] ** 2 + r[1] ** 2)),  # v_phi
            r[2],  # z
            v[2],  # v_z
            np.arctan2(r[1], r[0])])  # phi

        ts = np.linspace(0, -t0.to(u.Myr).value, 100) * u.Myr
    o.integrate(ts, pot)
    print(ts[-1])
    if plane == 'x-y':
        return o.R(ts) * np.cos(o.phi(ts)), o.R(ts) * np.sin(o.phi(ts))
    elif plane == 'x-z':
        return o.R(ts) * np.cos(o.phi(ts)), o.z(ts)


def vectorizedOrbits(potential, mean_psi=None, sigma_psi=None, mu_ti=150 * u.Myr, sigma_ti=10 * u.Myr, N=1e3):
    from galpy.orbit import Orbit

    T = np.linspace(0.0, mu_ti.to(u.Myr).value + 5 * sigma_ti.to(u.Myr).value, int(N))
    # sample final time distriution: set of final times (one per particle)
    Ti = np.random.normal(mu_ti.to(u.Myr).value, sigma_ti.to(u.Myr).value, int(N)) * u.Myr

    # sample the initial 6-dim phase space distribution
    mean_psi = np.zeros(6) if mean_psi is None else mean_psi
    if sigma_psi is None:  # (sigma_x, sigma_y, sigma_z, sigma_vx, sigma_vy, sigma_vz)
        sigma_psi = (0.25 * u.kpc, 0.25 * u.kpc, 0.1 * u.kpc,
                     15 * u.km / u.s, 15 * u.km / u.s, 15 * u.km / u.s)
    cov = np.diag(np.array(sigma_psi) ** 2)
    Psi_0 = np.random.multivariate_normal(mean_psi, cov, size=int(N))
    print(f'\te.g.\t\\Psi_1(t = {-mu_ti:.1f}) : {Psi_0[0]}')

    # reorder to galpy format and attach units
    vxvv = [
        np.sqrt(Psi_0[:, 0] ** 2 + Psi_0[:, 1] ** 2) * u.kpc,  # R
        ((Psi_0[:, 0] * Psi_0[:, 3] + Psi_0[:, 1] * Psi_0[:, 4]) /  # vR
         np.sqrt(Psi_0[:, 0] ** 2 + Psi_0[:, 1] ** 2)) * u.km / u.s,
        ((Psi_0[:, 0] * Psi_0[:, 4] - Psi_0[:, 1] * Psi_0[:, 3]) /  # vT
         np.sqrt(Psi_0[:, 0] ** 2 + Psi_0[:, 1] ** 2)) * u.km / u.s,
        Psi_0[:, 2] * u.kpc,  # z
        Psi_0[:, 5] * u.km / u.s,  # vz
        np.arctan2(Psi_0[:, 1], Psi_0[:, 0]) * u.rad  # phi
    ]
    orbits = Orbit(vxvv)
    orbits.integrate(T, potential)

    Psi_i = np.zeros(Psi_0.shape)
    for i in range(int(N)):
        psi_i = np.array([
            orbits.R(Ti[i])[i].to(u.kpc).value * np.cos(orbits.phi(Ti[i])[i].to(u.rad).value),  # x
            orbits.R(Ti[i])[i].to(u.kpc).value * np.sin(orbits.phi(Ti[i])[i].to(u.rad).value),  # y
            orbits.z(Ti[i])[i].to(u.kpc).value,  # z
            orbits.vR(Ti[i])[i].to(u.km / u.s).value * np.cos(orbits.phi(Ti[i])[i].to(u.rad).value)
            - orbits.vT(Ti[i])[i].to(u.km / u.s).value * np.sin(orbits.phi(Ti[i])[i].to(u.rad).value),  # vx
            orbits.vR(Ti[i])[i].to(u.km / u.s).value * np.sin(orbits.phi(Ti[i])[i].to(u.rad).value)
            + orbits.vT(Ti[i])[i].to(u.km / u.s).value * np.cos(orbits.phi(Ti[i])[i].to(u.rad).value),  # vy
            orbits.vz(Ti[i])[i].to(u.km / u.s).value  # vz
        ])
        Psi_i[i] = psi_i

    print(f'\t\t\\Psi_1(t = {(Ti[i] - mu_ti):.1f}) : {Psi_i[0]}')
    return Psi_0, Psi_i  # Psi is (N, 6), dimensionless cartesian coordinates (in kpc)
