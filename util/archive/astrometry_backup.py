import inspect
import numpy as np
from numpy import sin, cos
import astropy.units as u
from astropy.coordinates import (BaseCoordinateFrame, frame_transform_graph,
                                 CartesianRepresentation, CartesianDifferential, ICRS, Attribute)
from astropy.coordinates.transformations import FunctionTransform

def rad(a): return np.radians(a)


def R_x(a): return np.array([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])


def R_y(a): return np.array([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])


def R_z(a): return np.array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])


class Measurements:
    def __init__(self, obj, tracer=None):
        # (ra, dec) all in J2000 epoch unless otherwise specified

        if obj == 'LMC':
            if tracer == 'bar':  # stellar dynamical (completeness-corrected, red clump stars)
                # rathore+2025 (https://iopscience.iop.org/article/10.3847/1538-4357/ad93ae/pdf)
                self.ra = 80.27
                self.dec = -69.65
                self.sigma_ra = 0.02
                self.sigma_dec = 0.02
                self.inclination = 25.86
                self.nodes = 121.26
                self.distance = 49.9
                self.R_bar = 2.13  # kpc
                self.axisRatio = 0.54
                self.barStrength = 0.27  # +/- 0.01
                self.barOffset = 0.76  # =/- 0.01, in kpc;  "the separation between the center of the bar ellipse
                # and the center of the iso-ellipse of the outer LMC disk"
                self.barOffsetDirection = np.array([0.149, -0.602, 0])  # in kpc, taken from Fig 8a plot digitizer
                self.patternSpeed = 18.5  # km/s/kpc; Jimenez-Arranz+2024
            elif tracer == 'pm':  # stellar kinematical center;
                # Choi+2022 (https://iopscience.iop.org/article/10.3847/1538-4357/ac4e90/pdf)
                self.ra = 80.443
                self.dec = -69.272
                self.inclination = 23.396
                self.nodes = 138.856
                self.distance = 50.1  # Freedman+2001
            elif tracer == 'HI':
                # Kim+1998 (https://iopscience.iop.org/article/10.1086/306030/pdf)
                self.ra = 79.25
                self.dec = -69.03
                self.sigma_ra = 0.08  # taken from pixel size of 20"
                self.sigma_dec = 0.08
                self.distance = 49.8  # assuming no significant vertical offset (assumption not from reference)
            elif tracer == 'photometric':
                # vanDerMarel+2001 (iopscience.iop.org/article/10.1086/323100/pdf for (ra, dec),
                # https://iopscience.iop.org/article/10.1086/323099/pdf for (i, theta))
                self.ra = 82.25
                self.dec = -69.5
                self.inclination = 34.7  # +- 6.2
                self.nodes = 122.5  # +- 8.3
            elif tracer == 'disk':
                # Kacharov+2024 (https://www.aanda.org/articles/aa/pdf/2024/12/aa51578-24.pdf)
                # jeans modeling, gaia dr3
                self.ra = 80.29
                self.sigma_ra = 0.04
                self.dec = -69.25
                self.sigma_dec = 0.02
                self.inclination = 25.5
                self.sigma_i = 0.2
                self.nodes = 124
                self.sigma_nodes = 0.4
                self.distance = 49.9  # adopted
                self.axisRatio = 0.23
                self.mu_alpha = 1.88  # mu_alpha*
                self.mu_delta = 0.32  # mas/yr,
                self.vlsr = 264.83  # km/s
            elif tracer == 'HVSs':
                # Lucchini+2025 (https://iopscience.iop.org/article/10.3847/2041-8213/ae109d/pdf)
                self.ra = 80.72
                self.dec = -67.79
                self.sigma_ra = 0.44
                self.sigma_dec = 0.80
                self.distance = 49.9  # temp test

            elif tracer == 'cepheids':
                # Bhuyan+2024, disk of LMC from cepheid light curves
                self.ra = 80.78  # Nikolaev+2004
                self.dec = -69.03
                self.distance = 49.59  # Pietrzyński+2019
                self.inclination = 22.87
                self.nodes = 154.76
            else:
                print(f'Error: {tracer} is an invalid tracer for util/astrometry.py/Measurements/{obj}; '
                      f'line {inspect.currentframe().f_lineno}')
                exit(1)
        elif obj == 'SMC':
            if tracer == 'stars':
                # Graczyk+2020
                self.ra = 12.54  # +- 0.30
                self.dec = -73.11  # +- 0.15
                self.distance = 62.44
                self.mu_alpha = 0.82  # +- 0.10 mas/yr *, -d(alpha)/dt cos(delta)  convert everything to mu_W
                self.mu_delta = -1.21  # +- 0.03 mas/yr
                self.vlsr = 145.6  # +- 0.6 km/s
            else:
                print(f'Error: {tracer} is an invalid tracer for util/astrometry.py/Measurements/{obj}; '
                      f'line {inspect.currentframe().f_lineno}')
                exit(1)
        elif obj == 'MW':
            if tracer == 'SagA*':
                # https://ui.adsabs.harvard.edu/abs/2023AJ....165...49G/abstract
                self.ra = 266.4168
                self.dec = -29.0078
                self.distance = 8.178  # https://ui.adsabs.harvard.edu/abs/2019A%26A...625L..10G/abstract
            else:
                print(f'Error: {tracer} is an invalid tracer for util/astrometry.py/Measurements/{obj}; '
                      f'line {inspect.currentframe().f_lineno}')
                exit(1)

        elif obj == 'Sun':
            if True:  # Asplund et al. 2009
                self.X = 0.7154
                self.Y = 0.2703
                self.Z = 0.0142
                self.Ox = 0.0054
                self.Ne = 0.00133
                self.Mg = 0.000705
                self.Si = 0.000683
                self.Fe = 0.00129
        else:
            print(f'Error: {obj} is an invalid object for util/astrometry.py/Measurements; '
                  f'line {inspect.currentframe().f_lineno}')
            exit(1)


class LMCTangent(BaseCoordinateFrame):
    """
    Cartesian coordinate system centered on the LMC.
    Axes are aligned with ICRS.
    Origin is the LMC center defined above.
    """
    default_representation = CartesianRepresentation
    default_differential = CartesianDifferential

    LMC = Attribute(default=None)  # pass your LMC object here

class LMCDisk(BaseCoordinateFrame):
    default_representation = CartesianRepresentation
    default_differential = CartesianDifferential

    LMC = Attribute(default=None)  # pass your LMC object here

def rotationMatrix(alpha0, delta0):
    # R : ra,dec --> projected coordinates (cos\rho = e_los, tan\phi = e_north / e_east)
    # R^-1 = R^T : projected coordinates --> ra,dec
    R = np.array([
            [-sin(alpha0), cos(alpha0), 0],
            [-sin(delta0) * cos(alpha0), -sin(delta0) * sin(alpha0), cos(delta0)],
            [cos(delta0) * cos(alpha0), cos(delta0) * sin(alpha0), sin(delta0)]
        ])
    return R

def v_lmc():
    # V : pm_ra*, pm_dec, vlsr --> (vx, vy, vz)_equitorial
    alpha, delta = rad(LMC.ra), rad(LMC.dec)
    v_alpha = (LMC.mu_alpha * (u.mas / u.yr) * LMC.distance * u.kpc)
    v_delta = (LMC.mu_delta * (u.mas / u.yr) * LMC.distance * u.kpc)
    V = rotationMatrix(alpha, delta).T
    v = np.array([
        v_alpha.to(u.km / u.s, equivalencies=u.dimensionless_angles()).value,
        v_delta.to(u.km / u.s, equivalencies=u.dimensionless_angles()).value,
        LMC.vlsr  # km/s
    ])
    # vx = -v_alpha*sin(alpha) - v_delta*sin(delta)*cos(alpha) + rv*cos(delta)*cos(alpha)
    # vy = v_alpha*cos(alpha) - v_delta*sin(delta)*sin(alpha) + rv*cos(delta)*sin(alpha)
    # vz = v_delta*cos(delta) + rv*sin(delta)
    return V @ v


# ICRS -> LMCCentric
@frame_transform_graph.transform(FunctionTransform, ICRS, LMCTangent)
def icrs_to_lmc(icrs_coord, lmc_coord):
    LMC = LMCTangent.LMC  # <-- pulled from the frame instance
    alpha, delta = rad(icrs_coord.ra.value), rad(icrs_coord.dec.value)
    R = rotationMatrix(rad(LMC.ra), rad(LMC.dec))  # 3x3 rotation
    v = np.array([
            cos(delta) * cos(alpha),
            cos(delta) * sin(alpha),
            sin(delta)
    ])
    v_proj = icrs_coord.distance * (R @ v)  # D * R @ v
    v_proj[2] -= LMC.distance * u.kpc

    if icrs_coord.has_data and icrs_coord.data.differentials:  # if velocity is given
        cart = icrs_coord.cartesian
        vel = cart.differentials['s'].d_xyz.to(u.km / u.s)
        v_dot = (R @ (vel.value - v_lmc())) * u.km / u.s
        return LMCTangent(CartesianRepresentation(v_proj, differentials={'s': CartesianDifferential(v_dot)}))

    return LMCTangent(CartesianRepresentation(v_proj), LMC=LMCTangent.LMC)


@frame_transform_graph.transform(FunctionTransform, LMCTangent, ICRS)
def lmc_to_icrs(lmc_coord, icrs_frame):
    LMC = LMCTangent.LMC
    v = np.array([
        lmc_coord.x.to(u.kpc).value,
        lmc_coord.y.to(u.kpc).value,
        (lmc_coord.z + LMC.distance * u.kpc).to(u.kpc).value
    ])
    R = rotationMatrix(rad(LMC.ra), rad(LMC.dec))  # 3x3 rotation
    w = R.T @ v  # cartesian equatorial

    if lmc_coord.has_data and lmc_coord.data.differentials:  # if velocity is given
        vel = lmc_coord.data.differentials['s'].d_xyz.to(u.km / u.s)
        w_dot = (R.T @ vel.value + v_lmc()) * u.km / u.s
        return ICRS(CartesianRepresentation(w * u.kpc, differentials={'s': CartesianDifferential(w_dot)}))

    return ICRS(CartesianRepresentation(w * u.kpc), LMC=LMCTangent.LMC)


@frame_transform_graph.transform(FunctionTransform, LMCTangent, LMCDisk)  # @ is kinda like a wrapper
def lmc_to_disk(lmc_coord, disk_frame):
    LMC = LMCDisk.LMC
    # tangent plane (x,y,z) --> disk plane (x',y',z')
    i, theta = rad(LMC.inclination), rad(-90) + rad(LMC.nodes)
    r_tangent = lmc_coord.cartesian.xyz.to(u.kpc).value
    r_disk = R_x(i) @ R_z(theta) @ r_tangent

    if lmc_coord.has_data and lmc_coord.data.differentials:  # if velocity is given
        vel = lmc_coord.data.differentials['s'].d_xyz.to(u.km / u.s)
        v_disk = ((R_x(i) @ R_z(theta)) @ vel.value) * u.km / u.s
        return LMCDisk(CartesianRepresentation(r_disk * u.kpc, differentials={'s': CartesianDifferential(v_disk)}))

    return LMCDisk(CartesianRepresentation(r_disk * u.kpc), LMC=LMCDisk.LMC)


@frame_transform_graph.transform(FunctionTransform, LMCDisk, LMCTangent)
def disk_to_lmc(disk_coord, lmc_frame):
    LMC = LMCDisk.LMC
    # disk plane (x',y',z') --> tangent plane (x,y,z)
    i, theta = rad(LMC.inclination), rad(-90) + rad(LMC.nodes)
    r_disk = disk_coord.cartesian.xyz.to(u.kpc).value
    r_tan = R_z(-theta) @ R_x(-i) @ r_disk

    if disk_coord.has_data and disk_coord.data.differentials:  # if velocity is given
        vel = disk_coord.data.differentials['s'].d_xyz.to(u.km / u.s)
        v_tan = ((R_z(-theta) @ R_x(-i)) @ vel.value) * u.km / u.s
        return LMCTangent(CartesianRepresentation(r_tan * u.kpc, differentials={'s': CartesianDifferential(v_tan)}))

    return LMCTangent(CartesianRepresentation(r_tan * u.kpc), LMC=LMCDisk.LMC)


# ---------------------------------------------
# LMC = Measurements('LMC', 'disk')
# ---------------------------------------------



