# -----------------------------------------------------------------------------------
# Title: Sphere models for gravity and magnetic methods
# Description: Revised and expanded routines for solid spheres.
# Author: Nelson Ribeiro Filho
# -----------------------------------------------------------------------------------

from __future__ import division
import numpy

try:
    from . import auxiliars
except Exception:  # pragma: no cover
    try:
        import auxiliars
    except Exception:
        auxiliars = None

try:
    from . import constants
except Exception:  # pragma: no cover
    try:
        import constants
    except Exception:
        constants = None


# -----------------------------------------------------------------------------
# CONSTANTS AND INTERNAL AUXILIARY ROUTINES
# -----------------------------------------------------------------------------

G = getattr(constants, "G", 6.673e-11)
SI2MGAL = getattr(constants, "si2mGal", 100000.0)
CM = getattr(constants, "cm", 1.0e-7)
T2NT = getattr(constants, "T2nT", 1.0e9)
EPS = getattr(constants, "EPS", 1.0e-12)


def _as_array(a):
    '''Return input as a numpy array without unnecessary copying.'''
    return numpy.asarray(a, dtype=float)


def _check_observation_points(x, y, z=None):
    '''
    Check and broadcast observation coordinates.

    Inputs:
    x, y - numpy arrays - observation coordinates
    z - float or numpy array - observation level

    Outputs:
    x, y, z - numpy arrays with compatible shapes
    '''
    x = _as_array(x)
    y = _as_array(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape!")

    if z is None:
        return x, y, None

    if numpy.isscalar(z):
        z = numpy.zeros_like(x) + float(z)
    else:
        z = _as_array(z)
        if z.shape != x.shape:
            raise ValueError("z must be a scalar or have the same shape as x and y!")

    return x, y, z


def _dircos(inc, dec):
    '''
    Return direction cosines for inclination and declination in degrees.
    This function preserves compatibility with auxiliars.my_dircos.
    '''
    if auxiliars is not None and hasattr(auxiliars, "my_dircos"):
        return auxiliars.my_dircos(inc, dec)

    inc_rad = numpy.deg2rad(inc)
    dec_rad = numpy.deg2rad(dec)

    fx = numpy.cos(inc_rad)*numpy.cos(dec_rad)
    fy = numpy.cos(inc_rad)*numpy.sin(dec_rad)
    fz = numpy.sin(inc_rad)

    return fx, fy, fz


def _regional(field, inc, dec):
    '''Return regional field components.'''
    if auxiliars is not None and hasattr(auxiliars, "my_regional"):
        return auxiliars.my_regional(field, inc, dec)

    fx, fy, fz = _dircos(inc, dec)
    return field*fx, field*fy, field*fz


def _sphere_parameters(sphere, default_property=None):
    '''
    Return sphere parameters as xe, ye, ze, radius and optional property.

    The sphere format can be:
    [xe, ye, ze, radius]
    [xe, ye, ze, radius, property]
    '''
    if len(sphere) < 4:
        raise ValueError("sphere must contain at least [xe, ye, ze, radius]!")

    xe = float(sphere[0])
    ye = float(sphere[1])
    ze = float(sphere[2])
    radius = float(sphere[3])

    if radius <= 0.0:
        raise ValueError("sphere radius must be positive!")

    if len(sphere) >= 5:
        prop = float(sphere[4])
    else:
        prop = default_property

    return xe, ye, ze, radius, prop


def my_sphere_volume(sphere):
    '''
    Return the volume of a solid sphere.

    Input:
    sphere - list/array - [xe, ye, ze, radius, optional_property]

    Output:
    volume - float - sphere volume
    '''
    _, _, _, radius, _ = _sphere_parameters(sphere)
    return (4.0/3.0)*numpy.pi*radius**3


def my_sphere_distance_components(x, y, z, sphere):
    '''
    Return distance components from the sphere center to observation points.

    Inputs:
    x, y, z - numpy arrays - observation coordinates
    sphere - list/array - [xe, ye, ze, radius]

    Outputs:
    rx, ry, rz, r2, r - numpy arrays
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, _ = _sphere_parameters(sphere)

    rx = x - xe
    ry = y - ye
    rz = z - ze

    r2 = rx**2 + ry**2 + rz**2
    r = numpy.sqrt(r2)

    if numpy.any(r2 == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    return rx, ry, rz, r2, r


# -----------------------------------------------------------------------------
# MAGNETIC FIELD OF ONE SPHERE
# -----------------------------------------------------------------------------

def my_sphere_bx(x, y, z, sphere, mag=None, incs=0.0, decs=0.0):
    '''
    Compute the X component of magnetic induction caused by a uniformly
    magnetized solid sphere.

    Inputs:
    x, y, z - numpy arrays - observation points in meters
    sphere - list/array - [xe, ye, ze, radius] or [xe, ye, ze, radius, mag]
    mag - float/None - magnetization intensity. If None, sphere[4] is used.
    incs, decs - floats - source inclination and declination in degrees

    Output:
    bx - numpy array - X component in nT
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_mag = _sphere_parameters(sphere, mag)

    if mag is None:
        mag = sphere_mag
    if mag is None:
        raise ValueError("mag must be given or stored in sphere[4]!")

    rx = x - xe
    ry = y - ye
    rz = z - ze

    r2 = rx**2 + ry**2 + rz**2
    if numpy.any(r2 == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    mx, my, mz = _dircos(incs, decs)
    dot = rx*mx + ry*my + rz*mz
    moment = (4.0*numpy.pi*radius**3*mag)/3.0

    bx = moment*(3.0*dot*rx - r2*mx)/(r2**2.5)
    bx *= CM*T2NT

    return bx


def my_sphere_by(x, y, z, sphere, mag=None, incs=0.0, decs=0.0):
    '''
    Compute the Y component of magnetic induction caused by a uniformly
    magnetized solid sphere.

    Output is in nT.
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_mag = _sphere_parameters(sphere, mag)

    if mag is None:
        mag = sphere_mag
    if mag is None:
        raise ValueError("mag must be given or stored in sphere[4]!")

    rx = x - xe
    ry = y - ye
    rz = z - ze

    r2 = rx**2 + ry**2 + rz**2
    if numpy.any(r2 == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    mx, my, mz = _dircos(incs, decs)
    dot = rx*mx + ry*my + rz*mz
    moment = (4.0*numpy.pi*radius**3*mag)/3.0

    by = moment*(3.0*dot*ry - r2*my)/(r2**2.5)
    by *= CM*T2NT

    return by


def my_sphere_bz(x, y, z, sphere, mag=None, incs=0.0, decs=0.0):
    '''
    Compute the Z component of magnetic induction caused by a uniformly
    magnetized solid sphere.

    Output is in nT.
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_mag = _sphere_parameters(sphere, mag)

    if mag is None:
        mag = sphere_mag
    if mag is None:
        raise ValueError("mag must be given or stored in sphere[4]!")

    rx = x - xe
    ry = y - ye
    rz = z - ze

    r2 = rx**2 + ry**2 + rz**2
    if numpy.any(r2 == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    mx, my, mz = _dircos(incs, decs)
    dot = rx*mx + ry*my + rz*mz
    moment = (4.0*numpy.pi*radius**3*mag)/3.0

    bz = moment*(3.0*dot*rz - r2*mz)/(r2**2.5)
    bz *= CM*T2NT

    return bz


def my_sphere_bxyz(x, y, z, sphere, mag=None, incs=0.0, decs=0.0):
    '''
    Compute the three magnetic components Bx, By and Bz for one sphere.

    Outputs:
    bx, by, bz - numpy arrays - magnetic components in nT
    '''
    bx = my_sphere_bx(x, y, z, sphere, mag=mag, incs=incs, decs=decs)
    by = my_sphere_by(x, y, z, sphere, mag=mag, incs=incs, decs=decs)
    bz = my_sphere_bz(x, y, z, sphere, mag=mag, incs=incs, decs=decs)
    return bx, by, bz


def my_tfa(x, y, z, sphere, mag=None, inc=0.0, dec=0.0, incs=None, decs=None):
    '''
    Compute the approximated total field anomaly produced by a solid sphere.

    Inputs:
    x, y, z - numpy arrays - observation points
    sphere - list/array - [xe, ye, ze, radius] or [xe, ye, ze, radius, mag]
    mag - float/None - magnetization intensity. If None, sphere[4] is used.
    inc, dec - floats - regional field inclination and declination in degrees
    incs, decs - floats/None - source magnetization direction. If None, induced
                  magnetization is assumed.

    Output:
    tfa - numpy array - total field anomaly in nT
    '''
    _check_observation_points(x, y, z)

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    fx, fy, fz = _dircos(inc, dec)

    bx, by, bz = my_sphere_bxyz(x, y, z, sphere, mag=mag, incs=incs, decs=decs)
    tfa = fx*bx + fy*by + fz*bz

    return tfa


def my_sphere_tf(x, y, z, sphere, mag=None, field=50000.0,
                 inc=0.0, dec=0.0, incs=None, decs=None):
    '''
    Compute the exact total field anomaly from the vector sum between the
    regional field and the anomalous magnetic field of the sphere.

    Inputs:
    x, y, z - numpy arrays - observation points
    sphere - list/array - [xe, ye, ze, radius] or [xe, ye, ze, radius, mag]
    mag - float/None - magnetization intensity. If None, sphere[4] is used.
    field - float - regional field intensity in nT
    inc, dec - floats - regional field inclination and declination in degrees
    incs, decs - floats/None - source magnetization direction

    Output:
    tf - numpy array - total field anomaly in nT
    '''
    _check_observation_points(x, y, z)

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    fx, fy, fz = _regional(field, inc, dec)

    bx, by, bz = my_sphere_bxyz(x, y, z, sphere, mag=mag, incs=incs, decs=decs)

    bx_total = bx + fx
    by_total = by + fy
    bz_total = bz + fz

    tf = numpy.sqrt(bx_total**2 + by_total**2 + bz_total**2) - field

    return tf


# -----------------------------------------------------------------------------
# GRAVITY FIELD OF ONE SPHERE
# -----------------------------------------------------------------------------

def my_sphere_potential(x, y, z, sphere, rho=None):
    '''
    Compute the gravitational potential produced by a homogeneous solid sphere.

    Inputs:
    x, y, z - numpy arrays - observation points in meters
    sphere - list/array - [xe, ye, ze, radius] or [xe, ye, ze, radius, rho]
    rho - float/None - density in g/cm^3. If None, sphere[4] is used.

    Output:
    potential - numpy array - gravitational potential in SI units
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_rho = _sphere_parameters(sphere, rho)

    if rho is None:
        rho = sphere_rho
    if rho is None:
        raise ValueError("rho must be given or stored in sphere[4]!")

    rho_si = rho*1000.0
    volume = (4.0/3.0)*numpy.pi*radius**3
    mass = rho_si*volume

    r = numpy.sqrt((x - xe)**2 + (y - ye)**2 + (z - ze)**2)
    if numpy.any(r == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    potential = G*mass/r

    return potential


def my_sphere_gx(x, y, z, sphere, rho=None):
    '''
    Compute the X component of gravitational attraction of a solid sphere.

    Output is in mGal.
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_rho = _sphere_parameters(sphere, rho)

    if rho is None:
        rho = sphere_rho
    if rho is None:
        raise ValueError("rho must be given or stored in sphere[4]!")

    rho_si = rho*1000.0
    volume = (4.0/3.0)*numpy.pi*radius**3

    dx = xe - x
    dy = ye - y
    dz = ze - z
    r = numpy.sqrt(dx**2 + dy**2 + dz**2)

    if numpy.any(r == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    gx = volume*rho_si*dx/(r**3)
    gx *= G*SI2MGAL

    return gx


def my_sphere_gy(x, y, z, sphere, rho=None):
    '''
    Compute the Y component of gravitational attraction of a solid sphere.

    Output is in mGal.
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_rho = _sphere_parameters(sphere, rho)

    if rho is None:
        rho = sphere_rho
    if rho is None:
        raise ValueError("rho must be given or stored in sphere[4]!")

    rho_si = rho*1000.0
    volume = (4.0/3.0)*numpy.pi*radius**3

    dx = xe - x
    dy = ye - y
    dz = ze - z
    r = numpy.sqrt(dx**2 + dy**2 + dz**2)

    if numpy.any(r == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    gy = volume*rho_si*dy/(r**3)
    gy *= G*SI2MGAL

    return gy


def my_sphere_gz(x, y, z, sphere, rho=None):
    '''
    Compute the vertical component of gravitational attraction of a solid sphere.

    Inputs:
    x, y, z - numpy arrays - observation points in meters
    sphere - list/array - [xe, ye, ze, radius] or [xe, ye, ze, radius, rho]
    rho - float/None - density in g/cm^3. If None, sphere[4] is used.

    Output:
    gz - numpy array - vertical component in mGal
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_rho = _sphere_parameters(sphere, rho)

    if rho is None:
        rho = sphere_rho
    if rho is None:
        raise ValueError("rho must be given or stored in sphere[4]!")

    rho_si = rho*1000.0
    volume = (4.0/3.0)*numpy.pi*radius**3

    dx = xe - x
    dy = ye - y
    dz = ze - z
    r = numpy.sqrt(dx**2 + dy**2 + dz**2)

    if numpy.any(r == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    gz = volume*rho_si*dz/(r**3)
    gz *= G*SI2MGAL

    return gz


def my_sphere_gxyz(x, y, z, sphere, rho=None):
    '''
    Compute gx, gy and gz for one solid sphere.

    Outputs are in mGal.
    '''
    gx = my_sphere_gx(x, y, z, sphere, rho=rho)
    gy = my_sphere_gy(x, y, z, sphere, rho=rho)
    gz = my_sphere_gz(x, y, z, sphere, rho=rho)
    return gx, gy, gz


def my_sphere_gravity_gradient(x, y, z, sphere, rho=None, component="zz"):
    '''
    Compute gravity-gradient tensor components of a homogeneous sphere.

    Inputs:
    component - string - one of 'xx', 'xy', 'xz', 'yy', 'yz', 'zz'

    Output:
    tensor_component - numpy array - gradient component in mGal/m
    '''
    x, y, z = _check_observation_points(x, y, z)
    xe, ye, ze, radius, sphere_rho = _sphere_parameters(sphere, rho)

    if rho is None:
        rho = sphere_rho
    if rho is None:
        raise ValueError("rho must be given or stored in sphere[4]!")

    rho_si = rho*1000.0
    volume = (4.0/3.0)*numpy.pi*radius**3

    # Components from source to observation for derivatives of 1/r.
    rx = x - xe
    ry = y - ye
    rz = z - ze
    r2 = rx**2 + ry**2 + rz**2
    r = numpy.sqrt(r2)

    if numpy.any(r2 == 0.0):
        raise ZeroDivisionError("Observation point coincides with sphere center!")

    comp = component.lower()
    if comp == "xx":
        kernel = (3.0*rx*rx/r**5) - (1.0/r**3)
    elif comp == "xy":
        kernel = 3.0*rx*ry/r**5
    elif comp == "xz":
        kernel = 3.0*rx*rz/r**5
    elif comp == "yy":
        kernel = (3.0*ry*ry/r**5) - (1.0/r**3)
    elif comp == "yz":
        kernel = 3.0*ry*rz/r**5
    elif comp == "zz":
        kernel = (3.0*rz*rz/r**5) - (1.0/r**3)
    else:
        raise ValueError("component must be 'xx', 'xy', 'xz', 'yy', 'yz' or 'zz'!")

    result = G*rho_si*volume*kernel*SI2MGAL

    return result


def my_sphere_tensor(x, y, z, sphere, rho=None):
    '''
    Compute the complete gravity-gradient tensor for one sphere.

    Outputs:
    gxx, gxy, gxz, gyy, gyz, gzz - numpy arrays in mGal/m
    '''
    gxx = my_sphere_gravity_gradient(x, y, z, sphere, rho=rho, component="xx")
    gxy = my_sphere_gravity_gradient(x, y, z, sphere, rho=rho, component="xy")
    gxz = my_sphere_gravity_gradient(x, y, z, sphere, rho=rho, component="xz")
    gyy = my_sphere_gravity_gradient(x, y, z, sphere, rho=rho, component="yy")
    gyz = my_sphere_gravity_gradient(x, y, z, sphere, rho=rho, component="yz")
    gzz = my_sphere_gravity_gradient(x, y, z, sphere, rho=rho, component="zz")

    return gxx, gxy, gxz, gyy, gyz, gzz


# -----------------------------------------------------------------------------
# MULTIPLE SPHERES
# -----------------------------------------------------------------------------

def my_spheres_gz(x, y, z, spheres, rho=None):
    '''
    Compute the total gz produced by a list of solid spheres.

    Inputs:
    spheres - list - each element is [xe, ye, ze, radius] or
              [xe, ye, ze, radius, rho]
    rho - float/array/None - density. If None, each sphere[4] is used.

    Output:
    gz_total - numpy array - total gz in mGal
    '''
    x, y, z = _check_observation_points(x, y, z)
    total = numpy.zeros_like(x, dtype=float)

    for i, sph in enumerate(spheres):
        if rho is None:
            rhoi = None
        elif numpy.isscalar(rho):
            rhoi = rho
        else:
            rhoi = rho[i]
        total += my_sphere_gz(x, y, z, sph, rho=rhoi)

    return total


def my_spheres_gx(x, y, z, spheres, rho=None):
    '''Compute total gx produced by a list of solid spheres. Output in mGal.'''
    x, y, z = _check_observation_points(x, y, z)
    total = numpy.zeros_like(x, dtype=float)

    for i, sph in enumerate(spheres):
        if rho is None:
            rhoi = None
        elif numpy.isscalar(rho):
            rhoi = rho
        else:
            rhoi = rho[i]
        total += my_sphere_gx(x, y, z, sph, rho=rhoi)

    return total


def my_spheres_gy(x, y, z, spheres, rho=None):
    '''Compute total gy produced by a list of solid spheres. Output in mGal.'''
    x, y, z = _check_observation_points(x, y, z)
    total = numpy.zeros_like(x, dtype=float)

    for i, sph in enumerate(spheres):
        if rho is None:
            rhoi = None
        elif numpy.isscalar(rho):
            rhoi = rho
        else:
            rhoi = rho[i]
        total += my_sphere_gy(x, y, z, sph, rho=rhoi)

    return total


def my_spheres_gxyz(x, y, z, spheres, rho=None):
    '''Compute total gx, gy and gz produced by a list of spheres.'''
    gx = my_spheres_gx(x, y, z, spheres, rho=rho)
    gy = my_spheres_gy(x, y, z, spheres, rho=rho)
    gz = my_spheres_gz(x, y, z, spheres, rho=rho)
    return gx, gy, gz


def my_spheres_tfa(x, y, z, spheres, mag=None, inc=0.0, dec=0.0,
                   incs=None, decs=None):
    '''
    Compute the total approximated magnetic total field anomaly produced by
    a list of spheres.

    Output is in nT.
    '''
    x, y, z = _check_observation_points(x, y, z)
    total = numpy.zeros_like(x, dtype=float)

    for i, sph in enumerate(spheres):
        if mag is None:
            magi = None
        elif numpy.isscalar(mag):
            magi = mag
        else:
            magi = mag[i]
        total += my_tfa(x, y, z, sph, mag=magi, inc=inc, dec=dec,
                        incs=incs, decs=decs)

    return total


def my_spheres_tf(x, y, z, spheres, mag=None, field=50000.0,
                  inc=0.0, dec=0.0, incs=None, decs=None):
    '''
    Compute the total exact magnetic total field anomaly produced by a list
    of spheres.

    Output is in nT.
    '''
    x, y, z = _check_observation_points(x, y, z)
    total = numpy.zeros_like(x, dtype=float)

    for i, sph in enumerate(spheres):
        if mag is None:
            magi = None
        elif numpy.isscalar(mag):
            magi = mag
        else:
            magi = mag[i]
        total += my_sphere_tf(x, y, z, sph, mag=magi, field=field,
                              inc=inc, dec=dec, incs=incs, decs=decs)

    return total


def my_spheres_bxyz(x, y, z, spheres, mag=None, incs=0.0, decs=0.0):
    '''Compute total Bx, By and Bz produced by a list of spheres.'''
    x, y, z = _check_observation_points(x, y, z)
    bx = numpy.zeros_like(x, dtype=float)
    by = numpy.zeros_like(x, dtype=float)
    bz = numpy.zeros_like(x, dtype=float)

    for i, sph in enumerate(spheres):
        if mag is None:
            magi = None
        elif numpy.isscalar(mag):
            magi = mag
        else:
            magi = mag[i]
        bxi, byi, bzi = my_sphere_bxyz(x, y, z, sph, mag=magi,
                                       incs=incs, decs=decs)
        bx += bxi
        by += byi
        bz += bzi

    return bx, by, bz


# -----------------------------------------------------------------------------
# MODEL BUILDERS
# -----------------------------------------------------------------------------

def my_sphere_model(xc, yc, zc, radius, physical_property=None):
    '''
    Build one sphere model.

    Inputs:
    xc, yc, zc - floats - sphere center coordinates
    radius - float - sphere radius
    physical_property - float/None - density or magnetization

    Output:
    sphere - list
    '''
    if radius <= 0.0:
        raise ValueError("radius must be positive!")

    if physical_property is None:
        return [float(xc), float(yc), float(zc), float(radius)]

    return [float(xc), float(yc), float(zc), float(radius), float(physical_property)]


def my_build_spheres_from_centers(xc, yc, zc, radius, physical_property=None):
    '''
    Build a list of spheres from center coordinates.

    Inputs:
    xc, yc, zc - arrays - center coordinates
    radius - float or array - sphere radius
    physical_property - float, array or None - density or magnetization

    Output:
    spheres - list of sphere models
    '''
    xc = numpy.asarray(xc, dtype=float).ravel()
    yc = numpy.asarray(yc, dtype=float).ravel()
    zc = numpy.asarray(zc, dtype=float).ravel()

    if xc.shape != yc.shape or xc.shape != zc.shape:
        raise ValueError("xc, yc and zc must have the same size!")

    n = xc.size

    if numpy.isscalar(radius):
        radius_values = numpy.zeros(n) + float(radius)
    else:
        radius_values = numpy.asarray(radius, dtype=float).ravel()
        if radius_values.size != n:
            raise ValueError("radius must be scalar or have the same size as centers!")

    if physical_property is None:
        prop_values = [None]*n
    elif numpy.isscalar(physical_property):
        prop_values = numpy.zeros(n) + float(physical_property)
    else:
        prop_values = numpy.asarray(physical_property, dtype=float).ravel()
        if prop_values.size != n:
            raise ValueError("physical_property must be scalar or have the same size as centers!")

    spheres = []
    for i in range(n):
        spheres.append(my_sphere_model(xc[i], yc[i], zc[i], radius_values[i], prop_values[i]))

    return spheres


# -----------------------------------------------------------------------------
# COMPATIBILITY ALIASES
# -----------------------------------------------------------------------------

sphere_bx = my_sphere_bx
sphere_by = my_sphere_by
sphere_bz = my_sphere_bz
sphere_tf = my_sphere_tf
sphere_tfa = my_tfa
sphere_gx = my_sphere_gx
sphere_gy = my_sphere_gy
sphere_gz = my_sphere_gz
sphere_potential = my_sphere_potential

my_bx_sphere = my_sphere_bx
my_by_sphere = my_sphere_by
my_bz_sphere = my_sphere_bz
my_tf_sphere = my_sphere_tf
my_gx_sphere = my_sphere_gx
my_gy_sphere = my_sphere_gy
my_gz_sphere = my_sphere_gz

my_spheres_g = my_spheres_gxyz
my_spheres_mag = my_spheres_bxyz
