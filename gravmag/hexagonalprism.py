# -*- coding: utf-8 -*-
"""
Hexagonal prism modelling functions for gravity and magnetic applications.

This module follows the same style used in the revised grav-mag package:
- functions start with the prefix ``my_``;
- vertical coordinate is positive downward;
- density is expected in g/cm^3;
- magnetization is expected in A/m;
- gravity components are returned in mGal;
- magnetic components and total-field anomaly are returned in nT.

A vertical hexagonal prism is represented by

    hexprism = [xc, yc, top, bottom, radius, physical_property, rotation]

where ``radius`` is the circumradius of the regular hexagon, i.e., the
horizontal distance from the center to each vertex. The optional ``rotation``
is given in degrees. If absent, it is assumed to be zero.

The calculations are performed by discretizing the hexagonal prism into small
volume elements and summing the contribution of each element. This numerical
strategy is robust, general and compatible with later polygonal-prism routines.
"""

from __future__ import division

import numpy

try:
    from . import auxiliars
except Exception:  # pragma: no cover - compatibility fallback
    try:
        import auxiliars
    except Exception:  # pragma: no cover
        auxiliars = None

try:
    from . import constants
except Exception:  # pragma: no cover - compatibility fallback
    try:
        import constants
    except Exception:  # pragma: no cover
        constants = None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

if constants is not None and hasattr(constants, "G"):
    G_CONST = constants.G
else:
    G_CONST = 6.673e-11

if constants is not None and hasattr(constants, "si2mGal"):
    SI2MGAL = constants.si2mGal
else:
    SI2MGAL = 100000.0

if constants is not None and hasattr(constants, "cm"):
    CM = constants.cm
else:
    CM = 1.0e-7

if constants is not None and hasattr(constants, "T2nT"):
    T2NT = constants.T2nT
else:
    T2NT = 1.0e9

EPS = 1.0e-12


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _as_array_like_z(z, x):
    """
    Internal helper that converts a scalar z to an array compatible with x.
    """

    if numpy.isscalar(z):
        return z*numpy.ones_like(x, dtype=float)

    z = numpy.asarray(z, dtype=float)

    if z.shape != x.shape:
        raise ValueError("z must be scalar or have the same shape as x!")

    return z


def _check_observation_points(x, y, z):
    """
    Internal helper that checks observation coordinates.
    """

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape!")

    z = _as_array_like_z(z, x)

    return x, y, z


def _get_physical_property(hexprism, value, index=5, name="physical_property"):
    """
    Internal helper used to retrieve density or magnetization.
    """

    if value is not None:
        return value

    if len(hexprism) > index:
        return hexprism[index]

    raise ValueError("%s must be supplied or stored in the model!" % name)


def _direction_cosines(inc, dec):
    """
    Internal direction cosines using inclination and declination in degrees.
    The x axis is North, y is East and z is positive downward.
    """

    if auxiliars is not None:
        if hasattr(auxiliars, "my_dircos"):
            return auxiliars.my_dircos(inc, dec)
        if hasattr(auxiliars, "my_regional"):
            return auxiliars.my_regional(1.0, inc, dec)

    inc_rad = numpy.deg2rad(inc)
    dec_rad = numpy.deg2rad(dec)

    mx = numpy.cos(inc_rad)*numpy.cos(dec_rad)
    my = numpy.cos(inc_rad)*numpy.sin(dec_rad)
    mz = numpy.sin(inc_rad)

    return mx, my, mz


def _safe_radius(rx, ry, rz):
    """
    Internal safe distance used to avoid singularities.
    """

    r2 = rx**2 + ry**2 + rz**2
    r2 = numpy.where(r2 < EPS, EPS, r2)
    r = numpy.sqrt(r2)

    return r, r2


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def my_hexagonal_prism_model(xc, yc, top, bottom, radius,
                             physical_property=None, rotation=0.0):
    """
    This function creates a vertical hexagonal prism model.

    Inputs:
    xc, yc - floats - horizontal center of the prism
    top, bottom - floats - top and bottom depths, positive downward
    radius - float - circumradius of the hexagon
    physical_property - float/None - density or magnetization
    rotation - float - rotation angle in degrees

    Output:
    hexprism - list - [xc, yc, top, bottom, radius, property, rotation]
    """

    if radius <= 0.0:
        raise ValueError("radius must be positive!")

    if bottom <= top:
        raise ValueError("bottom must be greater than top!")

    if physical_property is None:
        return [float(xc), float(yc), float(top), float(bottom),
                float(radius), None, float(rotation)]

    return [float(xc), float(yc), float(top), float(bottom),
            float(radius), float(physical_property), float(rotation)]


def my_hexagonal_vertices(hexprism):
    """
    This function returns the six vertices of a regular hexagonal prism.

    Input:
    hexprism - list - [xc, yc, top, bottom, radius, property, rotation]

    Outputs:
    xv, yv - numpy arrays - polygon vertices in counterclockwise order
    """

    xc = hexprism[0]
    yc = hexprism[1]
    radius = hexprism[4]

    rotation = 0.0
    if len(hexprism) > 6 and hexprism[6] is not None:
        rotation = hexprism[6]

    angles = numpy.deg2rad(rotation) + numpy.arange(6)*numpy.pi/3.0

    xv = xc + radius*numpy.cos(angles)
    yv = yc + radius*numpy.sin(angles)

    return xv, yv


def my_hexagonal_area(hexprism):
    """
    This function calculates the horizontal area of a regular hexagon.

    Input:
    hexprism - list - hexagonal prism model

    Output:
    area - float - horizontal area
    """

    radius = hexprism[4]
    area = (3.0*numpy.sqrt(3.0)/2.0)*radius**2

    return area


def my_hexagonal_volume(hexprism):
    """
    This function calculates the volume of a regular hexagonal prism.

    Input:
    hexprism - list - hexagonal prism model

    Output:
    volume - float - prism volume
    """

    top = hexprism[2]
    bottom = hexprism[3]

    return my_hexagonal_area(hexprism)*(bottom - top)


def my_hexagonal_center(hexprism):
    """
    This function returns the geometric center of the hexagonal prism.

    Input:
    hexprism - list - hexagonal prism model

    Output:
    center - tuple - (xc, yc, zc)
    """

    xc, yc, top, bottom = hexprism[0], hexprism[1], hexprism[2], hexprism[3]
    zc = 0.5*(top + bottom)

    return xc, yc, zc


def my_point_in_polygon(x, y, xv, yv):
    """
    This function tests if points are inside a polygon using the ray method.

    Inputs:
    x, y - numpy arrays - points to be tested
    xv, yv - numpy arrays - polygon vertices

    Output:
    inside - numpy boolean array - True for points inside the polygon
    """

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    xv = numpy.asarray(xv, dtype=float)
    yv = numpy.asarray(yv, dtype=float)

    inside = numpy.zeros_like(x, dtype=bool)
    nvert = xv.size
    j = nvert - 1

    for i in range(nvert):
        cond = ((yv[i] > y) != (yv[j] > y))
        x_intersect = (xv[j] - xv[i])*(y - yv[i])/(yv[j] - yv[i] + EPS) + xv[i]
        inside ^= cond & (x < x_intersect)
        j = i

    return inside


def my_discretize_hexagonal_prism(hexprism, spacing=None, nz=1,
                                  nx=None, ny=None, keep_property=True):
    """
    This function discretizes a hexagonal prism into small volume elements.

    Inputs:
    hexprism - list - [xc, yc, top, bottom, radius, property, rotation]
    spacing - float/None - horizontal cell size. If None, it is estimated
    nz - int - number of vertical layers
    nx, ny - int/None - number of cells in x and y if spacing is not supplied
    keep_property - bool - if True, appends the physical property to each element

    Outputs:
    elements - numpy array - columns [x, y, z, volume] or [x, y, z, volume, property]
    """

    if nz < 1:
        raise ValueError("nz must be at least 1!")

    xv, yv = my_hexagonal_vertices(hexprism)
    xmin, xmax = numpy.min(xv), numpy.max(xv)
    ymin, ymax = numpy.min(yv), numpy.max(yv)

    radius = hexprism[4]

    if spacing is None:
        if nx is None:
            nx = 35
        if ny is None:
            ny = 35
        xs = numpy.linspace(xmin, xmax, int(nx))
        ys = numpy.linspace(ymin, ymax, int(ny))
        dx = numpy.mean(numpy.diff(xs)) if xs.size > 1 else 2.0*radius
        dy = numpy.mean(numpy.diff(ys)) if ys.size > 1 else 2.0*radius
    else:
        if spacing <= 0.0:
            raise ValueError("spacing must be positive!")
        xs = numpy.arange(xmin + 0.5*spacing, xmax, spacing)
        ys = numpy.arange(ymin + 0.5*spacing, ymax, spacing)
        dx = spacing
        dy = spacing

    xg, yg = numpy.meshgrid(xs, ys)
    inside = my_point_in_polygon(xg, yg, xv, yv)

    xh = xg[inside]
    yh = yg[inside]

    if xh.size == 0:
        raise ValueError("No discretization points were created inside the hexagon!")

    top = hexprism[2]
    bottom = hexprism[3]
    dz = (bottom - top)/float(nz)
    zc = top + (numpy.arange(nz) + 0.5)*dz

    cell_volume = dx*dy*dz

    elements = []
    for zk in zc:
        volume = cell_volume*numpy.ones_like(xh)
        zarr = zk*numpy.ones_like(xh)
        if keep_property is True:
            prop = hexprism[5]*numpy.ones_like(xh)
            block = numpy.vstack([xh, yh, zarr, volume, prop]).T
        else:
            block = numpy.vstack([xh, yh, zarr, volume]).T
        elements.append(block)

    elements = numpy.vstack(elements)

    return elements


# -----------------------------------------------------------------------------
# Gravity of one hexagonal prism
# -----------------------------------------------------------------------------

def my_hexagonal_prism_potential(x, y, z, hexprism, rho=None,
                                 spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates the gravitational potential due to a vertical
    hexagonal prism using volume discretization.

    Inputs:
    x, y, z - numpy arrays/scalar - observation points
    hexprism - list - hexagonal prism model
    rho - float/None - density in g/cm^3. If None, uses hexprism[5]
    spacing - float/None - horizontal discretization size
    nz - int - number of vertical layers
    nx, ny - int/None - horizontal discretization size control

    Output:
    potential - numpy array - gravitational potential in SI units
    """

    x, y, z = _check_observation_points(x, y, z)
    rho = _get_physical_property(hexprism, rho, name="density")
    rho_si = rho*1000.0

    elements = my_discretize_hexagonal_prism(
        hexprism, spacing=spacing, nz=nz, nx=nx, ny=ny, keep_property=False
    )

    potential = numpy.zeros_like(x, dtype=float)

    for xe, ye, ze, volume in elements:
        rx = xe - x
        ry = ye - y
        rz = ze - z
        r, r2 = _safe_radius(rx, ry, rz)
        potential += volume/r

    potential *= G_CONST*rho_si

    return potential


def my_hexagonal_prism_gx(x, y, z, hexprism, rho=None,
                          spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates gx due to a vertical hexagonal prism.

    Output:
    gx - numpy array - x component in mGal
    """

    x, y, z = _check_observation_points(x, y, z)
    rho = _get_physical_property(hexprism, rho, name="density")
    rho_si = rho*1000.0

    elements = my_discretize_hexagonal_prism(
        hexprism, spacing=spacing, nz=nz, nx=nx, ny=ny, keep_property=False
    )

    gx = numpy.zeros_like(x, dtype=float)

    for xe, ye, ze, volume in elements:
        rx = xe - x
        ry = ye - y
        rz = ze - z
        r, r2 = _safe_radius(rx, ry, rz)
        gx += volume*rx/(r**3)

    gx *= G_CONST*rho_si*SI2MGAL

    return gx


def my_hexagonal_prism_gy(x, y, z, hexprism, rho=None,
                          spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates gy due to a vertical hexagonal prism.

    Output:
    gy - numpy array - y component in mGal
    """

    x, y, z = _check_observation_points(x, y, z)
    rho = _get_physical_property(hexprism, rho, name="density")
    rho_si = rho*1000.0

    elements = my_discretize_hexagonal_prism(
        hexprism, spacing=spacing, nz=nz, nx=nx, ny=ny, keep_property=False
    )

    gy = numpy.zeros_like(x, dtype=float)

    for xe, ye, ze, volume in elements:
        rx = xe - x
        ry = ye - y
        rz = ze - z
        r, r2 = _safe_radius(rx, ry, rz)
        gy += volume*ry/(r**3)

    gy *= G_CONST*rho_si*SI2MGAL

    return gy


def my_hexagonal_prism_gz(x, y, z, hexprism, rho=None,
                          spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates gz due to a vertical hexagonal prism.

    Output:
    gz - numpy array - vertical component in mGal
    """

    x, y, z = _check_observation_points(x, y, z)
    rho = _get_physical_property(hexprism, rho, name="density")
    rho_si = rho*1000.0

    elements = my_discretize_hexagonal_prism(
        hexprism, spacing=spacing, nz=nz, nx=nx, ny=ny, keep_property=False
    )

    gz = numpy.zeros_like(x, dtype=float)

    for xe, ye, ze, volume in elements:
        rx = xe - x
        ry = ye - y
        rz = ze - z
        r, r2 = _safe_radius(rx, ry, rz)
        gz += volume*rz/(r**3)

    gz *= G_CONST*rho_si*SI2MGAL

    return gz


def my_hexagonal_prism_gxyz(x, y, z, hexprism, rho=None,
                            spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates gx, gy and gz due to a vertical hexagonal prism.

    Outputs:
    gx, gy, gz - numpy arrays - gravity components in mGal
    """

    gx = my_hexagonal_prism_gx(x, y, z, hexprism, rho=rho,
                               spacing=spacing, nz=nz, nx=nx, ny=ny)
    gy = my_hexagonal_prism_gy(x, y, z, hexprism, rho=rho,
                               spacing=spacing, nz=nz, nx=nx, ny=ny)
    gz = my_hexagonal_prism_gz(x, y, z, hexprism, rho=rho,
                               spacing=spacing, nz=nz, nx=nx, ny=ny)

    return gx, gy, gz


# -----------------------------------------------------------------------------
# Magnetic field of one hexagonal prism
# -----------------------------------------------------------------------------

def my_hexagonal_prism_bxyz(x, y, z, hexprism, mag=None,
                            incs=90.0, decs=0.0,
                            spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates Bx, By and Bz due to a uniformly magnetized
    vertical hexagonal prism using dipole-volume discretization.

    Inputs:
    x, y, z - numpy arrays/scalar - observation points
    hexprism - list - hexagonal prism model
    mag - float/None - magnetization intensity in A/m. If None, uses hexprism[5]
    incs, decs - floats - source magnetization inclination and declination
    spacing - float/None - horizontal discretization size
    nz - int - number of vertical layers
    nx, ny - int/None - horizontal discretization size control

    Outputs:
    bx, by, bz - numpy arrays - magnetic components in nT
    """

    x, y, z = _check_observation_points(x, y, z)
    mag = _get_physical_property(hexprism, mag, name="magnetization")

    mx, my, mz = _direction_cosines(incs, decs)

    elements = my_discretize_hexagonal_prism(
        hexprism, spacing=spacing, nz=nz, nx=nx, ny=ny, keep_property=False
    )

    bx = numpy.zeros_like(x, dtype=float)
    by = numpy.zeros_like(x, dtype=float)
    bz = numpy.zeros_like(x, dtype=float)

    for xe, ye, ze, volume in elements:
        rx = x - xe
        ry = y - ye
        rz = z - ze
        r, r2 = _safe_radius(rx, ry, rz)

        moment = volume*mag
        dot = rx*mx + ry*my + rz*mz
        r5 = r2**2.5

        bx += moment*(3.0*dot*rx - r2*mx)/r5
        by += moment*(3.0*dot*ry - r2*my)/r5
        bz += moment*(3.0*dot*rz - r2*mz)/r5

    bx *= CM*T2NT
    by *= CM*T2NT
    bz *= CM*T2NT

    return bx, by, bz


def my_hexagonal_prism_bx(x, y, z, hexprism, mag=None,
                          incs=90.0, decs=0.0,
                          spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates Bx due to a hexagonal prism.
    """

    bx, by, bz = my_hexagonal_prism_bxyz(
        x, y, z, hexprism, mag=mag, incs=incs, decs=decs,
        spacing=spacing, nz=nz, nx=nx, ny=ny
    )

    return bx


def my_hexagonal_prism_by(x, y, z, hexprism, mag=None,
                          incs=90.0, decs=0.0,
                          spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates By due to a hexagonal prism.
    """

    bx, by, bz = my_hexagonal_prism_bxyz(
        x, y, z, hexprism, mag=mag, incs=incs, decs=decs,
        spacing=spacing, nz=nz, nx=nx, ny=ny
    )

    return by


def my_hexagonal_prism_bz(x, y, z, hexprism, mag=None,
                          incs=90.0, decs=0.0,
                          spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates Bz due to a hexagonal prism.
    """

    bx, by, bz = my_hexagonal_prism_bxyz(
        x, y, z, hexprism, mag=mag, incs=incs, decs=decs,
        spacing=spacing, nz=nz, nx=nx, ny=ny
    )

    return bz


def my_hexagonal_prism_tfa(x, y, z, hexprism, mag=None,
                           inc=-20.0, dec=0.0,
                           incs=None, decs=None,
                           spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates the total-field magnetic anomaly of a vertical
    hexagonal prism.

    Inputs:
    x, y, z - numpy arrays/scalar - observation points
    hexprism - list - hexagonal prism model
    mag - float/None - magnetization intensity in A/m
    inc, dec - floats - regional field inclination and declination
    incs, decs - floats/None - source magnetization direction. If None,
                 induced magnetization is assumed.
    spacing, nz, nx, ny - discretization controls

    Output:
    tfa - numpy array - total-field anomaly in nT
    """

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    bx, by, bz = my_hexagonal_prism_bxyz(
        x, y, z, hexprism, mag=mag, incs=incs, decs=decs,
        spacing=spacing, nz=nz, nx=nx, ny=ny
    )

    fx, fy, fz = _direction_cosines(inc, dec)

    tfa = fx*bx + fy*by + fz*bz

    return tfa


def my_hexagonal_prism_tf(*args, **kwargs):
    """
    Compatibility alias for total-field anomaly.
    """

    return my_hexagonal_prism_tfa(*args, **kwargs)


# -----------------------------------------------------------------------------
# Multiple hexagonal prisms
# -----------------------------------------------------------------------------

def my_hexagonal_prisms_potential(x, y, z, hexprisms, rho=None,
                                  spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates the total gravitational potential of several
    hexagonal prisms.
    """

    x, y, z = _check_observation_points(x, y, z)
    result = numpy.zeros_like(x, dtype=float)

    for i, hp in enumerate(hexprisms):
        rhoi = rho[i] if hasattr(rho, "__len__") else rho
        result += my_hexagonal_prism_potential(
            x, y, z, hp, rho=rhoi, spacing=spacing, nz=nz, nx=nx, ny=ny
        )

    return result


def my_hexagonal_prisms_gx(x, y, z, hexprisms, rho=None,
                           spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates total gx of several hexagonal prisms.
    """

    x, y, z = _check_observation_points(x, y, z)
    result = numpy.zeros_like(x, dtype=float)

    for i, hp in enumerate(hexprisms):
        rhoi = rho[i] if hasattr(rho, "__len__") else rho
        result += my_hexagonal_prism_gx(
            x, y, z, hp, rho=rhoi, spacing=spacing, nz=nz, nx=nx, ny=ny
        )

    return result


def my_hexagonal_prisms_gy(x, y, z, hexprisms, rho=None,
                           spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates total gy of several hexagonal prisms.
    """

    x, y, z = _check_observation_points(x, y, z)
    result = numpy.zeros_like(x, dtype=float)

    for i, hp in enumerate(hexprisms):
        rhoi = rho[i] if hasattr(rho, "__len__") else rho
        result += my_hexagonal_prism_gy(
            x, y, z, hp, rho=rhoi, spacing=spacing, nz=nz, nx=nx, ny=ny
        )

    return result


def my_hexagonal_prisms_gz(x, y, z, hexprisms, rho=None,
                           spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates total gz of several hexagonal prisms.
    """

    x, y, z = _check_observation_points(x, y, z)
    result = numpy.zeros_like(x, dtype=float)

    for i, hp in enumerate(hexprisms):
        rhoi = rho[i] if hasattr(rho, "__len__") else rho
        result += my_hexagonal_prism_gz(
            x, y, z, hp, rho=rhoi, spacing=spacing, nz=nz, nx=nx, ny=ny
        )

    return result


def my_hexagonal_prisms_gxyz(x, y, z, hexprisms, rho=None,
                             spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates total gx, gy and gz of several hexagonal prisms.
    """

    gx = my_hexagonal_prisms_gx(x, y, z, hexprisms, rho=rho,
                                spacing=spacing, nz=nz, nx=nx, ny=ny)
    gy = my_hexagonal_prisms_gy(x, y, z, hexprisms, rho=rho,
                                spacing=spacing, nz=nz, nx=nx, ny=ny)
    gz = my_hexagonal_prisms_gz(x, y, z, hexprisms, rho=rho,
                                spacing=spacing, nz=nz, nx=nx, ny=ny)

    return gx, gy, gz


def my_hexagonal_prisms_bxyz(x, y, z, hexprisms, mag=None,
                             incs=90.0, decs=0.0,
                             spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates total Bx, By and Bz of several hexagonal prisms.
    """

    x, y, z = _check_observation_points(x, y, z)

    bx_total = numpy.zeros_like(x, dtype=float)
    by_total = numpy.zeros_like(x, dtype=float)
    bz_total = numpy.zeros_like(x, dtype=float)

    for i, hp in enumerate(hexprisms):
        magi = mag[i] if hasattr(mag, "__len__") else mag
        bx, by, bz = my_hexagonal_prism_bxyz(
            x, y, z, hp, mag=magi, incs=incs, decs=decs,
            spacing=spacing, nz=nz, nx=nx, ny=ny
        )
        bx_total += bx
        by_total += by
        bz_total += bz

    return bx_total, by_total, bz_total


def my_hexagonal_prisms_tfa(x, y, z, hexprisms, mag=None,
                            inc=-20.0, dec=0.0,
                            incs=None, decs=None,
                            spacing=None, nz=1, nx=None, ny=None):
    """
    This function calculates the total-field magnetic anomaly of several
    hexagonal prisms.
    """

    x, y, z = _check_observation_points(x, y, z)
    result = numpy.zeros_like(x, dtype=float)

    for i, hp in enumerate(hexprisms):
        magi = mag[i] if hasattr(mag, "__len__") else mag
        result += my_hexagonal_prism_tfa(
            x, y, z, hp, mag=magi, inc=inc, dec=dec,
            incs=incs, decs=decs, spacing=spacing, nz=nz, nx=nx, ny=ny
        )

    return result


def my_hexagonal_prisms_tf(*args, **kwargs):
    """
    Compatibility alias for total-field anomaly of multiple prisms.
    """

    return my_hexagonal_prisms_tfa(*args, **kwargs)


# -----------------------------------------------------------------------------
# Conversion to rectangular prisms
# -----------------------------------------------------------------------------

def my_hexagonal_prism_to_prisms(hexprism, spacing=None, nz=1,
                                 nx=None, ny=None, physical_property=None):
    """
    This function approximates a hexagonal prism by a list of rectangular prisms.

    The returned list is compatible with the revised prism module:
    [xi, xf, yi, yf, top, bottom, physical_property]

    Inputs:
    hexprism - list - hexagonal prism model
    spacing - float/None - horizontal cell size
    nz - int - number of vertical layers
    nx, ny - int/None - horizontal discretization control
    physical_property - float/None - density or magnetization

    Output:
    prisms - list - rectangular prisms approximating the hexagonal prism
    """

    if physical_property is None:
        physical_property = hexprism[5]

    elements = my_discretize_hexagonal_prism(
        hexprism, spacing=spacing, nz=nz, nx=nx, ny=ny, keep_property=False
    )

    if spacing is None:
        xv, yv = my_hexagonal_vertices(hexprism)
        xmin, xmax = numpy.min(xv), numpy.max(xv)
        ymin, ymax = numpy.min(yv), numpy.max(yv)
        nx_eff = 35 if nx is None else int(nx)
        ny_eff = 35 if ny is None else int(ny)
        dx = (xmax - xmin)/max(nx_eff - 1, 1)
        dy = (ymax - ymin)/max(ny_eff - 1, 1)
    else:
        dx = spacing
        dy = spacing

    top = hexprism[2]
    bottom = hexprism[3]
    dz = (bottom - top)/float(nz)

    prisms = []
    for xe, ye, ze, volume in elements:
        xi = xe - 0.5*dx
        xf = xe + 0.5*dx
        yi = ye - 0.5*dy
        yf = ye + 0.5*dy
        zt = ze - 0.5*dz
        zb = ze + 0.5*dz
        prisms.append([xi, xf, yi, yf, zt, zb, physical_property])

    return prisms


# -----------------------------------------------------------------------------
# Compatibility aliases
# -----------------------------------------------------------------------------

my_hexprism_model = my_hexagonal_prism_model
my_hexprism_vertices = my_hexagonal_vertices
my_hexprism_area = my_hexagonal_area
my_hexprism_volume = my_hexagonal_volume
my_hexprism_center = my_hexagonal_center
my_hexprism_potential = my_hexagonal_prism_potential
my_hexprism_gx = my_hexagonal_prism_gx
my_hexprism_gy = my_hexagonal_prism_gy
my_hexprism_gz = my_hexagonal_prism_gz
my_hexprism_gxyz = my_hexagonal_prism_gxyz
my_hexprism_bx = my_hexagonal_prism_bx
my_hexprism_by = my_hexagonal_prism_by
my_hexprism_bz = my_hexagonal_prism_bz
my_hexprism_bxyz = my_hexagonal_prism_bxyz
my_hexprism_tfa = my_hexagonal_prism_tfa
my_hexprism_tf = my_hexagonal_prism_tf

my_hexprisms_gx = my_hexagonal_prisms_gx
my_hexprisms_gy = my_hexagonal_prisms_gy
my_hexprisms_gz = my_hexagonal_prisms_gz
my_hexprisms_gxyz = my_hexagonal_prisms_gxyz
my_hexprisms_tfa = my_hexagonal_prisms_tfa
my_hexprisms_tf = my_hexagonal_prisms_tf

hexagonal_prism_gz = my_hexagonal_prism_gz
hexagonal_prism_tfa = my_hexagonal_prism_tfa
hexprism_gz = my_hexagonal_prism_gz
hexprism_tfa = my_hexagonal_prism_tfa
