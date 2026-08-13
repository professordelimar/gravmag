# -----------------------------------------------------------------------------------
# Title: Cylinder modelling routines
# Description: Gravitational and magnetic fields of vertical finite cylinders.
# Author: Nelson Ribeiro Filho
# Revised/expanded with compatibility-oriented functions.
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


# -----------------------------------------------------------------------------------
# CONSTANTS AND SMALL UTILITIES
# -----------------------------------------------------------------------------------

G = getattr(constants, "G", 6.673e-11) if constants is not None else 6.673e-11
SI2MGAL = getattr(constants, "si2mGal", 100000.0) if constants is not None else 100000.0
CM = getattr(constants, "cm", 1.0e-7) if constants is not None else 1.0e-7
T2NT = getattr(constants, "T2nT", 1.0e9) if constants is not None else 1.0e9
EPS = getattr(constants, "EPS", 1.0e-12) if constants is not None else 1.0e-12


def _check_observation_arrays(x, y):
    '''
    Check if x and y observation arrays have compatible shapes.
    '''

    if numpy.shape(x) != numpy.shape(y):
        raise ValueError("x and y must have the same shape!")


def _prepare_z(x, z):
    '''
    Convert scalar z to an array compatible with x.
    '''

    if numpy.isscalar(z):
        return z*numpy.ones_like(x, dtype=float)

    z = numpy.asarray(z, dtype=float)

    if z.shape != numpy.shape(x):
        raise ValueError("z must be scalar or have the same shape as x and y!")

    return z


def _get_property(model, value, index, default=None, name="physical property"):
    '''
    Get density or magnetization from function input or from the model list.
    '''

    if value is not None:
        return value

    if len(model) > index:
        return model[index]

    if default is not None:
        return default

    raise ValueError("%s must be provided or stored in the model!" % name)


def _dircos(inc, dec):
    '''
    Direction cosines for inclination and declination in degrees.

    Coordinate convention:
    x = North, y = East, z = positive downward.
    '''

    if auxiliars is not None and hasattr(auxiliars, "my_dircos"):
        return auxiliars.my_dircos(inc, dec)

    inc_rad = numpy.deg2rad(inc)
    dec_rad = numpy.deg2rad(dec)

    fx = numpy.cos(inc_rad)*numpy.cos(dec_rad)
    fy = numpy.cos(inc_rad)*numpy.sin(dec_rad)
    fz = numpy.sin(inc_rad)

    return fx, fy, fz


def my_cylinder_model(xc, yc, top, bottom, radius, physical_property=None):
    '''
    Create a vertical finite cylinder model.

    Inputs:
    xc, yc - float - horizontal coordinates of the cylinder center
    top - float - depth to the cylinder top, positive downward
    bottom - float - depth to the cylinder bottom, positive downward
    radius - float - cylinder radius
    physical_property - float/None - density in g/cm^3 or magnetization in A/m

    Output:
    cylinder - list - [xc, yc, top, bottom, radius] or
                      [xc, yc, top, bottom, radius, physical_property]
    '''

    if radius <= 0.0:
        raise ValueError("radius must be positive!")

    if bottom <= top:
        raise ValueError("bottom must be greater than top for z positive downward!")

    if physical_property is None:
        return [float(xc), float(yc), float(top), float(bottom), float(radius)]

    return [float(xc), float(yc), float(top), float(bottom), float(radius),
            float(physical_property)]


def my_cylinder_volume(cylinder):
    '''
    Calculate the volume of a vertical finite cylinder.

    Input:
    cylinder - list - [xc, yc, top, bottom, radius, optional_property]

    Output:
    volume - float - cylinder volume
    '''

    xc, yc, top, bottom, radius = cylinder[:5]
    return numpy.pi*(radius**2)*(bottom - top)


def my_cylinder_cells(cylinder, nradial=20, ntheta=72, nz=1):
    '''
    Discretize a vertical finite cylinder into point-volume cells.

    The horizontal discretization is made with polar sectors. The vertical
    direction can also be discretized with nz layers. This routine is useful
    for gravitational and magnetic numerical integration.

    Inputs:
    cylinder - list - [xc, yc, top, bottom, radius, optional_property]
    nradial - int - number of radial divisions
    ntheta - int - number of angular divisions
    nz - int - number of vertical divisions

    Outputs:
    xcell, ycell, zcell - numpy arrays - cell center coordinates
    volume - numpy array - cell volumes
    '''

    xc, yc, top, bottom, radius = cylinder[:5]

    if nradial < 1:
        raise ValueError("nradial must be greater than or equal to 1!")
    if ntheta < 3:
        raise ValueError("ntheta must be greater than or equal to 3!")
    if nz < 1:
        raise ValueError("nz must be greater than or equal to 1!")

    height = bottom - top
    dz = height/float(nz)
    dtheta = 2.0*numpy.pi/float(ntheta)

    xcells = []
    ycells = []
    zcells = []
    volumes = []

    for iz in range(nz):
        zc = top + (iz + 0.5)*dz

        for ir in range(nradial):
            r1 = radius*float(ir)/float(nradial)
            r2 = radius*float(ir + 1)/float(nradial)

            # Centroid radius of an annular sector.
            if abs(r2**2 - r1**2) < EPS:
                rmid = 0.0
            else:
                rmid = (2.0/3.0)*(r2**3 - r1**3)/(r2**2 - r1**2)

            area_sector = 0.5*(r2**2 - r1**2)*dtheta
            cell_volume = area_sector*dz

            for it in range(ntheta):
                theta = (it + 0.5)*dtheta
                xcells.append(xc + rmid*numpy.cos(theta))
                ycells.append(yc + rmid*numpy.sin(theta))
                zcells.append(zc)
                volumes.append(cell_volume)

    return (numpy.asarray(xcells, dtype=float),
            numpy.asarray(ycells, dtype=float),
            numpy.asarray(zcells, dtype=float),
            numpy.asarray(volumes, dtype=float))


def my_cylinder_to_prisms(cylinder, spacing=None, shape=None,
                          physical_property=None, include_boundary=True):
    '''
    Approximate a vertical cylinder by a set of small rectangular prisms.

    This routine is useful when the user wants to reuse prism-based routines.
    The function creates a Cartesian grid of rectangular cells and keeps only
    cells whose centers fall inside the circular cylinder.

    Inputs:
    cylinder - list - [xc, yc, top, bottom, radius, optional_property]
    spacing - float/None - horizontal cell size. If None, it is defined from shape.
    shape - int/tuple/None - number of cells along x and y across diameter.
    physical_property - float/None - density or magnetization. If None, uses cylinder[5]
    include_boundary - bool - if True, includes cells at r <= radius

    Output:
    prisms - list - each prism is [xi, xf, yi, yf, top, bottom, physical_property]
    '''

    xc, yc, top, bottom, radius = cylinder[:5]
    prop = _get_property(cylinder, physical_property, 5, default=0.0)

    if shape is None and spacing is None:
        shape = 41

    if shape is not None:
        if numpy.isscalar(shape):
            nx = int(shape)
            ny = int(shape)
        else:
            nx, ny = shape

        if nx < 2 or ny < 2:
            raise ValueError("shape must define at least 2 cells in each direction!")

        dx = 2.0*radius/float(nx)
        dy = 2.0*radius/float(ny)
    else:
        if spacing <= 0.0:
            raise ValueError("spacing must be positive!")
        dx = float(spacing)
        dy = float(spacing)
        nx = int(numpy.ceil(2.0*radius/dx))
        ny = int(numpy.ceil(2.0*radius/dy))

    xcenters = numpy.linspace(xc - radius + 0.5*dx, xc + radius - 0.5*dx, nx)
    ycenters = numpy.linspace(yc - radius + 0.5*dy, yc + radius - 0.5*dy, ny)

    prisms = []

    for xx in xcenters:
        for yy in ycenters:
            rr = numpy.sqrt((xx - xc)**2 + (yy - yc)**2)

            inside = rr <= radius if include_boundary else rr < radius

            if inside:
                prisms.append([xx - 0.5*dx, xx + 0.5*dx,
                               yy - 0.5*dy, yy + 0.5*dy,
                               top, bottom, prop])

    return prisms


# -----------------------------------------------------------------------------------
# GRAVITY OF ONE CYLINDER
# -----------------------------------------------------------------------------------

def my_cylinder_potential(x, y, z, cylinder, rho=None,
                          nradial=20, ntheta=72, nz=1):
    '''
    Calculate the gravitational potential due to a vertical finite cylinder.

    The cylinder is discretized into point-volume cells. The density must be
    given in g/cm^3 and coordinates in meters.

    Inputs:
    x, y - numpy arrays - observation coordinates
    z - float or numpy array - observation height/depth, positive downward
    cylinder - list - [xc, yc, top, bottom, radius, optional_density]
    rho - float/None - density in g/cm^3. If None, uses cylinder[5]
    nradial, ntheta, nz - ints - cylinder discretization parameters

    Output:
    potential - numpy array - gravitational potential in SI units
    '''

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    _check_observation_arrays(x, y)
    z = _prepare_z(x, z)

    rho_si = 1000.0*_get_property(cylinder, rho, 5, name="density")
    xcell, ycell, zcell, volume = my_cylinder_cells(cylinder, nradial, ntheta, nz)

    potential = numpy.zeros_like(x, dtype=float)

    for xc, yc, zc, vol in zip(xcell, ycell, zcell, volume):
        dx = x - xc
        dy = y - yc
        dz = z - zc
        r = numpy.sqrt(dx**2 + dy**2 + dz**2 + EPS**2)
        potential += vol/r

    potential *= G*rho_si

    return potential


def my_cylinder_gx(x, y, z, cylinder, rho=None,
                   nradial=20, ntheta=72, nz=1):
    '''
    Calculate the x component of the gravitational attraction of a cylinder.

    Inputs are the same as my_cylinder_potential.

    Output:
    gx - numpy array - x gravity component in mGal
    '''

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    _check_observation_arrays(x, y)
    z = _prepare_z(x, z)

    rho_si = 1000.0*_get_property(cylinder, rho, 5, name="density")
    xcell, ycell, zcell, volume = my_cylinder_cells(cylinder, nradial, ntheta, nz)

    gx = numpy.zeros_like(x, dtype=float)

    for xc, yc, zc, vol in zip(xcell, ycell, zcell, volume):
        dx = xc - x
        dy = yc - y
        dz = zc - z
        r2 = dx**2 + dy**2 + dz**2 + EPS**2
        r3 = r2**1.5
        gx += vol*dx/r3

    gx *= G*rho_si*SI2MGAL

    return gx


def my_cylinder_gy(x, y, z, cylinder, rho=None,
                   nradial=20, ntheta=72, nz=1):
    '''
    Calculate the y component of the gravitational attraction of a cylinder.

    Output:
    gy - numpy array - y gravity component in mGal
    '''

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    _check_observation_arrays(x, y)
    z = _prepare_z(x, z)

    rho_si = 1000.0*_get_property(cylinder, rho, 5, name="density")
    xcell, ycell, zcell, volume = my_cylinder_cells(cylinder, nradial, ntheta, nz)

    gy = numpy.zeros_like(x, dtype=float)

    for xc, yc, zc, vol in zip(xcell, ycell, zcell, volume):
        dx = xc - x
        dy = yc - y
        dz = zc - z
        r2 = dx**2 + dy**2 + dz**2 + EPS**2
        r3 = r2**1.5
        gy += vol*dy/r3

    gy *= G*rho_si*SI2MGAL

    return gy


def my_cylinder_gz(x, y, z, cylinder, rho=None,
                   nradial=20, ntheta=72, nz=1):
    '''
    Calculate the vertical component of the gravitational attraction of a cylinder.

    The z-axis is positive downward. Therefore, the vertical component is
    proportional to (zc - zobs), consistently with the sphere kernels.

    Output:
    gz - numpy array - vertical gravity component in mGal
    '''

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    _check_observation_arrays(x, y)
    z = _prepare_z(x, z)

    rho_si = 1000.0*_get_property(cylinder, rho, 5, name="density")
    xcell, ycell, zcell, volume = my_cylinder_cells(cylinder, nradial, ntheta, nz)

    gz = numpy.zeros_like(x, dtype=float)

    for xc, yc, zc, vol in zip(xcell, ycell, zcell, volume):
        dx = xc - x
        dy = yc - y
        dz = zc - z
        r2 = dx**2 + dy**2 + dz**2 + EPS**2
        r3 = r2**1.5
        gz += vol*dz/r3

    gz *= G*rho_si*SI2MGAL

    return gz


def my_cylinder_gxyz(x, y, z, cylinder, rho=None,
                     nradial=20, ntheta=72, nz=1):
    '''
    Calculate gx, gy and gz of one vertical finite cylinder.

    Output:
    gx, gy, gz - numpy arrays - gravity components in mGal
    '''

    gx = my_cylinder_gx(x, y, z, cylinder, rho, nradial, ntheta, nz)
    gy = my_cylinder_gy(x, y, z, cylinder, rho, nradial, ntheta, nz)
    gz = my_cylinder_gz(x, y, z, cylinder, rho, nradial, ntheta, nz)

    return gx, gy, gz


# -----------------------------------------------------------------------------------
# MAGNETIC FIELD OF ONE CYLINDER
# -----------------------------------------------------------------------------------

def my_cylinder_bxyz(x, y, z, cylinder, mag=None, incs=90.0, decs=0.0,
                     nradial=20, ntheta=72, nz=1):
    '''
    Calculate the magnetic induction components of a magnetized cylinder.

    The cylinder is approximated by point dipole volume cells. The magnetization
    intensity must be given in A/m. Output is in nT.

    Inputs:
    x, y - numpy arrays - observation coordinates
    z - float or numpy array - observation height/depth, positive downward
    cylinder - list - [xc, yc, top, bottom, radius, optional_magnetization]
    mag - float/None - magnetization intensity in A/m. If None, uses cylinder[5]
    incs, decs - floats - source magnetization inclination and declination in degrees
    nradial, ntheta, nz - ints - cylinder discretization parameters

    Outputs:
    bx, by, bz - numpy arrays - magnetic components in nT
    '''

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    _check_observation_arrays(x, y)
    z = _prepare_z(x, z)

    mag_value = _get_property(cylinder, mag, 5, name="magnetization")
    mx, my, mz = _dircos(incs, decs)

    xcell, ycell, zcell, volume = my_cylinder_cells(cylinder, nradial, ntheta, nz)

    bx = numpy.zeros_like(x, dtype=float)
    by = numpy.zeros_like(x, dtype=float)
    bz = numpy.zeros_like(x, dtype=float)

    for xc, yc, zc, vol in zip(xcell, ycell, zcell, volume):
        rx = x - xc
        ry = y - yc
        rz = z - zc

        r2 = rx**2 + ry**2 + rz**2 + EPS**2
        r5 = r2**2.5

        moment = vol*mag_value
        dot = rx*mx + ry*my + rz*mz

        bx += moment*(3.0*dot*rx - r2*mx)/r5
        by += moment*(3.0*dot*ry - r2*my)/r5
        bz += moment*(3.0*dot*rz - r2*mz)/r5

    bx *= CM*T2NT
    by *= CM*T2NT
    bz *= CM*T2NT

    return bx, by, bz


def my_cylinder_bx(x, y, z, cylinder, mag=None, incs=90.0, decs=0.0,
                   nradial=20, ntheta=72, nz=1):
    '''
    Calculate Bx of one magnetized cylinder in nT.
    '''

    bx, by, bz = my_cylinder_bxyz(x, y, z, cylinder, mag, incs, decs,
                                  nradial, ntheta, nz)
    return bx


def my_cylinder_by(x, y, z, cylinder, mag=None, incs=90.0, decs=0.0,
                   nradial=20, ntheta=72, nz=1):
    '''
    Calculate By of one magnetized cylinder in nT.
    '''

    bx, by, bz = my_cylinder_bxyz(x, y, z, cylinder, mag, incs, decs,
                                  nradial, ntheta, nz)
    return by


def my_cylinder_bz(x, y, z, cylinder, mag=None, incs=90.0, decs=0.0,
                   nradial=20, ntheta=72, nz=1):
    '''
    Calculate Bz of one magnetized cylinder in nT.
    '''

    bx, by, bz = my_cylinder_bxyz(x, y, z, cylinder, mag, incs, decs,
                                  nradial, ntheta, nz)
    return bz


def my_cylinder_tfa(x, y, z, cylinder, mag=None,
                    inc=-20.0, dec=0.0, incs=None, decs=None,
                    nradial=20, ntheta=72, nz=1):
    '''
    Calculate the total-field magnetic anomaly of a cylinder.

    The total-field anomaly is approximated by projecting the magnetic induction
    vector produced by the cylinder onto the regional geomagnetic field direction.

    Inputs:
    x, y - numpy arrays - observation coordinates
    z - float or numpy array - observation height/depth
    cylinder - list - cylinder model
    mag - float/None - magnetization intensity in A/m
    inc, dec - floats - regional field inclination and declination in degrees
    incs, decs - floats/None - source magnetization inclination and declination.
                 If None, induced magnetization is assumed.
    nradial, ntheta, nz - ints - cylinder discretization parameters

    Output:
    tfa - numpy array - total-field anomaly in nT
    '''

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    bx, by, bz = my_cylinder_bxyz(x, y, z, cylinder, mag, incs, decs,
                                  nradial, ntheta, nz)

    fx, fy, fz = _dircos(inc, dec)
    tfa = bx*fx + by*fy + bz*fz

    return tfa


def my_cylinder_tf(x, y, z, cylinder, mag=None,
                   inc=-20.0, dec=0.0, incs=None, decs=None,
                   nradial=20, ntheta=72, nz=1):
    '''
    Compatibility alias for total-field anomaly of a cylinder.
    '''

    return my_cylinder_tfa(x, y, z, cylinder, mag, inc, dec, incs, decs,
                           nradial, ntheta, nz)


# -----------------------------------------------------------------------------------
# MULTIPLE CYLINDERS
# -----------------------------------------------------------------------------------

def my_cylinders_potential(x, y, z, cylinders, rho=None,
                           nradial=20, ntheta=72, nz=1):
    '''
    Calculate gravitational potential produced by multiple cylinders.
    '''

    result = numpy.zeros_like(numpy.asarray(x, dtype=float), dtype=float)

    for cylinder in cylinders:
        result += my_cylinder_potential(x, y, z, cylinder, rho,
                                        nradial, ntheta, nz)

    return result


def my_cylinders_gx(x, y, z, cylinders, rho=None,
                    nradial=20, ntheta=72, nz=1):
    '''
    Calculate gx produced by multiple cylinders.
    '''

    result = numpy.zeros_like(numpy.asarray(x, dtype=float), dtype=float)

    for cylinder in cylinders:
        result += my_cylinder_gx(x, y, z, cylinder, rho,
                                 nradial, ntheta, nz)

    return result


def my_cylinders_gy(x, y, z, cylinders, rho=None,
                    nradial=20, ntheta=72, nz=1):
    '''
    Calculate gy produced by multiple cylinders.
    '''

    result = numpy.zeros_like(numpy.asarray(x, dtype=float), dtype=float)

    for cylinder in cylinders:
        result += my_cylinder_gy(x, y, z, cylinder, rho,
                                 nradial, ntheta, nz)

    return result


def my_cylinders_gz(x, y, z, cylinders, rho=None,
                    nradial=20, ntheta=72, nz=1):
    '''
    Calculate gz produced by multiple cylinders.
    '''

    result = numpy.zeros_like(numpy.asarray(x, dtype=float), dtype=float)

    for cylinder in cylinders:
        result += my_cylinder_gz(x, y, z, cylinder, rho,
                                 nradial, ntheta, nz)

    return result


def my_cylinders_gxyz(x, y, z, cylinders, rho=None,
                      nradial=20, ntheta=72, nz=1):
    '''
    Calculate gx, gy and gz produced by multiple cylinders.
    '''

    gx = my_cylinders_gx(x, y, z, cylinders, rho, nradial, ntheta, nz)
    gy = my_cylinders_gy(x, y, z, cylinders, rho, nradial, ntheta, nz)
    gz = my_cylinders_gz(x, y, z, cylinders, rho, nradial, ntheta, nz)

    return gx, gy, gz


def my_cylinders_bxyz(x, y, z, cylinders, mag=None,
                      incs=90.0, decs=0.0,
                      nradial=20, ntheta=72, nz=1):
    '''
    Calculate Bx, By and Bz produced by multiple magnetized cylinders.
    '''

    xarr = numpy.asarray(x, dtype=float)
    bx = numpy.zeros_like(xarr, dtype=float)
    by = numpy.zeros_like(xarr, dtype=float)
    bz = numpy.zeros_like(xarr, dtype=float)

    for cylinder in cylinders:
        bx_i, by_i, bz_i = my_cylinder_bxyz(x, y, z, cylinder, mag,
                                            incs, decs, nradial, ntheta, nz)
        bx += bx_i
        by += by_i
        bz += bz_i

    return bx, by, bz


def my_cylinders_tfa(x, y, z, cylinders, mag=None,
                     inc=-20.0, dec=0.0, incs=None, decs=None,
                     nradial=20, ntheta=72, nz=1):
    '''
    Calculate total-field magnetic anomaly produced by multiple cylinders.
    '''

    result = numpy.zeros_like(numpy.asarray(x, dtype=float), dtype=float)

    for cylinder in cylinders:
        result += my_cylinder_tfa(x, y, z, cylinder, mag,
                                  inc, dec, incs, decs,
                                  nradial, ntheta, nz)

    return result


def my_cylinders_tf(x, y, z, cylinders, mag=None,
                    inc=-20.0, dec=0.0, incs=None, decs=None,
                    nradial=20, ntheta=72, nz=1):
    '''
    Compatibility alias for total-field anomaly of multiple cylinders.
    '''

    return my_cylinders_tfa(x, y, z, cylinders, mag,
                            inc, dec, incs, decs,
                            nradial, ntheta, nz)


# -----------------------------------------------------------------------------------
# COMPATIBILITY ALIASES
# -----------------------------------------------------------------------------------

cylinder_model = my_cylinder_model
cylinder_volume = my_cylinder_volume
cylinder_cells = my_cylinder_cells
cylinder_to_prisms = my_cylinder_to_prisms

cylinder_potential = my_cylinder_potential
cylinder_gx = my_cylinder_gx
cylinder_gy = my_cylinder_gy
cylinder_gz = my_cylinder_gz
cylinder_gxyz = my_cylinder_gxyz

cylinder_bx = my_cylinder_bx
cylinder_by = my_cylinder_by
cylinder_bz = my_cylinder_bz
cylinder_bxyz = my_cylinder_bxyz
cylinder_tf = my_cylinder_tf
cylinder_tfa = my_cylinder_tfa

my_cylinder_totalfield = my_cylinder_tfa
my_cylinder_total_field = my_cylinder_tfa
my_cylinder_magnetic = my_cylinder_tfa
my_cylinder_mag = my_cylinder_tfa

my_cylinders_totalfield = my_cylinders_tfa
my_cylinders_total_field = my_cylinders_tfa
my_cylinders_magnetic = my_cylinders_tfa
my_cylinders_mag = my_cylinders_tfa
