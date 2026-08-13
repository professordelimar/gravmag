# -----------------------------------------------------------------------------------
# Title: Polygonal Prism Forward Modeling
# Description: Gravity and magnetic forward modeling for vertical polygonal prisms.
# Author: Nelson Ribeiro Filho
# Revised/expanded with compatibility functions
# -----------------------------------------------------------------------------------

from __future__ import division
import numpy

try:
    from . import auxiliars
except Exception:
    try:
        import auxiliars
    except Exception:
        auxiliars = None


# -----------------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------------

G = 6.673e-11          # gravitational constant [SI]
SI2MGAL = 100000.0    # m/s^2 to mGal
CM = 1.0e-7           # magnetic constant [SI]
T2NT = 1.0e9          # Tesla to nT
EPS = 1.0e-12


# -----------------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------------

def _as_array(a):
    '''Return input as numpy array with float dtype.'''
    return numpy.asarray(a, dtype=float)


def _check_observation_points(x, y):
    '''Check if x and y have the same shape.'''
    if numpy.shape(x) != numpy.shape(y):
        raise ValueError('x and y must have the same shape!')


def _z_array(z, shape):
    '''Convert scalar z to an array compatible with x and y.'''
    if numpy.isscalar(z):
        return z*numpy.ones(shape, dtype=float)
    z = _as_array(z)
    if z.shape != shape:
        raise ValueError('z must be scalar or have the same shape as x and y!')
    return z


def _get_physical_property(model, value):
    '''Return explicit physical property or model[5].'''
    if value is not None:
        return value
    if len(model) >= 6:
        return model[5]
    raise ValueError('physical property must be supplied or stored in model[5]!')


def _dircos(inc, dec):
    '''Direction cosines for inclination and declination in degrees.'''
    if auxiliars is not None and hasattr(auxiliars, 'my_dircos'):
        return auxiliars.my_dircos(inc, dec)

    inc_rad = numpy.deg2rad(inc)
    dec_rad = numpy.deg2rad(dec)
    mx = numpy.cos(inc_rad)*numpy.cos(dec_rad)
    my = numpy.cos(inc_rad)*numpy.sin(dec_rad)
    mz = numpy.sin(inc_rad)
    return mx, my, mz


def _safe_radius(rx, ry, rz):
    '''Euclidean distance with small stabilization.'''
    r2 = rx*rx + ry*ry + rz*rz
    r2 = numpy.where(r2 < EPS, EPS, r2)
    return numpy.sqrt(r2), r2


# -----------------------------------------------------------------------------------
# Polygon geometry
# -----------------------------------------------------------------------------------

def my_polygonal_prism_model(vertices, top, bottom, physical_property=None):
    '''
    Create a vertical polygonal prism model.

    Inputs:
    vertices - array/list - polygon vertices as [[x1, y1], [x2, y2], ...]
    top - float - top depth, positive downward
    bottom - float - bottom depth, positive downward
    physical_property - float/None - density [g/cm^3] or magnetization [A/m]

    Output:
    model - list - [vertices, top, bottom, physical_property]
    '''

    vertices = _as_array(vertices)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError('vertices must be an array with shape (n_vertices, 2)!')

    if vertices.shape[0] < 3:
        raise ValueError('a polygon must have at least three vertices!')

    if bottom <= top:
        raise ValueError('bottom must be greater than top!')

    return [vertices, float(top), float(bottom), physical_property]


def my_regular_polygon_vertices(xc, yc, radius, n_vertices=6, rotation=0.0):
    '''
    Create vertices of a regular polygon.

    Inputs:
    xc, yc - float - polygon center
    radius - float - circumradius
    n_vertices - int - number of polygon vertices
    rotation - float - rotation angle in degrees

    Output:
    vertices - numpy array - shape (n_vertices, 2)
    '''

    if n_vertices < 3:
        raise ValueError('n_vertices must be at least 3!')

    angles = numpy.linspace(0.0, 2.0*numpy.pi, n_vertices, endpoint=False)
    angles = angles + numpy.deg2rad(rotation)

    xv = xc + radius*numpy.cos(angles)
    yv = yc + radius*numpy.sin(angles)

    return numpy.column_stack((xv, yv))


def my_polygon_area(vertices):
    '''
    Compute polygon area using the shoelace formula.

    Input:
    vertices - array - polygon vertices [[x1, y1], ...]

    Output:
    area - float - polygon area
    '''

    vertices = _as_array(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]

    area = 0.5*numpy.abs(numpy.dot(x, numpy.roll(y, -1)) -
                         numpy.dot(y, numpy.roll(x, -1)))
    return area


def my_polygon_centroid(vertices):
    '''
    Compute polygon centroid.

    Input:
    vertices - array - polygon vertices [[x1, y1], ...]

    Output:
    xc, yc - floats - centroid coordinates
    '''

    vertices = _as_array(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]

    cross = x*numpy.roll(y, -1) - numpy.roll(x, -1)*y
    A = 0.5*numpy.sum(cross)

    if numpy.abs(A) < EPS:
        return numpy.mean(x), numpy.mean(y)

    xc = numpy.sum((x + numpy.roll(x, -1))*cross)/(6.0*A)
    yc = numpy.sum((y + numpy.roll(y, -1))*cross)/(6.0*A)

    return xc, yc


def my_polygon_bounds(vertices):
    '''Return xmin, xmax, ymin, ymax for a polygon.'''

    vertices = _as_array(vertices)
    xmin = numpy.min(vertices[:, 0])
    xmax = numpy.max(vertices[:, 0])
    ymin = numpy.min(vertices[:, 1])
    ymax = numpy.max(vertices[:, 1])
    return xmin, xmax, ymin, ymax


def my_polygon_volume(vertices, top, bottom):
    '''Compute vertical polygonal prism volume.'''

    return my_polygon_area(vertices)*(bottom - top)


def my_points_in_polygon(xp, yp, vertices):
    '''
    Ray-casting point-in-polygon test.

    Inputs:
    xp, yp - arrays - point coordinates
    vertices - array - polygon vertices

    Output:
    inside - boolean array
    '''

    xp = _as_array(xp)
    yp = _as_array(yp)
    vertices = _as_array(vertices)

    xv = vertices[:, 0]
    yv = vertices[:, 1]
    n = len(vertices)

    inside = numpy.zeros_like(xp, dtype=bool)
    j = n - 1

    for i in range(n):
        cond = ((yv[i] > yp) != (yv[j] > yp))
        x_inter = (xv[j] - xv[i])*(yp - yv[i])/(yv[j] - yv[i] + EPS) + xv[i]
        inside = inside ^ (cond & (xp < x_inter))
        j = i

    return inside


# -----------------------------------------------------------------------------------
# Discretization
# -----------------------------------------------------------------------------------

def my_discretize_polygonal_prism(model, spacing=None, nz=1, nx=None, ny=None):
    '''
    Discretize a vertical polygonal prism into small volume elements.

    Inputs:
    model - list - [vertices, top, bottom, physical_property]
    spacing - float/None - horizontal sampling interval
    nz - int - number of vertical layers
    nx, ny - int/None - number of samples in x and y if spacing is None

    Outputs:
    xc, yc, zc - arrays - centers of volume elements
    dv - array - volume of each element
    '''

    vertices, top, bottom, _ = model
    vertices = _as_array(vertices)

    if nz < 1:
        raise ValueError('nz must be >= 1!')

    xmin, xmax, ymin, ymax = my_polygon_bounds(vertices)

    if spacing is None:
        if nx is None:
            nx = 40
        if ny is None:
            ny = 40
        xs = numpy.linspace(xmin, xmax, int(nx))
        ys = numpy.linspace(ymin, ymax, int(ny))
        dx = abs(xs[1] - xs[0]) if len(xs) > 1 else (xmax - xmin)
        dy = abs(ys[1] - ys[0]) if len(ys) > 1 else (ymax - ymin)
    else:
        if spacing <= 0.0:
            raise ValueError('spacing must be positive!')
        xs = numpy.arange(xmin + 0.5*spacing, xmax, spacing)
        ys = numpy.arange(ymin + 0.5*spacing, ymax, spacing)
        dx = spacing
        dy = spacing

    xg, yg = numpy.meshgrid(xs, ys)
    inside = my_points_in_polygon(xg.ravel(), yg.ravel(), vertices)

    x_inside = xg.ravel()[inside]
    y_inside = yg.ravel()[inside]

    if x_inside.size == 0:
        raise ValueError('no discretization points found inside polygon! Use smaller spacing or larger nx/ny.')

    z_edges = numpy.linspace(top, bottom, nz + 1)
    dz = numpy.diff(z_edges)
    z_centers = 0.5*(z_edges[:-1] + z_edges[1:])

    x_list = []
    y_list = []
    z_list = []
    v_list = []

    area_cell = dx*dy

    for k in range(nz):
        x_list.append(x_inside.copy())
        y_list.append(y_inside.copy())
        z_list.append(z_centers[k]*numpy.ones_like(x_inside))
        v_list.append(area_cell*dz[k]*numpy.ones_like(x_inside))

    xc = numpy.concatenate(x_list)
    yc = numpy.concatenate(y_list)
    zc = numpy.concatenate(z_list)
    dv = numpy.concatenate(v_list)

    return xc, yc, zc, dv


# -----------------------------------------------------------------------------------
# Gravity of volume elements
# -----------------------------------------------------------------------------------

def _point_mass_potential(x, y, z, xc, yc, zc, mass):
    rx = x - xc
    ry = y - yc
    rz = z - zc
    r, r2 = _safe_radius(rx, ry, rz)
    return G*mass/r


def _point_mass_gravity(x, y, z, xc, yc, zc, mass, component='gz'):
    rx = x - xc
    ry = y - yc
    rz = z - zc
    r, r2 = _safe_radius(rx, ry, rz)
    r3 = r2*r

    if component in ['gx', 'x']:
        val = -G*mass*rx/r3
    elif component in ['gy', 'y']:
        val = -G*mass*ry/r3
    elif component in ['gz', 'z']:
        val = -G*mass*rz/r3
    else:
        raise ValueError('component must be gx, gy or gz!')

    return val*SI2MGAL


def my_polygonal_prism_potential(x, y, z, model, rho=None,
                                  spacing=None, nz=1, nx=None, ny=None):
    '''
    Compute gravitational potential of a vertical polygonal prism.

    Inputs:
    x, y - arrays - observation coordinates
    z - scalar/array - observation level, positive downward
    model - list - [vertices, top, bottom, density]
    rho - float/None - density in g/cm^3. If None, uses model[3]
    spacing, nz, nx, ny - discretization parameters

    Output:
    potential - array - gravitational potential [SI]
    '''

    _check_observation_points(x, y)
    x = _as_array(x)
    y = _as_array(y)
    z = _z_array(z, x.shape)

    rho = _get_physical_property(model, rho)
    rho_si = 1000.0*rho

    xc, yc, zc, dv = my_discretize_polygonal_prism(model, spacing=spacing, nz=nz, nx=nx, ny=ny)
    mass = rho_si*dv

    result = numpy.zeros_like(x, dtype=float)
    for i in range(xc.size):
        result += _point_mass_potential(x, y, z, xc[i], yc[i], zc[i], mass[i])

    return result


def my_polygonal_prism_gx(x, y, z, model, rho=None,
                           spacing=None, nz=1, nx=None, ny=None):
    '''Compute gx of a vertical polygonal prism in mGal.'''

    return my_polygonal_prism_gravity(x, y, z, model, rho=rho, component='gx',
                                      spacing=spacing, nz=nz, nx=nx, ny=ny)


def my_polygonal_prism_gy(x, y, z, model, rho=None,
                           spacing=None, nz=1, nx=None, ny=None):
    '''Compute gy of a vertical polygonal prism in mGal.'''

    return my_polygonal_prism_gravity(x, y, z, model, rho=rho, component='gy',
                                      spacing=spacing, nz=nz, nx=nx, ny=ny)


def my_polygonal_prism_gz(x, y, z, model, rho=None,
                           spacing=None, nz=1, nx=None, ny=None):
    '''Compute gz of a vertical polygonal prism in mGal.'''

    return my_polygonal_prism_gravity(x, y, z, model, rho=rho, component='gz',
                                      spacing=spacing, nz=nz, nx=nx, ny=ny)


def my_polygonal_prism_gravity(x, y, z, model, rho=None, component='gz',
                                spacing=None, nz=1, nx=None, ny=None):
    '''
    Compute one gravity component of a vertical polygonal prism.

    Inputs:
    component - string - 'gx', 'gy' or 'gz'

    Output:
    gravity - array - component in mGal
    '''

    _check_observation_points(x, y)
    x = _as_array(x)
    y = _as_array(y)
    z = _z_array(z, x.shape)

    rho = _get_physical_property(model, rho)
    rho_si = 1000.0*rho

    xc, yc, zc, dv = my_discretize_polygonal_prism(model, spacing=spacing, nz=nz, nx=nx, ny=ny)
    mass = rho_si*dv

    result = numpy.zeros_like(x, dtype=float)
    for i in range(xc.size):
        result += _point_mass_gravity(x, y, z, xc[i], yc[i], zc[i], mass[i], component=component)

    return result


def my_polygonal_prism_gxyz(x, y, z, model, rho=None,
                             spacing=None, nz=1, nx=None, ny=None):
    '''Compute gx, gy and gz of a vertical polygonal prism in mGal.'''

    gx = my_polygonal_prism_gx(x, y, z, model, rho=rho, spacing=spacing, nz=nz, nx=nx, ny=ny)
    gy = my_polygonal_prism_gy(x, y, z, model, rho=rho, spacing=spacing, nz=nz, nx=nx, ny=ny)
    gz = my_polygonal_prism_gz(x, y, z, model, rho=rho, spacing=spacing, nz=nz, nx=nx, ny=ny)
    return gx, gy, gz


# -----------------------------------------------------------------------------------
# Magnetic field of volume elements
# -----------------------------------------------------------------------------------

def _dipole_field(x, y, z, xc, yc, zc, volume, mag, incs, decs):
    '''Magnetic induction from a small uniformly magnetized volume element in nT.'''

    rx = x - xc
    ry = y - yc
    rz = z - zc
    r, r2 = _safe_radius(rx, ry, rz)

    mx, my, mz = _dircos(incs, decs)
    moment = mag*volume

    dot = rx*mx + ry*my + rz*mz
    r5 = r2*r2*r
    r3 = r2*r

    bx = CM*moment*(3.0*dot*rx/r5 - mx/r3)*T2NT
    by = CM*moment*(3.0*dot*ry/r5 - my/r3)*T2NT
    bz = CM*moment*(3.0*dot*rz/r5 - mz/r3)*T2NT

    return bx, by, bz


def my_polygonal_prism_bxyz(x, y, z, model, mag=None,
                             incs=90.0, decs=0.0,
                             spacing=None, nz=1, nx=None, ny=None):
    '''
    Compute magnetic induction components of a vertical polygonal prism.

    Inputs:
    mag - float/None - magnetization intensity [A/m]. If None, uses model[3]
    incs, decs - float - source magnetization inclination and declination [degrees]

    Outputs:
    bx, by, bz - arrays - magnetic components [nT]
    '''

    _check_observation_points(x, y)
    x = _as_array(x)
    y = _as_array(y)
    z = _z_array(z, x.shape)

    mag = _get_physical_property(model, mag)

    xc, yc, zc, dv = my_discretize_polygonal_prism(model, spacing=spacing, nz=nz, nx=nx, ny=ny)

    bx_total = numpy.zeros_like(x, dtype=float)
    by_total = numpy.zeros_like(x, dtype=float)
    bz_total = numpy.zeros_like(x, dtype=float)

    for i in range(xc.size):
        bx, by, bz = _dipole_field(x, y, z, xc[i], yc[i], zc[i], dv[i], mag, incs, decs)
        bx_total += bx
        by_total += by
        bz_total += bz

    return bx_total, by_total, bz_total


def my_polygonal_prism_bx(x, y, z, model, mag=None,
                           incs=90.0, decs=0.0,
                           spacing=None, nz=1, nx=None, ny=None):
    '''Compute Bx of a polygonal prism in nT.'''

    bx, by, bz = my_polygonal_prism_bxyz(x, y, z, model, mag=mag,
                                         incs=incs, decs=decs,
                                         spacing=spacing, nz=nz, nx=nx, ny=ny)
    return bx


def my_polygonal_prism_by(x, y, z, model, mag=None,
                           incs=90.0, decs=0.0,
                           spacing=None, nz=1, nx=None, ny=None):
    '''Compute By of a polygonal prism in nT.'''

    bx, by, bz = my_polygonal_prism_bxyz(x, y, z, model, mag=mag,
                                         incs=incs, decs=decs,
                                         spacing=spacing, nz=nz, nx=nx, ny=ny)
    return by


def my_polygonal_prism_bz(x, y, z, model, mag=None,
                           incs=90.0, decs=0.0,
                           spacing=None, nz=1, nx=None, ny=None):
    '''Compute Bz of a polygonal prism in nT.'''

    bx, by, bz = my_polygonal_prism_bxyz(x, y, z, model, mag=mag,
                                         incs=incs, decs=decs,
                                         spacing=spacing, nz=nz, nx=nx, ny=ny)
    return bz


def my_polygonal_prism_tfa(x, y, z, model, mag=None,
                            inc=-20.0, dec=0.0,
                            incs=None, decs=None,
                            spacing=None, nz=1, nx=None, ny=None):
    '''
    Compute magnetic total-field anomaly of a polygonal prism.

    Inputs:
    inc, dec - float - regional field inclination and declination [degrees]
    incs, decs - float/None - source magnetization inclination and declination.
                 If None, induced magnetization is assumed.

    Output:
    tfa - array - total-field anomaly [nT]
    '''

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    bx, by, bz = my_polygonal_prism_bxyz(x, y, z, model, mag=mag,
                                         incs=incs, decs=decs,
                                         spacing=spacing, nz=nz, nx=nx, ny=ny)
    fx, fy, fz = _dircos(inc, dec)

    return bx*fx + by*fy + bz*fz


def my_polygonal_prism_tf(*args, **kwargs):
    '''Alias for my_polygonal_prism_tfa.'''
    return my_polygonal_prism_tfa(*args, **kwargs)


# -----------------------------------------------------------------------------------
# Multiple polygonal prisms
# -----------------------------------------------------------------------------------

def my_polygonal_prisms_potential(x, y, z, models, rho=None,
                                   spacing=None, nz=1, nx=None, ny=None):
    '''Compute gravitational potential of multiple polygonal prisms.'''

    total = numpy.zeros_like(_as_array(x), dtype=float)
    for model in models:
        total += my_polygonal_prism_potential(x, y, z, model, rho=rho,
                                              spacing=spacing, nz=nz, nx=nx, ny=ny)
    return total


def my_polygonal_prisms_gx(x, y, z, models, rho=None,
                            spacing=None, nz=1, nx=None, ny=None):
    '''Compute gx of multiple polygonal prisms.'''

    total = numpy.zeros_like(_as_array(x), dtype=float)
    for model in models:
        total += my_polygonal_prism_gx(x, y, z, model, rho=rho,
                                       spacing=spacing, nz=nz, nx=nx, ny=ny)
    return total


def my_polygonal_prisms_gy(x, y, z, models, rho=None,
                            spacing=None, nz=1, nx=None, ny=None):
    '''Compute gy of multiple polygonal prisms.'''

    total = numpy.zeros_like(_as_array(x), dtype=float)
    for model in models:
        total += my_polygonal_prism_gy(x, y, z, model, rho=rho,
                                       spacing=spacing, nz=nz, nx=nx, ny=ny)
    return total


def my_polygonal_prisms_gz(x, y, z, models, rho=None,
                            spacing=None, nz=1, nx=None, ny=None):
    '''Compute gz of multiple polygonal prisms.'''

    total = numpy.zeros_like(_as_array(x), dtype=float)
    for model in models:
        total += my_polygonal_prism_gz(x, y, z, model, rho=rho,
                                       spacing=spacing, nz=nz, nx=nx, ny=ny)
    return total


def my_polygonal_prisms_gxyz(x, y, z, models, rho=None,
                              spacing=None, nz=1, nx=None, ny=None):
    '''Compute gx, gy and gz of multiple polygonal prisms.'''

    gx = my_polygonal_prisms_gx(x, y, z, models, rho=rho, spacing=spacing, nz=nz, nx=nx, ny=ny)
    gy = my_polygonal_prisms_gy(x, y, z, models, rho=rho, spacing=spacing, nz=nz, nx=nx, ny=ny)
    gz = my_polygonal_prisms_gz(x, y, z, models, rho=rho, spacing=spacing, nz=nz, nx=nx, ny=ny)
    return gx, gy, gz


def my_polygonal_prisms_bxyz(x, y, z, models, mag=None,
                              incs=90.0, decs=0.0,
                              spacing=None, nz=1, nx=None, ny=None):
    '''Compute magnetic components of multiple polygonal prisms.'''

    bx_total = numpy.zeros_like(_as_array(x), dtype=float)
    by_total = numpy.zeros_like(_as_array(x), dtype=float)
    bz_total = numpy.zeros_like(_as_array(x), dtype=float)

    for model in models:
        bx, by, bz = my_polygonal_prism_bxyz(x, y, z, model, mag=mag,
                                             incs=incs, decs=decs,
                                             spacing=spacing, nz=nz, nx=nx, ny=ny)
        bx_total += bx
        by_total += by
        bz_total += bz

    return bx_total, by_total, bz_total


def my_polygonal_prisms_tfa(x, y, z, models, mag=None,
                             inc=-20.0, dec=0.0,
                             incs=None, decs=None,
                             spacing=None, nz=1, nx=None, ny=None):
    '''Compute total-field anomaly of multiple polygonal prisms.'''

    total = numpy.zeros_like(_as_array(x), dtype=float)
    for model in models:
        total += my_polygonal_prism_tfa(x, y, z, model, mag=mag,
                                        inc=inc, dec=dec,
                                        incs=incs, decs=decs,
                                        spacing=spacing, nz=nz, nx=nx, ny=ny)
    return total


def my_polygonal_prisms_tf(*args, **kwargs):
    '''Alias for my_polygonal_prisms_tfa.'''
    return my_polygonal_prisms_tfa(*args, **kwargs)


# -----------------------------------------------------------------------------------
# Conversion to rectangular prisms
# -----------------------------------------------------------------------------------

def my_polygonal_prism_to_prisms(model, spacing=None, nz=1, nx=None, ny=None,
                                  physical_property=None):
    '''
    Convert a polygonal prism into a list of small rectangular prisms.

    This is useful to reuse rectangular-prism routines from prism.py.

    Inputs:
    model - list - [vertices, top, bottom, physical_property]
    spacing - float/None - horizontal cell size
    nz - int - vertical subdivisions
    nx, ny - int/None - number of horizontal samples if spacing is None
    physical_property - float/None - property stored in each prism

    Output:
    prisms - list - each prism as [xi, xf, yi, yf, top, bottom, property]
    '''

    vertices, top, bottom, prop = model

    if physical_property is None:
        physical_property = prop

    vertices = _as_array(vertices)
    xmin, xmax, ymin, ymax = my_polygon_bounds(vertices)

    if spacing is None:
        if nx is None:
            nx = 40
        if ny is None:
            ny = 40
        xs = numpy.linspace(xmin, xmax, int(nx))
        ys = numpy.linspace(ymin, ymax, int(ny))
        dx = abs(xs[1] - xs[0]) if len(xs) > 1 else (xmax - xmin)
        dy = abs(ys[1] - ys[0]) if len(ys) > 1 else (ymax - ymin)
    else:
        dx = spacing
        dy = spacing
        xs = numpy.arange(xmin + 0.5*dx, xmax, dx)
        ys = numpy.arange(ymin + 0.5*dy, ymax, dy)

    xg, yg = numpy.meshgrid(xs, ys)
    inside = my_points_in_polygon(xg.ravel(), yg.ravel(), vertices)

    x_inside = xg.ravel()[inside]
    y_inside = yg.ravel()[inside]

    z_edges = numpy.linspace(top, bottom, nz + 1)

    prisms = []
    for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
        for xi, yi in zip(x_inside, y_inside):
            prisms.append([xi - 0.5*dx, xi + 0.5*dx,
                           yi - 0.5*dy, yi + 0.5*dy,
                           z0, z1, physical_property])

    return prisms


# -----------------------------------------------------------------------------------
# Compatibility aliases
# -----------------------------------------------------------------------------------

my_polyprism_model = my_polygonal_prism_model
my_polyprism_vertices = my_regular_polygon_vertices
my_polyprism_area = my_polygon_area
my_polyprism_centroid = my_polygon_centroid
my_polyprism_volume = my_polygon_volume

my_polyprism_potential = my_polygonal_prism_potential
my_polyprism_gx = my_polygonal_prism_gx
my_polyprism_gy = my_polygonal_prism_gy
my_polyprism_gz = my_polygonal_prism_gz
my_polyprism_gxyz = my_polygonal_prism_gxyz

my_polyprism_bx = my_polygonal_prism_bx
my_polyprism_by = my_polygonal_prism_by
my_polyprism_bz = my_polygonal_prism_bz
my_polyprism_bxyz = my_polygonal_prism_bxyz
my_polyprism_tfa = my_polygonal_prism_tfa
my_polyprism_tf = my_polygonal_prism_tf

my_polyprisms_gz = my_polygonal_prisms_gz
my_polyprisms_tfa = my_polygonal_prisms_tfa
my_polyprisms_tf = my_polygonal_prisms_tf

polygonal_prism_gz = my_polygonal_prism_gz
polygonal_prism_tfa = my_polygonal_prism_tfa
polygonal_prisms_gz = my_polygonal_prisms_gz
polygonal_prisms_tfa = my_polygonal_prisms_tfa
