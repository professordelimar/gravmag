# -----------------------------------------------------------------------------------
# Title: Grid utilities
# Description: Regular, irregular and special grids for gravimetric/magnetic modelling
# Author: Nelson Ribeiro Filho
# Revised with compatibility improvements and additional grid generators
# -----------------------------------------------------------------------------------

from __future__ import division

import numpy
import warnings

try:
    from scipy.interpolate import griddata, interp1d, RegularGridInterpolator
except ImportError:  # pragma: no cover
    griddata = None
    interp1d = None
    RegularGridInterpolator = None


# ============================================================
# INTERNAL CHECKS AND BASIC UTILITIES
# ============================================================

def _check_area(area):
    '''
    Check and return the limits of a rectangular area.

    Inputs:
    area - list/tuple/array - [xi, xf, yi, yf]

    Outputs:
    xi, xf, yi, yf - floats - area limits
    '''

    if len(area) != 4:
        raise ValueError("area must have four elements: [xi, xf, yi, yf]")

    xi, xf, yi, yf = area

    if xi >= xf:
        raise ValueError("xf must be greater than xi")

    if yi >= yf:
        raise ValueError("yf must be greater than yi")

    return float(xi), float(xf), float(yi), float(yf)


def _check_shape(shape):
    '''
    Check and return the number of points in x and y directions.

    Inputs:
    shape - tuple/list - (nx, ny)

    Outputs:
    nx, ny - ints - number of points in x and y directions
    '''

    if len(shape) != 2:
        raise ValueError("shape must have two elements: (nx, ny)")

    nx, ny = int(shape[0]), int(shape[1])

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")

    return nx, ny


def _as_level_array(level, size):
    '''
    Convert a scalar or array level to a 1D array with a chosen size.
    '''

    if level is None:
        return None

    if numpy.isscalar(level):
        return float(level)*numpy.ones(size)

    level_array = numpy.asarray(level, dtype=float).ravel()

    if level_array.size != size:
        raise ValueError("level array must have the same size as the grid")

    return level_array


def my_shape_to_size(shape):
    '''
    Return nx*ny for a grid shape.
    '''

    nx, ny = _check_shape(shape)
    return nx*ny


def my_grid_spacing(x, y):
    '''
    Estimate grid spacing from a regular grid.

    Inputs:
    x, y - numpy arrays - 1D flattened or 2D coordinate grids

    Outputs:
    dx, dy - floats - spacing in x and y directions
    '''

    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim == 1:
        # Try to infer shape from unique coordinates.
        xu = numpy.unique(x)
        yu = numpy.unique(y)
        if xu.size < 2 or yu.size < 2:
            raise ValueError("x and y must contain at least two unique values")
        dx = numpy.mean(numpy.diff(numpy.sort(xu)))
        dy = numpy.mean(numpy.diff(numpy.sort(yu)))
    elif x.ndim == 2:
        dx = numpy.mean(numpy.diff(x[:, 0])) if x.shape[0] > 1 else numpy.mean(numpy.diff(x[0, :]))
        dy = numpy.mean(numpy.diff(y[0, :])) if y.shape[1] > 1 else numpy.mean(numpy.diff(y[:, 0]))
    else:
        raise ValueError("x and y must be 1D or 2D arrays")

    return abs(float(dx)), abs(float(dy))


def my_grid_area(x, y):
    '''
    Return the area limits [xi, xf, yi, yf] from coordinate arrays.
    '''

    return [float(numpy.nanmin(x)), float(numpy.nanmax(x)),
            float(numpy.nanmin(y)), float(numpy.nanmax(y))]


def my_grid_shape_from_vectors(x, y):
    '''
    Infer the regular grid shape from flattened coordinate vectors.

    Inputs:
    x, y - numpy arrays - flattened coordinate vectors

    Output:
    shape - tuple - (nx, ny)
    '''

    x = numpy.asarray(x).ravel()
    y = numpy.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("x and y must have the same size")

    nx = numpy.unique(x).size
    ny = numpy.unique(y).size

    if nx*ny != x.size:
        raise ValueError("x and y do not seem to define a complete regular grid")

    return (nx, ny)


def my_as_grid(x, y, data=None, shape=None):
    '''
    Reshape flattened x, y and optionally data to 2D grids.

    Important:
    This function follows the package convention used by my_regular:
    x varies along the first axis and y varies along the second axis.

    Inputs:
    x, y - numpy arrays - flattened or 2D coordinates
    data - None or numpy array - flattened or 2D data
    shape - None or tuple - (nx, ny)

    Outputs:
    xg, yg or xg, yg, dg - 2D arrays
    '''

    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim == 2:
        xg = x.copy()
        yg = y.copy()
        if data is None:
            return xg, yg
        data = numpy.asarray(data)
        if data.shape == x.shape:
            return xg, yg, data.copy()
        return xg, yg, data.reshape(x.shape)

    if shape is None:
        shape = my_grid_shape_from_vectors(x, y)

    nx, ny = _check_shape(shape)
    xg = x.reshape(nx, ny)
    yg = y.reshape(nx, ny)

    if data is None:
        return xg, yg

    dg = numpy.asarray(data).reshape(nx, ny)
    return xg, yg, dg


def my_as_vector(*arrays):
    '''
    Return flattened versions of all input arrays.
    '''

    if len(arrays) == 1:
        return numpy.asarray(arrays[0]).ravel()

    return tuple(numpy.asarray(array).ravel() for array in arrays)


# ============================================================
# ORIGINAL GRID FUNCTIONS - COMPATIBLE VERSIONS
# ============================================================

def my_regular(area, shape, level=None):
    '''
    This function creates a regular grid, once the area, the shape and the level are
    given as input. The area must have four elements named as [xi, xf, yi, yf].
    The shape represents the grid size. The level indicates the value over the grid,
    which is converted to an array with same size as x and y.

    Inputs:
    area - list - [xi, xf, yi, yf]
    shape - tuple - (nx, ny)
    level - float or array or None - observation level, positive downward

    Outputs:
    xp, yp - numpy 1D arrays - grid points
    zp - numpy 1D array - observation level, returned only if level is not None

    Notes:
    The output is flattened to preserve compatibility with the original code.
    '''

    xi, xf, yi, yf = _check_area(area)
    nx, ny = _check_shape(shape)

    x = numpy.linspace(xi, xf, nx)
    y = numpy.linspace(yi, yf, ny)

    # Original convention preserved: xp and yp have shape (nx, ny).
    yp, xp = numpy.meshgrid(y, x)

    xp = xp.reshape(nx*ny)
    yp = yp.reshape(nx*ny)

    if level is not None:
        zp = _as_level_array(level, nx*ny)
        return xp, yp, zp

    return xp, yp


def my_regular_grid(area, shape, level=None):
    '''
    Create a regular grid and return 2D arrays.

    Inputs:
    area - list - [xi, xf, yi, yf]
    shape - tuple - (nx, ny)
    level - float or array or None

    Outputs:
    xp, yp - numpy 2D arrays
    zp - numpy 2D array, returned only if level is not None
    '''

    xi, xf, yi, yf = _check_area(area)
    nx, ny = _check_shape(shape)

    x = numpy.linspace(xi, xf, nx)
    y = numpy.linspace(yi, yf, ny)
    yp, xp = numpy.meshgrid(y, x)

    if level is not None:
        zp = _as_level_array(level, nx*ny).reshape(nx, ny)
        return xp, yp, zp

    return xp, yp


def my_irregular(area, n, z=None, seed=None):
    '''
    This function creates an irregular random grid inside a rectangular area.

    Inputs:
    area - list - [xi, xf, yi, yf]
    n - int - number of points
    z - float or array or None - observation level
    seed - int or None - random seed

    Outputs:
    xarray, yarray - numpy arrays
    zarray - numpy array, returned only if z is not None
    '''

    xi, xf, yi, yf = _check_area(area)

    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = numpy.random.default_rng(seed)
    xarray = rng.uniform(xi, xf, n)
    yarray = rng.uniform(yi, yf, n)

    if z is not None:
        zarray = _as_level_array(z, n)
        return xarray, yarray, zarray

    return xarray, yarray


def my_random(area, n, level=None, seed=None):
    '''
    Alias for my_irregular, with level as a more explicit name.
    '''

    return my_irregular(area, n, z=level, seed=seed)


def my_profile(x, y, data, p1, p2, size, method='cubic', extrapolate=True):
    '''
    It draws an interpolated profile between two data points.

    Inputs:
    x, y - numpy arrays - observation points
    data - numpy array - observed data
    p1 - list/tuple - initial profile point (x, y)
    p2 - list/tuple - final profile point (x, y)
    size - int - number of points along profile
    method - string - interpolation method: 'linear', 'nearest' or 'cubic'
    extrapolate - bool - if True, fills NaNs with nearest interpolation

    Outputs:
    xp, yp - numpy arrays - profile coordinates
    profile - numpy array - interpolated profile
    '''

    if griddata is None:
        raise ImportError("scipy is required for my_profile")

    x = numpy.asarray(x).ravel()
    y = numpy.asarray(y).ravel()
    data = numpy.asarray(data).ravel()

    if not (x.size == y.size == data.size):
        raise ValueError("x, y and data must have the same size")

    if size <= 1:
        raise ValueError("size must be greater than 1")

    x1, y1 = p1
    x2, y2 = p2

    maxdist = numpy.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    distances = numpy.linspace(0.0, maxdist, int(size))

    angle = numpy.arctan2(y2 - y1, x2 - x1)
    xp = x1 + distances*numpy.cos(angle)
    yp = y1 + distances*numpy.sin(angle)

    profile = griddata((x, y), data, (xp, yp), method=method)

    if extrapolate and method != 'nearest' and numpy.any(numpy.isnan(profile)):
        nans = numpy.isnan(profile)
        profile[nans] = griddata((x, y), data, (xp[nans], yp[nans]), method='nearest')

    return xp, yp, profile


def my_padzeros(vector, width, ax, kwargs):
    '''
    This function pads an array with zeros. It can be used with numpy.pad.
    '''

    vector[:width[0]] = 0.0
    vector[-width[1]:] = 0.0
    return vector


def my_padones(vector, width, ax, kwargs):
    '''
    This function pads an array with ones. It can be used with numpy.pad.
    '''

    vector[:width[0]] = 1.0
    vector[-width[1]:] = 1.0
    return vector


def my_1Dinterpolation(x, y, n, kind='linear', fill_value='extrapolate'):
    '''
    It returns a 1D interpolation of an array.

    Inputs:
    x - numpy array - 1D array of variable x
    y - numpy array - 1D array of function y
    n - int - number of points to interpolate
    kind - string - interpolation kind accepted by scipy.interpolate.interp1d
    fill_value - string/float - extrapolation control

    Outputs:
    xi, yi - numpy arrays - interpolated variable and function
    '''

    if interp1d is None:
        raise ImportError("scipy is required for my_1Dinterpolation")

    x = numpy.asarray(x).ravel()
    y = numpy.asarray(y).ravel()

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if int(n) <= 0:
        raise ValueError("Number of interpolated points must be positive")

    order = numpy.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    f = interp1d(x_sorted, y_sorted, kind=kind, bounds_error=False,
                 fill_value=fill_value)

    xi = numpy.linspace(x_sorted.min(), x_sorted.max(), int(n))
    yi = f(xi)

    return xi, yi


def my_griddata(x, y, values, datashape, area=None, method='cubic',
                extrapolate=True, fill_method='nearest'):
    '''
    This function creates a regular grid and interpolates irregular data.

    Inputs:
    x, y - numpy arrays - coordinates
    values - numpy array - values to interpolate
    datashape - tuple - (nx, ny)
    area - None or list - [xi, xf, yi, yf]
    method - string - 'linear', 'nearest' or 'cubic'
    extrapolate - bool - if True, fills NaNs with nearest interpolation
    fill_method - string - interpolation method for NaN filling

    Outputs:
    xp, yp, grid - numpy 2D arrays
    '''

    if griddata is None:
        raise ImportError("scipy is required for my_griddata")

    x = numpy.asarray(x).ravel()
    y = numpy.asarray(y).ravel()
    values = numpy.asarray(values).ravel()

    if not (x.size == y.size == values.size):
        raise ValueError("x, y and values must have the same size")

    if area is None:
        area = [x.min(), x.max(), y.min(), y.max()]

    xi, xf, yi, yf = _check_area(area)
    nx, ny = _check_shape(datashape)

    xp, yp = my_regular_grid([xi, xf, yi, yf], (nx, ny))

    valid = numpy.isfinite(x) & numpy.isfinite(y) & numpy.isfinite(values)
    if valid.sum() < 3 and method != 'nearest':
        warnings.warn("Not enough valid points for selected method. Using nearest.")
        method = 'nearest'

    grid = griddata((x[valid], y[valid]), values[valid], (xp, yp), method=method)

    if extrapolate and method != 'nearest' and numpy.any(numpy.isnan(grid)):
        nans = numpy.isnan(grid)
        grid[nans] = griddata((x[valid], y[valid]), values[valid],
                              (xp[nans], yp[nans]), method=fill_method)

    return xp, yp, grid


# ============================================================
# ADDITIONAL GRID GENERATORS
# ============================================================

def my_circular_grid(center, radius, nr, ntheta, level=None,
                     include_center=True, endpoint=False):
    '''
    Create a circular/polar grid.

    Inputs:
    center - tuple/list - (xc, yc)
    radius - float - maximum radius
    nr - int - number of radial samples
    ntheta - int - number of angular samples
    level - float or None - z level
    include_center - bool - if True, starts radial vector at zero
    endpoint - bool - if True, includes 2*pi in angular vector

    Outputs:
    x, y - numpy 1D arrays
    z - numpy 1D array, returned only if level is not None
    '''

    xc, yc = center

    if radius <= 0.0:
        raise ValueError("radius must be positive")

    nr = int(nr)
    ntheta = int(ntheta)

    if nr <= 0 or ntheta <= 0:
        raise ValueError("nr and ntheta must be positive integers")

    if include_center:
        r = numpy.linspace(0.0, radius, nr)
    else:
        r = numpy.linspace(radius/nr, radius, nr)

    theta = numpy.linspace(0.0, 2.0*numpy.pi, ntheta, endpoint=endpoint)
    tt, rr = numpy.meshgrid(theta, r)

    x = xc + rr*numpy.cos(tt)
    y = yc + rr*numpy.sin(tt)

    x = x.ravel()
    y = y.ravel()

    if level is not None:
        z = _as_level_array(level, x.size)
        return x, y, z

    return x, y


def my_concentric_circles(center, radii, points_per_circle, level=None,
                          include_center=False):
    '''
    Create points distributed over several concentric circles.

    Inputs:
    center - tuple/list - (xc, yc)
    radii - array/list - circle radii
    points_per_circle - int or list - points on each circle
    level - float or None - z level
    include_center - bool - if True, includes the center point

    Outputs:
    x, y or x, y, z - numpy arrays
    '''

    xc, yc = center
    radii = numpy.asarray(radii, dtype=float).ravel()

    if numpy.any(radii < 0.0):
        raise ValueError("radii must be non-negative")

    if numpy.isscalar(points_per_circle):
        npc = numpy.full(radii.size, int(points_per_circle), dtype=int)
    else:
        npc = numpy.asarray(points_per_circle, dtype=int).ravel()
        if npc.size != radii.size:
            raise ValueError("points_per_circle must be scalar or have same size as radii")

    xs = []
    ys = []

    if include_center:
        xs.append(numpy.array([xc], dtype=float))
        ys.append(numpy.array([yc], dtype=float))

    for r, n in zip(radii, npc):
        if n <= 0:
            continue
        theta = numpy.linspace(0.0, 2.0*numpy.pi, n, endpoint=False)
        xs.append(xc + r*numpy.cos(theta))
        ys.append(yc + r*numpy.sin(theta))

    x = numpy.concatenate(xs) if xs else numpy.array([], dtype=float)
    y = numpy.concatenate(ys) if ys else numpy.array([], dtype=float)

    if level is not None:
        z = _as_level_array(level, x.size)
        return x, y, z

    return x, y


def my_radial_random_grid(center, radius, n, level=None, seed=None,
                          uniform_area=True):
    '''
    Create random points inside a circle.

    Inputs:
    center - tuple/list - (xc, yc)
    radius - float - circle radius
    n - int - number of points
    level - float or None - z level
    seed - int or None - random seed
    uniform_area - bool - if True, samples uniformly in area

    Outputs:
    x, y or x, y, z - numpy arrays
    '''

    xc, yc = center

    if radius <= 0.0:
        raise ValueError("radius must be positive")

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    rng = numpy.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0*numpy.pi, n)

    if uniform_area:
        r = radius*numpy.sqrt(rng.uniform(0.0, 1.0, n))
    else:
        r = radius*rng.uniform(0.0, 1.0, n)

    x = xc + r*numpy.cos(theta)
    y = yc + r*numpy.sin(theta)

    if level is not None:
        z = _as_level_array(level, n)
        return x, y, z

    return x, y


def my_triangular_grid(area, spacing=None, shape=None, level=None):
    '''
    Create a triangular/staggered grid inside a rectangular area.

    Inputs:
    area - list - [xi, xf, yi, yf]
    spacing - float or tuple or None - dx or (dx, dy)
    shape - tuple or None - approximate (nx, ny), used if spacing is None
    level - float or None - z level

    Outputs:
    x, y or x, y, z - numpy arrays
    '''

    xi, xf, yi, yf = _check_area(area)

    if spacing is None:
        if shape is None:
            raise ValueError("Provide either spacing or shape")
        nx, ny = _check_shape(shape)
        xvec = numpy.linspace(xi, xf, nx)
        yvec = numpy.linspace(yi, yf, ny)
        dx = xvec[1] - xvec[0] if nx > 1 else xf - xi
    else:
        if numpy.isscalar(spacing):
            dx = dy = float(spacing)
        else:
            dx, dy = float(spacing[0]), float(spacing[1])
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("spacing must be positive")
        xvec = numpy.arange(xi, xf + 0.5*dx, dx)
        yvec = numpy.arange(yi, yf + 0.5*dy, dy)

    xs = []
    ys = []

    for j, yy in enumerate(yvec):
        offset = 0.5*dx if (j % 2 == 1) else 0.0
        xx = xvec + offset
        valid = (xx >= xi) & (xx <= xf)
        xs.append(xx[valid])
        ys.append(numpy.full(valid.sum(), yy))

    x = numpy.concatenate(xs)
    y = numpy.concatenate(ys)

    if level is not None:
        z = _as_level_array(level, x.size)
        return x, y, z

    return x, y


def my_hexagonal_grid(area, spacing, level=None):
    '''
    Create a hexagonal sampling grid inside a rectangular area.

    Inputs:
    area - list - [xi, xf, yi, yf]
    spacing - float - distance between neighboring points in x direction
    level - float or None - z level

    Outputs:
    x, y or x, y, z - numpy arrays
    '''

    xi, xf, yi, yf = _check_area(area)

    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    dx = spacing
    dy = spacing*numpy.sqrt(3.0)/2.0

    return my_triangular_grid(area, spacing=(dx, dy), level=level)


def my_line_grid(p1, p2, n, level=None):
    '''
    Create points along a straight line.

    Inputs:
    p1, p2 - tuple/list - initial and final points (x, y)
    n - int - number of points
    level - float or None - z level

    Outputs:
    x, y or x, y, z - numpy arrays
    '''

    n = int(n)
    if n <= 1:
        raise ValueError("n must be greater than 1")

    x = numpy.linspace(p1[0], p2[0], n)
    y = numpy.linspace(p1[1], p2[1], n)

    if level is not None:
        z = _as_level_array(level, n)
        return x, y, z

    return x, y


def my_cross_grid(center, length, n_per_line, level=None, angle=0.0):
    '''
    Create a cross-shaped grid formed by two perpendicular lines.

    Inputs:
    center - tuple/list - (xc, yc)
    length - float - total length of each line
    n_per_line - int - number of points per line
    level - float or None - z level
    angle - float - rotation angle in degrees

    Outputs:
    x, y or x, y, z - numpy arrays
    '''

    xc, yc = center
    half = 0.5*float(length)
    n = int(n_per_line)

    t = numpy.linspace(-half, half, n)
    x1, y1 = t, numpy.zeros_like(t)
    x2, y2 = numpy.zeros_like(t), t

    x = numpy.concatenate([x1, x2])
    y = numpy.concatenate([y1, y2])

    ang = numpy.deg2rad(angle)
    xr = x*numpy.cos(ang) - y*numpy.sin(ang) + xc
    yr = x*numpy.sin(ang) + y*numpy.cos(ang) + yc

    if level is not None:
        z = _as_level_array(level, xr.size)
        return xr, yr, z

    return xr, yr


def my_spiral_grid(center, radius, n, turns=3.0, level=None):
    '''
    Create points along an Archimedean spiral.

    Inputs:
    center - tuple/list - (xc, yc)
    radius - float - maximum radius
    n - int - number of points
    turns - float - number of spiral turns
    level - float or None - z level

    Outputs:
    x, y or x, y, z - numpy arrays
    '''

    xc, yc = center
    n = int(n)

    if n <= 1:
        raise ValueError("n must be greater than 1")

    theta = numpy.linspace(0.0, 2.0*numpy.pi*turns, n)
    r = numpy.linspace(0.0, radius, n)

    x = xc + r*numpy.cos(theta)
    y = yc + r*numpy.sin(theta)

    if level is not None:
        z = _as_level_array(level, n)
        return x, y, z

    return x, y


def my_checkerboard_grid(area, shape, level=None, return_mask=False):
    '''
    Create a regular grid and a checkerboard mask, useful for synthetic tests.

    Inputs:
    area - list - [xi, xf, yi, yf]
    shape - tuple - (nx, ny)
    level - float or None - z level
    return_mask - bool - if True, returns the checkerboard mask

    Outputs:
    x, y or x, y, z or x, y, mask or x, y, z, mask
    '''

    if level is None:
        x, y = my_regular(area, shape)
    else:
        x, y, z = my_regular(area, shape, level=level)

    nx, ny = _check_shape(shape)
    ii, jj = numpy.indices((nx, ny))
    mask = ((ii + jj) % 2).ravel()

    if level is None:
        if return_mask:
            return x, y, mask
        return x, y

    if return_mask:
        return x, y, z, mask

    return x, y, z


# ============================================================
# GRID TRANSFORMATIONS AND INTERPOLATION HELPERS
# ============================================================

def my_rotate_grid(x, y, angle, center=None):
    '''
    Rotate coordinates around a center.

    Inputs:
    x, y - numpy arrays - coordinates
    angle - float - rotation angle in degrees, counterclockwise positive
    center - None or tuple - (xc, yc). If None, uses the mean coordinate.

    Outputs:
    xr, yr - numpy arrays - rotated coordinates
    '''

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if center is None:
        xc = numpy.nanmean(x)
        yc = numpy.nanmean(y)
    else:
        xc, yc = center

    angle = numpy.deg2rad(angle)
    x0 = x - xc
    y0 = y - yc

    xr = x0*numpy.cos(angle) - y0*numpy.sin(angle) + xc
    yr = x0*numpy.sin(angle) + y0*numpy.cos(angle) + yc

    return xr, yr


def my_translate_grid(x, y, dx=0.0, dy=0.0):
    '''
    Translate coordinates by dx and dy.
    '''

    return numpy.asarray(x) + dx, numpy.asarray(y) + dy


def my_resample_grid(x, y, data, new_shape, method='cubic', extrapolate=True):
    '''
    Resample gridded or irregular data to a new regular grid.

    Inputs:
    x, y - numpy arrays - coordinates
    data - numpy array - data values
    new_shape - tuple - (nx, ny)
    method - string - interpolation method
    extrapolate - bool - fill NaNs with nearest interpolation

    Outputs:
    xn, yn, dn - numpy 2D arrays
    '''

    area = my_grid_area(x, y)
    return my_griddata(x, y, data, new_shape, area=area,
                       method=method, extrapolate=extrapolate)


def my_extract_subgrid(x, y, data, area):
    '''
    Extract points inside a rectangular area.

    Inputs:
    x, y - numpy arrays - coordinates
    data - numpy array - values
    area - list - [xi, xf, yi, yf]

    Outputs:
    xs, ys, ds - numpy arrays - selected points and values
    '''

    xi, xf, yi, yf = _check_area(area)
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    data = numpy.asarray(data)

    if not (x.shape == y.shape == data.shape):
        raise ValueError("x, y and data must have the same shape")

    mask = (x >= xi) & (x <= xf) & (y >= yi) & (y <= yf)

    return x[mask], y[mask], data[mask]


def my_nearest_point(x, y, point):
    '''
    Find the index of the nearest point to a chosen coordinate.

    Inputs:
    x, y - numpy arrays - coordinates
    point - tuple/list - (xp, yp)

    Outputs:
    index - int - flattened index of nearest point
    distance - float - distance to nearest point
    '''

    x = numpy.asarray(x).ravel()
    y = numpy.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("x and y must have the same size")

    xp, yp = point
    dist = numpy.sqrt((x - xp)**2 + (y - yp)**2)
    index = int(numpy.nanargmin(dist))

    return index, float(dist[index])


def my_grid_to_table(x, y, data, level=None):
    '''
    Convert grid arrays to a table with columns x, y, data or x, y, z, data.
    '''

    x = numpy.asarray(x).ravel()
    y = numpy.asarray(y).ravel()
    data = numpy.asarray(data).ravel()

    if not (x.size == y.size == data.size):
        raise ValueError("x, y and data must have the same size")

    if level is None:
        return numpy.column_stack((x, y, data))

    z = _as_level_array(level, x.size)
    return numpy.column_stack((x, y, z, data))


def my_save_grid(filename, x, y, data, level=None, header=None, fmt='%.10e'):
    '''
    Save grid/table data to a text file.

    Inputs:
    filename - string - output file name
    x, y - numpy arrays - coordinates
    data - numpy array - data values
    level - None, scalar or array - optional z coordinate
    header - string or None - file header
    fmt - string - number format
    '''

    table = my_grid_to_table(x, y, data, level=level)

    if header is None:
        header = 'x y value' if level is None else 'x y z value'

    numpy.savetxt(filename, table, fmt=fmt, header=header)


def my_load_grid(filename, usecols=(0, 1, 2), skiprows=0):
    '''
    Load x, y and data columns from a text file.

    Inputs:
    filename - string - input file name
    usecols - tuple - columns to read
    skiprows - int - number of header rows

    Outputs:
    x, y, data - numpy arrays
    '''

    table = numpy.loadtxt(filename, usecols=usecols, skiprows=skiprows)

    if table.ndim == 1:
        table = table.reshape(1, -1)

    return table[:, 0], table[:, 1], table[:, 2]


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

my_regular2d = my_regular_grid
my_grid = my_regular
my_grid2d = my_regular_grid
my_random_grid = my_random
my_irregular_grid = my_irregular
my_circular = my_circular_grid
my_triangular = my_triangular_grid
my_hexagonal = my_hexagonal_grid
my_profile_grid = my_profile
my_interpolate = my_griddata
my_interp1d = my_1Dinterpolation
