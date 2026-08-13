# -----------------------------------------------------------------------------------
# Title: Auxiliary functions
# Description: General mathematical, geometric, numerical and geophysical utilities.
# Author: Nelson Ribeiro Filho
# Notes:
#   - Function names were preserved whenever possible to maintain compatibility.
#   - Numerical safety was improved for scalar and array inputs.
#   - No intentional change was made to the physical meaning of the existing routines.
# -----------------------------------------------------------------------------------

from __future__ import division

import warnings
import numpy
import scipy.linalg


# ============================================================
# BASIC VALIDATION UTILITIES
# ============================================================

def my_asarray(x, dtype=float):
    '''
    Convert input to numpy array without forcing copy when unnecessary.

    Inputs:
    x - scalar, list or numpy array
    dtype - data type

    Output:
    arr - numpy array
    '''
    return numpy.asarray(x, dtype=dtype)


def my_check_same_shape(*arrays):
    '''
    Check if all input arrays have the same shape.

    Inputs:
    arrays - sequence of array-like objects

    Output:
    None
    '''
    shapes = [numpy.asarray(a).shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError("All inputs must have the same shape!")


def my_check_same_size(*arrays):
    '''
    Check if all input arrays have the same size.

    Inputs:
    arrays - sequence of array-like objects

    Output:
    None
    '''
    sizes = [numpy.asarray(a).size for a in arrays]
    if len(set(sizes)) != 1:
        raise ValueError("All inputs must have the same size!")


# ============================================================
# ANGLE CONVERSIONS AND TRIGONOMETRIC FUNCTIONS
# ============================================================

def my_deg2rad(angle):
    '''
    Convert angle from degrees to radians.

    Input:
    angle - float or numpy array - angle in degrees

    Output:
    argument - float or numpy array - angle in radians
    '''
    return numpy.deg2rad(angle)


def my_rad2deg(argument):
    '''
    Convert angle from radians to degrees.

    Input:
    argument - float or numpy array - angle in radians

    Output:
    angle - float or numpy array - angle in degrees
    '''
    return numpy.rad2deg(argument)


# Backward-compatible aliases used by older codes.
deg2rad = my_deg2rad
rad2deg = my_rad2deg


def my_trigonometrics_deg(angle):
    '''
    Return sine, cosine and tangent of an angle given in degrees.

    Input:
    angle - float or numpy array - angle in degrees

    Outputs:
    mysin, mycos, mytan - float or numpy array
    '''
    angle_rad = my_deg2rad(angle)
    mysin = numpy.sin(angle_rad)
    mycos = numpy.cos(angle_rad)
    mytan = numpy.tan(angle_rad)
    return mysin, mycos, mytan


def my_trigonometrics_rad(angle):
    '''
    Return sine, cosine and tangent of an angle given in radians.

    Input:
    angle - float or numpy array - angle in radians

    Outputs:
    mysin, mycos, mytan - float or numpy array
    '''
    mysin = numpy.sin(angle)
    mycos = numpy.cos(angle)
    mytan = numpy.tan(angle)
    return mysin, mycos, mytan


def my_asin(x):
    '''
    Return arcsine using numpy.
    '''
    return numpy.arcsin(x)


def my_acos(x):
    '''
    Return arccosine using numpy.
    '''
    return numpy.arccos(x)


def my_atan(x, y):
    '''
    Return a stable arctangent using numpy.arctan2.

    This function is used in prism formulas. It accepts scalars or arrays.
    It preserves the older convention in which values with x == 0 return 0.

    Inputs:
    x, y - float or numpy array

    Output:
    arctan - float or numpy array
    '''
    x_arr = numpy.asarray(x)
    y_arr = numpy.asarray(y)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        arctan = numpy.arctan2(x_arr, y_arr)

    arctan = numpy.where(x_arr == 0, 0.0, arctan)
    arctan = numpy.where((x_arr > 0) & (y_arr < 0), arctan - numpy.pi, arctan)
    arctan = numpy.where((x_arr < 0) & (y_arr < 0), arctan + numpy.pi, arctan)

    if numpy.isscalar(x) and numpy.isscalar(y):
        return float(arctan)

    return arctan


def my_sqrt(x):
    '''
    Return square root using numpy.

    Input:
    x - float or numpy array

    Output:
    mysqrt - float or numpy array
    '''
    return numpy.sqrt(x)


def my_log(x):
    '''
    Return log(x), replacing log(0) by 0.

    This behavior is useful in closed-form prism formulas because terms
    multiplied by log arguments have finite limiting values.

    Input:
    x - float or numpy array

    Output:
    log - float or numpy array
    '''
    x_arr = numpy.asarray(x)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        log = numpy.log(x_arr)

    log = numpy.where(x_arr == 0, 0.0, log)

    if numpy.isscalar(x):
        return float(log)

    return log


def my_safe_divide(numerator, denominator, fill_value=0.0):
    '''
    Safely divide numerator by denominator.

    Inputs:
    numerator - float or numpy array
    denominator - float or numpy array
    fill_value - value used where denominator is zero

    Output:
    result - float or numpy array
    '''
    numerator = numpy.asarray(numerator)
    denominator = numpy.asarray(denominator)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        result = numerator/denominator

    result = numpy.where(numpy.isfinite(result), result, fill_value)

    if result.size == 1:
        return float(result)

    return result


# ============================================================
# LINEAR ALGEBRA UTILITIES
# ============================================================

def my_dot(x, y):
    '''
    Return dot product between two arrays.
    '''
    return numpy.dot(x, y)


def my_hadamard(x, y):
    '''
    Return Hadamard product between two arrays.
    '''
    return numpy.multiply(x, y)


def my_outer(x, y):
    '''
    Return outer product between two vectors.
    '''
    return numpy.outer(x, y)


def my_inverse(mat):
    '''
    Return matrix inverse.
    '''
    return numpy.linalg.inv(mat)


def my_pinv(mat, rcond=1.0e-15):
    '''
    Return Moore-Penrose pseudo-inverse.

    Inputs:
    mat - numpy array - input matrix
    rcond - float - cutoff for small singular values

    Output:
    invmat - numpy array - pseudo-inverse matrix
    '''
    return numpy.linalg.pinv(mat, rcond=rcond)


def my_LU(mat):
    '''
    Return LU decomposition.

    Output:
    p, l, u - permutation, lower and upper matrices
    '''
    p, l, u = scipy.linalg.lu(mat)
    return p, l, u


# ============================================================
# ROTATION MATRICES AND COORDINATE TRANSFORMATIONS
# ============================================================

def my_xrotation(angle):
    '''
    Return 3D rotation matrix around x-axis.

    Input:
    angle - float - angle in degrees

    Output:
    rotx - numpy 2D array
    '''
    c = numpy.cos(my_deg2rad(angle))
    s = numpy.sin(my_deg2rad(angle))

    rotx = numpy.array([[1.0, 0.0, 0.0],
                        [0.0, c, s],
                        [0.0, -s, c]])
    return rotx


def my_yrotation(angle):
    '''
    Return 3D rotation matrix around y-axis.

    Input:
    angle - float - angle in degrees

    Output:
    roty - numpy 2D array
    '''
    c = numpy.cos(my_deg2rad(angle))
    s = numpy.sin(my_deg2rad(angle))

    roty = numpy.array([[c, 0.0, s],
                        [0.0, 1.0, 0.0],
                        [-s, 0.0, c]])
    return roty


def my_zrotation(angle):
    '''
    Return 3D rotation matrix around z-axis.

    Input:
    angle - float - angle in degrees

    Output:
    rotz - numpy 2D array
    '''
    c = numpy.cos(my_deg2rad(angle))
    s = numpy.sin(my_deg2rad(angle))

    rotz = numpy.array([[c, s, 0.0],
                        [-s, c, 0.0],
                        [0.0, 0.0, 1.0]])
    return rotz


def rotate3D_xyz(x, y, z, angle, direction='z'):
    '''
    Rotate 3D coordinate points around x, y or z axis.

    Inputs:
    x, y, z - numpy arrays - coordinate points
    angle - float - rotation angle in degrees
    direction - string - 'x', 'y' or 'z'

    Outputs:
    xr, yr, zr - numpy arrays - rotated coordinates
    '''
    if direction == 'x':
        rot = my_xrotation(angle)
    elif direction == 'y':
        rot = my_yrotation(angle)
    elif direction == 'z':
        rot = my_zrotation(angle)
    else:
        raise ValueError("direction must be 'x', 'y' or 'z'!")

    x_arr = numpy.asarray(x)
    y_arr = numpy.asarray(y)
    z_arr = numpy.asarray(z)

    my_check_same_shape(x_arr, y_arr, z_arr)

    shape = x_arr.shape
    mat = numpy.vstack([x_arr.ravel(), y_arr.ravel(), z_arr.ravel()])
    res = rot.dot(mat)

    xr = res[0].reshape(shape)
    yr = res[1].reshape(shape)
    zr = res[2].reshape(shape)

    return xr, yr, zr


def my_spherical2cartesian(longitude, latitude, level):
    '''
    Convert spherical coordinates to geocentric Cartesian coordinates.

    Inputs:
    longitude - float or numpy array - longitude in degrees
    latitude - float or numpy array - latitude in degrees
    level - float or numpy array - height above Earth radius in meters

    Outputs:
    x, y, z - float or numpy arrays - geocentric coordinates
    '''
    R = 6378137.0 + level

    lon_rad = my_deg2rad(longitude)
    lat_rad = my_deg2rad(latitude)

    x = numpy.cos(lat_rad)*numpy.cos(lon_rad)*R
    y = numpy.cos(lat_rad)*numpy.sin(lon_rad)*R
    z = numpy.sin(lat_rad)*R

    return x, y, z


# ============================================================
# NOISE, RESIDUALS AND BASIC STATISTICS
# ============================================================

def my_normalnoise(xi, vi=0.0, std=0.0, seed=None):
    '''
    Contaminate data with Gaussian noise.

    Inputs:
    xi - float or numpy array - input data
    vi - float - noise mean
    std - float - standard deviation
    seed - int/None - random seed

    Output:
    noisy - numpy array - contaminated data
    '''
    if std < 0.0:
        raise ValueError('std must be positive or zero!')

    rng = numpy.random.default_rng(seed)
    noise = rng.normal(loc=vi, scale=std, size=numpy.asarray(xi).shape)

    return xi + noise


def my_uniformnoise(xi, vmin, vmax, seed=None):
    '''
    Contaminate data with uniform noise.

    Inputs:
    xi - float or numpy array - input data
    vmin, vmax - floats - limits of uniform distribution
    seed - int/None - random seed

    Output:
    noisy - numpy array - contaminated data
    '''
    if vmax < vmin:
        raise ValueError('vmax must be greater than or equal to vmin!')

    rng = numpy.random.default_rng(seed)
    noise = rng.uniform(vmin, vmax, size=numpy.asarray(xi).shape)

    return xi + noise


def my_residual(do, dp, eps=1.0e-15):
    '''
    Calculate residual between observed and predicted data.

    Inputs:
    do - numpy array - observed data
    dp - numpy array - predicted data
    eps - float - small number to avoid division by zero

    Outputs:
    res - numpy array - residual
    mean - float - mean residual
    norm - numpy array - normalized residual
    std - float - residual standard deviation
    '''
    do = numpy.asarray(do)
    dp = numpy.asarray(dp)

    my_check_same_shape(do, dp)

    res = do - dp
    mean = numpy.nanmean(res)
    std = numpy.nanstd(res)

    if std < eps:
        norm = numpy.zeros_like(res)
    else:
        norm = (res - mean)/std

    return res, mean, norm, std


def my_rms(data):
    '''
    Return root mean square of data.
    '''
    data = numpy.asarray(data)
    return numpy.sqrt(numpy.nanmean(data**2))


def my_rmse(observed, predicted):
    '''
    Return root mean square error between observed and predicted data.
    '''
    observed = numpy.asarray(observed)
    predicted = numpy.asarray(predicted)
    my_check_same_shape(observed, predicted)
    return my_rms(observed - predicted)


def my_mae(observed, predicted):
    '''
    Return mean absolute error between observed and predicted data.
    '''
    observed = numpy.asarray(observed)
    predicted = numpy.asarray(predicted)
    my_check_same_shape(observed, predicted)
    return numpy.nanmean(numpy.abs(observed - predicted))


# ============================================================
# MAGNETIC DIRECTION UTILITIES
# ============================================================

def my_dircos(inc, dec, azm=0.0):
    '''
    Calculate direction cosines from inclination and declination.

    Inputs:
    inc - float or numpy array - inclination in degrees
    dec - float or numpy array - declination in degrees
    azm - float - azimuth correction in degrees

    Outputs:
    xdir, ydir, zdir - projected direction cosines
    '''
    inc_rad = my_deg2rad(inc)
    dec_rad = my_deg2rad(dec)
    azm_rad = my_deg2rad(azm)

    xdir = numpy.cos(inc_rad)*numpy.cos(dec_rad - azm_rad)
    ydir = numpy.cos(inc_rad)*numpy.sin(dec_rad - azm_rad)
    zdir = numpy.sin(inc_rad)

    return xdir, ydir, zdir


def my_regional(field, inc, dec, azm=0.0):
    '''
    Calculate regional magnetic field components.

    Inputs:
    field - float - regional magnetic field intensity
    inc - float - magnetic inclination in degrees
    dec - float - magnetic declination in degrees
    azm - float - azimuth correction in degrees

    Outputs:
    fx, fy, fz - magnetic field components
    '''
    xdir, ydir, zdir = my_dircos(inc, dec, azm)

    fx = field*xdir
    fy = field*ydir
    fz = field*zdir

    return fx, fy, fz


def my_theta(inc, dec, u, v, azim=0.0, eps=1.0e-15):
    '''
    Return Fourier-domain magnetization or field-direction operator.

    Inputs:
    inc, dec - floats - inclination and declination in degrees
    u, v - numpy arrays - wavenumbers in x and y directions
    azim - float - azimuth correction
    eps - float - small number to avoid division by zero

    Output:
    theta - complex numpy array - direction operator
    '''
    u = numpy.asarray(u)
    v = numpy.asarray(v)

    k = numpy.sqrt(u**2 + v**2)
    xdir, ydir, zdir = my_dircos(inc, dec, azim)

    horizontal_projection = my_safe_divide(xdir*u + ydir*v, k, fill_value=0.0)
    theta = zdir + 1j*horizontal_projection

    return theta


# ============================================================
# WAVENUMBER UTILITIES
# ============================================================

def my_grid_spacing(x, y):
    '''
    Estimate regular grid spacing from 1D or 2D coordinate arrays.

    Inputs:
    x, y - numpy arrays - coordinates

    Outputs:
    dx, dy - floats - grid spacing
    '''
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if x.ndim == 1:
        dx = numpy.nanmean(numpy.diff(x))
    elif x.ndim == 2:
        dx = numpy.nanmean(numpy.diff(x, axis=1))
    else:
        raise ValueError('x must be a 1D or 2D array!')

    if y.ndim == 1:
        dy = numpy.nanmean(numpy.diff(y))
    elif y.ndim == 2:
        dy = numpy.nanmean(numpy.diff(y, axis=0))
    else:
        raise ValueError('y must be a 1D or 2D array!')

    if dx == 0.0 or dy == 0.0:
        raise ValueError('Grid spacing cannot be zero!')

    return abs(float(dx)), abs(float(dy))


def my_wavenumber(x, y):
    '''
    Return wavenumber grids in x and y directions.

    Inputs:
    x, y - numpy arrays - coordinate arrays. They can be 1D vectors or 2D grids.

    Outputs:
    kx, ky - numpy 2D arrays - wavenumber grids in x and y directions
    '''
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if x.ndim == 1:
        nx = x.size
    elif x.ndim == 2:
        nx = x.shape[1]
    else:
        raise ValueError('x must be a 1D or 2D array!')

    if y.ndim == 1:
        ny = y.size
    elif y.ndim == 2:
        ny = y.shape[0]
    else:
        raise ValueError('y must be a 1D or 2D array!')

    dx, dy = my_grid_spacing(x, y)

    kx_1d = 2.0*numpy.pi*numpy.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0*numpy.pi*numpy.fft.fftfreq(ny, d=dy)

    kx, ky = numpy.meshgrid(kx_1d, ky_1d)

    return kx, ky


def my_wavenumber_modulus(x, y):
    '''
    Return kx, ky and total wavenumber modulus.

    Inputs:
    x, y - numpy arrays - coordinate arrays

    Outputs:
    kx, ky, k - numpy arrays
    '''
    kx, ky = my_wavenumber(x, y)
    k = numpy.sqrt(kx**2 + ky**2)
    return kx, ky, k


# ============================================================
# PADDING AND TAPERING UTILITIES FOR FFT FILTERS
# ============================================================

def my_cosine_taper_1d(n, p=0.10):
    '''
    Create a 1D cosine taper.

    Inputs:
    n - int - number of samples
    p - float - taper fraction at each border

    Output:
    w - numpy array - taper weights
    '''
    if n <= 0:
        raise ValueError('n must be positive!')

    w = numpy.ones(n)
    m = int(p*n)

    if m < 1:
        return w

    t = numpy.linspace(0.0, numpy.pi/2.0, m)
    taper = numpy.sin(t)**2

    w[:m] = taper
    w[-m:] = taper[::-1]

    return w


def my_cosine_taper_2d(data, p=0.10):
    '''
    Apply a 2D cosine taper.

    Inputs:
    data - numpy 2D array
    p - float - taper fraction at each border

    Output:
    tapered - numpy 2D array
    '''
    data = numpy.asarray(data)

    if data.ndim != 2:
        raise ValueError('data must be a 2D array!')

    ny, nx = data.shape
    wx = my_cosine_taper_1d(nx, p=p)
    wy = my_cosine_taper_1d(ny, p=p)

    return data*numpy.outer(wy, wx)


def my_pad_grid(data, pad_factor=0.50, pad_mode='reflect', taper=True, taper_fraction=0.05):
    '''
    Expand a regular grid before FFT filtering.

    Inputs:
    data - numpy 2D array
    pad_factor - float - fraction of grid size added to each border
    pad_mode - string - numpy.pad mode
    taper - bool - if True, apply cosine taper after padding
    taper_fraction - float - taper fraction

    Outputs:
    data_pad - numpy 2D array - expanded grid
    pad_width - tuple - padding information
    mean_data - float - removed mean value
    '''
    data = numpy.asarray(data)

    if data.ndim != 2:
        raise ValueError('data must be a 2D array!')

    if pad_factor < 0.0:
        raise ValueError('pad_factor must be positive or zero!')

    ny, nx = data.shape
    py = int(pad_factor*ny)
    px = int(pad_factor*nx)

    mean_data = numpy.nanmean(data)
    data0 = data - mean_data

    pad_width = ((py, py), (px, px))
    data_pad = numpy.pad(data0, pad_width=pad_width, mode=pad_mode)

    if taper is True:
        data_pad = my_cosine_taper_2d(data_pad, p=taper_fraction)

    return data_pad, pad_width, mean_data


def my_crop_grid(data_pad, pad_width):
    '''
    Crop padded grid to original size.

    Inputs:
    data_pad - numpy 2D array
    pad_width - tuple - padding information

    Output:
    data - numpy 2D array - cropped grid
    '''
    py0, py1 = pad_width[0]
    px0, px1 = pad_width[1]

    y_slice = slice(py0, data_pad.shape[0] - py1 if py1 > 0 else data_pad.shape[0])
    x_slice = slice(px0, data_pad.shape[1] - px1 if px1 > 0 else data_pad.shape[1])

    return data_pad[y_slice, x_slice]


# ============================================================
# LEGACY PADDING FUNCTIONS USED WITH numpy.pad
# ============================================================

def my_padzeros(vector, width, ax=None, kwargs=None):
    '''
    Pad vector borders with zeros. Designed for legacy numpy.pad workflows.
    '''
    vector[:width[0]] = 0.0
    vector[-width[1]:] = 0.0
    return vector


def my_padones(vector, width, ax=None, kwargs=None):
    '''
    Pad vector borders with ones. Designed for legacy numpy.pad workflows.
    '''
    vector[:width[0]] = 1.0
    vector[-width[1]:] = 1.0
    return vector


# ============================================================
# COMPATIBILITY WRAPPERS
# ============================================================

def my_shape(data):
    '''
    Return shape of a numpy array.
    '''
    return numpy.asarray(data).shape


def my_size(data):
    '''
    Return size of a numpy array.
    '''
    return numpy.asarray(data).size


