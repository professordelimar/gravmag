# -----------------------------------------------------------------------------------
# Title: Derivative tools for potential-field data
# Description: Fourier-domain and space-domain numerical derivatives for regular grids
# Author: Nelson Ribeiro Filho
# -----------------------------------------------------------------------------------

from __future__ import division

import warnings
import numpy

try:
    from . import auxiliars
except ImportError:  # Allows running this file outside a package named "codes".
    import auxiliars


# ============================================================
# INTERNAL CHECKING AND SHAPE UTILITIES
# ============================================================

def _check_grid_inputs(x, y, data):
    '''
    Internal function that checks if x, y and data are 2D arrays with the same shape.
    '''

    if not hasattr(x, "shape") or not hasattr(y, "shape") or not hasattr(data, "shape"):
        raise ValueError("x, y and data must be numpy arrays!")

    if x.shape != y.shape or x.shape != data.shape:
        raise ValueError("All inputs must have same shape!")

    if data.ndim != 2:
        raise ValueError("x, y and data must be 2D arrays representing a regular grid!")


def _as_output(result, original_shape=None, flatten=True):
    '''
    Internal function that controls the output format.

    The original functions in this file returned 1D arrays. Therefore, flatten=True
    is the default for backward compatibility. New functions can use flatten=False.
    '''

    if flatten is True:
        return numpy.asarray(result).reshape(numpy.asarray(result).size)

    if original_shape is not None:
        return numpy.asarray(result).reshape(original_shape)

    return numpy.asarray(result)


def _grid_spacing_from_coordinates(x, y):
    '''
    Internal function that estimates dx and dy from 2D regular coordinate grids.
    '''

    _check_grid_inputs(x, y, numpy.zeros_like(x))

    # x varies along columns for common meshgrid indexing='xy'. In the user's
    # grids.py, xp and yp are flattened, but when reshaped to 2D this convention
    # can vary. Therefore, both alternatives are checked.
    dx_candidates = []
    dy_candidates = []

    if x.shape[1] > 1:
        dx_candidates.append(numpy.nanmean(numpy.diff(x, axis=1)))
    if x.shape[0] > 1:
        dx_candidates.append(numpy.nanmean(numpy.diff(x, axis=0)))

    if y.shape[0] > 1:
        dy_candidates.append(numpy.nanmean(numpy.diff(y, axis=0)))
    if y.shape[1] > 1:
        dy_candidates.append(numpy.nanmean(numpy.diff(y, axis=1)))

    dx_candidates = [abs(v) for v in dx_candidates if numpy.isfinite(v) and abs(v) > 0.0]
    dy_candidates = [abs(v) for v in dy_candidates if numpy.isfinite(v) and abs(v) > 0.0]

    if len(dx_candidates) == 0 or len(dy_candidates) == 0:
        raise ValueError("Unable to estimate grid spacing from x and y.")

    dx = dx_candidates[0]
    dy = dy_candidates[0]

    return dx, dy


def _get_wavenumbers(x, y):
    '''
    Internal function that obtains ky, kx from auxiliars.my_wavenumber if available.
    A local fallback is used if the auxiliary function is not available.
    '''

    if hasattr(auxiliars, "my_wavenumber"):
        kx, ky = auxiliars.my_wavenumber(x, y)
        return ky, kx

    dx, dy = _grid_spacing_from_coordinates(x, y)
    ny, nx = x.shape

    kx_1d = 2.0*numpy.pi*numpy.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0*numpy.pi*numpy.fft.fftfreq(ny, d=dy)

    kx, ky = numpy.meshgrid(kx_1d, ky_1d)

    return ky, kx


def _cosine_taper_1d(n, percent=0.10):
    '''
    Internal 1D cosine taper.
    '''

    w = numpy.ones(n)
    m = int(percent*n)

    if m < 1:
        return w

    t = numpy.linspace(0.0, numpy.pi/2.0, m)
    taper = numpy.sin(t)**2

    w[:m] = taper
    w[-m:] = taper[::-1]

    return w


def _cosine_taper_2d(data, percent=0.10):
    '''
    Internal 2D cosine taper.
    '''

    ny, nx = data.shape
    wx = _cosine_taper_1d(nx, percent=percent)
    wy = _cosine_taper_1d(ny, percent=percent)

    return data*numpy.outer(wy, wx)


def _pad_data(data, pad_factor=0.0, pad_mode="reflect", taper=False):
    '''
    Internal padding function for optional edge-effect reduction.
    '''

    if pad_factor is None or pad_factor <= 0.0:
        return data.copy(), ((0, 0), (0, 0))

    ny, nx = data.shape
    py = max(1, int(pad_factor*ny))
    px = max(1, int(pad_factor*nx))

    pad_width = ((py, py), (px, px))
    data_pad = numpy.pad(data, pad_width=pad_width, mode=pad_mode)

    if taper is True:
        data_pad = _cosine_taper_2d(data_pad, percent=0.05)

    return data_pad, pad_width


def _crop_data(data_pad, pad_width):
    '''
    Internal cropping function after padding.
    '''

    py0, py1 = pad_width[0]
    px0, px1 = pad_width[1]

    if py0 == 0 and py1 == 0 and px0 == 0 and px1 == 0:
        return data_pad

    return data_pad[py0:data_pad.shape[0]-py1, px0:data_pad.shape[1]-px1]


def _fft_derivative_operator(x, y, data, operator_function,
                             pad_factor=0.0, pad_mode="reflect", taper=False):
    '''
    Internal FFT filtering core for derivatives.
    '''

    _check_grid_inputs(x, y, data)

    original_shape = data.shape

    data0 = numpy.asarray(data, dtype=float)
    mean_value = numpy.nanmean(data0)
    data0 = data0 - mean_value

    data_pad, pad_width = _pad_data(data0, pad_factor=pad_factor,
                                    pad_mode=pad_mode, taper=taper)

    if pad_factor is None or pad_factor <= 0.0:
        xp = x
        yp = y
    else:
        dx, dy = _grid_spacing_from_coordinates(x, y)
        ny_pad, nx_pad = data_pad.shape

        # Build auxiliary padded coordinates only to obtain correct FFT frequencies.
        xp = numpy.tile(numpy.arange(nx_pad)*dx, (ny_pad, 1))
        yp = numpy.tile((numpy.arange(ny_pad)*dy).reshape(ny_pad, 1), (1, nx_pad))

    ky, kx = _get_wavenumbers(xp, yp)
    k = numpy.sqrt(kx**2 + ky**2)

    F = numpy.fft.fft2(data_pad)
    operator = operator_function(kx, ky, k)

    result_pad = numpy.real(numpy.fft.ifft2(F*operator))
    result = _crop_data(result_pad, pad_width)

    return result.reshape(original_shape)


# ============================================================
# FOURIER-DOMAIN DERIVATIVES - BACKWARD-COMPATIBLE FUNCTIONS
# ============================================================

def my_xderiv(x, y, data, n=1, pad_factor=0.0, pad_mode="reflect",
              taper=False, flatten=True):
    '''
    Return the horizontal derivative in x direction for n order in Fourier domain.

    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    n - int/float - order of the derivative
    pad_factor - float - optional grid expansion factor for FFT filtering
    pad_mode - string - numpy padding mode
    taper - bool - if True, applies a cosine taper to the expanded grid
    flatten - bool - if True, returns a 1D array to preserve the original behavior

    Output:
    xder - numpy array - derivative in x direction
    '''

    _check_grid_inputs(x, y, data)

    if n < 0:
        raise ValueError("Order of the derivative must be positive!")

    if n == 0:
        res = numpy.asarray(data, dtype=float).copy()
    else:
        def operator(kx, ky, k):
            return (1j*kx)**n

        res = _fft_derivative_operator(x, y, data, operator,
                                       pad_factor=pad_factor,
                                       pad_mode=pad_mode,
                                       taper=taper)

    return _as_output(res, original_shape=data.shape, flatten=flatten)


def my_yderiv(x, y, data, n=1, pad_factor=0.0, pad_mode="reflect",
              taper=False, flatten=True):
    '''
    Return the horizontal derivative in y direction for n order in Fourier domain.

    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    n - int/float - order of the derivative
    pad_factor - float - optional grid expansion factor for FFT filtering
    pad_mode - string - numpy padding mode
    taper - bool - if True, applies a cosine taper to the expanded grid
    flatten - bool - if True, returns a 1D array to preserve the original behavior

    Output:
    yder - numpy array - derivative in y direction
    '''

    _check_grid_inputs(x, y, data)

    if n < 0:
        raise ValueError("Order of the derivative must be positive!")

    if n == 0:
        res = numpy.asarray(data, dtype=float).copy()
    else:
        def operator(kx, ky, k):
            return (1j*ky)**n

        res = _fft_derivative_operator(x, y, data, operator,
                                       pad_factor=pad_factor,
                                       pad_mode=pad_mode,
                                       taper=taper)

    return _as_output(res, original_shape=data.shape, flatten=flatten)


def my_zderiv(x, y, data, n=1, pad_factor=0.0, pad_mode="reflect",
              taper=False, flatten=True, z_positive_down=True):
    '''
    Return the vertical derivative in z direction for n order in Fourier domain.

    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    n - int/float - order of the derivative
    pad_factor - float - optional grid expansion factor for FFT filtering
    pad_mode - string - numpy padding mode
    taper - bool - if True, applies a cosine taper to the expanded grid
    flatten - bool - if True, returns a 1D array to preserve the original behavior
    z_positive_down - bool - vertical coordinate convention

    Output:
    zder - numpy array - derivative in z direction
    '''

    _check_grid_inputs(x, y, data)

    if n < 0:
        raise ValueError("Order of the derivative must be positive!")

    if n == 0:
        res = numpy.asarray(data, dtype=float).copy()
    else:
        sign = 1.0
        if z_positive_down is False:
            sign = -1.0

        def operator(kx, ky, k):
            return (sign*k)**n

        res = _fft_derivative_operator(x, y, data, operator,
                                       pad_factor=pad_factor,
                                       pad_mode=pad_mode,
                                       taper=taper)

    return _as_output(res, original_shape=data.shape, flatten=flatten)


def my_hgrad(x, y, data, pad_factor=0.0, pad_mode="reflect",
             taper=False, flatten=True):
    '''
    Return the horizontal gradient amplitude for a potential data on a regular grid.
    All calculation is done in Fourier domain.

    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    pad_factor - float - optional grid expansion factor for FFT filtering
    pad_mode - string - numpy padding mode
    taper - bool - if True, applies a cosine taper to the expanded grid
    flatten - bool - if True, returns a 1D array to preserve the original behavior

    Output:
    hgrad - numpy array - horizontal gradient amplitude
    '''

    _check_grid_inputs(x, y, data)

    diffx = my_xderiv(x, y, data, n=1, pad_factor=pad_factor,
                      pad_mode=pad_mode, taper=taper, flatten=False)
    diffy = my_yderiv(x, y, data, n=1, pad_factor=pad_factor,
                      pad_mode=pad_mode, taper=taper, flatten=False)

    hgrad = numpy.sqrt(diffx**2 + diffy**2)

    return _as_output(hgrad, original_shape=data.shape, flatten=flatten)


def my_totalgrad(x, y, data, pad_factor=0.0, pad_mode="reflect",
                 taper=False, flatten=True, z_positive_down=True):
    '''
    Return the total gradient amplitude for a potential data on a regular grid.

    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    pad_factor - float - optional grid expansion factor for FFT filtering
    pad_mode - string - numpy padding mode
    taper - bool - if True, applies a cosine taper to the expanded grid
    flatten - bool - if True, returns a 1D array to preserve the original behavior
    z_positive_down - bool - vertical coordinate convention

    Output:
    tga - numpy array - total gradient amplitude
    '''

    _check_grid_inputs(x, y, data)

    diffx = my_xderiv(x, y, data, n=1, pad_factor=pad_factor,
                      pad_mode=pad_mode, taper=taper, flatten=False)
    diffy = my_yderiv(x, y, data, n=1, pad_factor=pad_factor,
                      pad_mode=pad_mode, taper=taper, flatten=False)
    diffz = my_zderiv(x, y, data, n=1, pad_factor=pad_factor,
                      pad_mode=pad_mode, taper=taper, flatten=False,
                      z_positive_down=z_positive_down)

    res = numpy.sqrt(diffx**2 + diffy**2 + diffz**2)

    return _as_output(res, original_shape=data.shape, flatten=flatten)


# ============================================================
# FOURIER-DOMAIN DERIVATIVES - GRID OUTPUT CONVENIENCE ALIASES
# ============================================================

def my_xderiv_grid(x, y, data, n=1, pad_factor=0.0,
                   pad_mode="reflect", taper=False):
    '''
    Return the x derivative as a 2D grid.
    '''

    return my_xderiv(x, y, data, n=n, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False)


def my_yderiv_grid(x, y, data, n=1, pad_factor=0.0,
                   pad_mode="reflect", taper=False):
    '''
    Return the y derivative as a 2D grid.
    '''

    return my_yderiv(x, y, data, n=n, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False)


def my_zderiv_grid(x, y, data, n=1, pad_factor=0.0,
                   pad_mode="reflect", taper=False, z_positive_down=True):
    '''
    Return the z derivative as a 2D grid.
    '''

    return my_zderiv(x, y, data, n=n, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False,
                     z_positive_down=z_positive_down)


def my_hgrad_grid(x, y, data, pad_factor=0.0,
                  pad_mode="reflect", taper=False):
    '''
    Return the horizontal gradient amplitude as a 2D grid.
    '''

    return my_hgrad(x, y, data, pad_factor=pad_factor,
                    pad_mode=pad_mode, taper=taper, flatten=False)


def my_totalgrad_grid(x, y, data, pad_factor=0.0,
                      pad_mode="reflect", taper=False,
                      z_positive_down=True):
    '''
    Return the total gradient amplitude as a 2D grid.
    '''

    return my_totalgrad(x, y, data, pad_factor=pad_factor,
                        pad_mode=pad_mode, taper=taper,
                        flatten=False, z_positive_down=z_positive_down)


# ============================================================
# MIXED AND HIGHER-ORDER DERIVATIVES IN FOURIER DOMAIN
# ============================================================

def my_mixed_deriv_fft(x, y, data, nx_order=0, ny_order=0, nz_order=0,
                       pad_factor=0.0, pad_mode="reflect", taper=False,
                       flatten=True, z_positive_down=True):
    '''
    Return a general mixed derivative in the Fourier domain.

    It computes:
    d^(nx_order + ny_order + nz_order) data / dx^nx_order dy^ny_order dz^nz_order

    Inputs:
    x, y - numpy 2D arrays - regular grid coordinates
    data - numpy 2D array - potential field data
    nx_order - int - derivative order in x
    ny_order - int - derivative order in y
    nz_order - int - derivative order in z
    pad_factor - float - optional grid expansion factor
    pad_mode - string - numpy padding mode
    taper - bool - if True, applies taper
    flatten - bool - if True, returns a 1D array
    z_positive_down - bool - vertical coordinate convention

    Output:
    derivative - numpy array - mixed derivative
    '''

    _check_grid_inputs(x, y, data)

    if nx_order < 0 or ny_order < 0 or nz_order < 0:
        raise ValueError("Derivative orders must be non-negative!")

    if nx_order == 0 and ny_order == 0 and nz_order == 0:
        res = numpy.asarray(data, dtype=float).copy()
    else:
        sign = 1.0
        if z_positive_down is False:
            sign = -1.0

        def operator(kx, ky, k):
            op = numpy.ones_like(k, dtype=complex)
            if nx_order > 0:
                op *= (1j*kx)**nx_order
            if ny_order > 0:
                op *= (1j*ky)**ny_order
            if nz_order > 0:
                op *= (sign*k)**nz_order
            return op

        res = _fft_derivative_operator(x, y, data, operator,
                                       pad_factor=pad_factor,
                                       pad_mode=pad_mode,
                                       taper=taper)

    return _as_output(res, original_shape=data.shape, flatten=flatten)


def my_xzderiv(x, y, data, pad_factor=0.0, pad_mode="reflect",
               taper=False, flatten=True, z_positive_down=True):
    '''
    Return the mixed derivative d2(data)/dxdz in Fourier domain.
    '''

    return my_mixed_deriv_fft(x, y, data, nx_order=1, ny_order=0, nz_order=1,
                              pad_factor=pad_factor, pad_mode=pad_mode,
                              taper=taper, flatten=flatten,
                              z_positive_down=z_positive_down)


def my_yzderiv(x, y, data, pad_factor=0.0, pad_mode="reflect",
               taper=False, flatten=True, z_positive_down=True):
    '''
    Return the mixed derivative d2(data)/dydz in Fourier domain.
    '''

    return my_mixed_deriv_fft(x, y, data, nx_order=0, ny_order=1, nz_order=1,
                              pad_factor=pad_factor, pad_mode=pad_mode,
                              taper=taper, flatten=flatten,
                              z_positive_down=z_positive_down)


def my_xxderiv(x, y, data, pad_factor=0.0, pad_mode="reflect",
               taper=False, flatten=True):
    '''
    Return the second derivative d2(data)/dx2 in Fourier domain.
    '''

    return my_xderiv(x, y, data, n=2, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=flatten)


def my_yyderiv(x, y, data, pad_factor=0.0, pad_mode="reflect",
               taper=False, flatten=True):
    '''
    Return the second derivative d2(data)/dy2 in Fourier domain.
    '''

    return my_yderiv(x, y, data, n=2, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=flatten)


def my_zzderiv(x, y, data, pad_factor=0.0, pad_mode="reflect",
               taper=False, flatten=True, z_positive_down=True):
    '''
    Return the second derivative d2(data)/dz2 in Fourier domain.
    '''

    return my_zderiv(x, y, data, n=2, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=flatten,
                     z_positive_down=z_positive_down)


# ============================================================
# SPACE-DOMAIN DERIVATIVES
# ============================================================

def my_xderiv_space(x, y, data, edge_order=2, flatten=True):
    '''
    Return the x derivative using finite differences in the space domain.

    Inputs:
    x, y - numpy 2D arrays - regular grid coordinates
    data - numpy 2D array - potential field data
    edge_order - int - edge order used by numpy.gradient
    flatten - bool - if True, returns a 1D array

    Output:
    derivative - numpy array - x derivative
    '''

    _check_grid_inputs(x, y, data)

    dx, dy = _grid_spacing_from_coordinates(x, y)
    df_dy, df_dx = numpy.gradient(data, dy, dx, edge_order=edge_order)

    return _as_output(df_dx, original_shape=data.shape, flatten=flatten)


def my_yderiv_space(x, y, data, edge_order=2, flatten=True):
    '''
    Return the y derivative using finite differences in the space domain.
    '''

    _check_grid_inputs(x, y, data)

    dx, dy = _grid_spacing_from_coordinates(x, y)
    df_dy, df_dx = numpy.gradient(data, dy, dx, edge_order=edge_order)

    return _as_output(df_dy, original_shape=data.shape, flatten=flatten)


def my_hgrad_space(x, y, data, edge_order=2, flatten=True):
    '''
    Return the horizontal gradient amplitude using finite differences.
    '''

    _check_grid_inputs(x, y, data)

    dx_field = my_xderiv_space(x, y, data, edge_order=edge_order, flatten=False)
    dy_field = my_yderiv_space(x, y, data, edge_order=edge_order, flatten=False)

    hgrad = numpy.sqrt(dx_field**2 + dy_field**2)

    return _as_output(hgrad, original_shape=data.shape, flatten=flatten)


def my_laplacian_space(x, y, data, edge_order=2, flatten=True):
    '''
    Return the horizontal Laplacian using finite differences.

    Output:
    laplacian = d2(data)/dx2 + d2(data)/dy2
    '''

    _check_grid_inputs(x, y, data)

    dfdx = my_xderiv_space(x, y, data, edge_order=edge_order, flatten=False)
    dfdy = my_yderiv_space(x, y, data, edge_order=edge_order, flatten=False)

    d2fdx2 = my_xderiv_space(x, y, dfdx, edge_order=edge_order, flatten=False)
    d2fdy2 = my_yderiv_space(x, y, dfdy, edge_order=edge_order, flatten=False)

    lap = d2fdx2 + d2fdy2

    return _as_output(lap, original_shape=data.shape, flatten=flatten)


def my_totalgrad_space(x, y, data, dz_data, edge_order=2, flatten=True):
    '''
    Return the total gradient amplitude in the space domain using a supplied
    vertical derivative dz_data.
    '''

    _check_grid_inputs(x, y, data)

    if dz_data.shape != data.shape:
        raise ValueError("dz_data must have the same shape as data!")

    dx_field = my_xderiv_space(x, y, data, edge_order=edge_order, flatten=False)
    dy_field = my_yderiv_space(x, y, data, edge_order=edge_order, flatten=False)

    total = numpy.sqrt(dx_field**2 + dy_field**2 + dz_data**2)

    return _as_output(total, original_shape=data.shape, flatten=flatten)


# ============================================================
# USEFUL DERIVATIVE-BASED FILTERS
# ============================================================

def my_tilt_derivative(x, y, data, pad_factor=0.0, pad_mode="reflect",
                       taper=False, eps=1.0e-12, degrees=False,
                       flatten=True, z_positive_down=True):
    '''
    Return the tilt derivative / tilt angle of a potential field.

    tilt = arctan( dz / sqrt(dx**2 + dy**2) )
    '''

    _check_grid_inputs(x, y, data)

    dz = my_zderiv(x, y, data, n=1, pad_factor=pad_factor,
                   pad_mode=pad_mode, taper=taper, flatten=False,
                   z_positive_down=z_positive_down)
    hgrad = my_hgrad(x, y, data, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False)

    tilt = numpy.arctan2(dz, hgrad + eps)

    if degrees is True:
        tilt = numpy.degrees(tilt)

    return _as_output(tilt, original_shape=data.shape, flatten=flatten)


def my_tilt_vertical_derivative(x, y, data, pad_factor=0.0,
                                pad_mode="reflect", taper=False,
                                eps=1.0e-12, degrees=False,
                                flatten=True, z_positive_down=True):
    '''
    Return the tilt angle of the first vertical derivative.

    This implements:
    phi = arctan( d2F/dz2 / sqrt((d2F/dxdz)^2 + (d2F/dydz)^2) )
    '''

    _check_grid_inputs(x, y, data)

    dzz = my_zzderiv(x, y, data, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False,
                     z_positive_down=z_positive_down)
    dxz = my_xzderiv(x, y, data, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False,
                     z_positive_down=z_positive_down)
    dyz = my_yzderiv(x, y, data, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False,
                     z_positive_down=z_positive_down)

    hgm_vd = numpy.sqrt(dxz**2 + dyz**2)
    phi = numpy.arctan2(dzz, hgm_vd + eps)

    if degrees is True:
        phi = numpy.degrees(phi)

    return _as_output(phi, original_shape=data.shape, flatten=flatten)


def my_theta_map(x, y, data, pad_factor=0.0, pad_mode="reflect",
                 taper=False, eps=1.0e-12, flatten=True,
                 z_positive_down=True):
    '''
    Return the theta map, defined as cos(tilt angle):

    theta = HGM / sqrt(HGM**2 + dz**2)
    '''

    _check_grid_inputs(x, y, data)

    dz = my_zderiv(x, y, data, n=1, pad_factor=pad_factor,
                   pad_mode=pad_mode, taper=taper, flatten=False,
                   z_positive_down=z_positive_down)
    hgrad = my_hgrad(x, y, data, pad_factor=pad_factor,
                     pad_mode=pad_mode, taper=taper, flatten=False)

    theta = hgrad/(numpy.sqrt(hgrad**2 + dz**2) + eps)

    return _as_output(theta, original_shape=data.shape, flatten=flatten)


def my_thdr_tilt(x, y, data, pad_factor=0.0, pad_mode="reflect",
                 taper=False, flatten=True, z_positive_down=True):
    '''
    Return the total horizontal derivative of the tilt angle.
    '''

    _check_grid_inputs(x, y, data)

    tilt = my_tilt_derivative(x, y, data, pad_factor=pad_factor,
                              pad_mode=pad_mode, taper=taper,
                              flatten=False, z_positive_down=z_positive_down)

    thdr = my_hgrad(x, y, tilt, pad_factor=pad_factor,
                    pad_mode=pad_mode, taper=taper, flatten=False)

    return _as_output(thdr, original_shape=data.shape, flatten=flatten)


# ============================================================
# ALIASES FOR COMPATIBILITY AND READABILITY
# ============================================================

my_hga = my_hgrad
my_tga = my_totalgrad
my_hgm = my_hgrad
my_asa = my_totalgrad
my_tilt = my_tilt_derivative
my_tdr = my_tilt_derivative
