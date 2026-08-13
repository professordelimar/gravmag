# -----------------------------------------------------------------------------------
# Title: Filtering
# Description: Fourier-domain and space-domain filters for gravity and magnetic data.
# Author: Nelson Ribeiro Filho
# -----------------------------------------------------------------------------------

from __future__ import division

import numpy

try:
    from . import auxiliars, derivative
except Exception:
    import auxiliars
    import derivative


# ============================================================
# INTERNAL CHECKS AND UTILITIES
# ============================================================

def _check_grid(x, y, data):
    '''
    Check if x, y and data are 2D arrays with the same shape.
    '''

    if x.shape != y.shape or x.shape != data.shape:
        raise ValueError("x, y and data must have the same shape!")

    if data.ndim != 2:
        raise ValueError("x, y and data must be 2D regular grids!")


def _as_grid(result, shape, flatten=True):
    '''
    Return result as 1D vector or 2D grid.
    '''

    result = numpy.asarray(result).reshape(shape)

    if flatten is True:
        return result.reshape(result.size)

    return result


def _grid_spacing(x, y):
    '''
    Estimate grid spacing in x and y directions.
    '''

    try:
        dx, dy = auxiliars.my_grid_spacing(x, y)
    except Exception:
        dx = numpy.mean(numpy.diff(x[0, :]))
        dy = numpy.mean(numpy.diff(y[:, 0]))
        dx = abs(dx)
        dy = abs(dy)

    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Invalid grid spacing. dx and dy must be positive!")

    return dx, dy


def _wavenumbers(x, y, shape=None):
    '''
    Compute kx, ky and k for a 2D regular grid.
    '''

    if shape is None:
        ny, nx = x.shape
    else:
        ny, nx = shape

    dx, dy = _grid_spacing(x, y)

    kx_1d = 2.0*numpy.pi*numpy.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0*numpy.pi*numpy.fft.fftfreq(ny, d=dy)

    kx, ky = numpy.meshgrid(kx_1d, ky_1d)
    k = numpy.sqrt(kx**2 + ky**2)

    return kx, ky, k


def _cosine_taper_1d(n, p=0.05):
    '''
    Build a 1D cosine taper.
    '''

    w = numpy.ones(n)
    m = int(p*n)

    if m < 1:
        return w

    t = numpy.linspace(0.0, numpy.pi/2.0, m)
    taper = numpy.sin(t)**2

    w[:m] = taper
    w[-m:] = taper[::-1]

    return w


def _cosine_taper_2d(data, p=0.05):
    '''
    Apply a 2D cosine taper.
    '''

    ny, nx = data.shape
    wx = _cosine_taper_1d(nx, p=p)
    wy = _cosine_taper_1d(ny, p=p)

    return data*numpy.outer(wy, wx)


def _pad_grid(data, pad_factor=0.5, pad_mode="reflect", taper=True):
    '''
    Expand a grid before FFT filtering.
    '''

    if pad_factor is None or pad_factor <= 0.0:
        mean_data = numpy.nanmean(data)
        data0 = data - mean_data
        return data0.copy(), ((0, 0), (0, 0)), mean_data

    ny, nx = data.shape
    py = max(1, int(pad_factor*ny))
    px = max(1, int(pad_factor*nx))

    mean_data = numpy.nanmean(data)
    data0 = data - mean_data

    pad_width = ((py, py), (px, px))
    data_pad = numpy.pad(data0, pad_width=pad_width, mode=pad_mode)

    if taper is True:
        data_pad = _cosine_taper_2d(data_pad, p=0.05)

    return data_pad, pad_width, mean_data


def _crop_grid(data_pad, pad_width):
    '''
    Crop a padded grid back to the original size.
    '''

    py0, py1 = pad_width[0]
    px0, px1 = pad_width[1]

    if py0 == 0 and py1 == 0 and px0 == 0 and px1 == 0:
        return data_pad

    return data_pad[py0:data_pad.shape[0]-py1,
                    px0:data_pad.shape[1]-px1]


def _apply_fft_operator(x, y, data, operator_function,
                        pad_factor=0.5, pad_mode="reflect",
                        taper=True, restore_mean=False):
    '''
    Apply a generic Fourier-domain operator to a regular grid.
    '''

    _check_grid(x, y, data)

    data_pad, pad_width, mean_data = _pad_grid(
        data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    kx, ky, k = _wavenumbers(x, y, shape=data_pad.shape)

    F = numpy.fft.fft2(data_pad)
    operator = operator_function(kx, ky, k)
    result_pad = numpy.real(numpy.fft.ifft2(F*operator))
    result = _crop_grid(result_pad, pad_width)

    if restore_mean is True:
        result = result + mean_data

    return result


# ============================================================
# CONTINUATION FILTERS
# ============================================================

def my_continuation_grid(x, y, data, level, pad_factor=0.5,
                         pad_mode="reflect", taper=True):
    '''
    Compute upward or downward continuation of a potential field.

    Positive level applies upward continuation by exp(-k*level).
    Negative level applies downward continuation by exp(+k*abs(level)),
    which is unstable and should be used carefully.

    Inputs:
    x, y - numpy 2D arrays - regular grid coordinates
    data - numpy 2D array - gravity or magnetic data
    level - float - continuation distance in the same unit as x and y
    pad_factor - float - grid expansion factor to reduce border effects
    pad_mode - string - numpy padding mode
    taper - bool - if True, applies a cosine taper after padding

    Output:
    continued - numpy 2D array - continued field
    '''

    if level == 0.0:
        return data.copy()

    def operator(kx, ky, k):
        return numpy.exp(-level*k)

    continued = _apply_fft_operator(
        x, y, data, operator,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        restore_mean=True
    )

    return continued


def my_continuation(x, y, data, level, pad_factor=0.5,
                    pad_mode="reflect", taper=True, flatten=True):
    '''
    Compatibility function for upward/downward continuation.

    By default, returns a 1D vector, preserving the original behavior.
    Use flatten=False to return a 2D grid.
    '''

    result = my_continuation_grid(
        x, y, data, level,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_upward_continuation(x, y, data, height, pad_factor=0.5,
                           pad_mode="reflect", taper=True, flatten=True):
    '''
    Compute upward continuation using a positive height.
    '''

    if height < 0.0:
        raise ValueError("height must be positive for upward continuation!")

    return my_continuation(
        x, y, data, height,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=flatten
    )


def my_downward_continuation_regularized(x, y, data, height, k_cut=None,
                                         pad_factor=0.5, pad_mode="reflect",
                                         taper=True, flatten=True):
    '''
    Compute regularized downward continuation.

    Downward continuation is unstable. This function uses a Gaussian
    stabilizing factor exp(-(k/k_cut)^2).
    '''

    if height < 0.0:
        raise ValueError("height must be positive!")

    dx, dy = _grid_spacing(x, y)

    if k_cut is None:
        wavelength_min = 6.0*max(dx, dy)
        k_cut = 2.0*numpy.pi/wavelength_min

    def operator(kx, ky, k):
        return numpy.exp(k*height)*numpy.exp(-(k/k_cut)**2)

    result = _apply_fft_operator(
        x, y, data, operator,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        restore_mean=True
    )

    return _as_grid(result, data.shape, flatten=flatten)


# ============================================================
# GENERAL WAVENUMBER FILTERS
# ============================================================

def my_lowpass_gaussian(x, y, data, wavelength_cut, pad_factor=0.5,
                        pad_mode="reflect", taper=True, flatten=True):
    '''
    Apply a Gaussian low-pass filter.
    '''

    if wavelength_cut <= 0.0:
        raise ValueError("wavelength_cut must be positive!")

    k_cut = 2.0*numpy.pi/wavelength_cut

    def operator(kx, ky, k):
        return numpy.exp(-(k/k_cut)**2)

    result = _apply_fft_operator(
        x, y, data, operator,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        restore_mean=True
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_highpass_gaussian(x, y, data, wavelength_cut, pad_factor=0.5,
                         pad_mode="reflect", taper=True, flatten=True):
    '''
    Apply a Gaussian high-pass filter.
    '''

    low = my_lowpass_gaussian(
        x, y, data, wavelength_cut,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    result = data - low

    return _as_grid(result, data.shape, flatten=flatten)


def my_bandpass_gaussian(x, y, data, wavelength_min, wavelength_max,
                         pad_factor=0.5, pad_mode="reflect",
                         taper=True, flatten=True):
    '''
    Apply a Gaussian band-pass filter.
    '''

    if wavelength_min <= 0.0 or wavelength_max <= 0.0:
        raise ValueError("wavelength_min and wavelength_max must be positive!")

    if wavelength_min >= wavelength_max:
        raise ValueError("wavelength_min must be smaller than wavelength_max!")

    low_min = my_lowpass_gaussian(
        x, y, data, wavelength_min,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    low_max = my_lowpass_gaussian(
        x, y, data, wavelength_max,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    result = low_min - low_max

    return _as_grid(result, data.shape, flatten=flatten)


def my_directional_derivative(x, y, data, azimuth_degrees, pad_factor=0.5,
                              pad_mode="reflect", taper=True, flatten=True):
    '''
    Compute directional derivative in the Fourier domain.

    Azimuth is measured clockwise from North:
    0 degrees = y direction; 90 degrees = x direction.
    '''

    angle = numpy.deg2rad(azimuth_degrees)

    dx = derivative.my_xderiv(
        x, y, data, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    dy = derivative.my_yderiv(
        x, y, data, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    result = dx*numpy.sin(angle) + dy*numpy.cos(angle)

    return _as_grid(result, data.shape, flatten=flatten)


# ============================================================
# MAGNETIC REDUCTION AND PSEUDOGRAVITY
# ============================================================

def reduction_grid(x, y, data, inc, dec, incs=None, decs=None,
                   newinc=None, newdec=None, newincs=None, newdecs=None,
                   pad_factor=0.5, pad_mode="reflect", taper=True,
                   eps=1.0e-12):
    '''
    Reduce magnetic data to new field and source magnetization directions.

    Default behavior is reduction to the pole.
    '''

    _check_grid(x, y, data)

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    if newinc is None:
        newinc = 90.0
    if newdec is None:
        newdec = 0.0
    if newincs is None:
        newincs = 90.0
    if newdecs is None:
        newdecs = 0.0

    data_pad, pad_width, mean_data = _pad_grid(
        data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    kx, ky, k = _wavenumbers(x, y, shape=data_pad.shape)

    f0 = auxiliars.my_theta(inc, dec, kx, ky)
    m0 = auxiliars.my_theta(incs, decs, kx, ky)
    f1 = auxiliars.my_theta(newinc, newdec, kx, ky)
    m1 = auxiliars.my_theta(newincs, newdecs, kx, ky)

    denominator = f0*m0

    with numpy.errstate(divide="ignore", invalid="ignore"):
        operator = (f1*m1)/(denominator + eps)

    operator[0, 0] = 0.0

    result_pad = numpy.real(numpy.fft.ifft2(operator*numpy.fft.fft2(data_pad)))
    result = _crop_grid(result_pad, pad_width) + mean_data

    return result


def reduction(x, y, data, inc, dec, incs=None, decs=None,
              newinc=None, newdec=None, newincs=None, newdecs=None,
              pad_factor=0.5, pad_mode="reflect", taper=True,
              flatten=True):
    '''
    Compatibility function for magnetic reduction.
    '''

    result = reduction_grid(
        x, y, data, inc, dec,
        incs=incs, decs=decs,
        newinc=newinc, newdec=newdec,
        newincs=newincs, newdecs=newdecs,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_rtp(x, y, data, inc, dec, incs=None, decs=None,
           pad_factor=0.5, pad_mode="reflect", taper=True, flatten=True):
    '''
    Reduction to the pole.
    '''

    return reduction(
        x, y, data, inc, dec,
        incs=incs, decs=decs,
        newinc=90.0, newdec=0.0,
        newincs=90.0, newdecs=0.0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=flatten
    )


def my_rte(x, y, data, inc, dec, incs=None, decs=None,
           pad_factor=0.5, pad_mode="reflect", taper=True, flatten=True):
    '''
    Reduction to the equator.
    '''

    return reduction(
        x, y, data, inc, dec,
        incs=incs, decs=decs,
        newinc=0.0, newdec=0.0,
        newincs=0.0, newdecs=0.0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=flatten
    )


def my_pseudograv_grid(x, y, data, inc, dec, incs, decs,
                       rho=1000.0, mag=1.0, pad_factor=0.5,
                       pad_mode="reflect", taper=True, eps=1.0e-12):
    '''
    Compute pseudogravity transformation from magnetic total field anomaly.

    Inputs:
    x, y - numpy 2D arrays - regular grid coordinates
    data - numpy 2D array - magnetic anomaly, usually in nT
    inc, dec - float - field inclination and declination in degrees
    incs, decs - float - source inclination and declination in degrees
    rho - float - density contrast in kg/m^3
    mag - float - magnetization intensity in A/m

    Output:
    pseudo - numpy 2D array - pseudogravity anomaly in mGal-like scale
    '''

    _check_grid(x, y, data)

    if rho == 0.0:
        raise ValueError("rho must be nonzero!")
    if mag == 0.0:
        raise ValueError("mag must be nonzero!")

    G = 6.673e-11
    si2mGal = 100000.0
    t2nt = 1000000000.0
    cm = 1.0e-7

    C = G*rho*si2mGal/(cm*mag*t2nt)

    data_pad, pad_width, mean_data = _pad_grid(
        data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    kx, ky, k = _wavenumbers(x, y, shape=data_pad.shape)

    thetaf = auxiliars.my_theta(inc, dec, kx, ky)
    thetas = auxiliars.my_theta(incs, decs, kx, ky)

    with numpy.errstate(divide="ignore", invalid="ignore"):
        operator = 1.0/(thetaf*thetas*k + eps)

    operator[0, 0] = 0.0

    pseudo_pad = numpy.fft.fft2(data_pad)*operator*C
    pseudo = numpy.real(numpy.fft.ifft2(pseudo_pad))
    pseudo = _crop_grid(pseudo, pad_width)

    return pseudo


def my_pseudograv(x, y, data, inc, dec, incs, decs,
                  rho=1000.0, mag=1.0, pad_factor=0.5,
                  pad_mode="reflect", taper=True, flatten=True):
    '''
    Compatibility function for pseudogravity transformation.
    '''

    result = my_pseudograv_grid(
        x, y, data, inc, dec, incs, decs,
        rho=rho, mag=mag,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    return _as_grid(result, data.shape, flatten=flatten)


# ============================================================
# EDGE-DETECTION FILTERS
# ============================================================

def my_tilt_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                 taper=True, eps=1.0e-12, degrees=False):
    '''
    Compute the tilt angle of a potential field.

    tilt = arctan( vertical derivative / horizontal gradient magnitude )
    '''

    _check_grid(x, y, data)

    hgrad = derivative.my_hgrad(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    diffz = derivative.my_zderiv(
        x, y, data, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    tilt = numpy.arctan2(diffz, hgrad + eps)

    if degrees is True:
        tilt = numpy.degrees(tilt)

    return tilt


def my_tilt(x, y, data, pad_factor=0.5, pad_mode="reflect",
            taper=True, degrees=False, flatten=True):
    '''
    Compatibility function for tilt angle.
    '''

    result = my_tilt_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        degrees=degrees
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_tilt_vdr_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                     taper=True, eps=1.0e-12, degrees=False):
    '''
    Compute the tilt angle of the vertical derivative.

    This is the TAM applied to the first vertical derivative:
    phi = arctan( d2F/dz2 / sqrt((d2F/dxdz)^2 + (d2F/dydz)^2) )
    '''

    vdr = derivative.my_zderiv(
        x, y, data, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    phi = my_tilt_grid(
        x, y, vdr,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        eps=eps,
        degrees=degrees
    )

    return phi


def my_tilt_vdr(x, y, data, pad_factor=0.5, pad_mode="reflect",
                taper=True, degrees=False, flatten=True):
    '''
    Compatibility function for tilt angle of vertical derivative.
    '''

    result = my_tilt_vdr_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        degrees=degrees
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_thdr_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                 taper=True):
    '''
    Compute the total horizontal derivative of the tilt angle.
    '''

    tilt = my_tilt_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        degrees=False
    )

    thdr = derivative.my_hgrad(
        x, y, tilt,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    return thdr


def my_thdr(x, y, data, pad_factor=0.5, pad_mode="reflect",
            taper=True, flatten=True):
    '''
    Compatibility function for total horizontal derivative of tilt angle.
    '''

    result = my_thdr_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_hyperbolictilt_grid(x, y, data, pad_factor=0.5,
                           pad_mode="reflect", taper=True,
                           eps=1.0e-12):
    '''
    Compute the hyperbolic tilt angle.

    Here it is implemented as arctanh-normalized ratio:
    atanh( dz / sqrt(dz^2 + hgrad^2) ).
    '''

    hgrad = derivative.my_hgrad(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    diffz = derivative.my_zderiv(
        x, y, data, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    ratio = diffz/(numpy.sqrt(diffz**2 + hgrad**2) + eps)
    ratio = numpy.clip(ratio, -1.0 + eps, 1.0 - eps)
    hyptilt = numpy.arctanh(ratio)

    return hyptilt


def my_hyperbolictilt(x, y, data, pad_factor=0.5,
                      pad_mode="reflect", taper=True,
                      flatten=True):
    '''
    Compatibility function for hyperbolic tilt angle.
    '''

    result = my_hyperbolictilt_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_thetamap_grid(x, y, data, pad_factor=0.5,
                     pad_mode="reflect", taper=True,
                     eps=1.0e-12):
    '''
    Compute the theta map.

    theta = arccos( horizontal gradient magnitude / analytic signal amplitude )

    This returns an angular representation. If you need cos(tilt), use
    my_cos_tilt_grid.
    '''

    hgrad = derivative.my_hgrad(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    tgrad = derivative.my_totalgrad(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    ratio = hgrad/(tgrad + eps)
    ratio = numpy.clip(ratio, -1.0, 1.0)
    theta = numpy.arccos(ratio)

    return theta


def my_thetamap(x, y, data, pad_factor=0.5,
                pad_mode="reflect", taper=True, flatten=True):
    '''
    Compatibility function for theta map.
    '''

    result = my_thetamap_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_cos_tilt_grid(x, y, data, pad_factor=0.5,
                     pad_mode="reflect", taper=True,
                     eps=1.0e-12):
    '''
    Compute cos(tilt angle), another common theta-map representation.
    '''

    hgrad = derivative.my_hgrad(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    diffz = derivative.my_zderiv(
        x, y, data, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    return hgrad/(numpy.sqrt(hgrad**2 + diffz**2) + eps)


def my_cos_tilt(x, y, data, pad_factor=0.5,
                pad_mode="reflect", taper=True, flatten=True):
    '''
    Compatibility function for cos(tilt angle).
    '''

    result = my_cos_tilt_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper
    )

    return _as_grid(result, data.shape, flatten=flatten)


# ============================================================
# INTERPRETATION SUPPORT
# ============================================================

def my_tilt_depth_from_pm45(distance_minus45_plus45):
    '''
    Estimate depth from half distance between -45 and +45 degree contours.
    '''

    return 0.5*numpy.asarray(distance_minus45_plus45)


def my_tilt_depth_from_zero_to_45(distance_zero_to_45):
    '''
    Estimate depth from distance between 0 and +45 or -45 degree contours.
    '''

    return numpy.asarray(distance_zero_to_45)


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

my_upcontinue = my_upward_continuation
my_downcontinue = my_downward_continuation_regularized
my_tdr = my_tilt
my_tam = my_tilt
my_theta_map = my_thetamap
my_tilt_vertical_derivative = my_tilt_vdr
my_tilt_derivative_vertical = my_tilt_vdr
my_thdr_tilt = my_thdr


# ============================================================
# PROPOSED SOURCE-DETECTION FILTERS
# ============================================================

def _finite_data(data):
    '''
    Replace non-finite values by the finite mean before FFT-based filtering.
    The original non-finite positions are not restored here because the filters
    are designed to be used on already gridded and cleaned data.
    '''

    arr = numpy.asarray(data, dtype=float).copy()
    mask = numpy.isfinite(arr)

    if not numpy.any(mask):
        raise ValueError("data does not contain finite values!")

    fill_value = numpy.nanmean(arr[mask])
    arr[~mask] = fill_value

    return arr


def _robust_normalize(data, lower_percentile=2.0, upper_percentile=98.0,
                      eps=1.0e-12):
    '''
    Robustly normalize an array to the interval [0, 1] using percentiles.
    '''

    arr = numpy.asarray(data, dtype=float)
    finite = numpy.isfinite(arr)

    if not numpy.any(finite):
        return numpy.zeros_like(arr, dtype=float)

    pmin = numpy.nanpercentile(arr[finite], lower_percentile)
    pmax = numpy.nanpercentile(arr[finite], upper_percentile)

    if abs(pmax - pmin) < eps:
        return numpy.zeros_like(arr, dtype=float)

    out = (arr - pmin)/(pmax - pmin + eps)
    out = numpy.clip(out, 0.0, 1.0)
    out[~finite] = 0.0

    return out


def _angular_difference_rad(angle1, angle2):
    '''
    Smallest angular difference between two angles in radians.
    '''

    return numpy.arctan2(numpy.sin(angle1 - angle2),
                         numpy.cos(angle1 - angle2))


def _source_filter_components(x, y, data, pad_factor=0.5,
                              pad_mode="reflect", taper=True,
                              eps=1.0e-12):
    '''
    Compute the basic Fourier-domain components used by the proposed filters.

    Components:
    dx, dy, dz    - first derivatives in x, y and z
    hgm           - horizontal gradient magnitude
    asa           - analytic signal amplitude / total gradient amplitude
    tdr           - classical tilt derivative in radians
    thdr_tdr      - horizontal gradient magnitude of the tilt derivative
    theta_ratio   - hgm / asa, equivalent to cos(TDR)
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    dx = derivative.my_xderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    dy = derivative.my_yderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    dz = derivative.my_zderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    hgm = numpy.sqrt(dx**2 + dy**2)
    asa = numpy.sqrt(dx**2 + dy**2 + dz**2)
    tdr = numpy.arctan2(dz, hgm + eps)

    tdr_dx = derivative.my_xderiv(
        x, y, tdr, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    tdr_dy = derivative.my_yderiv(
        x, y, tdr, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    thdr_tdr = numpy.sqrt(tdr_dx**2 + tdr_dy**2)
    theta_ratio = hgm/(asa + eps)

    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "hgm": hgm,
        "asa": asa,
        "tdr": tdr,
        "tdr_dx": tdr_dx,
        "tdr_dy": tdr_dy,
        "thdr_tdr": thdr_tdr,
        "theta_ratio": theta_ratio,
    }


def my_cgtp_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                 taper=True, alpha=1.0, beta=1.0,
                 sigma_tilt=numpy.deg2rad(12.0), eps=1.0e-12,
                 return_components=False):
    '''
    Compute the proposed Coherence Gradient-Tilt-Depth filter (CGTP).

    The filter enhances probable source edges by combining:
    1) the normalized horizontal-to-total gradient ratio, hgm/asa;
    2) the normalized horizontal gradient of the tilt derivative;
    3) a Gaussian weight centered on TDR = 0, emphasizing source boundaries.

    All derivatives are computed in the wavenumber domain with optional grid
    expansion and tapering to reduce border effects.
    '''

    comps = _source_filter_components(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        eps=eps
    )

    edge_weight = numpy.exp(-((comps["tdr"]/(sigma_tilt + eps))**2))
    ratio_n = _robust_normalize(comps["theta_ratio"])
    thdr_n = _robust_normalize(comps["thdr_tdr"])

    cgtp = (ratio_n + eps)**alpha * (thdr_n + eps)**beta * edge_weight
    cgtp = _robust_normalize(cgtp)

    if return_components is True:
        comps.update({
            "edge_weight": edge_weight,
            "theta_ratio_normalized": ratio_n,
            "thdr_tdr_normalized": thdr_n,
            "cgtp": cgtp,
        })
        return comps

    return cgtp


def my_cgtp(x, y, data, pad_factor=0.5, pad_mode="reflect",
            taper=True, alpha=1.0, beta=1.0,
            sigma_tilt=numpy.deg2rad(12.0), flatten=True):
    '''
    Compatibility wrapper for my_cgtp_grid.
    '''

    result = my_cgtp_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        alpha=alpha,
        beta=beta,
        sigma_tilt=sigma_tilt,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_dri_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                taper=True, sigma_tilt=numpy.deg2rad(12.0),
                eps=1.0e-12, return_components=False):
    '''
    Compute the proposed Derivative Relative-Depth Index (DRI).

    DRI is a relative-depth indicator based on the inverse of the horizontal
    gradient magnitude of the tilt derivative. The Gaussian TDR = 0 weight
    restricts the response to probable contacts. Interpret this result only
    where edge filters such as CGTP are high.

    Smaller DRI values suggest sharper/shallower contacts. Larger DRI values
    suggest smoother/deeper contacts. The output is normalized to [0, 1].
    '''

    comps = _source_filter_components(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        eps=eps
    )

    edge_weight = numpy.exp(-((comps["tdr"]/(sigma_tilt + eps))**2))
    dri_raw = edge_weight/(comps["thdr_tdr"] + eps)
    dri = _robust_normalize(dri_raw)

    if return_components is True:
        comps.update({
            "edge_weight": edge_weight,
            "dri_raw": dri_raw,
            "dri": dri,
        })
        return comps

    return dri


def my_dri(x, y, data, pad_factor=0.5, pad_mode="reflect",
           taper=True, sigma_tilt=numpy.deg2rad(12.0), flatten=True):
    '''
    Compatibility wrapper for my_dri_grid.
    '''

    result = my_dri_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        sigma_tilt=sigma_tilt,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_cgtp_vdr_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                     taper=True, alpha=1.0, beta=1.0,
                     sigma_tilt=numpy.deg2rad(12.0), eps=1.0e-12,
                     return_components=False):
    '''
    Compute the proposed CGTP-VDR filter.

    This is the CGTP concept applied to the first vertical derivative of the
    input field. It uses mixed derivatives and the second vertical derivative
    through the same wavenumber-domain operators used by the derivative module.

    CGTP-VDR is more sensitive to shallow contacts and should be applied to
    smoothed or residual fields with caution because higher-order derivatives
    amplify short-wavelength noise.
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    vdr = derivative.my_zderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    xz = derivative.my_xzderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    yz = derivative.my_yzderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    zz = derivative.my_zzderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    hgm_v = numpy.sqrt(xz**2 + yz**2)
    asa_v = numpy.sqrt(xz**2 + yz**2 + zz**2)
    tdr_v = numpy.arctan2(zz, hgm_v + eps)

    tdrv_dx = derivative.my_xderiv(
        x, y, tdr_v, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    tdrv_dy = derivative.my_yderiv(
        x, y, tdr_v, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    thdr_tdrv = numpy.sqrt(tdrv_dx**2 + tdrv_dy**2)
    theta_ratio_v = hgm_v/(asa_v + eps)
    edge_weight_v = numpy.exp(-((tdr_v/(sigma_tilt + eps))**2))

    ratio_n = _robust_normalize(theta_ratio_v)
    thdr_n = _robust_normalize(thdr_tdrv)

    cgtp_v = (ratio_n + eps)**alpha * (thdr_n + eps)**beta * edge_weight_v
    cgtp_v = _robust_normalize(cgtp_v)

    if return_components is True:
        return {
            "vdr": vdr,
            "xz": xz,
            "yz": yz,
            "zz": zz,
            "hgm_v": hgm_v,
            "asa_v": asa_v,
            "tdr_v": tdr_v,
            "thdr_tdrv": thdr_tdrv,
            "theta_ratio_v": theta_ratio_v,
            "edge_weight_v": edge_weight_v,
            "cgtp_vdr": cgtp_v,
        }

    return cgtp_v


def my_cgtp_vdr(x, y, data, pad_factor=0.5, pad_mode="reflect",
                taper=True, alpha=1.0, beta=1.0,
                sigma_tilt=numpy.deg2rad(12.0), flatten=True):
    '''
    Compatibility wrapper for my_cgtp_vdr_grid.
    '''

    result = my_cgtp_vdr_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        alpha=alpha,
        beta=beta,
        sigma_tilt=sigma_tilt,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_multiscale_cgtp_grid(x, y, data, upward_levels=(0.0, 250.0, 500.0,
                            1000.0), pad_factor=0.5, pad_mode="reflect",
                            taper=True, alpha=1.0, beta=1.0,
                            sigma_tilt=numpy.deg2rad(12.0), eps=1.0e-12,
                            return_components=False):
    '''
    Compute the proposed Multiscale CGTP filter (MS-CGTP).

    The input field is upward continued to several levels and CGTP is computed
    for each scale. The final output is the average CGTP multiplied by a
    persistence index. The persistence index favors contacts that remain stable
    across scales.

    upward_levels must be in the same length unit as x and y. For example, use
    meters if x and y are in meters; use kilometers if x and y are in kilometers.
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    maps = []
    for level in upward_levels:
        if abs(float(level)) < eps:
            field_i = data0.copy()
        else:
            field_i = my_upward_continuation(
                x, y, data0, height=float(level),
                pad_factor=pad_factor,
                pad_mode=pad_mode,
                taper=taper,
                flatten=False
            )

        cgtp_i = my_cgtp_grid(
            x, y, field_i,
            pad_factor=pad_factor,
            pad_mode=pad_mode,
            taper=taper,
            alpha=alpha,
            beta=beta,
            sigma_tilt=sigma_tilt,
            return_components=False
        )
        maps.append(cgtp_i)

    stack = numpy.stack(maps, axis=0)
    mean_map = numpy.nanmean(stack, axis=0)
    std_map = numpy.nanstd(stack, axis=0)
    persistence = 1.0 - std_map/(mean_map + eps)
    persistence = numpy.clip(persistence, 0.0, 1.0)

    ms_cgtp = mean_map*persistence
    ms_cgtp = _robust_normalize(ms_cgtp)

    if return_components is True:
        return {
            "levels": tuple(upward_levels),
            "cgtp_stack": stack,
            "mean_cgtp": mean_map,
            "std_cgtp": std_map,
            "persistence": persistence,
            "ms_cgtp": ms_cgtp,
        }

    return ms_cgtp


def my_multiscale_cgtp(x, y, data, upward_levels=(0.0, 250.0, 500.0,
                       1000.0), pad_factor=0.5, pad_mode="reflect",
                       taper=True, alpha=1.0, beta=1.0,
                       sigma_tilt=numpy.deg2rad(12.0), flatten=True):
    '''
    Compatibility wrapper for my_multiscale_cgtp_grid.
    '''

    result = my_multiscale_cgtp_grid(
        x, y, data,
        upward_levels=upward_levels,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        alpha=alpha,
        beta=beta,
        sigma_tilt=sigma_tilt,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_directional_cgtp_grid(x, y, data, azimuth_degrees=45.0,
                            pad_factor=0.5, pad_mode="reflect",
                            taper=True, alpha=1.0, beta=1.0,
                            sigma_tilt=numpy.deg2rad(12.0),
                            directional_power=2.0, eps=1.0e-12,
                            return_components=False):
    '''
    Compute the proposed Directional CGTP filter (D-CGTP).

    D-CGTP multiplies CGTP by a directional weight based on the local structural
    strike inferred from the horizontal gradient. The gradient points normal to
    the contact, so the local strike is obtained by adding 90 degrees.

    azimuth_degrees is the target structural strike measured clockwise from the
    positive x direction in the x-y grid reference frame.
    '''

    comps = my_cgtp_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        alpha=alpha,
        beta=beta,
        sigma_tilt=sigma_tilt,
        return_components=True
    )

    gradient_azimuth = numpy.arctan2(comps["dy"], comps["dx"])
    structural_strike = gradient_azimuth + numpy.pi/2.0
    target = numpy.deg2rad(azimuth_degrees)
    delta = _angular_difference_rad(structural_strike, target)

    # Axial structures do not distinguish theta from theta + 180 degrees.
    weight = numpy.abs(numpy.cos(delta))**directional_power

    directional = comps["cgtp"]*weight
    directional = _robust_normalize(directional)

    if return_components is True:
        comps.update({
            "gradient_azimuth": gradient_azimuth,
            "structural_strike": structural_strike,
            "directional_weight": weight,
            "directional_cgtp": directional,
        })
        return comps

    return directional


def my_directional_cgtp(x, y, data, azimuth_degrees=45.0,
                        pad_factor=0.5, pad_mode="reflect",
                        taper=True, alpha=1.0, beta=1.0,
                        sigma_tilt=numpy.deg2rad(12.0),
                        directional_power=2.0, flatten=True):
    '''
    Compatibility wrapper for my_directional_cgtp_grid.
    '''

    result = my_directional_cgtp_grid(
        x, y, data,
        azimuth_degrees=azimuth_degrees,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        alpha=alpha,
        beta=beta,
        sigma_tilt=sigma_tilt,
        directional_power=directional_power,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)

# ============================================================
# ADVANCED PROPOSED FILTERS - FADG FAMILY
# ============================================================

def my_ncg_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                taper=True, eps=1.0e-12, normalize=True,
                return_components=False):
    '''
    Compute the Normalized Curvature Gravity filter (NCG).

    NCG enhances structural contacts, source flanks and basin edges by
    comparing the horizontal curvature energy with the total-gradient
    amplitude:

        NCG = sqrt(gxx^2 + 2*gxy^2 + gyy^2) / (ASA + eps)

    All derivatives are computed in the wavenumber domain. Grid expansion
    and tapering are controlled by pad_factor, pad_mode and taper.
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    gx = derivative.my_xderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gy = derivative.my_yderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gz = derivative.my_zderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gxx = derivative.my_xxderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gyy = derivative.my_yyderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gxy = derivative.my_mixed_deriv_fft(
        x, y, data0,
        nx_order=1, ny_order=1, nz_order=0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    asa = numpy.sqrt(gx**2 + gy**2 + gz**2)
    curvature = numpy.sqrt(gxx**2 + 2.0*gxy**2 + gyy**2)
    ncg_raw = curvature/(asa + eps)

    if normalize is True:
        ncg = _robust_normalize(ncg_raw)
    else:
        ncg = ncg_raw

    if return_components is True:
        return {
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "gxx": gxx,
            "gyy": gyy,
            "gxy": gxy,
            "asa": asa,
            "curvature": curvature,
            "ncg_raw": ncg_raw,
            "ncg": ncg,
        }

    return ncg


def my_ncg(x, y, data, pad_factor=0.5, pad_mode="reflect",
           taper=True, normalize=True, flatten=True):
    '''Compatibility wrapper for my_ncg_grid.'''

    result = my_ncg_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        normalize=normalize,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_avh_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                taper=True, eps=1.0e-12, normalize=False,
                return_components=False):
    '''
    Compute the Vertical-Horizontal Asymmetry filter (AVH).

    AVH compares vertical-derivative energy with horizontal-gradient energy:

        AVH = (abs(gz) - HGM) / (abs(gz) + HGM + eps)

    Positive values indicate relative dominance of the vertical derivative.
    Negative values indicate relative dominance of the horizontal gradient.
    All derivatives are computed in the wavenumber domain.
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    gx = derivative.my_xderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gy = derivative.my_yderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gz = derivative.my_zderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    hgm = numpy.sqrt(gx**2 + gy**2)
    avh_raw = (numpy.abs(gz) - hgm)/(numpy.abs(gz) + hgm + eps)

    if normalize is True:
        avh = _robust_normalize(avh_raw)
    else:
        avh = avh_raw

    if return_components is True:
        return {
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "hgm": hgm,
            "avh_raw": avh_raw,
            "avh": avh,
        }

    return avh


def my_avh(x, y, data, pad_factor=0.5, pad_mode="reflect",
           taper=True, normalize=False, flatten=True):
    '''Compatibility wrapper for my_avh_grid.'''

    result = my_avh_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        normalize=normalize,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_fseg_grid(x, y, data, pad_factor=0.5, pad_mode="reflect",
                 taper=True, eps=1.0e-12, normalize=False,
                 return_components=False):
    '''
    Compute the Structural Gravity Phase filter (FSEG).

    FSEG combines the phase of the classical tilt derivative with the phase
    of the tilt derivative applied to the vertical derivative:

        FSEG = sin(TDR) * cos(TDR_V)

    where
        TDR   = atan2(gz, sqrt(gx^2 + gy^2))
        TDR_V = atan2(gzz, sqrt(gxz^2 + gyz^2))

    The filter is designed to detect contacts where field phase and vertical
    derivative phase show coherent behavior. All derivatives are computed in
    the wavenumber domain.
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    gx = derivative.my_xderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gy = derivative.my_yderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gz = derivative.my_zderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gxz = derivative.my_xzderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gyz = derivative.my_yzderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gzz = derivative.my_zzderiv(
        x, y, data0,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    hgm = numpy.sqrt(gx**2 + gy**2)
    hgm_v = numpy.sqrt(gxz**2 + gyz**2)
    tdr = numpy.arctan2(gz, hgm + eps)
    tdr_v = numpy.arctan2(gzz, hgm_v + eps)
    fseg_raw = numpy.sin(tdr)*numpy.cos(tdr_v)

    if normalize is True:
        fseg = _robust_normalize(fseg_raw)
    else:
        fseg = fseg_raw

    if return_components is True:
        return {
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "gxz": gxz,
            "gyz": gyz,
            "gzz": gzz,
            "tdr": tdr,
            "tdr_v": tdr_v,
            "fseg_raw": fseg_raw,
            "fseg": fseg,
        }

    return fseg


def my_fseg(x, y, data, pad_factor=0.5, pad_mode="reflect",
            taper=True, normalize=False, flatten=True):
    '''Compatibility wrapper for my_fseg_grid.'''

    result = my_fseg_grid(
        x, y, data,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        normalize=normalize,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def my_adg_grid(x, y, data, azimuth_degrees=45.0,
                pad_factor=0.5, pad_mode="reflect", taper=True,
                eps=1.0e-12, normalize=False, return_components=False):
    '''
    Compute the Directional Gradient Anisotropy filter (ADG).

    The filter compares the directional derivative along a target azimuth
    with the directional derivative along the orthogonal direction:

        ADG = (abs(D_theta g) - abs(D_theta+90 g)) /
              (abs(D_theta g) + abs(D_theta+90 g) + eps)

    The azimuth is measured counterclockwise from the positive x direction in
    the x-y grid reference frame. All derivatives are computed in the
    wavenumber domain.
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    gx = derivative.my_xderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    gy = derivative.my_yderiv(
        x, y, data0, n=1,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        flatten=False
    )

    theta = numpy.deg2rad(azimuth_degrees)
    dtheta = gx*numpy.cos(theta) + gy*numpy.sin(theta)
    dorth = -gx*numpy.sin(theta) + gy*numpy.cos(theta)
    adg_raw = (numpy.abs(dtheta) - numpy.abs(dorth))/(numpy.abs(dtheta) + numpy.abs(dorth) + eps)

    if normalize is True:
        adg = _robust_normalize(adg_raw)
    else:
        adg = adg_raw

    if return_components is True:
        return {
            "gx": gx,
            "gy": gy,
            "dtheta": dtheta,
            "dorth": dorth,
            "adg_raw": adg_raw,
            "adg": adg,
        }

    return adg


def my_adg(x, y, data, azimuth_degrees=45.0,
           pad_factor=0.5, pad_mode="reflect", taper=True,
           normalize=False, flatten=True):
    '''Compatibility wrapper for my_adg_grid.'''

    result = my_adg_grid(
        x, y, data,
        azimuth_degrees=azimuth_degrees,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        normalize=normalize,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


def _edge_detector_for_peb(x, y, field, detector="hgm", pad_factor=0.5,
                           pad_mode="reflect", taper=True, eps=1.0e-12):
    '''Internal edge detector used by PEB.'''

    detector_name = str(detector).lower()

    if detector_name == "hgm":
        edge = derivative.my_hgrad(
            x, y, field,
            pad_factor=pad_factor,
            pad_mode=pad_mode,
            taper=taper,
            flatten=False
        )
        return _robust_normalize(edge)

    if detector_name in ("tilt_zero", "tdr_zero", "tilt"):
        tilt = my_tilt_grid(
            x, y, field,
            pad_factor=pad_factor,
            pad_mode=pad_mode,
            taper=taper,
            degrees=False
        )
        edge = numpy.exp(-((tilt/(numpy.deg2rad(12.0) + eps))**2))
        return _robust_normalize(edge)

    if detector_name == "cgtp":
        return my_cgtp_grid(
            x, y, field,
            pad_factor=pad_factor,
            pad_mode=pad_mode,
            taper=taper,
            return_components=False
        )

    raise ValueError("detector must be 'hgm', 'tilt_zero' or 'cgtp'.")


def my_peb_grid(x, y, data, wavelength_bands=((300.0, 700.0),
                (700.0, 1500.0), (1500.0, 3000.0)), detector="hgm",
                pad_factor=0.5, pad_mode="reflect", taper=True,
                eps=1.0e-12, return_components=False):
    '''
    Compute the Spectral Edge Persistence filter (PEB).

    The field is decomposed into Gaussian band-pass components. An edge
    detector is applied to each spectral band and the final output combines
    average edge strength with spectral persistence:

        PEB = mean(E_i) * (1 - std(E_i)/(mean(E_i) + eps))

    where E_i is the selected edge detector applied to the i-th band. This
    filter favors edges that remain coherent in different wavenumber bands.
    '''

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    edge_maps = []
    band_fields = []

    for band in wavelength_bands:
        if len(band) != 2:
            raise ValueError("Each wavelength band must have two values: (min, max).")

        wmin, wmax = float(band[0]), float(band[1])
        field_i = my_bandpass_gaussian(
            x, y, data0,
            wavelength_min=wmin,
            wavelength_max=wmax,
            pad_factor=pad_factor,
            pad_mode=pad_mode,
            taper=taper,
            flatten=False
        )

        edge_i = _edge_detector_for_peb(
            x, y, field_i,
            detector=detector,
            pad_factor=pad_factor,
            pad_mode=pad_mode,
            taper=taper,
            eps=eps
        )

        band_fields.append(field_i)
        edge_maps.append(edge_i)

    stack = numpy.stack(edge_maps, axis=0)
    mean_edge = numpy.nanmean(stack, axis=0)
    std_edge = numpy.nanstd(stack, axis=0)
    persistence = 1.0 - std_edge/(mean_edge + eps)
    persistence = numpy.clip(persistence, 0.0, 1.0)
    peb_raw = mean_edge*persistence
    peb = _robust_normalize(peb_raw)

    if return_components is True:
        return {
            "wavelength_bands": tuple(tuple(b) for b in wavelength_bands),
            "detector": detector,
            "band_fields": numpy.stack(band_fields, axis=0),
            "edge_stack": stack,
            "mean_edge": mean_edge,
            "std_edge": std_edge,
            "persistence": persistence,
            "peb_raw": peb_raw,
            "peb": peb,
        }

    return peb


def my_peb(x, y, data, wavelength_bands=((300.0, 700.0),
           (700.0, 1500.0), (1500.0, 3000.0)), detector="hgm",
           pad_factor=0.5, pad_mode="reflect", taper=True, flatten=True):
    '''Compatibility wrapper for my_peb_grid.'''

    result = my_peb_grid(
        x, y, data,
        wavelength_bands=wavelength_bands,
        detector=detector,
        pad_factor=pad_factor,
        pad_mode=pad_mode,
        taper=taper,
        return_components=False
    )

    return _as_grid(result, data.shape, flatten=flatten)


# Compatibility aliases for the FADG family
my_normalized_curvature_gravity = my_ncg
my_vertical_horizontal_asymmetry = my_avh
my_structural_gravity_phase = my_fseg
my_directional_gradient_anisotropy = my_adg
my_spectral_edge_persistence = my_peb


# ============================================================
# SPACE-DOMAIN SOURCE-DETECTION FILTERS AND DECONVOLUTIONS
# ============================================================
# This section adds space-domain versions of the 10 proposed filters.
# They use finite-difference derivatives and optional Gaussian smoothing
# computed by direct convolution in the space domain. No FFT operator is used
# inside these filters. The suffix _sd means "space domain".


def _gaussian_kernel_1d_space(sigma, radius=None):
    """Build a normalized 1D Gaussian kernel for space-domain convolution."""

    if sigma is None or sigma <= 0.0:
        return numpy.asarray([1.0], dtype=float)

    if radius is None:
        radius = int(numpy.ceil(3.0*float(sigma)))

    radius = max(1, int(radius))
    u = numpy.arange(-radius, radius + 1, dtype=float)
    kernel = numpy.exp(-0.5*(u/float(sigma))**2)
    kernel = kernel/numpy.sum(kernel)

    return kernel


def _convolve1d_reflect_space(arr, kernel, axis):
    """Apply 1D convolution using reflect padding along one axis."""

    arr = numpy.asarray(arr, dtype=float)
    kernel = numpy.asarray(kernel, dtype=float)
    radius = kernel.size//2

    if radius == 0:
        return arr.copy()

    pad_width = [(0, 0)]*arr.ndim
    pad_width[int(axis)] = (radius, radius)
    padded = numpy.pad(arr, pad_width=pad_width, mode='reflect')

    def _conv(v):
        return numpy.convolve(v, kernel, mode='valid')

    return numpy.apply_along_axis(_conv, int(axis), padded)


def _gaussian_smooth_space(data, sigma=None):
    """Smooth a 2D grid with separable Gaussian convolution in space domain."""

    arr = _finite_data(data)

    if sigma is None or sigma <= 0.0:
        return arr.copy()

    kernel = _gaussian_kernel_1d_space(float(sigma))
    out = _convolve1d_reflect_space(arr, kernel, axis=1)
    out = _convolve1d_reflect_space(out, kernel, axis=0)

    return out


def _space_bandpass_gaussian(data, sigma_min=1.0, sigma_max=3.0):
    """Approximate a space-domain band-pass by difference of Gaussian smooths."""

    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError('sigma_min and sigma_max must be positive!')
    if sigma_min >= sigma_max:
        raise ValueError('sigma_min must be smaller than sigma_max!')

    low_min = _gaussian_smooth_space(data, sigma=sigma_min)
    low_max = _gaussian_smooth_space(data, sigma=sigma_max)

    return low_min - low_max


def _space_first_derivatives(x, y, data, smooth_sigma=None):
    """Compute first horizontal derivatives using finite differences."""

    _check_grid(x, y, data)
    field = _gaussian_smooth_space(data, sigma=smooth_sigma)
    dx, dy = _grid_spacing(x, y)
    gy, gx = numpy.gradient(field, dy, dx, edge_order=2)

    return gx, gy, field


def _space_second_derivatives(x, y, data, smooth_sigma=None):
    """Compute first and second horizontal derivatives using finite differences."""

    _check_grid(x, y, data)
    field = _gaussian_smooth_space(data, sigma=smooth_sigma)
    dx, dy = _grid_spacing(x, y)

    gy, gx = numpy.gradient(field, dy, dx, edge_order=2)
    gxy_from_gx, gxx = numpy.gradient(gx, dy, dx, edge_order=2)
    gyy, gyx_from_gy = numpy.gradient(gy, dy, dx, edge_order=2)
    gxy = 0.5*(gxy_from_gx + gyx_from_gy)

    return gx, gy, gxx, gxy, gyy, field


def _space_pseudo_vertical_derivatives(x, y, data, smooth_sigma=None):
    """
    Estimate vertical-derivative proxies in the space domain.

    For a single observation level, true vertical derivatives are most commonly
    obtained in the wavenumber domain. To keep this section strictly in the
    space domain, we use horizontal-curvature proxies derived from the
    potential-field harmonic condition. These proxies are intended for
    comparative filter testing and not as a substitute for rigorous spectral
    continuation.
    """

    gx, gy, gxx, gxy, gyy, field = _space_second_derivatives(
        x, y, data, smooth_sigma=smooth_sigma)

    gz_proxy = -(gxx + gyy)

    dx, dy = _grid_spacing(x, y)
    gyz, gxz = numpy.gradient(gz_proxy, dy, dx, edge_order=2)
    gyzz, gxzz = numpy.gradient(gxz, dy, dx, edge_order=2)
    gzzz_proxy = -(gxzz + gyzz)

    return {
        'field': field,
        'gx': gx,
        'gy': gy,
        'gxx': gxx,
        'gxy': gxy,
        'gyy': gyy,
        'gz': gz_proxy,
        'gxz': gxz,
        'gyz': gyz,
        'gzz': gzzz_proxy,
    }


def _source_filter_components_sd(x, y, data, smooth_sigma=None, eps=1.0e-12):
    """Compute components used by the proposed source filters in space domain."""

    comps = _space_pseudo_vertical_derivatives(
        x, y, data, smooth_sigma=smooth_sigma)

    gx = comps['gx']
    gy = comps['gy']
    gz = comps['gz']

    hgm = numpy.sqrt(gx**2 + gy**2)
    asa = numpy.sqrt(gx**2 + gy**2 + gz**2)
    tdr = numpy.arctan2(gz, hgm + eps)

    dx, dy = _grid_spacing(x, y)
    tdr_dy, tdr_dx = numpy.gradient(tdr, dy, dx, edge_order=2)
    thdr_tdr = numpy.sqrt(tdr_dx**2 + tdr_dy**2)
    theta_ratio = hgm/(asa + eps)

    comps.update({
        'hgm': hgm,
        'asa': asa,
        'tdr': tdr,
        'tdr_dx': tdr_dx,
        'tdr_dy': tdr_dy,
        'thdr_tdr': thdr_tdr,
        'theta_ratio': theta_ratio,
    })

    return comps


def my_cgtp_sd_grid(x, y, data, smooth_sigma=None, alpha=1.0, beta=1.0,
                    sigma_tilt=numpy.deg2rad(12.0), eps=1.0e-12,
                    return_components=False):
    """
    Space-domain CGTP filter.

    CGTP_SD combines the horizontal-to-total gradient ratio, the horizontal
    gradient of the tilt derivative and a Gaussian weight centered on TDR = 0.
    """

    comps = _source_filter_components_sd(
        x, y, data, smooth_sigma=smooth_sigma, eps=eps)

    edge_weight = numpy.exp(-((comps['tdr']/(sigma_tilt + eps))**2))
    ratio_n = _robust_normalize(comps['theta_ratio'])
    thdr_n = _robust_normalize(comps['thdr_tdr'])

    cgtp = (ratio_n + eps)**alpha*(thdr_n + eps)**beta*edge_weight
    cgtp = _robust_normalize(cgtp)

    if return_components is True:
        comps.update({
            'edge_weight': edge_weight,
            'theta_ratio_normalized': ratio_n,
            'thdr_tdr_normalized': thdr_n,
            'cgtp_sd': cgtp,
        })
        return comps

    return cgtp


def my_cgtp_sd(x, y, data, smooth_sigma=None, alpha=1.0, beta=1.0,
               sigma_tilt=numpy.deg2rad(12.0), flatten=True):
    """Compatibility wrapper for my_cgtp_sd_grid."""

    result = my_cgtp_sd_grid(
        x, y, data, smooth_sigma=smooth_sigma,
        alpha=alpha, beta=beta, sigma_tilt=sigma_tilt,
        return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_dri_sd_grid(x, y, data, smooth_sigma=None,
                   sigma_tilt=numpy.deg2rad(12.0), eps=1.0e-12,
                   return_components=False):
    """Space-domain Derivative Relative-Depth Index."""

    comps = _source_filter_components_sd(
        x, y, data, smooth_sigma=smooth_sigma, eps=eps)

    edge_weight = numpy.exp(-((comps['tdr']/(sigma_tilt + eps))**2))
    dri_raw = edge_weight/(comps['thdr_tdr'] + eps)
    dri = _robust_normalize(dri_raw)

    if return_components is True:
        comps.update({'edge_weight': edge_weight, 'dri_raw': dri_raw,
                      'dri_sd': dri})
        return comps

    return dri


def my_dri_sd(x, y, data, smooth_sigma=None,
              sigma_tilt=numpy.deg2rad(12.0), flatten=True):
    """Compatibility wrapper for my_dri_sd_grid."""

    result = my_dri_sd_grid(
        x, y, data, smooth_sigma=smooth_sigma,
        sigma_tilt=sigma_tilt, return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_cgtp_vdr_sd_grid(x, y, data, smooth_sigma=None, alpha=1.0,
                        beta=1.0, sigma_tilt=numpy.deg2rad(12.0),
                        eps=1.0e-12, return_components=False):
    """Space-domain CGTP applied to the pseudo vertical-derivative field."""

    comps = _space_pseudo_vertical_derivatives(
        x, y, data, smooth_sigma=smooth_sigma)

    hgm_v = numpy.sqrt(comps['gxz']**2 + comps['gyz']**2)
    asa_v = numpy.sqrt(comps['gxz']**2 + comps['gyz']**2 + comps['gzz']**2)
    tdr_v = numpy.arctan2(comps['gzz'], hgm_v + eps)

    dx, dy = _grid_spacing(x, y)
    tdrv_dy, tdrv_dx = numpy.gradient(tdr_v, dy, dx, edge_order=2)
    thdr_tdrv = numpy.sqrt(tdrv_dx**2 + tdrv_dy**2)
    theta_ratio_v = hgm_v/(asa_v + eps)
    edge_weight_v = numpy.exp(-((tdr_v/(sigma_tilt + eps))**2))

    ratio_n = _robust_normalize(theta_ratio_v)
    thdr_n = _robust_normalize(thdr_tdrv)
    cgtp_v = (ratio_n + eps)**alpha*(thdr_n + eps)**beta*edge_weight_v
    cgtp_v = _robust_normalize(cgtp_v)

    if return_components is True:
        comps.update({
            'hgm_v': hgm_v,
            'asa_v': asa_v,
            'tdr_v': tdr_v,
            'tdrv_dx': tdrv_dx,
            'tdrv_dy': tdrv_dy,
            'thdr_tdrv': thdr_tdrv,
            'theta_ratio_v': theta_ratio_v,
            'edge_weight_v': edge_weight_v,
            'cgtp_vdr_sd': cgtp_v,
        })
        return comps

    return cgtp_v


def my_cgtp_vdr_sd(x, y, data, smooth_sigma=None, alpha=1.0,
                   beta=1.0, sigma_tilt=numpy.deg2rad(12.0),
                   flatten=True):
    """Compatibility wrapper for my_cgtp_vdr_sd_grid."""

    result = my_cgtp_vdr_sd_grid(
        x, y, data, smooth_sigma=smooth_sigma,
        alpha=alpha, beta=beta, sigma_tilt=sigma_tilt,
        return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_multiscale_cgtp_sd_grid(x, y, data, smooth_sigmas=(0.0, 1.0, 2.0,
                              4.0), alpha=1.0, beta=1.0,
                              sigma_tilt=numpy.deg2rad(12.0), eps=1.0e-12,
                              return_components=False):
    """
    Space-domain multiscale CGTP using Gaussian smoothing scales.

    Each smooth_sigma is a Gaussian standard deviation in grid samples.
    """

    maps = []
    for sigma in smooth_sigmas:
        s = None if float(sigma) <= 0.0 else float(sigma)
        maps.append(my_cgtp_sd_grid(
            x, y, data, smooth_sigma=s,
            alpha=alpha, beta=beta, sigma_tilt=sigma_tilt,
            return_components=False))

    stack = numpy.stack(maps, axis=0)
    mean_map = numpy.nanmean(stack, axis=0)
    std_map = numpy.nanstd(stack, axis=0)
    persistence = 1.0 - std_map/(mean_map + eps)
    persistence = numpy.clip(persistence, 0.0, 1.0)
    ms_cgtp = _robust_normalize(mean_map*persistence)

    if return_components is True:
        return {
            'smooth_sigmas': tuple(smooth_sigmas),
            'cgtp_stack': stack,
            'mean_cgtp': mean_map,
            'std_cgtp': std_map,
            'persistence': persistence,
            'ms_cgtp_sd': ms_cgtp,
        }

    return ms_cgtp


def my_multiscale_cgtp_sd(x, y, data, smooth_sigmas=(0.0, 1.0, 2.0, 4.0),
                          alpha=1.0, beta=1.0,
                          sigma_tilt=numpy.deg2rad(12.0), flatten=True):
    """Compatibility wrapper for my_multiscale_cgtp_sd_grid."""

    result = my_multiscale_cgtp_sd_grid(
        x, y, data, smooth_sigmas=smooth_sigmas,
        alpha=alpha, beta=beta, sigma_tilt=sigma_tilt,
        return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_directional_cgtp_sd_grid(x, y, data, azimuth_degrees=45.0,
                                smooth_sigma=None, alpha=1.0, beta=1.0,
                                sigma_tilt=numpy.deg2rad(12.0),
                                directional_power=2.0,
                                return_components=False):
    """Space-domain directional CGTP filter."""

    comps = my_cgtp_sd_grid(
        x, y, data, smooth_sigma=smooth_sigma,
        alpha=alpha, beta=beta, sigma_tilt=sigma_tilt,
        return_components=True)

    gradient_azimuth = numpy.arctan2(comps['gy'], comps['gx'])
    structural_strike = gradient_azimuth + numpy.pi/2.0
    target = numpy.deg2rad(azimuth_degrees)
    delta = _angular_difference_rad(structural_strike, target)
    weight = numpy.abs(numpy.cos(delta))**directional_power
    directional = _robust_normalize(comps['cgtp_sd']*weight)

    if return_components is True:
        comps.update({
            'gradient_azimuth': gradient_azimuth,
            'structural_strike': structural_strike,
            'directional_weight': weight,
            'directional_cgtp_sd': directional,
        })
        return comps

    return directional


def my_directional_cgtp_sd(x, y, data, azimuth_degrees=45.0,
                           smooth_sigma=None, alpha=1.0, beta=1.0,
                           sigma_tilt=numpy.deg2rad(12.0),
                           directional_power=2.0, flatten=True):
    """Compatibility wrapper for my_directional_cgtp_sd_grid."""

    result = my_directional_cgtp_sd_grid(
        x, y, data, azimuth_degrees=azimuth_degrees,
        smooth_sigma=smooth_sigma, alpha=alpha, beta=beta,
        sigma_tilt=sigma_tilt, directional_power=directional_power,
        return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_ncg_sd_grid(x, y, data, smooth_sigma=None, eps=1.0e-12,
                   normalize=True, return_components=False):
    """Space-domain Normalized Curvature Gravity filter."""

    comps = _space_pseudo_vertical_derivatives(
        x, y, data, smooth_sigma=smooth_sigma)

    asa = numpy.sqrt(comps['gx']**2 + comps['gy']**2 + comps['gz']**2)
    curvature = numpy.sqrt(comps['gxx']**2 + 2.0*comps['gxy']**2 + comps['gyy']**2)
    ncg_raw = curvature/(asa + eps)
    ncg = _robust_normalize(ncg_raw) if normalize is True else ncg_raw

    if return_components is True:
        comps.update({'asa': asa, 'curvature': curvature,
                      'ncg_raw': ncg_raw, 'ncg_sd': ncg})
        return comps

    return ncg


def my_ncg_sd(x, y, data, smooth_sigma=None, normalize=True, flatten=True):
    """Compatibility wrapper for my_ncg_sd_grid."""

    result = my_ncg_sd_grid(
        x, y, data, smooth_sigma=smooth_sigma,
        normalize=normalize, return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_avh_sd_grid(x, y, data, smooth_sigma=None, eps=1.0e-12,
                   normalize=False, return_components=False):
    """Space-domain Vertical-Horizontal Asymmetry filter."""

    comps = _space_pseudo_vertical_derivatives(
        x, y, data, smooth_sigma=smooth_sigma)

    hgm = numpy.sqrt(comps['gx']**2 + comps['gy']**2)
    avh_raw = (numpy.abs(comps['gz']) - hgm)/(numpy.abs(comps['gz']) + hgm + eps)
    avh = _robust_normalize(avh_raw) if normalize is True else avh_raw

    if return_components is True:
        comps.update({'hgm': hgm, 'avh_raw': avh_raw, 'avh_sd': avh})
        return comps

    return avh


def my_avh_sd(x, y, data, smooth_sigma=None, normalize=False,
              flatten=True):
    """Compatibility wrapper for my_avh_sd_grid."""

    result = my_avh_sd_grid(
        x, y, data, smooth_sigma=smooth_sigma,
        normalize=normalize, return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_fseg_sd_grid(x, y, data, smooth_sigma=None, eps=1.0e-12,
                    normalize=False, return_components=False):
    """Space-domain Structural Gravity Phase filter."""

    comps = _space_pseudo_vertical_derivatives(
        x, y, data, smooth_sigma=smooth_sigma)

    hgm = numpy.sqrt(comps['gx']**2 + comps['gy']**2)
    hgm_v = numpy.sqrt(comps['gxz']**2 + comps['gyz']**2)
    tdr = numpy.arctan2(comps['gz'], hgm + eps)
    tdr_v = numpy.arctan2(comps['gzz'], hgm_v + eps)
    fseg_raw = numpy.sin(tdr)*numpy.cos(tdr_v)
    fseg = _robust_normalize(fseg_raw) if normalize is True else fseg_raw

    if return_components is True:
        comps.update({'hgm': hgm, 'hgm_v': hgm_v, 'tdr': tdr,
                      'tdr_v': tdr_v, 'fseg_raw': fseg_raw,
                      'fseg_sd': fseg})
        return comps

    return fseg


def my_fseg_sd(x, y, data, smooth_sigma=None, normalize=False,
               flatten=True):
    """Compatibility wrapper for my_fseg_sd_grid."""

    result = my_fseg_sd_grid(
        x, y, data, smooth_sigma=smooth_sigma,
        normalize=normalize, return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def my_adg_sd_grid(x, y, data, azimuth_degrees=45.0, smooth_sigma=None,
                   eps=1.0e-12, normalize=False,
                   return_components=False):
    """Space-domain Directional Gradient Anisotropy filter."""

    gx, gy, field = _space_first_derivatives(
        x, y, data, smooth_sigma=smooth_sigma)

    theta = numpy.deg2rad(azimuth_degrees)
    dtheta = gx*numpy.cos(theta) + gy*numpy.sin(theta)
    dorth = -gx*numpy.sin(theta) + gy*numpy.cos(theta)
    adg_raw = (numpy.abs(dtheta) - numpy.abs(dorth))/(numpy.abs(dtheta) + numpy.abs(dorth) + eps)
    adg = _robust_normalize(adg_raw) if normalize is True else adg_raw

    if return_components is True:
        return {'field': field, 'gx': gx, 'gy': gy, 'dtheta': dtheta,
                'dorth': dorth, 'adg_raw': adg_raw, 'adg_sd': adg}

    return adg


def my_adg_sd(x, y, data, azimuth_degrees=45.0, smooth_sigma=None,
              normalize=False, flatten=True):
    """Compatibility wrapper for my_adg_sd_grid."""

    result = my_adg_sd_grid(
        x, y, data, azimuth_degrees=azimuth_degrees,
        smooth_sigma=smooth_sigma, normalize=normalize,
        return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


def _edge_detector_for_peb_sd(x, y, field, detector='hgm',
                              smooth_sigma=None, eps=1.0e-12):
    """Internal space-domain edge detector used by PEB-SD."""

    detector_name = str(detector).lower()

    if detector_name == 'hgm':
        gx, gy, _ = _space_first_derivatives(
            x, y, field, smooth_sigma=smooth_sigma)
        return _robust_normalize(numpy.sqrt(gx**2 + gy**2))

    if detector_name in ('tilt_zero', 'tdr_zero', 'tilt'):
        comps = _source_filter_components_sd(
            x, y, field, smooth_sigma=smooth_sigma, eps=eps)
        edge = numpy.exp(-((comps['tdr']/(numpy.deg2rad(12.0) + eps))**2))
        return _robust_normalize(edge)

    if detector_name == 'cgtp':
        return my_cgtp_sd_grid(
            x, y, field, smooth_sigma=smooth_sigma,
            return_components=False)

    raise ValueError("detector must be 'hgm', 'tilt_zero' or 'cgtp'.")


def my_peb_sd_grid(x, y, data, sigma_bands=((1.0, 2.0), (2.0, 4.0),
                   (4.0, 8.0)), detector='hgm', smooth_sigma=None,
                   eps=1.0e-12, return_components=False):
    """
    Space-domain Spectral/scale Edge Persistence filter.

    PEB_SD decomposes the field with differences of Gaussian smooths in the
    space domain and evaluates edge persistence among bands.
    """

    _check_grid(x, y, data)
    data0 = _finite_data(data)

    edge_maps = []
    band_fields = []
    for band in sigma_bands:
        if len(band) != 2:
            raise ValueError('Each sigma band must have two values: (min, max).')
        smin, smax = float(band[0]), float(band[1])
        band_i = _space_bandpass_gaussian(data0, sigma_min=smin,
                                          sigma_max=smax)
        edge_i = _edge_detector_for_peb_sd(
            x, y, band_i, detector=detector,
            smooth_sigma=smooth_sigma, eps=eps)
        band_fields.append(band_i)
        edge_maps.append(edge_i)

    stack = numpy.stack(edge_maps, axis=0)
    mean_edge = numpy.nanmean(stack, axis=0)
    std_edge = numpy.nanstd(stack, axis=0)
    persistence = 1.0 - std_edge/(mean_edge + eps)
    persistence = numpy.clip(persistence, 0.0, 1.0)
    peb_raw = mean_edge*persistence
    peb = _robust_normalize(peb_raw)

    if return_components is True:
        return {
            'sigma_bands': tuple(tuple(b) for b in sigma_bands),
            'detector': detector,
            'band_fields': numpy.stack(band_fields, axis=0),
            'edge_stack': stack,
            'mean_edge': mean_edge,
            'std_edge': std_edge,
            'persistence': persistence,
            'peb_raw': peb_raw,
            'peb_sd': peb,
        }

    return peb


def my_peb_sd(x, y, data, sigma_bands=((1.0, 2.0), (2.0, 4.0),
              (4.0, 8.0)), detector='hgm', smooth_sigma=None,
              flatten=True):
    """Compatibility wrapper for my_peb_sd_grid."""

    result = my_peb_sd_grid(
        x, y, data, sigma_bands=sigma_bands,
        detector=detector, smooth_sigma=smooth_sigma,
        return_components=False)

    return _as_grid(result, data.shape, flatten=flatten)


# ============================================================
# EULER AND WERNER DECONVOLUTION
# ============================================================

def my_euler_deconvolution_grid(x, y, data, structural_index=1.0,
                                z_obs=0.0, window_size=9, step=3,
                                smooth_sigma=None, min_points=None,
                                condition_max=1.0e10,
                                return_all=False):
    """
    Windowed Euler deconvolution for gridded potential-field data.

    The implementation solves, in each moving window:

        (x - x0) gx + (y - y0) gy + (z - z0) gz = N (B - T)

    which is rearranged as:

        x0 gx + y0 gy + z0 gz + N B = x gx + y gy + z gz + N T

    Output columns are:
        x0, y0, z0, B, rms, condition_number, x_center, y_center, n_points
    """

    _check_grid(x, y, data)
    if int(window_size) < 3:
        raise ValueError('window_size must be at least 3!')
    if int(window_size) % 2 == 0:
        raise ValueError('window_size must be odd!')
    if int(step) < 1:
        raise ValueError('step must be >= 1!')

    N = float(structural_index)
    field = _gaussian_smooth_space(data, sigma=smooth_sigma)
    comps = _space_pseudo_vertical_derivatives(
        x, y, field, smooth_sigma=None)
    gx = comps['gx']
    gy = comps['gy']
    gz = comps['gz']

    ny, nx = data.shape
    half = int(window_size)//2
    if min_points is None:
        min_points = max(6, int(0.65*window_size*window_size))

    solutions = []
    for iy in range(half, ny-half, int(step)):
        for ix in range(half, nx-half, int(step)):
            sl_y = slice(iy-half, iy+half+1)
            sl_x = slice(ix-half, ix+half+1)

            xv = x[sl_y, sl_x].ravel()
            yv = y[sl_y, sl_x].ravel()
            tv = field[sl_y, sl_x].ravel()
            gxv = gx[sl_y, sl_x].ravel()
            gyv = gy[sl_y, sl_x].ravel()
            gzv = gz[sl_y, sl_x].ravel()
            zv = numpy.zeros_like(xv) + float(z_obs)

            good = numpy.isfinite(xv + yv + tv + gxv + gyv + gzv)
            if numpy.sum(good) < min_points:
                continue

            xv = xv[good]
            yv = yv[good]
            zv = zv[good]
            tv = tv[good]
            gxv = gxv[good]
            gyv = gyv[good]
            gzv = gzv[good]

            A = numpy.column_stack([gxv, gyv, gzv,
                                    N*numpy.ones_like(tv)])
            b = xv*gxv + yv*gyv + zv*gzv + N*tv

            try:
                cond = numpy.linalg.cond(A)
                if not numpy.isfinite(cond) or cond > condition_max:
                    continue
                sol, residuals, rank, svals = numpy.linalg.lstsq(A, b, rcond=None)
                pred = A.dot(sol)
                rms = numpy.sqrt(numpy.mean((pred - b)**2))
                solutions.append([sol[0], sol[1], sol[2], sol[3], rms, cond,
                                  x[iy, ix], y[iy, ix], numpy.sum(good)])
            except Exception:
                continue

    out = numpy.asarray(solutions, dtype=float)
    if out.size == 0:
        out = numpy.empty((0, 9))

    if return_all is True:
        return {
            'solutions': out,
            'gx': gx,
            'gy': gy,
            'gz': gz,
            'field': field,
            'columns': ('x0', 'y0', 'z0', 'base_level', 'rms',
                        'condition', 'x_center', 'y_center', 'n_points'),
        }

    return out


def my_euler_deconvolution(x, y, data, structural_index=1.0,
                           z_obs=0.0, window_size=9, step=3,
                           smooth_sigma=None, min_points=None,
                           condition_max=1.0e10):
    """Compatibility wrapper for my_euler_deconvolution_grid."""

    return my_euler_deconvolution_grid(
        x, y, data, structural_index=structural_index,
        z_obs=z_obs, window_size=window_size, step=step,
        smooth_sigma=smooth_sigma, min_points=min_points,
        condition_max=condition_max, return_all=False)


def _hilbert_transform_1d(signal):
    """Return the Hilbert transform of a 1D real signal using FFT."""

    s = numpy.asarray(signal, dtype=float)
    n = s.size
    F = numpy.fft.fft(s)
    h = numpy.zeros(n)

    if n % 2 == 0:
        h[0] = 1.0
        h[n//2] = 1.0
        h[1:n//2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n+1)//2] = 2.0

    analytic = numpy.fft.ifft(F*h)
    return numpy.imag(analytic)


def my_werner_deconvolution_profile(x_profile, data_profile,
                                    structural_index=1.0,
                                    z_obs=0.0, window_size=15,
                                    step=1, background_order=0,
                                    vertical_derivative='hilbert',
                                    condition_max=1.0e10,
                                    min_points=None):
    """
    Simplified Werner-style moving-window deconvolution for profiles.

    This function applies an Euler-like linearization along a 1D profile:

        x0 Tx + z0 Tz + N B = x Tx + z Tz + N T

    Output columns for background_order = 0:
        x0, z0, B, rms, condition_number, x_center, n_points

    Output columns for background_order = 1:
        x0, z0, B0, B1, rms, condition_number, x_center, n_points
    """

    xp = numpy.asarray(x_profile, dtype=float).ravel()
    tp = numpy.asarray(data_profile, dtype=float).ravel()

    if xp.size != tp.size:
        raise ValueError('x_profile and data_profile must have the same size!')
    if xp.size < int(window_size):
        raise ValueError('profile is shorter than window_size!')
    if int(window_size) < 5:
        raise ValueError('window_size must be at least 5!')
    if int(window_size) % 2 == 0:
        raise ValueError('window_size must be odd!')

    order = numpy.argsort(xp)
    xp = xp[order]
    tp = tp[order]

    dx = numpy.nanmean(numpy.diff(xp))
    tx = numpy.gradient(tp, dx, edge_order=2)

    if str(vertical_derivative).lower() == 'hilbert':
        tz = -_hilbert_transform_1d(tx)
    else:
        txx = numpy.gradient(tx, dx, edge_order=2)
        tz = -txx

    N = float(structural_index)
    half = int(window_size)//2
    if min_points is None:
        min_points = max(5, int(0.70*window_size))

    solutions = []
    for center in range(half, xp.size-half, int(step)):
        sl = slice(center-half, center+half+1)
        xw = xp[sl]
        tw = tp[sl]
        txw = tx[sl]
        tzw = tz[sl]

        good = numpy.isfinite(xw + tw + txw + tzw)
        if numpy.sum(good) < min_points:
            continue

        xw = xw[good]
        tw = tw[good]
        txw = txw[good]
        tzw = tzw[good]

        b = xw*txw + float(z_obs)*tzw + N*tw

        if int(background_order) <= 0:
            A = numpy.column_stack([txw, tzw, N*numpy.ones_like(tw)])
        else:
            A = numpy.column_stack([txw, tzw, N*numpy.ones_like(tw), N*xw])

        try:
            cond = numpy.linalg.cond(A)
            if not numpy.isfinite(cond) or cond > condition_max:
                continue
            sol, residuals, rank, svals = numpy.linalg.lstsq(A, b, rcond=None)
            pred = A.dot(sol)
            rms = numpy.sqrt(numpy.mean((pred - b)**2))
            if int(background_order) <= 0:
                solutions.append([sol[0], abs(sol[1]), sol[2], rms, cond,
                                  xp[center], numpy.sum(good)])
            else:
                solutions.append([sol[0], abs(sol[1]), sol[2], sol[3], rms,
                                  cond, xp[center], numpy.sum(good)])
        except Exception:
            continue

    if len(solutions) == 0:
        if int(background_order) <= 0:
            return numpy.empty((0, 7))
        return numpy.empty((0, 8))

    return numpy.asarray(solutions, dtype=float)


def my_werner_deconvolution_grid_profile(x, y, data, profile_index=None,
                                         axis='x', structural_index=1.0,
                                         window_size=15, step=1,
                                         background_order=0,
                                         vertical_derivative='hilbert'):
    """Apply Werner-style deconvolution to a row or column of a 2D grid."""

    _check_grid(x, y, data)
    ny, nx = data.shape
    axis = str(axis).lower()

    if axis == 'x':
        if profile_index is None:
            profile_index = ny//2
        xp = x[int(profile_index), :]
        tp = data[int(profile_index), :]
    elif axis == 'y':
        if profile_index is None:
            profile_index = nx//2
        xp = y[:, int(profile_index)]
        tp = data[:, int(profile_index)]
    else:
        raise ValueError("axis must be 'x' or 'y'.")

    return my_werner_deconvolution_profile(
        xp, tp, structural_index=structural_index,
        window_size=window_size, step=step,
        background_order=background_order,
        vertical_derivative=vertical_derivative)


# Compatibility aliases for the space-domain family and deconvolutions
my_cgtp_space = my_cgtp_sd
my_dri_space = my_dri_sd
my_cgtp_vdr_space = my_cgtp_vdr_sd
my_multiscale_cgtp_space = my_multiscale_cgtp_sd
my_directional_cgtp_space = my_directional_cgtp_sd
my_ncg_space = my_ncg_sd
my_avh_space = my_avh_sd
my_fseg_space = my_fseg_sd
my_adg_space = my_adg_sd
my_peb_space = my_peb_sd

my_euler = my_euler_deconvolution
my_euler_grid = my_euler_deconvolution_grid
my_werner = my_werner_deconvolution_profile
my_werner_profile = my_werner_deconvolution_profile
my_werner_grid_profile = my_werner_deconvolution_grid_profile

