# ------------------------------------------------------------------------------------
# Title: Rectangular prism gravity and magnetic modelling
# Description: Forward modelling functions for one or many rectangular prisms.
# Author: Nelson Ribeiro Filho
# ------------------------------------------------------------------------------------

from __future__ import division
import numpy

try:
    from . import auxiliars
except Exception:
    try:
        import auxiliars
    except Exception:
        auxiliars = None

try:
    from . import constants
except Exception:
    try:
        import constants
    except Exception:
        constants = None


# ============================================================
# CONSTANTS AND SAFE AUXILIARY FUNCTIONS
# ============================================================

G = getattr(constants, "G", 6.673e-11) if constants is not None else 6.673e-11
SI2MGAL = getattr(constants, "si2mGal", 100000.0) if constants is not None else 100000.0
CM = getattr(constants, "cm", 1.0e-7) if constants is not None else 1.0e-7
T2NT = getattr(constants, "T2nT", 1.0e9) if constants is not None else 1.0e9
EPS = getattr(constants, "EPS", 1.0e-15) if constants is not None else 1.0e-15


def _safe_log(x):
    '''
    Return log(x), setting the limiting value log(0) contribution to zero.
    This follows the convention used in the original prism formulas.
    '''
    if auxiliars is not None and hasattr(auxiliars, "my_log"):
        return auxiliars.my_log(x)

    x = numpy.asarray(x)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        out = numpy.log(x)
    out = numpy.where((x == 0.0) | (~numpy.isfinite(out)), 0.0, out)
    return out


def _safe_atan(x, y):
    '''
    Return a stable arctan2(x, y), preserving the convention of the original code.
    '''
    if auxiliars is not None and hasattr(auxiliars, "my_atan"):
        return auxiliars.my_atan(x, y)

    x = numpy.asarray(x)
    y = numpy.asarray(y)
    out = numpy.arctan2(x, y)
    out = numpy.where(x == 0.0, 0.0, out)
    out = numpy.where((x > 0.0) & (y < 0.0), out - numpy.pi, out)
    out = numpy.where((x < 0.0) & (y < 0.0), out + numpy.pi, out)
    return out


def _dircos(inc, dec, azim=0.0):
    '''
    Return direction cosines for inclination and declination in degrees.
    If auxiliars.my_dircos exists, it is used to preserve compatibility.
    '''
    if auxiliars is not None and hasattr(auxiliars, "my_dircos"):
        try:
            return auxiliars.my_dircos(inc, dec, azim)
        except TypeError:
            return auxiliars.my_dircos(inc, dec)

    inc_rad = numpy.deg2rad(inc)
    dec_rad = numpy.deg2rad(dec - azim)
    mx = numpy.cos(inc_rad)*numpy.cos(dec_rad)
    my = numpy.cos(inc_rad)*numpy.sin(dec_rad)
    mz = numpy.sin(inc_rad)
    return mx, my, mz


def _check_xy(x, y):
    '''Check if x and y have the same shape.'''
    if numpy.shape(x) != numpy.shape(y):
        raise ValueError("x and y must have the same shape!")


def _prepare_xyz(x, y, z):
    '''
    Convert x, y, z to arrays compatible with broadcasting.
    z may be scalar or array with the same shape as x and y.
    '''
    _check_xy(x, y)
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)

    if numpy.isscalar(z):
        z = numpy.zeros_like(x, dtype=float) + float(z)
    else:
        z = numpy.asarray(z, dtype=float)
        if z.shape != x.shape:
            raise ValueError("z must be scalar or have the same shape as x and y!")

    return x, y, z


def _validate_prism(prism):
    '''
    Validate and return a prism as a 6-element numpy array.
    Prism order: [x1, x2, y1, y2, z1, z2].
    '''
    prism = numpy.asarray(prism, dtype=float).ravel()
    if prism.size < 6:
        raise ValueError("prism must have at least 6 elements: [x1, x2, y1, y2, z1, z2]!")
    prism = prism[:6].copy()

    if prism[0] > prism[1]:
        prism[0], prism[1] = prism[1], prism[0]
    if prism[2] > prism[3]:
        prism[2], prism[3] = prism[3], prism[2]
    if prism[4] > prism[5]:
        prism[4], prism[5] = prism[5], prism[4]

    return prism


def _density_value(prism, rho=None):
    '''
    Return density in g/cm3. If rho is None and prism has a 7th element,
    prism[6] is used.
    '''
    prism_array = numpy.asarray(prism, dtype=float).ravel()
    if rho is None:
        if prism_array.size >= 7:
            return prism_array[6]
        raise ValueError("rho must be provided when prism does not contain density as prism[6]!")
    return rho


def _mag_value(prism, mag=None):
    '''
    Return magnetization intensity. If mag is None and prism has a 7th element,
    prism[6] is used.
    '''
    prism_array = numpy.asarray(prism, dtype=float).ravel()
    if mag is None:
        if prism_array.size >= 7:
            return prism_array[6]
        raise ValueError("mag must be provided when prism does not contain magnetization as prism[6]!")
    return mag


def _prism_distances(x, y, z, prism):
    '''Return the integration-limit distance arrays used by the prism formulas.'''
    prism = _validate_prism(prism)
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    return xp, yp, zp


def my_prism_volume(prism):
    '''
    Return the volume of a rectangular prism.

    Inputs:
    prism - array/list - [x1, x2, y1, y2, z1, z2]

    Output:
    volume - float - prism volume
    '''
    p = _validate_prism(prism)
    return (p[1] - p[0])*(p[3] - p[2])*(p[5] - p[4])


def my_prism_center(prism):
    '''
    Return the center coordinates of a rectangular prism.
    '''
    p = _validate_prism(prism)
    xc = 0.5*(p[0] + p[1])
    yc = 0.5*(p[2] + p[3])
    zc = 0.5*(p[4] + p[5])
    return xc, yc, zc


# ============================================================
# GRAVITY OF ONE PRISM
# ============================================================

def my_potential(x, y, z, prism, rho=None):
    '''
    Calculate the gravitational potential due to a rectangular prism.

    Inputs:
    x, y - numpy arrays - observation coordinates
    z - scalar or numpy array - observation level, positive downward
    prism - array/list - [x1, x2, y1, y2, z1, z2] or [x1, x2, y1, y2, z1, z2, rho]
    rho - float/None - density in g/cm3. If None, prism[6] is used.

    Output:
    potential - numpy array - gravitational potential in SI units
    '''
    x, y, z = _prepare_xyz(x, y, z)
    rho = _density_value(prism, rho)*1000.0
    xp, yp, zp = _prism_distances(x, y, z, prism)

    potential = numpy.zeros_like(x, dtype=float)

    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = (xp[i]*yp[j]*_safe_log(zp[k] + r)
                          + yp[j]*zp[k]*_safe_log(xp[i] + r)
                          + xp[i]*zp[k]*_safe_log(yp[j] + r)
                          - 0.5*xp[i]**2*_safe_atan(zp[k]*yp[j], xp[i]*r)
                          - 0.5*yp[j]**2*_safe_atan(zp[k]*xp[i], yp[j]*r)
                          - 0.5*zp[k]**2*_safe_atan(xp[i]*yp[j], zp[k]*r))
                potential += ((-1.0)**(i + j + k))*result*rho

    potential *= G
    return potential


def my_prism_gx(x, y, z, prism, rho=None):
    '''
    Calculate the x component of gravitational attraction due to a rectangular prism.

    Output unit: mGal.
    '''
    x, y, z = _prepare_xyz(x, y, z)
    rho = _density_value(prism, rho)*1000.0
    xp, yp, zp = _prism_distances(x, y, z, prism)

    gx = numpy.zeros_like(x, dtype=float)

    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(yp[j]*_safe_log(zp[k] + r)
                           + zp[k]*_safe_log(yp[j] + r)
                           - xp[i]*_safe_atan(zp[k]*yp[j], xp[i]*r))
                gx += ((-1.0)**(i + j + k))*result*rho

    gx *= G*SI2MGAL
    return gx


def my_prism_gy(x, y, z, prism, rho=None):
    '''
    Calculate the y component of gravitational attraction due to a rectangular prism.

    Output unit: mGal.
    '''
    x, y, z = _prepare_xyz(x, y, z)
    rho = _density_value(prism, rho)*1000.0
    xp, yp, zp = _prism_distances(x, y, z, prism)

    gy = numpy.zeros_like(x, dtype=float)

    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(zp[k]*_safe_log(xp[i] + r)
                           + xp[i]*_safe_log(zp[k] + r)
                           - yp[j]*_safe_atan(xp[i]*zp[k], yp[j]*r))
                gy += ((-1.0)**(i + j + k))*result*rho

    gy *= G*SI2MGAL
    return gy


def my_prism_gz(x, y, z, prism, rho=None):
    '''
    Calculate the vertical component of gravitational attraction due to a rectangular prism.

    Output unit: mGal.
    '''
    x, y, z = _prepare_xyz(x, y, z)
    rho = _density_value(prism, rho)*1000.0
    xp, yp, zp = _prism_distances(x, y, z, prism)

    gz = numpy.zeros_like(x, dtype=float)

    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(xp[i]*_safe_log(yp[j] + r)
                           + yp[j]*_safe_log(xp[i] + r)
                           - zp[k]*_safe_atan(xp[i]*yp[j], zp[k]*r))
                gz += ((-1.0)**(i + j + k))*result*rho

    gz *= G*SI2MGAL
    return gz


def my_prism_gravity(x, y, z, prism, rho=None, component="gz"):
    '''
    General interface for prism gravity modelling.

    component options: "potential", "gx", "gy", "gz", "gxy", "gxz", "gyz", "gxx", "gyy", "gzz".
    Gravity components are returned in mGal; gravity-gradient kernels are returned without G scaling
    unless using my_prism_tensor_physical.
    '''
    component = component.lower()
    if component in ["potential", "pot", "v"]:
        return my_potential(x, y, z, prism, rho=rho)
    if component == "gx":
        return my_prism_gx(x, y, z, prism, rho=rho)
    if component == "gy":
        return my_prism_gy(x, y, z, prism, rho=rho)
    if component == "gz":
        return my_prism_gz(x, y, z, prism, rho=rho)
    if component == "gxx":
        return my_kernelxx(x, y, z, prism)
    if component == "gyy":
        return my_kernelyy(x, y, z, prism)
    if component == "gzz":
        return my_kernelzz(x, y, z, prism)
    if component == "gxy":
        return my_kernelxy(x, y, z, prism)
    if component == "gxz":
        return my_kernelxz(x, y, z, prism)
    if component == "gyz":
        return my_kernelyz(x, y, z, prism)
    raise ValueError("Invalid component: %s" % component)


# ============================================================
# GRAVITY OF MANY PRISMS
# ============================================================

def _as_array(value, size, name):
    '''Return value as a 1D array with a required size.'''
    arr = numpy.asarray(value, dtype=float)
    if arr.ndim == 0:
        return numpy.zeros(size, dtype=float) + float(arr)
    arr = arr.ravel()
    if arr.size != size:
        raise ValueError("%s must be scalar or have size equal to number of prisms!" % name)
    return arr


def my_build_prisms_from_centers(xprism, yprism, top, bottom, deltax, deltay, density=None):
    '''
    Build a list/array of prisms from center coordinates, top, bottom and cell dimensions.

    Inputs:
    xprism, yprism - arrays - prism center coordinates
    top, bottom - arrays/scalars - prism top and bottom depths
    deltax, deltay - floats/scalars/arrays - horizontal prism dimensions
    density - None/scalar/array - optional density appended as prism[6]

    Output:
    prisms - numpy array - shape (n_prisms, 6) or (n_prisms, 7)
    '''
    xprism = numpy.asarray(xprism, dtype=float).ravel()
    yprism = numpy.asarray(yprism, dtype=float).ravel()
    if xprism.size != yprism.size:
        raise ValueError("xprism and yprism must have the same size!")

    n = xprism.size
    top = _as_array(top, n, "top")
    bottom = _as_array(bottom, n, "bottom")
    deltax = _as_array(deltax, n, "deltax")
    deltay = _as_array(deltay, n, "deltay")

    prisms = numpy.column_stack([
        xprism - 0.5*deltax,
        xprism + 0.5*deltax,
        yprism - 0.5*deltay,
        yprism + 0.5*deltay,
        top,
        bottom
    ])

    if density is not None:
        density = _as_array(density, n, "density")
        prisms = numpy.column_stack([prisms, density])

    return prisms


def my_sum_prisms(x, y, z, prisms, physical_property=None, function=my_prism_gz,
                  progress=False):
    '''
    Sum the response of many prisms using a modelling function.

    Inputs:
    x, y, z - arrays/scalar - observation coordinates
    prisms - array/list - each row must contain [x1, x2, y1, y2, z1, z2] and optionally property
    physical_property - None/scalar/array - density or magnetization. If None, prism[6] is used.
    function - callable - modelling function with signature function(x, y, z, prism, property)
    progress - bool - print progress every 10% if True

    Output:
    total - numpy array - summed response
    '''
    x, y, z = _prepare_xyz(x, y, z)
    prisms = numpy.asarray(prisms, dtype=float)
    if prisms.ndim == 1:
        prisms = prisms.reshape(1, prisms.size)
    if prisms.shape[1] < 6:
        raise ValueError("prisms must have at least 6 columns!")

    n = prisms.shape[0]

    if physical_property is None:
        if prisms.shape[1] >= 7:
            props = prisms[:, 6]
        else:
            raise ValueError("physical_property must be provided if prisms do not contain a 7th column!")
    else:
        props = _as_array(physical_property, n, "physical_property")

    total = numpy.zeros_like(x, dtype=float)

    step = max(1, n//10)
    for i in range(n):
        total += function(x, y, z, prisms[i, :6], props[i])
        if progress and ((i + 1) % step == 0 or (i + 1) == n):
            print("Computed %d of %d prisms" % (i + 1, n))

    return total


def my_3Dpotential(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''
    Calculate gravitational potential due to a set of rectangular prisms.
    This function preserves the original interface.
    '''
    prisms = my_build_prisms_from_centers(xprism, yprism, top, bottom, deltax, deltay)
    return my_sum_prisms(x, y, z, prisms, density, function=my_potential)


def my_3Dgx(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''Calculate gx due to a set of rectangular prisms. Preserves the original interface.'''
    prisms = my_build_prisms_from_centers(xprism, yprism, top, bottom, deltax, deltay)
    return my_sum_prisms(x, y, z, prisms, density, function=my_prism_gx)


def my_3Dgy(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''Calculate gy due to a set of rectangular prisms. Preserves the original interface.'''
    prisms = my_build_prisms_from_centers(xprism, yprism, top, bottom, deltax, deltay)
    return my_sum_prisms(x, y, z, prisms, density, function=my_prism_gy)


def my_3Dgz(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''Calculate gz due to a set of rectangular prisms. Preserves the original interface.'''
    prisms = my_build_prisms_from_centers(xprism, yprism, top, bottom, deltax, deltay)
    return my_sum_prisms(x, y, z, prisms, density, function=my_prism_gz)


def my_prisms_gz(x, y, z, prisms, density=None, progress=False):
    '''Calculate gz due to many prisms using a prism table.'''
    return my_sum_prisms(x, y, z, prisms, density, function=my_prism_gz, progress=progress)


def my_prisms_gx(x, y, z, prisms, density=None, progress=False):
    '''Calculate gx due to many prisms using a prism table.'''
    return my_sum_prisms(x, y, z, prisms, density, function=my_prism_gx, progress=progress)


def my_prisms_gy(x, y, z, prisms, density=None, progress=False):
    '''Calculate gy due to many prisms using a prism table.'''
    return my_sum_prisms(x, y, z, prisms, density, function=my_prism_gy, progress=progress)


def my_prisms_potential(x, y, z, prisms, density=None, progress=False):
    '''Calculate gravitational potential due to many prisms using a prism table.'''
    return my_sum_prisms(x, y, z, prisms, density, function=my_potential, progress=progress)


# ============================================================
# PRISM GRAVITY-GRADIENT KERNELS
# ============================================================

def my_kernelxx(x, y, z, prism):
    '''Calculate the prism xx kernel.'''
    x, y, z = _prepare_xyz(x, y, z)
    xp, yp, zp = _prism_distances(x, y, z, prism)
    result = numpy.zeros_like(x, dtype=float)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = -_safe_atan(zp[k]*yp[j], xp[i]*r)
                result += ((-1.0)**(i + j + k))*kernel
    return result


def my_kernelyy(x, y, z, prism):
    '''Calculate the prism yy kernel.'''
    x, y, z = _prepare_xyz(x, y, z)
    xp, yp, zp = _prism_distances(x, y, z, prism)
    result = numpy.zeros_like(x, dtype=float)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = -_safe_atan(zp[k]*xp[i], yp[j]*r)
                result += ((-1.0)**(i + j + k))*kernel
    return result


def my_kernelzz(x, y, z, prism):
    '''Calculate the prism zz kernel.'''
    x, y, z = _prepare_xyz(x, y, z)
    xp, yp, zp = _prism_distances(x, y, z, prism)
    result = numpy.zeros_like(x, dtype=float)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = -_safe_atan(yp[j]*xp[i], zp[k]*r)
                result += ((-1.0)**(i + j + k))*kernel
    return result


def my_kernelxy(x, y, z, prism):
    '''Calculate the prism xy kernel.'''
    x, y, z = _prepare_xyz(x, y, z)
    xp, yp, zp = _prism_distances(x, y, z, prism)
    result = numpy.zeros_like(x, dtype=float)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = _safe_log(zp[k] + r)
                result += ((-1.0)**(i + j + k))*kernel
    return result


def my_kernelxz(x, y, z, prism):
    '''Calculate the prism xz kernel.'''
    x, y, z = _prepare_xyz(x, y, z)
    xp, yp, zp = _prism_distances(x, y, z, prism)
    result = numpy.zeros_like(x, dtype=float)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = _safe_log(yp[j] + r)
                result += ((-1.0)**(i + j + k))*kernel
    return result


def my_kernelyz(x, y, z, prism):
    '''Calculate the prism yz kernel.'''
    x, y, z = _prepare_xyz(x, y, z)
    xp, yp, zp = _prism_distances(x, y, z, prism)
    result = numpy.zeros_like(x, dtype=float)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = _safe_log(xp[i] + r)
                result += ((-1.0)**(i + j + k))*kernel
    return result


def my_prism_tensor(x, y, z, prism):
    '''
    Return the six independent prism tensor kernels: xx, xy, xz, yy, yz, zz.
    '''
    xx = my_kernelxx(x, y, z, prism)
    xy = my_kernelxy(x, y, z, prism)
    xz = my_kernelxz(x, y, z, prism)
    yy = my_kernelyy(x, y, z, prism)
    yz = my_kernelyz(x, y, z, prism)
    zz = my_kernelzz(x, y, z, prism)
    return xx, xy, xz, yy, yz, zz


def my_prism_tensor_physical(x, y, z, prism, rho=None, unit="Eotvos"):
    '''
    Return the physical gravity-gradient tensor of one prism.

    Inputs:
    rho - density in g/cm3
    unit - "SI" or "Eotvos"

    Output:
    gxx, gxy, gxz, gyy, gyz, gzz
    '''
    rho = _density_value(prism, rho)*1000.0
    xx, xy, xz, yy, yz, zz = my_prism_tensor(x, y, z, prism)
    factor = G*rho
    if str(unit).lower() in ["eotvos", "e", "eot"]:
        factor *= 1.0e9
    return factor*xx, factor*xy, factor*xz, factor*yy, factor*yz, factor*zz


# ============================================================
# MAGNETIC FIELD OF ONE PRISM
# ============================================================

def my_prism_tf(x, y, z, prism, mag=None, inc=0.0, dec=0.0,
                incs=None, decs=None, azim=0.0):
    '''
    Calculate the total-field magnetic anomaly produced by a rectangular prism.

    Inputs:
    x, y - arrays - observation coordinates
    z - scalar or array - observation level, positive downward
    prism - array/list - [x1, x2, y1, y2, z1, z2] or [x1, x2, y1, y2, z1, z2, mag]
    mag - float/None - magnetization intensity. If None, prism[6] is used.
    inc, dec - float - inducing field inclination and declination in degrees
    incs, decs - float/None - source magnetization inclination and declination in degrees
    azim - float - coordinate-system azimuth in degrees

    Output:
    tfa - numpy array - total-field anomaly in nT
    '''
    x, y, z = _prepare_xyz(x, y, z)
    mag = _mag_value(prism, mag)

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    Ma, Mb, Mc = _dircos(incs, decs, azim)
    Fa, Fb, Fc = _dircos(inc, dec, azim)

    MF = [Ma*Fb + Mb*Fa,
          Ma*Fc + Mc*Fa,
          Mb*Fc + Mc*Fb,
          Ma*Fa,
          Mb*Fb,
          Mc*Fc]

    xp, yp, zp = _prism_distances(x, y, z, prism)
    tfa = numpy.zeros_like(x, dtype=float)

    # The sign treatment below follows the original mbox-style implementation.
    local_mag = float(mag)
    for k in range(2):
        local_mag *= -1.0
        H2 = zp[k]**2
        for j in range(2):
            Y2 = yp[j]**2
            for i in range(2):
                X2 = xp[i]**2
                AxB = xp[i]*yp[j]
                R2 = X2 + Y2 + H2
                R = numpy.sqrt(R2)
                HxR = zp[k]*R

                tfa += ((-1.0)**(i + j))*local_mag*(
                    0.5*MF[2]*_safe_log((R - xp[i])/(R + xp[i]))
                    + 0.5*MF[1]*_safe_log((R - yp[j])/(R + yp[j]))
                    - MF[0]*_safe_log(R + zp[k])
                    - MF[3]*_safe_atan(AxB, X2 + HxR + H2)
                    - MF[4]*_safe_atan(AxB, R2 + HxR - X2)
                    + MF[5]*_safe_atan(AxB, HxR)
                )

    tfa *= T2NT*CM
    return tfa


def my_prism_bx(x, y, z, prism, mag=None, inc=0.0, dec=0.0,
                incs=None, decs=None, azim=0.0):
    '''Calculate the magnetic induction x component of a prism in nT.'''
    x, y, z = _prepare_xyz(x, y, z)
    mag = _mag_value(prism, mag)
    if incs is None:
        incs = inc
    if decs is None:
        decs = dec
    mx, my, mz = _dircos(incs, decs, azim)
    bx = (my_kernelxx(x, y, z, prism)*mx
          + my_kernelxy(x, y, z, prism)*my
          + my_kernelxz(x, y, z, prism)*mz)
    return bx*CM*T2NT*mag


def my_prism_by(x, y, z, prism, mag=None, inc=0.0, dec=0.0,
                incs=None, decs=None, azim=0.0):
    '''Calculate the magnetic induction y component of a prism in nT.'''
    x, y, z = _prepare_xyz(x, y, z)
    mag = _mag_value(prism, mag)
    if incs is None:
        incs = inc
    if decs is None:
        decs = dec
    mx, my, mz = _dircos(incs, decs, azim)
    by = (my_kernelxy(x, y, z, prism)*mx
          + my_kernelyy(x, y, z, prism)*my
          + my_kernelyz(x, y, z, prism)*mz)
    return by*CM*T2NT*mag


def my_prism_bz(x, y, z, prism, mag=None, inc=0.0, dec=0.0,
                incs=None, decs=None, azim=0.0):
    '''Calculate the magnetic induction z component of a prism in nT.'''
    x, y, z = _prepare_xyz(x, y, z)
    mag = _mag_value(prism, mag)
    if incs is None:
        incs = inc
    if decs is None:
        decs = dec
    mx, my, mz = _dircos(incs, decs, azim)
    bz = (my_kernelxz(x, y, z, prism)*mx
          + my_kernelyz(x, y, z, prism)*my
          + my_kernelzz(x, y, z, prism)*mz)
    return bz*CM*T2NT*mag


def my_prism_magnetic(x, y, z, prism, mag=None, inc=0.0, dec=0.0,
                      incs=None, decs=None, azim=0.0, component="tf"):
    '''
    General interface for magnetic prism modelling.

    component options: "tf", "bx", "by", "bz".
    '''
    component = component.lower()
    if component in ["tf", "tfa", "total", "totalfield"]:
        return my_prism_tf(x, y, z, prism, mag=mag, inc=inc, dec=dec,
                           incs=incs, decs=decs, azim=azim)
    if component == "bx":
        return my_prism_bx(x, y, z, prism, mag=mag, inc=inc, dec=dec,
                           incs=incs, decs=decs, azim=azim)
    if component == "by":
        return my_prism_by(x, y, z, prism, mag=mag, inc=inc, dec=dec,
                           incs=incs, decs=decs, azim=azim)
    if component == "bz":
        return my_prism_bz(x, y, z, prism, mag=mag, inc=inc, dec=dec,
                           incs=incs, decs=decs, azim=azim)
    raise ValueError("Invalid magnetic component: %s" % component)


# ============================================================
# MAGNETIC FIELD OF MANY PRISMS
# ============================================================

def my_prisms_tf(x, y, z, prisms, mag=None, inc=0.0, dec=0.0,
                 incs=None, decs=None, azim=0.0, progress=False):
    '''Calculate total-field anomaly due to many prisms.'''
    x, y, z = _prepare_xyz(x, y, z)
    prisms = numpy.asarray(prisms, dtype=float)
    if prisms.ndim == 1:
        prisms = prisms.reshape(1, prisms.size)
    n = prisms.shape[0]
    if mag is None:
        if prisms.shape[1] >= 7:
            props = prisms[:, 6]
        else:
            raise ValueError("mag must be provided if prisms do not contain a 7th column!")
    else:
        props = _as_array(mag, n, "mag")

    total = numpy.zeros_like(x, dtype=float)
    step = max(1, n//10)
    for i in range(n):
        total += my_prism_tf(x, y, z, prisms[i, :6], mag=props[i], inc=inc, dec=dec,
                             incs=incs, decs=decs, azim=azim)
        if progress and ((i + 1) % step == 0 or (i + 1) == n):
            print("Computed %d of %d prisms" % (i + 1, n))
    return total


def my_prisms_bx(x, y, z, prisms, mag=None, inc=0.0, dec=0.0,
                 incs=None, decs=None, azim=0.0, progress=False):
    '''Calculate Bx due to many prisms.'''
    return _sum_prism_magnetic_component(x, y, z, prisms, mag, inc, dec, incs, decs, azim,
                                         component="bx", progress=progress)


def my_prisms_by(x, y, z, prisms, mag=None, inc=0.0, dec=0.0,
                 incs=None, decs=None, azim=0.0, progress=False):
    '''Calculate By due to many prisms.'''
    return _sum_prism_magnetic_component(x, y, z, prisms, mag, inc, dec, incs, decs, azim,
                                         component="by", progress=progress)


def my_prisms_bz(x, y, z, prisms, mag=None, inc=0.0, dec=0.0,
                 incs=None, decs=None, azim=0.0, progress=False):
    '''Calculate Bz due to many prisms.'''
    return _sum_prism_magnetic_component(x, y, z, prisms, mag, inc, dec, incs, decs, azim,
                                         component="bz", progress=progress)


def _sum_prism_magnetic_component(x, y, z, prisms, mag, inc, dec, incs, decs, azim,
                                  component="tf", progress=False):
    '''Internal helper for summing magnetic components of many prisms.'''
    x, y, z = _prepare_xyz(x, y, z)
    prisms = numpy.asarray(prisms, dtype=float)
    if prisms.ndim == 1:
        prisms = prisms.reshape(1, prisms.size)
    n = prisms.shape[0]
    if mag is None:
        if prisms.shape[1] >= 7:
            props = prisms[:, 6]
        else:
            raise ValueError("mag must be provided if prisms do not contain a 7th column!")
    else:
        props = _as_array(mag, n, "mag")

    total = numpy.zeros_like(x, dtype=float)
    step = max(1, n//10)
    for i in range(n):
        total += my_prism_magnetic(x, y, z, prisms[i, :6], mag=props[i], inc=inc, dec=dec,
                                   incs=incs, decs=decs, azim=azim, component=component)
        if progress and ((i + 1) % step == 0 or (i + 1) == n):
            print("Computed %d of %d prisms" % (i + 1, n))
    return total


# ============================================================
# SYNTHETIC MODEL HELPERS
# ============================================================

def my_rectangular_prism(x1, x2, y1, y2, z1, z2, physical_property=None):
    '''
    Create a prism vector.
    '''
    if physical_property is None:
        return numpy.array([x1, x2, y1, y2, z1, z2], dtype=float)
    return numpy.array([x1, x2, y1, y2, z1, z2, physical_property], dtype=float)


def my_prism_grid(area, shape, top, bottom, density=None):
    '''
    Create a regular grid of prisms covering an area.

    Inputs:
    area - list - [x1, x2, y1, y2]
    shape - tuple - (nx, ny)
    top, bottom - scalar or array - prism top and bottom
    density - None/scalar/array - optional property column

    Output:
    prisms - numpy array - prism table
    '''
    x1, x2, y1, y2 = area
    nx, ny = shape
    x_edges = numpy.linspace(x1, x2, nx + 1)
    y_edges = numpy.linspace(y1, y2, ny + 1)
    xc = 0.5*(x_edges[:-1] + x_edges[1:])
    yc = 0.5*(y_edges[:-1] + y_edges[1:])
    yy, xx = numpy.meshgrid(yc, xc)
    dx = (x2 - x1)/float(nx)
    dy = (y2 - y1)/float(ny)
    return my_build_prisms_from_centers(xx.ravel(), yy.ravel(), top, bottom, dx, dy, density=density)


def my_depth_varying_prisms(xcenter, ycenter, top, bottom, dx, dy, density=None):
    '''
    Build prisms from arbitrary center coordinates and variable top/bottom surfaces.
    '''
    return my_build_prisms_from_centers(xcenter, ycenter, top, bottom, dx, dy, density=density)


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

# Gravity aliases
potential = my_potential
prism_potential = my_potential
prism_gx = my_prism_gx
prism_gy = my_prism_gy
prism_gz = my_prism_gz
my_gx = my_prism_gx
my_gy = my_prism_gy
my_gz = my_prism_gz
my_tf = my_prism_tf

# Magnetic aliases
prism_tf = my_prism_tf
prism_bx = my_prism_bx
prism_by = my_prism_by
prism_bz = my_prism_bz
my_prism_totalfield = my_prism_tf
my_totalfield = my_prism_tf
my_bx = my_prism_bx
my_by = my_prism_by
my_bz = my_prism_bz

# Kernel aliases used by old functions
kernelxx = my_kernelxx
kernelxy = my_kernelxy
kernelxz = my_kernelxz
kernelyy = my_kernelyy
kernelyz = my_kernelyz
kernelzz = my_kernelzz
my_kernel_xx = my_kernelxx
my_kernel_xy = my_kernelxy
my_kernel_xz = my_kernelxz
my_kernel_yy = my_kernelyy
my_kernel_yz = my_kernelyz
my_kernel_zz = my_kernelzz

# Many-prism aliases
my_3Dtf = my_prisms_tf
my_3Dtfa = my_prisms_tf
my_prisms_totalfield = my_prisms_tf
my_forward_gz_prisms = my_prisms_gz
my_forward_tf_prisms = my_prisms_tf
