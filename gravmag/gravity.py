# -----------------------------------------------------------------------------------
# Title: Gravity utilities
# Description: Normal gravity, gravity reductions and simple isostatic calculations.
# Author: Nelson Ribeiro Filho
# -----------------------------------------------------------------------------------

from __future__ import division
import numpy

try:
    from . import constants
except Exception:  # pragma: no cover - allows local use without package structure
    try:
        import constants
    except Exception:
        constants = None


# ============================================================
# INTERNAL CONSTANTS AND CHECKING UTILITIES
# ============================================================

_G = 6.67408e-11
_SI2MGAL = 1.0e5

if constants is not None:
    _G = getattr(constants, "G", _G)
    _SI2MGAL = getattr(constants, "si2mGal", _SI2MGAL)


def _asarray_float(value):
    '''
    Convert an input value to a numpy array with float dtype.
    Scalars, lists and numpy arrays are accepted.
    '''

    return numpy.asarray(value, dtype=float)


def _return_like_input(result, original):
    '''
    Return a scalar when the original input was scalar; otherwise return an array.
    '''

    if numpy.isscalar(original):
        return float(numpy.asarray(result))
    return result


def _check_same_shape(*arrays):
    '''
    Check if all numpy-compatible inputs have the same shape.
    '''

    shapes = [numpy.asarray(array).shape for array in arrays]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError("All inputs must have the same shape!")


# ============================================================
# REFERENCE ELLIPSOIDS
# ============================================================


def my_WGS84():
    '''
    Return the WGS84 reference ellipsoid parameters.

    Outputs:
    a - float - semimajor axis [m]
    f - float - flattening
    GM - float - geocentric gravitational constant [m^3/s^2]
    omega - float - angular velocity [rad/s]
    '''

    a = 6378137.0
    f = 1.0/298.257223563
    GM = 3986004.418e8
    omega = 7292115e-11

    return a, f, GM, omega


def my_GRS80():
    '''
    Return the GRS80 reference ellipsoid parameters.

    Outputs:
    a - float - semimajor axis [m]
    f - float - flattening
    GM - float - geocentric gravitational constant [m^3/s^2]
    omega - float - angular velocity [rad/s]
    '''

    a = 6378137.0
    f = 1.0/298.257222101
    GM = 3986005.0e8
    omega = 7292115e-11

    return a, f, GM, omega


def my_get_ellipsoid(name="WGS84"):
    '''
    Return reference ellipsoid parameters by name.

    Inputs:
    name - string - "WGS84" or "GRS80"

    Outputs:
    a, f, GM, omega - floats - ellipsoid parameters
    '''

    name = str(name).upper()

    if name == "WGS84":
        return my_WGS84()
    if name == "GRS80":
        return my_GRS80()

    raise ValueError("Unknown ellipsoid. Use 'WGS84' or 'GRS80'.")


# ============================================================
# NORMAL GRAVITY
# ============================================================


def my_somigliana(phi, a=None, f=None, GM=None, omega=None):
    '''
    Calculate normal gravity using Somigliana's formula.

    Inputs:
    phi - float or numpy array - geodetic latitude [degree]
    a - float/None - semimajor axis [m]
    f - float/None - flattening
    GM - float/None - geocentric gravitational constant [m^3/s^2]
    omega - float/None - angular velocity [rad/s]

    Output:
    gamma - float or numpy array - normal gravity on the ellipsoid [mGal]
    '''

    original = phi
    phi = _asarray_float(phi)

    if a is None or f is None or GM is None or omega is None:
        a0, f0, GM0, omega0 = my_WGS84()
        a = a0 if a is None else a
        f = f0 if f is None else f
        GM = GM0 if GM is None else GM
        omega = omega0 if omega is None else omega

    b = a*(1.0 - f)
    a2 = a**2
    b2 = b**2
    E = numpy.sqrt(a2 - b2)

    e_prime = E/b
    bE = b/E
    Eb = E/b
    atan_Eb = numpy.arctan(Eb)

    q0 = 0.5*((1.0 + 3.0*(bE**2))*atan_Eb - 3.0*bE)
    q0_prime = 3.0*(1.0 + bE**2)*(1.0 - bE*atan_Eb) - 1.0
    m = (omega**2)*(a2)*b/GM

    aux = e_prime*q0_prime/q0
    gamma_a = (GM/(a*b))*(1.0 - m - (m/6.0)*aux)
    gamma_b = (GM/a2)*(1.0 + (m/3.0)*aux)

    phirad = numpy.deg2rad(phi)
    sin2 = numpy.sin(phirad)**2
    cos2 = numpy.cos(phirad)**2

    gamma = _SI2MGAL*((a*gamma_a*cos2) + (b*gamma_b*sin2))/numpy.sqrt((a2*cos2) + (b2*sin2))

    return _return_like_input(gamma, original)


def my_closedform(phi, h, a=None, f=None, GM=None, omega=None):
    '''
    Calculate normal gravity using the closed-form formula of Li and Goetze (2001).

    Inputs:
    phi - float or numpy array - geodetic latitude [degree]
    h - float or numpy array - height [m]
    a - float/None - semimajor axis [m]
    f - float/None - flattening
    GM - float/None - geocentric gravitational constant [m^3/s^2]
    omega - float/None - angular velocity [rad/s]

    Output:
    gamma - float or numpy array - normal gravity [mGal]
    '''

    original = phi
    phi = _asarray_float(phi)
    h = _asarray_float(h)

    _check_same_shape(phi, h) if (phi.shape != () and h.shape != ()) else None

    if a is None or f is None or GM is None or omega is None:
        a0, f0, GM0, omega0 = my_WGS84()
        a = a0 if a is None else a
        f = f0 if f is None else f
        GM = GM0 if GM is None else GM
        omega = omega0 if omega is None else omega

    b = a*(1.0 - f)
    a2 = a**2
    b2 = b**2
    E = numpy.sqrt(a2 - b2)
    E2 = E**2

    bE = b/E
    Eb = E/b
    atan_Eb = numpy.arctan(Eb)

    phirad = numpy.deg2rad(phi)
    tanphi = numpy.tan(phirad)
    cosphi = numpy.cos(phirad)
    sinphi = numpy.sin(phirad)

    beta = numpy.arctan(b*tanphi/a)
    sinbeta = numpy.sin(beta)
    cosbeta = numpy.cos(beta)

    z_l = b*sinbeta + h*sinphi
    r_l = a*cosbeta + h*cosphi

    z_l2 = z_l**2
    r_l2 = r_l**2

    D = (r_l2 - z_l2)/E2
    R = (r_l2 + z_l2)/E2

    cos_beta_l = numpy.sqrt(0.5*(1.0 + R) - numpy.sqrt(0.25*(1.0 + R)**2 - 0.5*D))
    cos_beta_l2 = cos_beta_l**2
    sin_beta_l2 = 1.0 - cos_beta_l2

    b_l = numpy.sqrt(r_l2 + z_l2 - E2*cos_beta_l2)
    b_l2 = b_l**2

    b_lE = b_l/E
    E_b_l = E/b_l
    atan_E_b_l = numpy.arctan(E_b_l)

    q0 = 0.5*((1.0 + 3.0*(bE**2))*atan_Eb - 3.0*bE)
    q0_l = 3.0*(1.0 + b_lE**2)*(1.0 - b_lE*atan_E_b_l) - 1.0

    W = numpy.sqrt((b_l2 + E2*sin_beta_l2)/(b_l2 + E2))

    gamma = GM/(b_l2 + E2) - cos_beta_l2*b_l*(omega**2)
    gamma += (((omega**2)*(a2)*E*q0_l)/((b_l2 + E2)*q0))*(0.5*sin_beta_l2 - 1.0/6.0)
    gamma = _SI2MGAL*gamma/W

    return _return_like_input(gamma, original)


def my_normal_gravity(phi, h=0.0, method="somigliana", ellipsoid="WGS84"):
    '''
    Calculate normal gravity using a selected method.

    Inputs:
    phi - float or numpy array - geodetic latitude [degree]
    h - float or numpy array - height [m]
    method - string - "somigliana" or "closedform"
    ellipsoid - string - "WGS84" or "GRS80"

    Output:
    gamma - float or numpy array - normal gravity [mGal]
    '''

    a, f, GM, omega = my_get_ellipsoid(ellipsoid)
    method = str(method).lower()

    if method == "somigliana":
        if numpy.any(numpy.asarray(h) != 0.0):
            return my_closedform(phi, h, a=a, f=f, GM=GM, omega=omega)
        return my_somigliana(phi, a=a, f=f, GM=GM, omega=omega)

    if method in ("closed", "closedform", "li", "li_goetze"):
        return my_closedform(phi, h, a=a, f=f, GM=GM, omega=omega)

    raise ValueError("Invalid method. Use 'somigliana' or 'closedform'.")


# ============================================================
# GRAVITY CORRECTIONS AND ANOMALIES
# ============================================================


def my_freeair_correction(orthometric):
    '''
    Calculate the free-air correction using the original sign convention.

    Input:
    orthometric - float or numpy array - orthometric height [m]

    Output:
    fac - float or numpy array - free-air correction [mGal]

    Note:
    This function preserves the original project convention: fac = -0.3086*h.
    '''

    original = orthometric
    orthometric = _asarray_float(orthometric)
    fac = -0.3086*orthometric

    return _return_like_input(fac, original)


def my_freeair_anomaly(gobs, gamma, orthometric):
    '''
    Calculate the free-air anomaly using the correction defined in my_freeair_correction.

    Inputs:
    gobs - float or numpy array - observed gravity [mGal]
    gamma - float or numpy array - normal gravity [mGal]
    orthometric - float or numpy array - orthometric height [m]

    Output:
    faa - float or numpy array - free-air anomaly [mGal]
    '''

    gobs = _asarray_float(gobs)
    gamma = _asarray_float(gamma)
    orthometric = _asarray_float(orthometric)

    faa = gobs - gamma - my_freeair_correction(orthometric)

    return faa


def my_bouguer_correction(topography, rho_crust=2673.0, rho_oceanic=2950.0, rho_water=1040.0):
    '''
    Calculate the Bouguer slab correction for continental and oceanic areas.

    Inputs:
    topography - float or numpy array - topography [m]
    rho_crust - float - continental crust density [kg/m^3]
    rho_oceanic - float - oceanic crust density [kg/m^3]
    rho_water - float - water density [kg/m^3]

    Output:
    bgc - float or numpy array - Bouguer correction [mGal]
    '''

    original = topography
    topography = _asarray_float(topography)

    bgc = numpy.zeros_like(topography, dtype=float)

    continent = topography >= 0.0
    ocean = topography < 0.0

    bgc[continent] = 2.0*numpy.pi*_G*_SI2MGAL*rho_crust*topography[continent]
    bgc[ocean] = 2.0*numpy.pi*_G*_SI2MGAL*(rho_oceanic - rho_water)*topography[ocean]

    return _return_like_input(bgc, original)


def my_bouguer_anomaly(gobs, gamma, orthometric, topography=None,
                       rho_crust=2673.0, rho_oceanic=2950.0, rho_water=1040.0):
    '''
    Calculate a simple Bouguer anomaly.

    Inputs:
    gobs - float or numpy array - observed gravity [mGal]
    gamma - float or numpy array - normal gravity [mGal]
    orthometric - float or numpy array - orthometric height [m]
    topography - float or numpy array/None - topography [m]. If None, uses orthometric.
    rho_crust - float - continental crust density [kg/m^3]
    rho_oceanic - float - oceanic crust density [kg/m^3]
    rho_water - float - water density [kg/m^3]

    Output:
    bouguer - float or numpy array - Bouguer anomaly [mGal]
    '''

    if topography is None:
        topography = orthometric

    faa = my_freeair_anomaly(gobs, gamma, orthometric)
    bc = my_bouguer_correction(topography, rho_crust, rho_oceanic, rho_water)

    return faa - bc


# ============================================================
# AIRY ISOSTASY AND ISOSTATIC CORRECTION
# ============================================================


def my_Airy(topography, rho_mantle=3270.0, rho_crust=2673.0,
            rho_oceanic=2950.0, rho_water=1040.0):
    '''
    Calculate the isostatic root/anti-root thickness based on Airy's hypothesis.

    Inputs:
    topography - float or numpy array - topography [m]
    rho_mantle - float - mantle density [kg/m^3]
    rho_crust - float - continental crust density [kg/m^3]
    rho_oceanic - float - oceanic crust density [kg/m^3]
    rho_water - float - water density [kg/m^3]

    Output:
    root - float or numpy array - Airy root/anti-root thickness [m]
    '''

    original = topography
    topography = _asarray_float(topography)

    root = numpy.zeros_like(topography, dtype=float)

    continent = topography >= 0.0
    ocean = topography < 0.0

    root[continent] = rho_crust*topography[continent]/(rho_mantle - rho_crust)
    root[ocean] = (rho_oceanic - rho_water)*topography[ocean]/(rho_mantle - rho_oceanic)

    return _return_like_input(root, original)


def my_moho_airy(topography, reference_moho=30000.0, rho_mantle=3270.0,
                 rho_crust=2673.0, rho_oceanic=2950.0, rho_water=1040.0):
    '''
    Estimate Moho depth from Airy's isostatic root and a reference Moho depth.

    Inputs:
    topography - float or numpy array - topography [m]
    reference_moho - float - reference Moho depth [m]
    rho_mantle, rho_crust, rho_oceanic, rho_water - floats - densities [kg/m^3]

    Output:
    moho - float or numpy array - estimated Moho depth [m]
    '''

    return reference_moho + my_Airy(topography, rho_mantle, rho_crust, rho_oceanic, rho_water)


def my_isostatic_correction(topography, rho_crust=2673.0,
                            rho_oceanic=2950.0, rho_water=1040.0):
    '''
    Calculate a simple isostatic correction using the same slab approximation
    preserved from the original code structure.

    Inputs:
    topography - float or numpy array - topography [m]
    rho_crust - float - continental crust density [kg/m^3]
    rho_oceanic - float - oceanic crust density [kg/m^3]
    rho_water - float - water density [kg/m^3]

    Output:
    isostatic - float or numpy array - isostatic correction [mGal]
    '''

    return my_bouguer_correction(topography, rho_crust, rho_oceanic, rho_water)


def my_isostatic_anomaly(bouguer_anomaly, isostatic_correction):
    '''
    Calculate a simple isostatic anomaly.

    Inputs:
    bouguer_anomaly - float or numpy array - Bouguer anomaly [mGal]
    isostatic_correction - float or numpy array - isostatic correction [mGal]

    Output:
    anomaly - float or numpy array - isostatic anomaly [mGal]
    '''

    return _asarray_float(bouguer_anomaly) + _asarray_float(isostatic_correction)


# ============================================================
# USEFUL GEODETIC APPROXIMATIONS
# ============================================================


def my_gravity_disturbance(gobs, gamma):
    '''
    Calculate the gravity disturbance.

    Inputs:
    gobs - float or numpy array - observed gravity [mGal]
    gamma - float or numpy array - normal gravity [mGal]

    Output:
    disturbance - float or numpy array - gravity disturbance [mGal]
    '''

    return _asarray_float(gobs) - _asarray_float(gamma)


def my_eotvos_correction(velocity_east, latitude, velocity_north=0.0):
    '''
    Approximate Eotvos correction for moving gravity measurements.

    Inputs:
    velocity_east - float or numpy array - eastward velocity [m/s]
    latitude - float or numpy array - latitude [degree]
    velocity_north - float or numpy array - northward velocity [m/s]

    Output:
    correction - float or numpy array - Eotvos correction [mGal]
    '''

    omega = 7292115e-11
    radius = 6378137.0

    ve = _asarray_float(velocity_east)
    vn = _asarray_float(velocity_north)
    lat = numpy.deg2rad(_asarray_float(latitude))

    correction_si = 2.0*omega*ve*numpy.cos(lat) + (ve**2 + vn**2)/radius
    correction_mgal = correction_si*_SI2MGAL

    return correction_mgal


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

my_wgs84 = my_WGS84
my_grs80 = my_GRS80
my_airycroot = my_Airy
my_airy = my_Airy
my_airymoho = my_moho_airy
my_moho_air = my_moho_airy
my_freeair = my_freeair_correction
my_bouguer = my_bouguer_correction
my_disturbance = my_gravity_disturbance
