"""
constants.py
============

Physical, geophysical and unit-conversion constants used by the gravity,
magnetic, filtering and inversion routines.

The module keeps the original lowercase variable names for compatibility with
older scripts, but also provides clearer uppercase aliases for new code.

Important convention
--------------------
Distances are assumed to be in meters unless explicitly stated otherwise.
Densities used by the prism routines are usually provided in g/cm^3 and
converted internally to kg/m^3 when necessary.
"""

from __future__ import annotations

import numpy as numpy


# =============================================================================
# FUNDAMENTAL MATHEMATICAL CONSTANTS
# =============================================================================

PI = numpy.pi
TWO_PI = 2.0 * numpy.pi
DEG2RAD = numpy.pi / 180.0
RAD2DEG = 180.0 / numpy.pi

# Backward-compatible aliases
pi = PI
deg2rad = DEG2RAD
rad2deg = RAD2DEG


# =============================================================================
# GRAVITY CONSTANTS
# =============================================================================

# Universal gravitational constant in SI units: m^3 kg^-1 s^-2.
# The original code used 6.673e-11; it is preserved to avoid changing results.
G = 6.673e-11
GRAVITATIONAL_CONSTANT = G

# Standard SI gravity acceleration to mGal conversion.
# 1 m/s^2 = 1e5 mGal.
SI2MGAL = 1.0e5
MGAL2SI = 1.0e-5

# Backward-compatible aliases used in older files.
si2mGal = SI2MGAL
mGal2si = MGAL2SI


# =============================================================================
# MAGNETIC CONSTANTS
# =============================================================================

# Magnetic constant used in potential-field routines.
# Equivalent to mu0/(4*pi) in SI.
CM = 1.0e-7
cm = CM

# Tesla and nanoTesla conversions.
T2NT = 1.0e9
NT2T = 1.0e-9

# Backward-compatible aliases used in older files.
T2nT = T2NT
nT2T = NT2T
t2nT = T2NT
nt2T = NT2T

# Magnetic permeability of vacuum in T m A^-1.
MU0 = 4.0e-7 * numpy.pi
mu0 = MU0


# =============================================================================
# EARTH AND GEODETIC CONSTANTS
# =============================================================================

# WGS84 ellipsoid parameters.
WGS84_A = 6378137.0                  # Semi-major axis [m]
WGS84_F = 1.0 / 298.257223563        # Flattening
WGS84_GM = 3986004.418e8             # Geocentric gravitational constant [m^3/s^2]
WGS84_OMEGA = 7292115.0e-11          # Angular velocity [rad/s]

# GRS80 ellipsoid parameters.
GRS80_A = 6378137.0
GRS80_F = 1.0 / 298.257222101
GRS80_GM = 3986005.0e8
GRS80_OMEGA = 7292115.0e-11

# Earth parameters.
EARTH_RADIUS = WGS84_A               # Mean/reference radius used in legacy code [m]
EARTH_MASS = 5.972e24                # Earth mass [kg]
EARTH_VOLUME_KM3 = 1.08321e12        # Mean Earth volume [km^3]
EARTH_ESCAPE_VELOCITY = 11186.0      # Escape velocity near Earth surface [m/s]

# Backward-compatible aliases.
earth_radius = EARTH_RADIUS
earth_mass = EARTH_MASS
volume = EARTH_VOLUME_KM3
vescape = EARTH_ESCAPE_VELOCITY


# =============================================================================
# DENSITY CONSTANTS
# =============================================================================

# Densities in g/cm^3.
RHO_MEAN_EARTH = 5.514
RHO_CONTINENTAL_CRUST = 2.673
RHO_OCEANIC_CRUST = 2.92
RHO_WATER = 1.03
RHO_MANTLE = 3.35

# Useful conversion.
GCM3_TO_KGM3 = 1000.0
KGM3_TO_GCM3 = 0.001

# Backward-compatible aliases.
rhomean = RHO_MEAN_EARTH
rho_cc = RHO_CONTINENTAL_CRUST
rho_oc = RHO_OCEANIC_CRUST
rho_w = RHO_WATER
rho_m = RHO_MANTLE
gcm3_to_kgm3 = GCM3_TO_KGM3
kgm3_to_gcm3 = KGM3_TO_GCM3


# =============================================================================
# LENGTH, AREA AND VOLUME CONVERSIONS
# =============================================================================

# Length conversions.
M_TO_KM = 1.0e-3
KM_TO_M = 1.0e3
M_TO_CM = 1.0e2
CM_TO_M = 1.0e-2
M_TO_MM = 1.0e3
MM_TO_M = 1.0e-3

# Backward-compatible aliases.
# NOTE: The original file had m2km=1000 and km2m=0.001, which inverted the labels.
# Here they are corrected according to their names.
m2km = M_TO_KM
km2m = KM_TO_M
m2cm = M_TO_CM
cm2m = CM_TO_M
m2mm = M_TO_MM
mm2m = MM_TO_M

# Area conversions.
M2_TO_KM2 = 1.0e-6
KM2_TO_M2 = 1.0e6

m2_to_km2 = M2_TO_KM2
km2_to_m2 = KM2_TO_M2

# Volume conversions.
M3_TO_L = 1000.0
L_TO_M3 = 0.001
DM3_TO_L = 1.0
L_TO_DM3 = 1.0

# Backward-compatible aliases.
# NOTE: The original labels m3_L and L_m3 were inverted. They are corrected here.
m3_L = M3_TO_L
L_m3 = L_TO_M3
dm3_L = DM3_TO_L
L_dm3 = L_TO_DM3


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

LIGHT_SPEED = 299792458.0                 # m/s
PLANCK_CONSTANT = 6.62607015e-34          # J s
ELEMENTARY_CHARGE = 1.602176634e-19       # C
EPSILON0 = 8.8541878128e-12               # C^2/(N m^2)

# Backward-compatible aliases.
c = LIGHT_SPEED
h = PLANCK_CONSTANT
ec = ELEMENTARY_CHARGE
epsilon0 = EPSILON0


# =============================================================================
# SMALL NUMERICAL VALUES
# =============================================================================

EPS = 1.0e-12
TINY = 1.0e-30


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def my_show_constants():
    """
    Print the main constants used by the package.

    Output
    ------
    None
    """

    print("Gravity constants")
    print(f"  G        = {G:.12e} m^3 kg^-1 s^-2")
    print(f"  SI2MGAL  = {SI2MGAL:.6e}")
    print()
    print("Magnetic constants")
    print(f"  CM       = {CM:.12e}")
    print(f"  T2NT     = {T2NT:.6e}")
    print(f"  MU0      = {MU0:.12e} T m A^-1")
    print()
    print("Density constants [g/cm^3]")
    print(f"  rho_cc   = {rho_cc}")
    print(f"  rho_oc   = {rho_oc}")
    print(f"  rho_w    = {rho_w}")
    print(f"  rho_m    = {rho_m}")
    print()
    print("Length conversions")
    print(f"  m2km     = {m2km}")
    print(f"  km2m     = {km2m}")


def my_density_gcm3_to_kgm3(rho):
    """
    Convert density from g/cm^3 to kg/m^3.

    Input
    -----
    rho : float or numpy array
        Density in g/cm^3.

    Output
    ------
    converted : float or numpy array
        Density in kg/m^3.
    """

    return numpy.asarray(rho) * GCM3_TO_KGM3


def my_density_kgm3_to_gcm3(rho):
    """
    Convert density from kg/m^3 to g/cm^3.

    Input
    -----
    rho : float or numpy array
        Density in kg/m^3.

    Output
    ------
    converted : float or numpy array
        Density in g/cm^3.
    """

    return numpy.asarray(rho) * KGM3_TO_GCM3


def my_m_to_km(distance):
    """
    Convert distance from meters to kilometers.
    """

    return numpy.asarray(distance) * M_TO_KM


def my_km_to_m(distance):
    """
    Convert distance from kilometers to meters.
    """

    return numpy.asarray(distance) * KM_TO_M


def my_si_to_mgal(value):
    """
    Convert gravity acceleration from SI units to mGal.
    """

    return numpy.asarray(value) * SI2MGAL


def my_mgal_to_si(value):
    """
    Convert gravity acceleration from mGal to SI units.
    """

    return numpy.asarray(value) * MGAL2SI


def my_tesla_to_nt(value):
    """
    Convert magnetic field from Tesla to nanoTesla.
    """

    return numpy.asarray(value) * T2NT


def my_nt_to_tesla(value):
    """
    Convert magnetic field from nanoTesla to Tesla.
    """

    return numpy.asarray(value) * NT2T


__all__ = [
    "numpy",
    "PI", "TWO_PI", "DEG2RAD", "RAD2DEG",
    "G", "GRAVITATIONAL_CONSTANT", "SI2MGAL", "MGAL2SI",
    "CM", "T2NT", "NT2T", "MU0",
    "WGS84_A", "WGS84_F", "WGS84_GM", "WGS84_OMEGA",
    "GRS80_A", "GRS80_F", "GRS80_GM", "GRS80_OMEGA",
    "EARTH_RADIUS", "EARTH_MASS", "EARTH_VOLUME_KM3", "EARTH_ESCAPE_VELOCITY",
    "RHO_MEAN_EARTH", "RHO_CONTINENTAL_CRUST", "RHO_OCEANIC_CRUST", "RHO_WATER", "RHO_MANTLE",
    "GCM3_TO_KGM3", "KGM3_TO_GCM3",
    "M_TO_KM", "KM_TO_M", "M_TO_CM", "CM_TO_M", "M_TO_MM", "MM_TO_M",
    "M2_TO_KM2", "KM2_TO_M2", "M3_TO_L", "L_TO_M3", "DM3_TO_L", "L_TO_DM3",
    "LIGHT_SPEED", "PLANCK_CONSTANT", "ELEMENTARY_CHARGE", "EPSILON0", "EPS", "TINY",
    "T2nT", "nT2T", "t2nT", "nt2T", "cm", "si2mGal", "mGal2si",
    "earth_radius", "earth_mass", "rhomean", "rho_cc", "rho_oc", "rho_w", "rho_m",
    "volume", "vescape", "m2km", "km2m", "m3_L", "L_m3", "dm3_L", "L_dm3",
    "c", "h", "ec", "epsilon0", "mu0",
    "my_show_constants", "my_density_gcm3_to_kgm3", "my_density_kgm3_to_gcm3",
    "my_m_to_km", "my_km_to_m", "my_si_to_mgal", "my_mgal_to_si",
    "my_tesla_to_nt", "my_nt_to_tesla",
]
