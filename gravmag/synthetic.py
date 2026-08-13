# -*- coding: utf-8 -*-
"""
synthetic.py
============

Synthetic model builders and test utilities for gravimetric and magnetic modeling.

Author: Nelson Ribeiro Filho

Purpose
-------
This module creates practical synthetic models for geophysical tests using:

- rectangular prisms
- spheres
- cylinders
- hexagonal prisms
- polygonal prisms
- combinations of sources
- synthetic topography
- synthetic density and magnetization models
- noise addition
- basic comparison between true and predicted data

Design
------
The functions follow the same general style used in the Grav-Mag code package:

- functions start with "my_"
- z is positive downward
- coordinates are given in meters, unless explicitly stated
- density is usually in g/cm^3
- magnetization is usually in A/m
- gravity output from modeling modules is expected in mGal
- magnetic output from modeling modules is expected in nT

Important
---------
This module does not replace the physical modeling modules. It only organizes
synthetic examples and calls functions from:

    prism.py
    sphere.py
    cylinder.py
    hexagonalprism.py
    polygonalprism.py
    grids.py
    statistical.py

The imports are flexible and work both inside a package and as local files.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union, Dict, List

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]
AreaLike = Sequence[float]


# ============================================================
# FLEXIBLE IMPORTS
# ============================================================

try:
    from . import grids
except Exception:
    try:
        import grids
    except Exception:
        grids = None

try:
    from . import prism
except Exception:
    try:
        import prism
    except Exception:
        prism = None

try:
    from . import sphere
except Exception:
    try:
        import sphere
    except Exception:
        sphere = None

try:
    from . import cylinder
except Exception:
    try:
        import cylinder
    except Exception:
        cylinder = None

try:
    from . import hexagonalprism
except Exception:
    try:
        import hexagonalprism
    except Exception:
        hexagonalprism = None

try:
    from . import polygonalprism
except Exception:
    try:
        import polygonalprism
    except Exception:
        polygonalprism = None

try:
    from . import statistical
except Exception:
    try:
        import statistical
    except Exception:
        statistical = None


# ============================================================
# BASIC UTILITIES
# ============================================================

def _require_module(module, name: str):
    """
    Raise an informative error if a required module is unavailable.
    """
    if module is None:
        raise ImportError(
            f"The module '{name}' is required for this function, "
            f"but it could not be imported."
        )


def _as_array(data):
    """
    Convert input to numpy array.
    """
    return np.asarray(data)


def _check_same_shape(*arrays):
    """
    Check if all arrays have the same shape.
    """
    arrs = tuple(np.asarray(a) for a in arrays)
    shape0 = arrs[0].shape

    for arr in arrs[1:]:
        if arr.shape != shape0:
            raise ValueError("All input arrays must have the same shape!")

    return arrs


def my_make_observation_grid(
    area: AreaLike,
    shape: Tuple[int, int],
    level: float = 0.0,
    flatten: bool = False,
):
    """
    Create a regular observation grid.

    Inputs
    ------
    area : list
        [x_min, x_max, y_min, y_max].
    shape : tuple
        (nx, ny).
    level : float
        Observation level. z is positive downward.
    flatten : bool
        If True, returns 1D arrays. If False, returns 2D arrays.

    Outputs
    -------
    x, y, z : numpy arrays
        Observation coordinates.
    """
    x_min, x_max, y_min, y_max = area
    nx, ny = shape

    x_vec = np.linspace(x_min, x_max, nx)
    y_vec = np.linspace(y_min, y_max, ny)

    x, y = np.meshgrid(x_vec, y_vec, indexing="xy")
    z = level*np.ones_like(x)

    if flatten is True:
        return x.ravel(), y.ravel(), z.ravel()

    return x, y, z


def my_model_area_from_grid(x, y):
    """
    Return area = [x_min, x_max, y_min, y_max] from grid or scattered coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    return [
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
    ]


# ============================================================
# SYNTHETIC TOPOGRAPHY AND SURFACES
# ============================================================

def my_gaussian_surface(
    x,
    y,
    amplitude: float = 1000.0,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    angle: float = 0.0,
):
    """
    Create an elliptical Gaussian surface.

    Inputs
    ------
    x, y : arrays
        Coordinate arrays.
    amplitude : float
        Gaussian amplitude.
    x0, y0 : float or None
        Center. If None, the grid center is used.
    sigma_x, sigma_y : float or None
        Standard deviations in x and y.
    angle : float
        Rotation angle in degrees.

    Output
    ------
    surface : numpy array
        Gaussian surface.
    """
    x, y = _check_same_shape(x, y)

    if x0 is None:
        x0 = 0.5*(np.nanmin(x) + np.nanmax(x))
    if y0 is None:
        y0 = 0.5*(np.nanmin(y) + np.nanmax(y))

    if sigma_x is None:
        sigma_x = 0.15*(np.nanmax(x) - np.nanmin(x))
    if sigma_y is None:
        sigma_y = 0.15*(np.nanmax(y) - np.nanmin(y))

    theta = np.deg2rad(angle)

    xr = (x - x0)*np.cos(theta) + (y - y0)*np.sin(theta)
    yr = -(x - x0)*np.sin(theta) + (y - y0)*np.cos(theta)

    surface = amplitude*np.exp(-0.5*((xr/sigma_x)**2 + (yr/sigma_y)**2))

    return surface


def my_synthetic_topography(
    x,
    y,
    base: float = 0.0,
    mountain_amp: float = 900.0,
    valley_amp: float = -350.0,
    hill_amp: float = 250.0,
):
    """
    Create a synthetic topography with a mountain, a valley and secondary hills.

    Inputs
    ------
    x, y : arrays
        Coordinate grids.
    base : float
        Base elevation.
    mountain_amp : float
        Main mountain amplitude.
    valley_amp : float
        Valley amplitude. Use negative value.
    hill_amp : float
        Secondary hill amplitude.

    Output
    ------
    topo : numpy array
        Synthetic topography.
    """
    x, y = _check_same_shape(x, y)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    topo = base*np.ones_like(x, dtype=float)

    topo += my_gaussian_surface(
        x, y,
        amplitude=mountain_amp,
        x0=xmin + 0.30*(xmax - xmin),
        y0=ymin + 0.60*(ymax - ymin),
        sigma_x=0.12*(xmax - xmin),
        sigma_y=0.18*(ymax - ymin),
        angle=25.0,
    )

    topo += my_gaussian_surface(
        x, y,
        amplitude=valley_amp,
        x0=xmin + 0.72*(xmax - xmin),
        y0=ymin + 0.38*(ymax - ymin),
        sigma_x=0.18*(xmax - xmin),
        sigma_y=0.12*(ymax - ymin),
        angle=-20.0,
    )

    topo += my_gaussian_surface(
        x, y,
        amplitude=hill_amp,
        x0=xmin + 0.55*(xmax - xmin),
        y0=ymin + 0.75*(ymax - ymin),
        sigma_x=0.10*(xmax - xmin),
        sigma_y=0.10*(ymax - ymin),
        angle=0.0,
    )

    return topo


def my_synthetic_moho_airy(
    topography,
    reference_moho: float = 30000.0,
    rho_crust: float = 2.67,
    rho_mantle: float = 3.35,
    rho_air: float = 0.0,
):
    """
    Estimate a simple Airy-type Moho from topography.

    Inputs
    ------
    topography : array
        Topography in meters.
    reference_moho : float
        Reference Moho depth in meters.
    rho_crust : float
        Crust density in g/cm^3.
    rho_mantle : float
        Mantle density in g/cm^3.
    rho_air : float
        Air density in g/cm^3.

    Output
    ------
    moho : numpy array
        Moho depth in meters, positive downward.
    """
    topo = np.asarray(topography)
    drho = rho_mantle - rho_crust

    if drho == 0.0:
        raise ValueError("rho_mantle and rho_crust must be different.")

    root = ((rho_crust - rho_air)/drho)*topo
    moho = reference_moho + root

    return moho


# ============================================================
# SOURCE MODEL BUILDERS
# ============================================================

def my_synthetic_prism_model(
    centers_x,
    centers_y,
    top,
    bottom,
    dx,
    dy,
    density,
):
    """
    Build a list of rectangular prisms from center coordinates.

    Each prism has the format:
        [xi, xf, yi, yf, top, bottom, density]

    Inputs
    ------
    centers_x, centers_y : arrays
        Prism center coordinates.
    top, bottom : float or arrays
        Top and bottom depths.
    dx, dy : float
        Prism horizontal dimensions.
    density : float or array
        Density in g/cm^3.

    Output
    ------
    prisms : list
        List of rectangular prisms.
    """
    cx = np.asarray(centers_x, dtype=float)
    cy = np.asarray(centers_y, dtype=float)

    if cx.shape != cy.shape:
        raise ValueError("centers_x and centers_y must have the same shape.")

    n = cx.size

    top = np.full(n, top, dtype=float) if np.isscalar(top) else np.asarray(top, dtype=float).ravel()
    bottom = np.full(n, bottom, dtype=float) if np.isscalar(bottom) else np.asarray(bottom, dtype=float).ravel()
    density = np.full(n, density, dtype=float) if np.isscalar(density) else np.asarray(density, dtype=float).ravel()

    if top.size != n or bottom.size != n or density.size != n:
        raise ValueError("top, bottom and density must be scalar or have the same size as centers.")

    prisms = []

    for i in range(n):
        prisms.append([
            cx.ravel()[i] - 0.5*dx,
            cx.ravel()[i] + 0.5*dx,
            cy.ravel()[i] - 0.5*dy,
            cy.ravel()[i] + 0.5*dy,
            top[i],
            bottom[i],
            density[i],
        ])

    return prisms


def my_synthetic_prism_grid(
    area: AreaLike,
    shape: Tuple[int, int],
    top: float = 500.0,
    bottom: float = 2500.0,
    density: float = 2.67,
):
    """
    Create a regular grid of rectangular prisms.

    Inputs
    ------
    area : list
        [x_min, x_max, y_min, y_max].
    shape : tuple
        (nx, ny), number of prism centers in x and y.
    top, bottom : float
        Top and bottom depths.
    density : float
        Density in g/cm^3.

    Output
    ------
    prisms : list
        List of rectangular prisms.
    """
    x_min, x_max, y_min, y_max = area
    nx, ny = shape

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centers = 0.5*(y_edges[:-1] + y_edges[1:])

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")

    return my_synthetic_prism_model(
        Xc.ravel(),
        Yc.ravel(),
        top=top,
        bottom=bottom,
        dx=dx,
        dy=dy,
        density=density,
    )


def my_synthetic_spheres_model(
    centers_x,
    centers_y,
    centers_z,
    radius,
    physical_property,
):
    """
    Build a list of sphere models.

    Each sphere has the format:
        [xc, yc, zc, radius, physical_property]
    """
    cx = np.asarray(centers_x, dtype=float).ravel()
    cy = np.asarray(centers_y, dtype=float).ravel()
    cz = np.asarray(centers_z, dtype=float).ravel()

    if cx.size != cy.size or cx.size != cz.size:
        raise ValueError("centers_x, centers_y and centers_z must have the same size.")

    n = cx.size

    radius = np.full(n, radius, dtype=float) if np.isscalar(radius) else np.asarray(radius, dtype=float).ravel()
    prop = np.full(n, physical_property, dtype=float) if np.isscalar(physical_property) else np.asarray(physical_property, dtype=float).ravel()

    if radius.size != n or prop.size != n:
        raise ValueError("radius and physical_property must be scalar or have the same size as centers.")

    spheres = []

    for i in range(n):
        spheres.append([cx[i], cy[i], cz[i], radius[i], prop[i]])

    return spheres


def my_synthetic_cylinders_model(
    centers_x,
    centers_y,
    top,
    bottom,
    radius,
    physical_property,
):
    """
    Build a list of vertical cylinder models.

    Each cylinder has the format:
        [xc, yc, top, bottom, radius, physical_property]
    """
    cx = np.asarray(centers_x, dtype=float).ravel()
    cy = np.asarray(centers_y, dtype=float).ravel()

    if cx.size != cy.size:
        raise ValueError("centers_x and centers_y must have the same size.")

    n = cx.size

    top = np.full(n, top, dtype=float) if np.isscalar(top) else np.asarray(top, dtype=float).ravel()
    bottom = np.full(n, bottom, dtype=float) if np.isscalar(bottom) else np.asarray(bottom, dtype=float).ravel()
    radius = np.full(n, radius, dtype=float) if np.isscalar(radius) else np.asarray(radius, dtype=float).ravel()
    prop = np.full(n, physical_property, dtype=float) if np.isscalar(physical_property) else np.asarray(physical_property, dtype=float).ravel()

    if top.size != n or bottom.size != n or radius.size != n or prop.size != n:
        raise ValueError("top, bottom, radius and physical_property must be scalar or have the same size as centers.")

    cylinders = []

    for i in range(n):
        cylinders.append([cx[i], cy[i], top[i], bottom[i], radius[i], prop[i]])

    return cylinders


def my_synthetic_hexagonal_prisms_model(
    centers_x,
    centers_y,
    top,
    bottom,
    radius,
    physical_property,
    rotation: float = 0.0,
):
    """
    Build a list of vertical hexagonal prism models.

    Each hexagonal prism has the format:
        [xc, yc, top, bottom, radius, physical_property, rotation]
    """
    cx = np.asarray(centers_x, dtype=float).ravel()
    cy = np.asarray(centers_y, dtype=float).ravel()

    if cx.size != cy.size:
        raise ValueError("centers_x and centers_y must have the same size.")

    n = cx.size

    top = np.full(n, top, dtype=float) if np.isscalar(top) else np.asarray(top, dtype=float).ravel()
    bottom = np.full(n, bottom, dtype=float) if np.isscalar(bottom) else np.asarray(bottom, dtype=float).ravel()
    radius = np.full(n, radius, dtype=float) if np.isscalar(radius) else np.asarray(radius, dtype=float).ravel()
    prop = np.full(n, physical_property, dtype=float) if np.isscalar(physical_property) else np.asarray(physical_property, dtype=float).ravel()
    rotation = np.full(n, rotation, dtype=float) if np.isscalar(rotation) else np.asarray(rotation, dtype=float).ravel()

    hexprisms = []

    for i in range(n):
        hexprisms.append([cx[i], cy[i], top[i], bottom[i], radius[i], prop[i], rotation[i]])

    return hexprisms


def my_synthetic_polygonal_prism_model(
    vertices,
    top: float,
    bottom: float,
    physical_property: float,
):
    """
    Build a single polygonal prism model.

    Model format:
        [vertices, top, bottom, physical_property]
    """
    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (n_vertices, 2).")

    return [vertices, float(top), float(bottom), float(physical_property)]


# ============================================================
# NOISE AND DATA PERTURBATION
# ============================================================

def my_add_noise(
    data,
    percent: Optional[float] = None,
    std: Optional[float] = None,
    mean: float = 0.0,
    seed: Optional[int] = None,
    return_noise: bool = False,
):
    """
    Add Gaussian noise to data.

    Inputs
    ------
    data : array
        Original data.
    percent : float or None
        Noise standard deviation as percentage of data amplitude.
        Example: percent=2 means 2 percent of (max-min).
    std : float or None
        Absolute standard deviation.
    mean : float
        Noise mean.
    seed : int or None
        Random seed.
    return_noise : bool
        If True, returns noisy_data and noise.

    Output
    ------
    noisy_data : numpy array
        Data with added Gaussian noise.
    noise : numpy array, optional
        Generated noise.
    """
    data = np.asarray(data, dtype=float)

    rng = np.random.default_rng(seed)

    if std is None:
        if percent is None:
            raise ValueError("Either percent or std must be provided.")

        amplitude = np.nanmax(data) - np.nanmin(data)
        std = (percent/100.0)*amplitude

    noise = rng.normal(loc=mean, scale=std, size=data.shape)
    noisy = data + noise

    if return_noise is True:
        return noisy, noise

    return noisy


def my_add_outliers(
    data,
    n_outliers: int = 10,
    amplitude: float = 5.0,
    seed: Optional[int] = None,
):
    """
    Add sparse outliers to a dataset.

    Inputs
    ------
    data : array
        Original data.
    n_outliers : int
        Number of outliers.
    amplitude : float
        Outlier amplitude as a multiple of data standard deviation.
    seed : int or None
        Random seed.

    Output
    ------
    outlier_data : numpy array
        Data with outliers.
    """
    data = np.asarray(data, dtype=float).copy()
    flat = data.ravel()

    rng = np.random.default_rng(seed)

    n = flat.size
    n_outliers = min(n_outliers, n)

    indices = rng.choice(n, size=n_outliers, replace=False)

    std = np.nanstd(flat)
    signs = rng.choice([-1.0, 1.0], size=n_outliers)

    flat[indices] += signs*amplitude*std

    return flat.reshape(data.shape)


# ============================================================
# FORWARD MODELING WRAPPERS
# ============================================================

def my_forward_prisms_gz(x, y, z, prisms):
    """
    Compute gz from a list of rectangular prisms.
    """
    _require_module(prism, "prism")

    if hasattr(prism, "my_prisms_gz"):
        return prism.my_prisms_gz(x, y, z, prisms)

    result = np.zeros_like(np.asarray(x), dtype=float)
    for p in prisms:
        result += prism.my_prism_gz(x, y, z, p, p[6])
    return result


def my_forward_prisms_tfa(x, y, z, prisms, inc, dec, incs=None, decs=None):
    """
    Compute total-field anomaly from a list of rectangular prisms.
    """
    _require_module(prism, "prism")

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    if hasattr(prism, "my_prisms_tfa"):
        return prism.my_prisms_tfa(x, y, z, prisms, inc=inc, dec=dec, incs=incs, decs=decs)

    if hasattr(prism, "my_prisms_tf"):
        return prism.my_prisms_tf(x, y, z, prisms, inc=inc, dec=dec, incs=incs, decs=decs)

    result = np.zeros_like(np.asarray(x), dtype=float)
    for p in prisms:
        result += prism.my_prism_tf(x, y, z, p, inc, dec, incs, decs)
    return result


def my_forward_spheres_gz(x, y, z, spheres):
    """
    Compute gz from a list of spheres.
    """
    _require_module(sphere, "sphere")

    if hasattr(sphere, "my_spheres_gz"):
        return sphere.my_spheres_gz(x, y, z, spheres)

    result = np.zeros_like(np.asarray(x), dtype=float)
    for s in spheres:
        result += sphere.my_sphere_gz(x, y, z, s, s[4])
    return result


def my_forward_spheres_tfa(x, y, z, spheres, inc, dec, incs=None, decs=None):
    """
    Compute total-field anomaly from a list of spheres.
    """
    _require_module(sphere, "sphere")

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    if hasattr(sphere, "my_spheres_tfa"):
        return sphere.my_spheres_tfa(x, y, z, spheres, inc=inc, dec=dec, incs=incs, decs=decs)

    if hasattr(sphere, "my_spheres_tf"):
        return sphere.my_spheres_tf(x, y, z, spheres, inc=inc, dec=dec, incs=incs, decs=decs)

    result = np.zeros_like(np.asarray(x), dtype=float)
    for s in spheres:
        result += sphere.my_sphere_tfa(x, y, z, s, s[4], inc, dec, incs, decs)
    return result


def my_forward_cylinders_gz(x, y, z, cylinders, **kwargs):
    """
    Compute gz from a list of vertical cylinders.
    """
    _require_module(cylinder, "cylinder")

    if hasattr(cylinder, "my_cylinders_gz"):
        return cylinder.my_cylinders_gz(x, y, z, cylinders, **kwargs)

    result = np.zeros_like(np.asarray(x), dtype=float)
    for c in cylinders:
        result += cylinder.my_cylinder_gz(x, y, z, c, **kwargs)
    return result


def my_forward_cylinders_tfa(x, y, z, cylinders, inc, dec, incs=None, decs=None, **kwargs):
    """
    Compute total-field anomaly from a list of vertical cylinders.
    """
    _require_module(cylinder, "cylinder")

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    if hasattr(cylinder, "my_cylinders_tfa"):
        return cylinder.my_cylinders_tfa(
            x, y, z, cylinders, inc=inc, dec=dec, incs=incs, decs=decs, **kwargs
        )

    result = np.zeros_like(np.asarray(x), dtype=float)
    for c in cylinders:
        result += cylinder.my_cylinder_tfa(
            x, y, z, c, inc=inc, dec=dec, incs=incs, decs=decs, **kwargs
        )
    return result


def my_forward_hexagonal_prisms_gz(x, y, z, hexprisms, **kwargs):
    """
    Compute gz from a list of hexagonal prisms.
    """
    _require_module(hexagonalprism, "hexagonalprism")

    if hasattr(hexagonalprism, "my_hexagonal_prisms_gz"):
        return hexagonalprism.my_hexagonal_prisms_gz(x, y, z, hexprisms, **kwargs)

    result = np.zeros_like(np.asarray(x), dtype=float)
    for h in hexprisms:
        result += hexagonalprism.my_hexagonal_prism_gz(x, y, z, h, **kwargs)
    return result


def my_forward_polygonal_prisms_gz(x, y, z, polyprisms, **kwargs):
    """
    Compute gz from a list of polygonal prisms.
    """
    _require_module(polygonalprism, "polygonalprism")

    if hasattr(polygonalprism, "my_polygonal_prisms_gz"):
        return polygonalprism.my_polygonal_prisms_gz(x, y, z, polyprisms, **kwargs)

    result = np.zeros_like(np.asarray(x), dtype=float)
    for p in polyprisms:
        result += polygonalprism.my_polygonal_prism_gz(x, y, z, p, **kwargs)
    return result


# ============================================================
# COMPLETE SYNTHETIC TESTS
# ============================================================

def my_synthetic_prism_gravity_test(
    area: AreaLike = (0.0, 10000.0, 0.0, 10000.0),
    shape: Tuple[int, int] = (101, 101),
    observation_level: float = 0.0,
    prism_centers_x: Optional[ArrayLike] = None,
    prism_centers_y: Optional[ArrayLike] = None,
    top: Union[float, ArrayLike] = 500.0,
    bottom: Union[float, ArrayLike] = 2500.0,
    dx: float = 1000.0,
    dy: float = 1000.0,
    density: Union[float, ArrayLike] = 2.67,
    noise_percent: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Create a complete synthetic gravity test using rectangular prisms.

    Output
    ------
    result : dict
        Dictionary with grid, model and data.
    """
    x, y, z = my_make_observation_grid(area, shape, level=observation_level, flatten=False)

    if prism_centers_x is None:
        prism_centers_x = np.array([0.35*(area[1]-area[0]) + area[0],
                                    0.65*(area[1]-area[0]) + area[0]])
    if prism_centers_y is None:
        prism_centers_y = np.array([0.50*(area[3]-area[2]) + area[2],
                                    0.55*(area[3]-area[2]) + area[2]])

    prisms = my_synthetic_prism_model(
        prism_centers_x,
        prism_centers_y,
        top=top,
        bottom=bottom,
        dx=dx,
        dy=dy,
        density=density,
    )

    gz_true = my_forward_prisms_gz(x, y, z, prisms)

    if noise_percent is not None:
        gz_noisy, noise = my_add_noise(gz_true, percent=noise_percent, seed=seed, return_noise=True)
    else:
        gz_noisy = gz_true.copy()
        noise = np.zeros_like(gz_true)

    return {
        "x": x,
        "y": y,
        "z": z,
        "prisms": prisms,
        "gz_true": gz_true,
        "gz_noisy": gz_noisy,
        "noise": noise,
        "area": area,
        "shape": shape,
    }


def my_synthetic_magnetic_test_prisms(
    area: AreaLike = (0.0, 10000.0, 0.0, 10000.0),
    shape: Tuple[int, int] = (101, 101),
    observation_level: float = 0.0,
    prism_centers_x: Optional[ArrayLike] = None,
    prism_centers_y: Optional[ArrayLike] = None,
    top: Union[float, ArrayLike] = 500.0,
    bottom: Union[float, ArrayLike] = 2500.0,
    dx: float = 1000.0,
    dy: float = 1000.0,
    magnetization: Union[float, ArrayLike] = 2.0,
    inc: float = -20.0,
    dec: float = 0.0,
    incs: Optional[float] = None,
    decs: Optional[float] = None,
    noise_percent: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Create a complete synthetic magnetic test using rectangular prisms.

    Output
    ------
    result : dict
        Dictionary with grid, model and total-field anomaly.
    """
    x, y, z = my_make_observation_grid(area, shape, level=observation_level, flatten=False)

    if prism_centers_x is None:
        prism_centers_x = np.array([0.35*(area[1]-area[0]) + area[0],
                                    0.65*(area[1]-area[0]) + area[0]])
    if prism_centers_y is None:
        prism_centers_y = np.array([0.50*(area[3]-area[2]) + area[2],
                                    0.55*(area[3]-area[2]) + area[2]])

    prisms = my_synthetic_prism_model(
        prism_centers_x,
        prism_centers_y,
        top=top,
        bottom=bottom,
        dx=dx,
        dy=dy,
        density=magnetization,
    )

    tfa_true = my_forward_prisms_tfa(x, y, z, prisms, inc=inc, dec=dec, incs=incs, decs=decs)

    if noise_percent is not None:
        tfa_noisy, noise = my_add_noise(tfa_true, percent=noise_percent, seed=seed, return_noise=True)
    else:
        tfa_noisy = tfa_true.copy()
        noise = np.zeros_like(tfa_true)

    return {
        "x": x,
        "y": y,
        "z": z,
        "prisms": prisms,
        "tfa_true": tfa_true,
        "tfa_noisy": tfa_noisy,
        "noise": noise,
        "area": area,
        "shape": shape,
        "inc": inc,
        "dec": dec,
        "incs": inc if incs is None else incs,
        "decs": dec if decs is None else decs,
    }


# ============================================================
# MODEL COMPARISON
# ============================================================

def my_compare_data(observed, predicted):
    """
    Compare observed and predicted data using common error metrics.

    Output
    ------
    report : dict
        Dictionary with residual, rmse, mae, correlation and relative error.
    """
    observed, predicted = _check_same_shape(observed, predicted)

    residual = observed - predicted

    rmse = float(np.sqrt(np.nanmean(residual**2)))
    mae = float(np.nanmean(np.abs(residual)))

    obs_std = np.nanstd(observed)
    pred_std = np.nanstd(predicted)

    if obs_std == 0.0 or pred_std == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(observed.ravel(), predicted.ravel())[0, 1])

    denominator = np.nanmax(observed) - np.nanmin(observed)

    if denominator == 0.0:
        relative_rmse = np.nan
    else:
        relative_rmse = float(100.0*rmse/denominator)

    return {
        "residual": residual,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "relative_rmse_percent": relative_rmse,
    }


def my_save_synthetic_dataset(filename: str, x, y, z, data, header: Optional[str] = None):
    """
    Save synthetic dataset as a text file with columns x, y, z, data.

    Inputs
    ------
    filename : str
        Output filename.
    x, y, z, data : arrays
        Data arrays with the same shape.
    header : str or None
        File header.

    Output
    ------
    filename : str
        Saved filename.
    """
    x, y, z, data = _check_same_shape(x, y, z, data)

    folder = os.path.dirname(filename)

    if folder not in ["", "."] and not os.path.exists(folder):
        os.makedirs(folder)

    table = np.column_stack([x.ravel(), y.ravel(), z.ravel(), data.ravel()])

    if header is None:
        header = "x y z data"

    np.savetxt(filename, table, fmt="%.10e", header=header)

    return filename


# ============================================================
# SHORT ALIASES
# ============================================================

my_obs_grid = my_make_observation_grid
my_add_gaussian_noise = my_add_noise
my_prism_model = my_synthetic_prism_model
my_prism_grid_model = my_synthetic_prism_grid
my_sphere_model = my_synthetic_spheres_model
my_cylinder_model = my_synthetic_cylinders_model
my_hexprism_model = my_synthetic_hexagonal_prisms_model
my_polyprism_model = my_synthetic_polygonal_prism_model
my_compare = my_compare_data
