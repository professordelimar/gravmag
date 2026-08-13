# -*- coding: utf-8 -*-
"""
Kernel functions for equivalent-layer, gravity and magnetic modelling.

This module keeps the original public function names used by the previous
codes, especially:

    my_kernelx, my_kernely, my_kernelz,
    my_kernelxx, my_kernelxy, my_kernelxz,
    my_kernelyy, my_kernelyz, my_kernelzz

The original functions are preserved as compatibility functions. Internally,
the code was reorganized to reduce repetition, improve numerical safety and
make it easier to add new kernels.

Coordinate convention used in the original project is preserved:
    x, y, z are observation coordinates;
    model = [xe, ye, ze, radius];
    z is positive downward;
    distances are computed as observation minus source coordinates.

The kernels represent derivatives of the Newtonian kernel 1/r multiplied by
sphere volume, where V = 4*pi*radius^3/3. The physical constants G, cm and
T2nT are not included in the basic kernels because they are applied in the
calling modules, such as equivalentlayer.py.
"""

from __future__ import division
import numpy


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

_EPS = 1.0e-15


def _as_array(a):
    """Return input as numpy array without forcing a copy."""
    return numpy.asarray(a, dtype=float)


def _check_observation_arrays(x, y, z):
    """
    Check if x, y and z have compatible shapes.

    Inputs:
    x, y, z - scalar or numpy arrays - observation coordinates

    Outputs:
    x, y, z - numpy arrays with common shape
    """

    x = _as_array(x)
    y = _as_array(y)
    z = _as_array(z)

    try:
        x, y, z = numpy.broadcast_arrays(x, y, z)
    except ValueError:
        raise ValueError("x, y and z must be broadcastable to the same shape!")

    return x, y, z


def _parse_model(model):
    """
    Parse a sphere model.

    Inputs:
    model - list/tuple/array - [xe, ye, ze, radius]

    Outputs:
    xe, ye, ze, radius - floats
    """

    if len(model) < 4:
        raise ValueError("model must contain at least [xe, ye, ze, radius]!")

    xe = float(model[0])
    ye = float(model[1])
    ze = float(model[2])
    radius = float(model[3])

    if radius <= 0.0:
        raise ValueError("Sphere radius must be positive and nonzero!")

    return xe, ye, ze, radius


def my_sphere_volume(radius):
    """
    Calculate the volume of a sphere.

    Input:
    radius - float - sphere radius

    Output:
    volume - float - sphere volume
    """

    radius = float(radius)

    if radius <= 0.0:
        raise ValueError("Sphere radius must be positive and nonzero!")

    return (4.0/3.0)*numpy.pi*radius**3


def my_kernel_coordinates(x, y, z, model, eps=_EPS):
    """
    Calculate relative coordinates, distance and volume for a sphere kernel.

    Inputs:
    x, y, z - scalar or numpy arrays - observation coordinates
    model - list/tuple/array - [xe, ye, ze, radius]
    eps - float - small value used to avoid singularity at r=0

    Outputs:
    dx, dy, dz - numpy arrays - relative coordinates observation-source
    r - numpy array - Euclidean distance
    volume - float - sphere volume
    """

    x, y, z = _check_observation_arrays(x, y, z)
    xe, ye, ze, radius = _parse_model(model)

    dx = x - xe
    dy = y - ye
    dz = z - ze

    r2 = dx**2 + dy**2 + dz**2
    r = numpy.sqrt(r2)

    if eps is not None:
        r = numpy.where(r < eps, eps, r)

    volume = my_sphere_volume(radius)

    return dx, dy, dz, r, volume


def my_kernel_r(x, y, z, model, eps=_EPS):
    """
    Return the distance from observation points to the sphere center.

    Inputs:
    x, y, z - scalar or numpy arrays - observation coordinates
    model - list/tuple/array - [xe, ye, ze, radius]
    eps - float - small value used to avoid singularity at r=0

    Output:
    r - numpy array - distance
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model, eps=eps)
    return r


# -----------------------------------------------------------------------------
# Zeroth-order and first-order kernels
# -----------------------------------------------------------------------------

def my_kernel0(x, y, z, model):
    """
    Calculate the volume-scaled Newtonian kernel V/r.

    Inputs:
    x, y, z - scalar or numpy arrays - observation coordinates
    model - list/tuple/array - [xe, ye, ze, radius]

    Output:
    kernel - numpy array - V/r
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return volume/r


def my_kernel_potential(x, y, z, model):
    """
    Alias for the volume-scaled potential kernel V/r.
    """

    return my_kernel0(x, y, z, model)


def my_kernelx(x, y, z, model):
    """
    Calculate the first x-derivative of the volume-scaled 1/r kernel.

    Output:
    diffx = d(V/r)/dx = -V*(x - xe)/r^3
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return -volume*dx/(r**3)


def my_kernely(x, y, z, model):
    """
    Calculate the first y-derivative of the volume-scaled 1/r kernel.

    Output:
    diffy = d(V/r)/dy = -V*(y - ye)/r^3
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return -volume*dy/(r**3)


def my_kernelz(x, y, z, model):
    """
    Calculate the first z-derivative of the volume-scaled 1/r kernel.

    Output:
    diffz = d(V/r)/dz = -V*(z - ze)/r^3
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return -volume*dz/(r**3)


# -----------------------------------------------------------------------------
# Second-order kernels: Hessian of V/r
# -----------------------------------------------------------------------------

def my_kernelxx(x, y, z, model):
    """
    Calculate the second xx-derivative of the volume-scaled 1/r kernel.
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return volume*((3.0*dx*dx)/(r**5) - 1.0/(r**3))


def my_kernelxy(x, y, z, model):
    """
    Calculate the second xy-derivative of the volume-scaled 1/r kernel.
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return volume*(3.0*dx*dy/(r**5))


def my_kernelxz(x, y, z, model):
    """
    Calculate the second xz-derivative of the volume-scaled 1/r kernel.

    Note:
    The previous version omitted the multiplication by sphere volume. This
    implementation corrects that inconsistency so that xz, xy and yz kernels
    use the same physical scaling.
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return volume*(3.0*dx*dz/(r**5))


def my_kernelyy(x, y, z, model):
    """
    Calculate the second yy-derivative of the volume-scaled 1/r kernel.

    Note:
    The previous version had an operator-precedence inconsistency. This version
    applies the sphere volume to the complete expression.
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return volume*((3.0*dy*dy)/(r**5) - 1.0/(r**3))


def my_kernelyz(x, y, z, model):
    """
    Calculate the second yz-derivative of the volume-scaled 1/r kernel.
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return volume*(3.0*dy*dz/(r**5))


def my_kernelzz(x, y, z, model):
    """
    Calculate the second zz-derivative of the volume-scaled 1/r kernel.
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    return volume*((3.0*dz*dz)/(r**5) - 1.0/(r**3))


def my_kernel_gradient(x, y, z, model):
    """
    Return all first derivatives of V/r.

    Outputs:
    kx, ky, kz - numpy arrays
    """

    return (my_kernelx(x, y, z, model),
            my_kernely(x, y, z, model),
            my_kernelz(x, y, z, model))


def my_kernel_hessian(x, y, z, model):
    """
    Return all second derivatives of V/r.

    Outputs:
    kxx, kxy, kxz, kyy, kyz, kzz - numpy arrays
    """

    return (my_kernelxx(x, y, z, model),
            my_kernelxy(x, y, z, model),
            my_kernelxz(x, y, z, model),
            my_kernelyy(x, y, z, model),
            my_kernelyz(x, y, z, model),
            my_kernelzz(x, y, z, model))


def my_kernel_tensor_matrix(x, y, z, model):
    """
    Return the full 3x3 Hessian tensor of V/r.

    Output:
    tensor - numpy array with shape field_shape + (3, 3)
    """

    kxx, kxy, kxz, kyy, kyz, kzz = my_kernel_hessian(x, y, z, model)

    tensor = numpy.empty(kxx.shape + (3, 3), dtype=float)
    tensor[..., 0, 0] = kxx
    tensor[..., 0, 1] = kxy
    tensor[..., 0, 2] = kxz
    tensor[..., 1, 0] = kxy
    tensor[..., 1, 1] = kyy
    tensor[..., 1, 2] = kyz
    tensor[..., 2, 0] = kxz
    tensor[..., 2, 1] = kyz
    tensor[..., 2, 2] = kzz

    return tensor


def my_kernel_laplacian(x, y, z, model):
    """
    Return the Laplacian of V/r outside the source.

    For points outside the singularity, the Laplacian of 1/r is zero. This
    function is useful as a numerical consistency check:

        kxx + kyy + kzz ≈ 0
    """

    return (my_kernelxx(x, y, z, model) +
            my_kernelyy(x, y, z, model) +
            my_kernelzz(x, y, z, model))


# -----------------------------------------------------------------------------
# Third-order kernels: useful for Euler deconvolution and gradients of tensors
# -----------------------------------------------------------------------------

def _third_kernel_component(x, y, z, model, i, j, k):
    """
    Generic third derivative of V/r.

    Indices i, j, k follow: 0=x, 1=y, 2=z.
    """

    dx, dy, dz, r, volume = my_kernel_coordinates(x, y, z, model)
    q = [dx, dy, dz]

    delta_ij = 1.0 if i == j else 0.0
    delta_ik = 1.0 if i == k else 0.0
    delta_jk = 1.0 if j == k else 0.0

    term1 = 3.0*(delta_ij*q[k] + delta_ik*q[j] + delta_jk*q[i])/(r**5)
    term2 = 15.0*q[i]*q[j]*q[k]/(r**7)

    return volume*(term1 - term2)


def my_kernelxxx(x, y, z, model):
    """Calculate d3(V/r)/dx3."""
    return _third_kernel_component(x, y, z, model, 0, 0, 0)


def my_kernelxxy(x, y, z, model):
    """Calculate d3(V/r)/dx2dy."""
    return _third_kernel_component(x, y, z, model, 0, 0, 1)


def my_kernelxxz(x, y, z, model):
    """Calculate d3(V/r)/dx2dz."""
    return _third_kernel_component(x, y, z, model, 0, 0, 2)


def my_kernelxyy(x, y, z, model):
    """Calculate d3(V/r)/dxdy2."""
    return _third_kernel_component(x, y, z, model, 0, 1, 1)


def my_kernelxyz(x, y, z, model):
    """Calculate d3(V/r)/dxdydz."""
    return _third_kernel_component(x, y, z, model, 0, 1, 2)


def my_kernelxzz(x, y, z, model):
    """Calculate d3(V/r)/dxdz2."""
    return _third_kernel_component(x, y, z, model, 0, 2, 2)


def my_kernelyyy(x, y, z, model):
    """Calculate d3(V/r)/dy3."""
    return _third_kernel_component(x, y, z, model, 1, 1, 1)


def my_kernelyyz(x, y, z, model):
    """Calculate d3(V/r)/dy2dz."""
    return _third_kernel_component(x, y, z, model, 1, 1, 2)


def my_kernelyzz(x, y, z, model):
    """Calculate d3(V/r)/dydz2."""
    return _third_kernel_component(x, y, z, model, 1, 2, 2)


def my_kernelzzz(x, y, z, model):
    """Calculate d3(V/r)/dz3."""
    return _third_kernel_component(x, y, z, model, 2, 2, 2)


def my_kernel_third_order(x, y, z, model):
    """
    Return the independent third-order derivatives of V/r.

    Outputs:
    kxxx, kxxy, kxxz, kxyy, kxyz, kxzz, kyyy, kyyz, kyzz, kzzz
    """

    return (my_kernelxxx(x, y, z, model),
            my_kernelxxy(x, y, z, model),
            my_kernelxxz(x, y, z, model),
            my_kernelxyy(x, y, z, model),
            my_kernelxyz(x, y, z, model),
            my_kernelxzz(x, y, z, model),
            my_kernelyyy(x, y, z, model),
            my_kernelyyz(x, y, z, model),
            my_kernelyzz(x, y, z, model),
            my_kernelzzz(x, y, z, model))


# -----------------------------------------------------------------------------
# Directional and projected kernels
# -----------------------------------------------------------------------------

def my_direction_vector(inc, dec):
    """
    Return direction cosines for inclination and declination in degrees.

    Inputs:
    inc - float - inclination in degrees, positive downward
    dec - float - declination in degrees, clockwise from x/North

    Outputs:
    vx, vy, vz - floats - direction cosines
    """

    inc_rad = numpy.deg2rad(inc)
    dec_rad = numpy.deg2rad(dec)

    vx = numpy.cos(inc_rad)*numpy.cos(dec_rad)
    vy = numpy.cos(inc_rad)*numpy.sin(dec_rad)
    vz = numpy.sin(inc_rad)

    return vx, vy, vz


def my_directional_first_kernel(x, y, z, model, inc, dec):
    """
    Project the first-derivative kernel along a direction.

    Output:
    kd = vx*kx + vy*ky + vz*kz
    """

    vx, vy, vz = my_direction_vector(inc, dec)
    kx, ky, kz = my_kernel_gradient(x, y, z, model)

    return vx*kx + vy*ky + vz*kz


def my_directional_second_kernel(x, y, z, model, inc1, dec1, inc2=None, dec2=None):
    """
    Project the second-derivative tensor along two directions.

    This is useful for magnetic total-field kernels, where the Hessian is
    projected along the inducing-field and magnetization directions.

    Inputs:
    inc1, dec1 - first projection direction
    inc2, dec2 - second projection direction. If None, uses inc1, dec1.

    Output:
    kd - numpy array - projected second-order kernel
    """

    if inc2 is None:
        inc2 = inc1
    if dec2 is None:
        dec2 = dec1

    f = my_direction_vector(inc1, dec1)
    m = my_direction_vector(inc2, dec2)

    kxx, kxy, kxz, kyy, kyz, kzz = my_kernel_hessian(x, y, z, model)

    kd = (f[0]*m[0]*kxx +
          (f[0]*m[1] + f[1]*m[0])*kxy +
          (f[0]*m[2] + f[2]*m[0])*kxz +
          f[1]*m[1]*kyy +
          (f[1]*m[2] + f[2]*m[1])*kyz +
          f[2]*m[2]*kzz)

    return kd


def my_totalfield_kernel(x, y, z, model, inc, dec, incs=None, decs=None):
    """
    Return the magnetic total-field kernel of a uniformly magnetized sphere.

    This function returns only the projected Hessian kernel. The physical
    constants and magnetization intensity can be applied outside:

        anomaly_nT = cm*T2nT*mag*my_totalfield_kernel(...)

    Inputs:
    x, y, z - observation coordinates
    model - [xe, ye, ze, radius]
    inc, dec - inducing-field inclination and declination
    incs, decs - source magnetization inclination and declination. If None,
                 induced magnetization is assumed.

    Output:
    kernel - numpy array - projected magnetic total-field kernel
    """

    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    return my_directional_second_kernel(x, y, z, model, inc, dec, incs, decs)


# -----------------------------------------------------------------------------
# Physical kernels for unit density or unit magnetization
# -----------------------------------------------------------------------------

def my_gravity_gx_kernel(x, y, z, model, G=6.673e-11, si2mGal=100000.0):
    """
    Gravity gx kernel in mGal for unit SI density contrast of 1 kg/m^3.
    """

    return G*si2mGal*my_kernelx(x, y, z, model)


def my_gravity_gy_kernel(x, y, z, model, G=6.673e-11, si2mGal=100000.0):
    """
    Gravity gy kernel in mGal for unit SI density contrast of 1 kg/m^3.
    """

    return G*si2mGal*my_kernely(x, y, z, model)


def my_gravity_gz_kernel(x, y, z, model, G=6.673e-11, si2mGal=100000.0):
    """
    Gravity gz kernel in mGal for unit SI density contrast of 1 kg/m^3.
    """

    return G*si2mGal*my_kernelz(x, y, z, model)


def my_gravity_gradient_kernel(x, y, z, model, component="zz", G=6.673e-11, si2Eotvos=1.0e9):
    """
    Gravity-gradient kernel in Eotvos for unit SI density contrast of 1 kg/m^3.

    Inputs:
    component - string - one of 'xx', 'xy', 'xz', 'yy', 'yz', 'zz'

    Output:
    kernel - numpy array - gravity-gradient kernel in Eotvos per kg/m^3
    """

    component = component.lower()

    kernels = {
        "xx": my_kernelxx,
        "xy": my_kernelxy,
        "xz": my_kernelxz,
        "yy": my_kernelyy,
        "yz": my_kernelyz,
        "zz": my_kernelzz,
    }

    if component not in kernels:
        raise ValueError("component must be one of: xx, xy, xz, yy, yz, zz")

    return G*si2Eotvos*kernels[component](x, y, z, model)


def my_magnetic_totalfield_kernel(x, y, z, model, inc, dec, incs=None, decs=None,
                                  cm=1.0e-7, T2nT=1.0e9):
    """
    Magnetic total-field anomaly kernel in nT for unit magnetization A/m.
    """

    return cm*T2nT*my_totalfield_kernel(x, y, z, model, inc, dec, incs, decs)


# -----------------------------------------------------------------------------
# Sensitivity matrix builders
# -----------------------------------------------------------------------------

def my_kernel_matrix(x, y, z, models, kernel_function=my_kernelz, dtype=float):
    """
    Build a sensitivity matrix for a list of sphere models.

    Inputs:
    x, y, z - observation coordinates. They are flattened internally.
    models - list - list of [xe, ye, ze, radius]
    kernel_function - function - kernel to be evaluated for each model
    dtype - data type of the output matrix

    Output:
    matrix - numpy array - shape (n_observations, n_models)
    """

    x, y, z = _check_observation_arrays(x, y, z)
    x_vec = x.ravel()
    y_vec = y.ravel()
    z_vec = z.ravel()

    n = x_vec.size
    m = len(models)

    matrix = numpy.empty((n, m), dtype=dtype)

    for j, model in enumerate(models):
        matrix[:, j] = numpy.asarray(kernel_function(x_vec, y_vec, z_vec, model)).ravel()

    return matrix


def my_gravity_kernel_matrix(x, y, z, models, component="z", G=6.673e-11, si2mGal=100000.0):
    """
    Build a gravity sensitivity matrix for sphere models.

    Inputs:
    component - string - 'x', 'y' or 'z'

    Output:
    matrix - numpy array - gravity sensitivity in mGal per kg/m^3
    """

    component = component.lower()

    if component == "x":
        func = lambda xo, yo, zo, model: my_gravity_gx_kernel(xo, yo, zo, model, G=G, si2mGal=si2mGal)
    elif component == "y":
        func = lambda xo, yo, zo, model: my_gravity_gy_kernel(xo, yo, zo, model, G=G, si2mGal=si2mGal)
    elif component == "z":
        func = lambda xo, yo, zo, model: my_gravity_gz_kernel(xo, yo, zo, model, G=G, si2mGal=si2mGal)
    else:
        raise ValueError("component must be 'x', 'y' or 'z'!")

    return my_kernel_matrix(x, y, z, models, kernel_function=func)


def my_totalfield_kernel_matrix(x, y, z, models, inc, dec, incs=None, decs=None,
                                cm=1.0e-7, T2nT=1.0e9):
    """
    Build a magnetic total-field sensitivity matrix for sphere models.

    Output:
    matrix - numpy array - nT per A/m
    """

    func = lambda xo, yo, zo, model: my_magnetic_totalfield_kernel(
        xo, yo, zo, model, inc, dec, incs=incs, decs=decs, cm=cm, T2nT=T2nT
    )

    return my_kernel_matrix(x, y, z, models, kernel_function=func)


# -----------------------------------------------------------------------------
# Compatibility aliases
# -----------------------------------------------------------------------------

my_kernel = my_kernel0
my_kernelv = my_kernel0
my_kernel_pot = my_kernel0

my_kernel_x = my_kernelx
my_kernel_y = my_kernely
my_kernel_z = my_kernelz

my_kernel_xx = my_kernelxx
my_kernel_xy = my_kernelxy
my_kernel_xz = my_kernelxz
my_kernel_yy = my_kernelyy
my_kernel_yz = my_kernelyz
my_kernel_zz = my_kernelzz

my_kernelgrad = my_kernel_gradient
my_kernelhessian = my_kernel_hessian
my_kerneltensor = my_kernel_tensor_matrix
my_kernellap = my_kernel_laplacian
