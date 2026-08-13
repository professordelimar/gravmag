# -----------------------------------------------------------------------------------
# Title: Equivalent Layer Methods - revised version
# Description: Classical equivalent layer routines for gravity and magnetic data.
# Author: Nelson Ribeiro Filho
# Revised with compatibility-preserving improvements
# -----------------------------------------------------------------------------------

from __future__ import division
import numpy

try:
    from . import auxiliars, grids, kernel
except ImportError:  # Allows using the file outside a package named "codes"
    import auxiliars
    import grids
    import kernel

try:
    from . import constants
except ImportError:
    try:
        import constants
    except ImportError:
        constants = None


# -----------------------------------------------------------------------------
# CONSTANTS AND INTERNAL UTILITIES
# -----------------------------------------------------------------------------

def _gravity_constant():
    if constants is not None and hasattr(constants, "G"):
        return constants.G
    return 6.673e-11


def _si2mgal():
    if constants is not None and hasattr(constants, "si2mGal"):
        return constants.si2mGal
    return 100000.0


def _cm_constant():
    if constants is not None and hasattr(constants, "cm"):
        return constants.cm
    return 1.0e-7


def _t2nt():
    if constants is not None and hasattr(constants, "T2nT"):
        return constants.T2nT
    return 1.0e9


def _as_array_1d(data, name):
    '''
    Convert input data to a 1D numpy array without changing numerical values.
    '''
    arr = numpy.asarray(data, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(name + " must not be empty!")
    return arr


def _check_observations(xo, yo, zo):
    '''
    Check and return observation coordinates as 1D arrays.
    '''
    xo = _as_array_1d(xo, "xo")
    yo = _as_array_1d(yo, "yo")
    zo = _as_array_1d(zo, "zo")

    if xo.size != yo.size or xo.size != zo.size:
        raise ValueError("xo, yo and zo must have the same size!")

    return xo, yo, zo


def _check_dataset(dataset):
    '''
    Check a dataset in the form [x, y, z, data].
    '''
    if len(dataset) != 4:
        raise ValueError("dataset must be a list or tuple with [x, y, z, data]!")

    x = _as_array_1d(dataset[0], "dataset[0]")
    y = _as_array_1d(dataset[1], "dataset[1]")
    z = _as_array_1d(dataset[2], "dataset[2]")
    data = _as_array_1d(dataset[3], "dataset[3]")

    if x.size != y.size or x.size != z.size or x.size != data.size:
        raise ValueError("x, y, z and data in dataset must have the same size!")

    return x, y, z, data


def _shape_size(shape, name):
    '''
    Return the number of elements implied by a shape tuple.
    '''
    if shape is None:
        raise ValueError(name + " must not be None!")
    if len(shape) != 2:
        raise ValueError(name + " must be a tuple with two elements: (nx, ny)!")
    return int(shape[0])*int(shape[1])


def _safe_trace_normalization(mat, size):
    '''
    Compute trace(A.T A)/size with a safe fallback.
    '''
    if size <= 0:
        raise ValueError("size must be positive!")

    trace = numpy.trace(numpy.dot(mat.T, mat))/float(size)

    if not numpy.isfinite(trace) or trace == 0.0:
        trace = 1.0

    return trace


def _solve_tikhonov(mat, data, regulator=0.0):
    '''
    Solve an equivalent-layer linear system using zero-order Tikhonov
    regularization. This routine preserves the original overdetermined and
    underdetermined strategy used in the original code.

    Inputs:
    mat - numpy 2D array - sensitivity matrix
    data - numpy 1D array - observed data
    regulator - float - zero-order Tikhonov parameter

    Output:
    pvec - numpy 1D array - estimated equivalent layer parameters
    '''
    mat = numpy.asarray(mat, dtype=float)
    data = _as_array_1d(data, "data")

    if mat.ndim != 2:
        raise ValueError("mat must be a 2D array!")

    n_data, n_par = mat.shape

    if data.size != n_data:
        raise ValueError("data size must be equal to the number of rows of mat!")

    if regulator is None:
        regulator = 0.0

    regulator = float(regulator)

    if regulator < 0.0:
        raise ValueError("regulator must be non-negative!")

    if n_data >= n_par:
        identity = numpy.identity(n_par)
        trace = _safe_trace_normalization(mat, n_par)
        lhs = numpy.dot(mat.T, mat) + regulator*trace*identity
        rhs = numpy.dot(mat.T, data)
        pvec = numpy.linalg.solve(lhs, rhs)
    else:
        identity = numpy.identity(n_data)
        trace = _safe_trace_normalization(mat, n_data)
        lhs = numpy.dot(mat, mat.T) + regulator*trace*identity
        aux = numpy.linalg.solve(lhs, data)
        pvec = numpy.dot(mat.T, aux)

    return pvec


def my_tikhonov_solve(mat, data, regulator=0.0):
    '''
    Public compatibility-friendly wrapper for zero-order Tikhonov inversion.

    Inputs:
    mat - numpy 2D array - sensitivity matrix
    data - numpy 1D array - observed data
    regulator - float - regularization parameter

    Output:
    pvec - numpy 1D array - estimated parameter vector
    '''
    return _solve_tikhonov(mat, data, regulator=regulator)


# -----------------------------------------------------------------------------
# BUILDING THE CLASSICAL EQUIVALENT LAYER
# -----------------------------------------------------------------------------

def my_layer(area, shape, level):
    '''
    It generates a list with all 3D sphere positions used as an equivalent layer.

    Inputs:
    area - list - [xi, xf, yi, yf]
    shape - tuple - grid shape
    level - float - layer depth, positive downward

    Output:
    layer - list - each element is [x, y, z, radius]
    '''
    if level <= 0.0:
        raise ValueError("Depth of the layer must be positive and non-null!")

    radius = (3.0/(4.0*numpy.pi))**(1.0/3.0)
    xo, yo, zo = grids.my_regular(area, shape, level)

    layer = []
    for i in range(len(xo)):
        layer.append([xo[i], yo[i], zo[i], radius])

    return layer


def my_layer_array(area, shape, level):
    '''
    It generates the equivalent layer as a 2D numpy array.

    Inputs:
    area - list - [xi, xf, yi, yf]
    shape - tuple - grid shape
    level - float - layer depth, positive downward

    Output:
    layer - numpy array - columns: x, y, z, radius
    '''
    return numpy.asarray(my_layer(area, shape, level), dtype=float)


# -----------------------------------------------------------------------------
# GRAVITY EQUIVALENT LAYER MATRICES
# -----------------------------------------------------------------------------

def my_gz_layer(xo, yo, zo, layer):
    '''
    It calculates the sensitivity matrix for gravity data as the vertical
    gravitational component gz.

    Inputs:
    xo, yo, zo - numpy arrays - observation points
    layer - list - equivalent layer model

    Output:
    mat - numpy matrix - gravity sensitivity matrix
    '''
    xo, yo, zo = _check_observations(xo, yo, zo)

    g = _gravity_constant()
    si2mGal = _si2mgal()

    n = xo.size
    m = len(layer)

    if m == 0:
        raise ValueError("layer must contain at least one source!")

    mat = numpy.zeros((n, m))

    for i, source in enumerate(layer):
        mat[:, i] = kernel.my_kernelz(xo, yo, zo, source)

    mat *= g*si2mGal

    return mat


def my_gxyz_layer(xo, yo, zo, layer):
    '''
    It calculates sensitivity matrices for the first derivatives of gz:
    dgz/dx, dgz/dy and dgz/dz.

    Inputs:
    xo, yo, zo - numpy arrays - observation points
    layer - list - equivalent layer model

    Outputs:
    gzx, gzy, gzz - numpy matrices - derivative sensitivity matrices
    '''
    xo, yo, zo = _check_observations(xo, yo, zo)

    g = _gravity_constant()
    si2mGal = _si2mgal()

    n = xo.size
    m = len(layer)

    if m == 0:
        raise ValueError("layer must contain at least one source!")

    gzx = numpy.zeros((n, m))
    gzy = numpy.zeros((n, m))
    gzz = numpy.zeros((n, m))

    for i, source in enumerate(layer):
        gzx[:, i] = kernel.my_kernelxz(xo, yo, zo, source)
        gzy[:, i] = kernel.my_kernelyz(xo, yo, zo, source)
        gzz[:, i] = kernel.my_kernelzz(xo, yo, zo, source)

    gzx *= g*si2mGal
    gzy *= g*si2mGal
    gzz *= g*si2mGal

    return gzx, gzy, gzz


def my_predict_grav(xo, yo, zo, layermodel, pvec):
    '''
    Predict gz data from an equivalent layer parameter vector.

    Inputs:
    xo, yo, zo - numpy arrays - observation points
    layermodel - list - equivalent layer
    pvec - numpy array - layer parameters

    Output:
    predicted - numpy array - predicted gz data
    '''
    mat = my_gz_layer(xo, yo, zo, layermodel)
    pvec = _as_array_1d(pvec, "pvec")

    if pvec.size != mat.shape[1]:
        raise ValueError("pvec size must be equal to the number of layer sources!")

    return numpy.dot(mat, pvec)


def my_fitdata_grav(dataset, datashape, layermodel, layershape, regulator):
    '''
    It returns the estimated equivalent layer parameters and predicted gravity
    data using the classical equivalent layer technique.

    Inputs:
    dataset - list - [xobs, yobs, zobs, gz]
    datashape - tuple - shape of input data
    layermodel - list - equivalent layer model
    layershape - tuple - shape of equivalent layer
    regulator - float - zero-order Tikhonov parameter

    Outputs:
    pvec - numpy array - estimated parameter vector
    predicted - numpy array - predicted gravity data
    '''
    xp, yp, zp, gz = _check_dataset(dataset)

    n_expected = _shape_size(datashape, "datashape")
    m_expected = _shape_size(layershape, "layershape")

    if xp.size != n_expected:
        raise ValueError("datashape is inconsistent with the number of observations!")
    if len(layermodel) != m_expected:
        raise ValueError("layershape is inconsistent with the number of layer sources!")

    mat = my_gz_layer(xp, yp, zp, layermodel)
    pvec = _solve_tikhonov(mat, gz, regulator=regulator)
    predicted = numpy.dot(mat, pvec)

    return pvec, predicted


def my_fitdata_grav_full(dataset, datashape, layermodel, layershape, regulator):
    '''
    Extended gravity fitting routine that also returns the sensitivity matrix
    and residuals.

    Outputs:
    pvec, predicted, residual, mat
    '''
    xp, yp, zp, gz = _check_dataset(dataset)
    pvec, predicted = my_fitdata_grav(dataset, datashape, layermodel, layershape, regulator)
    residual = gz - predicted
    mat = my_gz_layer(xp, yp, zp, layermodel)
    return pvec, predicted, residual, mat


# -----------------------------------------------------------------------------
# MAGNETIC EQUIVALENT LAYER MATRICES
# -----------------------------------------------------------------------------

def my_totalfield_layer(xo, yo, zo, layer, inc, dec, incs, decs):
    '''
    It calculates the sensitivity matrix for magnetic total-field anomaly.

    Inputs:
    xo, yo, zo - numpy arrays - observation points
    layer - list - equivalent layer model
    inc, dec - float - inducing field inclination and declination in degrees
    incs, decs - float - source magnetization inclination and declination in degrees

    Output:
    mat - numpy matrix - magnetic sensitivity matrix
    '''
    xo, yo, zo = _check_observations(xo, yo, zo)

    cm = _cm_constant()
    t2nT = _t2nt()

    n = xo.size
    m = len(layer)

    if m == 0:
        raise ValueError("layer must contain at least one source!")

    mat = numpy.zeros((n, m))

    fx, fy, fz = auxiliars.my_regional(1.0, inc, dec)
    mx, my, mz = auxiliars.my_regional(1.0, incs, decs)

    for i, source in enumerate(layer):
        phi_xx = kernel.my_kernelxx(xo, yo, zo, source)
        phi_yy = kernel.my_kernelyy(xo, yo, zo, source)
        phi_zz = kernel.my_kernelzz(xo, yo, zo, source)
        phi_xy = kernel.my_kernelxy(xo, yo, zo, source)
        phi_xz = kernel.my_kernelxz(xo, yo, zo, source)
        phi_yz = kernel.my_kernelyz(xo, yo, zo, source)

        mat[:, i] = (
            fx*phi_xx*mx + fx*phi_xy*my + fx*phi_xz*mz +
            fy*phi_xy*mx + fy*phi_yy*my + fy*phi_yz*mz +
            fz*phi_xz*mx + fz*phi_yz*my + fz*phi_zz*mz
        )

    mat *= cm*t2nT

    return mat


def my_predict_mag(xo, yo, zo, layermodel, pvec, inc, dec, incs=None, decs=None):
    '''
    Predict magnetic total-field anomaly from an equivalent layer.

    Inputs:
    xo, yo, zo - numpy arrays - observation points
    layermodel - list - equivalent layer
    pvec - numpy array - layer parameters
    inc, dec - float - inducing field inclination and declination
    incs, decs - float/None - source inclination and declination

    Output:
    predicted - numpy array - predicted magnetic total-field anomaly
    '''
    if incs is None:
        incs = inc
    if decs is None:
        decs = dec

    mat = my_totalfield_layer(xo, yo, zo, layermodel, inc, dec, incs, decs)
    pvec = _as_array_1d(pvec, "pvec")

    if pvec.size != mat.shape[1]:
        raise ValueError("pvec size must be equal to the number of layer sources!")

    return numpy.dot(mat, pvec)


def my_fitdata_mag(dataset, datashape, layermodel, layershape, regulator,
                   inc, dec, incl=None, decl=None):
    '''
    It returns estimated equivalent layer parameters and predicted magnetic
    total-field data.

    Inputs:
    dataset - list - [xobs, yobs, zobs, totalfield]
    datashape - tuple - shape of input data
    layermodel - list - equivalent layer model
    layershape - tuple - shape of equivalent layer
    regulator - float - zero-order Tikhonov parameter
    inc, dec - float - inducing field inclination and declination
    incl, decl - float/None - equivalent layer magnetization direction

    Outputs:
    pvec - numpy array - estimated parameter vector
    predicted - numpy array - predicted magnetic total-field data
    '''
    if incl is None:
        incl = inc
    if decl is None:
        decl = dec

    xp, yp, zp, tf = _check_dataset(dataset)

    n_expected = _shape_size(datashape, "datashape")
    m_expected = _shape_size(layershape, "layershape")

    if xp.size != n_expected:
        raise ValueError("datashape is inconsistent with the number of observations!")
    if len(layermodel) != m_expected:
        raise ValueError("layershape is inconsistent with the number of layer sources!")

    mat = my_totalfield_layer(xp, yp, zp, layermodel, inc, dec, incl, decl)
    pvec = _solve_tikhonov(mat, tf, regulator=regulator)
    predicted = numpy.dot(mat, pvec)

    return pvec, predicted


def my_fitdata_mag_full(dataset, datashape, layermodel, layershape, regulator,
                        inc, dec, incl=None, decl=None):
    '''
    Extended magnetic fitting routine that also returns residuals and the
    sensitivity matrix.

    Outputs:
    pvec, predicted, residual, mat
    '''
    if incl is None:
        incl = inc
    if decl is None:
        decl = dec

    xp, yp, zp, tf = _check_dataset(dataset)
    pvec, predicted = my_fitdata_mag(
        dataset, datashape, layermodel, layershape, regulator,
        inc, dec, incl=incl, decl=decl
    )
    residual = tf - predicted
    mat = my_totalfield_layer(xp, yp, zp, layermodel, inc, dec, incl, decl)
    return pvec, predicted, residual, mat


# -----------------------------------------------------------------------------
# MAGNETIC TRANSFORMATIONS BY EQUIVALENT LAYER
# -----------------------------------------------------------------------------

def my_transform_layer(datasets, datashape, layermodel, layershape, regulator,
                       inc, dec, newinc, newdec,
                       incl=None, decl=None, newincl=None, newdecl=None):
    '''
    General equivalent-layer magnetic transformation.

    It estimates an equivalent layer using the original field/source directions
    and then predicts the field for new field/source directions.

    Inputs:
    datasets - list - [xobs, yobs, zobs, totalfield]
    datashape - tuple - shape of data
    layermodel - list - equivalent layer
    layershape - tuple - shape of layer
    regulator - float - regularization parameter
    inc, dec - float - original inducing field direction
    newinc, newdec - float - target inducing field direction
    incl, decl - float/None - original source direction
    newincl, newdecl - float/None - target source direction

    Output:
    transformed - numpy array - transformed magnetic data
    '''
    if incl is None:
        incl = inc
    if decl is None:
        decl = dec
    if newincl is None:
        newincl = newinc
    if newdecl is None:
        newdecl = newdec

    xp, yp, zp, tf = _check_dataset(datasets)

    n_expected = _shape_size(datashape, "datashape")
    m_expected = _shape_size(layershape, "layershape")

    if xp.size != n_expected:
        raise ValueError("datashape is inconsistent with the number of observations!")
    if len(layermodel) != m_expected:
        raise ValueError("layershape is inconsistent with the number of layer sources!")

    mat = my_totalfield_layer(xp, yp, zp, layermodel, inc, dec, incl, decl)
    pvec = _solve_tikhonov(mat, tf, regulator=regulator)

    transform_mat = my_totalfield_layer(
        xp, yp, zp, layermodel, newinc, newdec, newincl, newdecl
    )

    transformed = numpy.dot(transform_mat, pvec)

    return transformed


def my_rtp_layer(datasets, datashape, layermodel, layershape, regulator,
                 inc, dec, incl=None, decl=None):
    '''
    It returns reduction-to-pole data using the equivalent layer technique.

    Inputs:
    datasets - list - [xobs, yobs, zobs, totalfield]
    datashape - tuple - shape of data
    layermodel - list - equivalent layer model
    layershape - tuple - equivalent layer shape
    regulator - float - regularization parameter
    inc, dec - float - field inclination and declination
    incl, decl - float/None - source inclination and declination

    Output:
    rtp - numpy array - reduction-to-pole data
    '''
    return my_transform_layer(
        datasets, datashape, layermodel, layershape, regulator,
        inc, dec, 90.0, 0.0,
        incl=incl, decl=decl, newincl=90.0, newdecl=0.0
    )


def my_rte_layer(datasets, datashape, layermodel, layershape, regulator,
                 incf, decf, inceql=None, deceql=None):
    '''
    It returns reduction-to-equator data using the equivalent layer technique.

    Important compatibility note:
    The original routine had variable-name inconsistencies. This revised
    function preserves the intended behavior: estimate the layer using the
    provided original field direction and predict the field at 45 deg
    inclination and 0 deg declination, matching the target used originally.

    Inputs:
    datasets - list - [xobs, yobs, zobs, totalfield]
    datashape - tuple - shape of data
    layermodel - list - equivalent layer model
    layershape - tuple - equivalent layer shape
    regulator - float - regularization parameter
    incf, decf - float - original field inclination and declination
    inceql, deceql - float/None - original layer magnetization direction

    Output:
    rteq - numpy array - transformed data
    '''
    if inceql is None:
        inceql = incf
    if deceql is None:
        deceql = decf

    return my_transform_layer(
        datasets, datashape, layermodel, layershape, regulator,
        incf, decf, 45.0, 0.0,
        incl=inceql, decl=deceql, newincl=45.0, newdecl=0.0
    )


# -----------------------------------------------------------------------------
# USEFUL DIAGNOSTIC AND QUALITY-CONTROL FUNCTIONS
# -----------------------------------------------------------------------------

def my_residual(observed, predicted):
    '''
    Return observed minus predicted data.
    '''
    observed = _as_array_1d(observed, "observed")
    predicted = _as_array_1d(predicted, "predicted")

    if observed.size != predicted.size:
        raise ValueError("observed and predicted must have the same size!")

    return observed - predicted


def my_misfit(observed, predicted):
    '''
    Return basic misfit metrics: RMSE, MAE and correlation.
    '''
    residual = my_residual(observed, predicted)
    observed = _as_array_1d(observed, "observed")
    predicted = _as_array_1d(predicted, "predicted")

    rmse = numpy.sqrt(numpy.mean(residual**2))
    mae = numpy.mean(numpy.abs(residual))

    if numpy.std(observed) == 0.0 or numpy.std(predicted) == 0.0:
        corr = numpy.nan
    else:
        corr = numpy.corrcoef(observed, predicted)[0, 1]

    return rmse, mae, corr


# -----------------------------------------------------------------------------
# COMPATIBILITY ALIASES
# -----------------------------------------------------------------------------

my_eqlayer = my_layer
my_eqlayer_array = my_layer_array
my_grav_layer = my_gz_layer
my_mag_layer = my_totalfield_layer
my_fitgrav_layer = my_fitdata_grav
my_fitmag_layer = my_fitdata_mag
