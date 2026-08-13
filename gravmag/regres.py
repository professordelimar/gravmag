# -*- coding: utf-8 -*-
"""
Regression and polynomial fitting utilities for potential-field data.

This module preserves the original public functions used in the project:

    my_poly
    my_robust_poly

and adds safer, more general routines for polynomial regional-residual
separation, linear regression, weighted least squares and Tikhonov/ridge
regularization. The central objective is to improve numerical robustness
without changing the basic interpretation of the original routines.

Coordinate convention and units are inherited from the calling code.
"""

from __future__ import division

import warnings
import numpy


# ============================================================
# BASIC VALIDATION AND SHAPE UTILITIES
# ============================================================

def _as_array(a, dtype=float):
    """
    Convert input to a numpy array without forcing flattening.
    """
    return numpy.asarray(a, dtype=dtype)


def _check_same_shape(*arrays):
    """
    Check if all arrays have the same shape.
    """
    shapes = [numpy.shape(a) for a in arrays]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError("All inputs must have the same shape!")


def _valid_mask(*arrays):
    """
    Return a mask selecting only finite values from all arrays.
    """
    mask = numpy.ones(numpy.asarray(arrays[0]).size, dtype=bool)
    for a in arrays:
        mask &= numpy.isfinite(numpy.asarray(a).ravel())
    return mask


def _restore_shape(vector, shape):
    """
    Restore a vector to the original data shape.
    """
    return numpy.asarray(vector).reshape(shape)


# ============================================================
# DESIGN MATRICES
# ============================================================

def my_polynomial_terms(degree=1, full=True):
    '''
    Return the list of polynomial powers used in a 2D polynomial model.

    Inputs:
    degree - int - polynomial degree
    full - bool - if True, includes cross terms x^i y^j with i+j <= degree.
                  if False, uses the legacy separated structure
                  [1, x, y, x^2, y^2, ...].

    Output:
    terms - list of tuples - [(i, j), ...] representing x^i y^j
    '''

    if degree < 0:
        raise ValueError("degree must be non-negative!")

    terms = []

    if full is True:
        for total_degree in range(degree + 1):
            for i in range(total_degree + 1):
                j = total_degree - i
                terms.append((i, j))
    else:
        terms.append((0, 0))
        for k in range(1, degree + 1):
            terms.append((k, 0))
            terms.append((0, k))

    return terms


def my_polynomial_design_matrix(x, y, degree=1, full=True, normalize=False):
    '''
    Build the design matrix for 2D polynomial regression.

    Inputs:
    x, y - numpy arrays - observation coordinates
    degree - int - polynomial degree
    full - bool - if True, includes cross terms
    normalize - bool - if True, internally centers and scales x and y

    Outputs:
    A - numpy array - design matrix
    terms - list - polynomial terms used
    scale_info - dict - normalization information
    '''

    x = _as_array(x)
    y = _as_array(y)
    _check_same_shape(x, y)

    xv = x.ravel()
    yv = y.ravel()

    scale_info = {
        "normalize": normalize,
        "x_mean": 0.0,
        "y_mean": 0.0,
        "x_scale": 1.0,
        "y_scale": 1.0,
    }

    if normalize is True:
        x_mean = numpy.nanmean(xv)
        y_mean = numpy.nanmean(yv)
        x_scale = numpy.nanstd(xv)
        y_scale = numpy.nanstd(yv)

        if x_scale == 0.0 or not numpy.isfinite(x_scale):
            x_scale = 1.0
        if y_scale == 0.0 or not numpy.isfinite(y_scale):
            y_scale = 1.0

        xv = (xv - x_mean)/x_scale
        yv = (yv - y_mean)/y_scale

        scale_info.update({
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_scale": x_scale,
            "y_scale": y_scale,
        })

    terms = my_polynomial_terms(degree=degree, full=full)
    A = numpy.zeros((xv.size, len(terms)), dtype=float)

    for col, (i, j) in enumerate(terms):
        A[:, col] = (xv**i)*(yv**j)

    return A, terms, scale_info


def my_linear_design_matrix(x, include_intercept=True):
    '''
    Build a design matrix for linear regression.

    Inputs:
    x - numpy array - predictor vector or matrix
    include_intercept - bool - if True, adds a column of ones

    Output:
    A - numpy array - design matrix
    '''

    x = _as_array(x)

    if x.ndim == 1:
        X = x.reshape((-1, 1))
    else:
        X = x.reshape((x.shape[0], -1))

    if include_intercept is True:
        A = numpy.column_stack([numpy.ones(X.shape[0]), X])
    else:
        A = X

    return A


# ============================================================
# LINEAR SOLVERS
# ============================================================

def my_least_squares(A, data, rcond=None):
    '''
    Solve a linear least-squares problem using numpy.linalg.lstsq.

    Inputs:
    A - numpy array - design/sensitivity matrix
    data - numpy array - observed data
    rcond - float/None - cutoff for small singular values

    Outputs:
    p - numpy array - estimated parameters
    predicted - numpy array - predicted data
    residual - numpy array - data minus predicted
    '''

    A = _as_array(A)
    d = _as_array(data).ravel()

    if A.shape[0] != d.size:
        raise ValueError("A rows must match data size!")

    mask = _valid_mask(d)
    mask &= numpy.all(numpy.isfinite(A), axis=1)

    Av = A[mask, :]
    dv = d[mask]

    p, residuals, rank, s = numpy.linalg.lstsq(Av, dv, rcond=rcond)

    predicted = A.dot(p)
    residual = d - predicted

    return p, predicted, residual


def my_weighted_least_squares(A, data, weights=None, rcond=None):
    '''
    Solve a weighted linear least-squares problem.

    Inputs:
    A - numpy array - design/sensitivity matrix
    data - numpy array - observed data
    weights - numpy array/None - data weights. If None, all weights are one.
    rcond - float/None - cutoff for small singular values

    Outputs:
    p - numpy array - estimated parameters
    predicted - numpy array - predicted data
    residual - numpy array - data minus predicted
    '''

    A = _as_array(A)
    d = _as_array(data).ravel()

    if A.shape[0] != d.size:
        raise ValueError("A rows must match data size!")

    if weights is None:
        w = numpy.ones_like(d)
    else:
        w = _as_array(weights).ravel()
        if w.size != d.size:
            raise ValueError("weights must have the same size as data!")

    mask = _valid_mask(d, w)
    mask &= numpy.all(numpy.isfinite(A), axis=1)

    Av = A[mask, :]
    dv = d[mask]
    wv = w[mask]

    wv = numpy.sqrt(numpy.maximum(wv, 0.0))

    Aw = Av*wv[:, None]
    dw = dv*wv

    p, residuals, rank, s = numpy.linalg.lstsq(Aw, dw, rcond=rcond)

    predicted = A.dot(p)
    residual = d - predicted

    return p, predicted, residual


def my_tikhonov(A, data, regulator=0.0, order=0, weights=None,
                damping_matrix=None, scale_by_trace=True):
    '''
    Solve a Tikhonov-regularized least-squares problem.

    The problem solved is approximately:

        minimize || W (A p - d) ||^2 + lambda * || L p ||^2

    Inputs:
    A - numpy array - design/sensitivity matrix
    data - numpy array - observed data
    regulator - float - regularization parameter
    order - int - regularization order. Currently 0 uses identity.
    weights - numpy array/None - data weights
    damping_matrix - numpy array/None - custom regularization matrix L
    scale_by_trace - bool - if True, scales regulator by trace(A.T A)/M

    Outputs:
    p - numpy array - estimated parameters
    predicted - numpy array - predicted data
    residual - numpy array - data minus predicted
    '''

    A = _as_array(A)
    d = _as_array(data).ravel()

    if A.shape[0] != d.size:
        raise ValueError("A rows must match data size!")

    if regulator < 0.0:
        raise ValueError("regulator must be non-negative!")

    if weights is None:
        Aw = A.copy()
        dw = d.copy()
    else:
        w = _as_array(weights).ravel()
        if w.size != d.size:
            raise ValueError("weights must have the same size as data!")
        w = numpy.sqrt(numpy.maximum(w, 0.0))
        Aw = A*w[:, None]
        dw = d*w

    mask = _valid_mask(dw)
    mask &= numpy.all(numpy.isfinite(Aw), axis=1)

    Aw = Aw[mask, :]
    dw = dw[mask]

    n_params = A.shape[1]

    if damping_matrix is not None:
        L = _as_array(damping_matrix)
        if L.shape[1] != n_params:
            raise ValueError("damping_matrix must have A.shape[1] columns!")
    else:
        if order != 0:
            warnings.warn("Only zero-order Tikhonov is implemented here. Using identity matrix.")
        L = numpy.identity(n_params)

    ATA = Aw.T.dot(Aw)
    ATd = Aw.T.dot(dw)

    if scale_by_trace is True:
        trace = numpy.trace(ATA)/float(n_params)
        if not numpy.isfinite(trace) or trace == 0.0:
            trace = 1.0
    else:
        trace = 1.0

    system = ATA + regulator*trace*L.T.dot(L)

    try:
        p = numpy.linalg.solve(system, ATd)
    except numpy.linalg.LinAlgError:
        p = numpy.linalg.lstsq(system, ATd, rcond=None)[0]

    predicted = A.dot(p)
    residual = d - predicted

    return p, predicted, residual


def my_ridge(A, data, alpha=0.0, weights=None):
    '''
    Alias for zero-order Tikhonov/ridge regression.
    '''

    return my_tikhonov(A, data, regulator=alpha, order=0, weights=weights)


# ============================================================
# POLYNOMIAL FITTING AND REGIONAL-RESIDUAL SEPARATION
# ============================================================

def my_poly(x, y, data):
    '''
    It calculates the regional and residual signal by applying a first-degree
    polynomial plane to fit the observed data.

    This function preserves the original behavior and output order:

        poly, reg, res = my_poly(x, y, data)

    Inputs:
    x, y - numpy arrays - observation points
    data - numpy array - gravity or magnetic data

    Outputs:
    poly - numpy array - polynomial coefficients [a0, ax, ay]
    reg - numpy array - regional signal, same shape as data
    res - numpy array - residual signal, same shape as data
    '''

    x = _as_array(x)
    y = _as_array(y)
    data = _as_array(data)

    _check_same_shape(x, y, data)

    original_shape = data.shape

    A = numpy.vstack((
        numpy.ones(x.size),
        x.ravel(),
        y.ravel()
    )).T

    poly, reg_vec, res_vec = my_least_squares(A, data.ravel())

    reg = _restore_shape(reg_vec, original_shape)
    res = _restore_shape(res_vec, original_shape)

    return poly, reg, res


def my_polynomial_fit(x, y, data, degree=1, full=True,
                      normalize=False, regulator=0.0,
                      weights=None, flatten=False):
    '''
    Fit a 2D polynomial surface to potential-field data.

    Inputs:
    x, y - numpy arrays - observation coordinates
    data - numpy array - observed potential-field data
    degree - int - polynomial degree
    full - bool - if True, includes cross terms
    normalize - bool - if True, centers and scales x and y internally
    regulator - float - zero-order Tikhonov parameter
    weights - numpy array/None - data weights
    flatten - bool - if True, returns regional and residual as vectors

    Outputs:
    coeffs - numpy array - polynomial coefficients
    regional - numpy array - fitted regional field
    residual - numpy array - data minus regional
    info - dict - terms and normalization metadata
    '''

    x = _as_array(x)
    y = _as_array(y)
    data = _as_array(data)

    _check_same_shape(x, y, data)

    shape = data.shape
    A, terms, scale_info = my_polynomial_design_matrix(
        x, y, degree=degree, full=full, normalize=normalize
    )

    if regulator is None or regulator == 0.0:
        coeffs, regional_vec, residual_vec = my_weighted_least_squares(
            A, data.ravel(), weights=weights
        )
    else:
        coeffs, regional_vec, residual_vec = my_tikhonov(
            A, data.ravel(), regulator=regulator, weights=weights
        )

    if flatten is True:
        regional = regional_vec
        residual = residual_vec
    else:
        regional = _restore_shape(regional_vec, shape)
        residual = _restore_shape(residual_vec, shape)

    info = {
        "degree": degree,
        "full": full,
        "terms": terms,
        "scale_info": scale_info,
        "regulator": regulator,
    }

    return coeffs, regional, residual, info


def my_robust_poly(x, y, data, degree=2, iterations=20):
    '''
    It calculates the robust polynomial fitting on regional-residual separation
    for gravity or magnetic data.

    This function keeps the original name and output order:

        poly_rob, reg_rob, res_rob = my_robust_poly(...)

    The revised implementation uses an iteratively reweighted least-squares
    scheme with Huber-like weights, which is more stable than explicitly
    building large diagonal matrices.

    Inputs:
    x, y - numpy arrays - observation points
    data - numpy array - gravity or magnetic data
    degree - int - polynomial degree
    iterations - int - number of robust iterations

    Outputs:
    poly_rob - numpy array - robust polynomial coefficients
    reg_rob - numpy array - robust regional signal
    res_rob - numpy array - robust residual signal
    '''

    coeffs, reg, res, info = my_robust_polynomial_fit(
        x, y, data,
        degree=degree,
        iterations=iterations,
        full=False,
        normalize=False,
        flatten=False
    )

    return coeffs, reg, res


def my_robust_polynomial_fit(x, y, data, degree=2, iterations=20,
                             full=True, normalize=False,
                             tuning=1.345, flatten=False):
    '''
    Robust 2D polynomial fitting using iteratively reweighted least squares.

    Inputs:
    x, y - numpy arrays - observation coordinates
    data - numpy array - observed data
    degree - int - polynomial degree
    iterations - int - maximum number of IRLS iterations
    full - bool - if True, includes cross terms
    normalize - bool - if True, centers and scales x and y internally
    tuning - float - Huber tuning constant
    flatten - bool - if True, returns regional and residual as vectors

    Outputs:
    coeffs - numpy array - robust polynomial coefficients
    regional - numpy array - robust regional field
    residual - numpy array - data minus regional
    info - dict - metadata
    '''

    x = _as_array(x)
    y = _as_array(y)
    data = _as_array(data)

    _check_same_shape(x, y, data)

    shape = data.shape
    d = data.ravel()

    A, terms, scale_info = my_polynomial_design_matrix(
        x, y, degree=degree, full=full, normalize=normalize
    )

    coeffs, predicted, residual = my_least_squares(A, d)

    weights = numpy.ones_like(d)

    for i in range(int(iterations)):
        residual = d - A.dot(coeffs)
        med = numpy.nanmedian(residual)
        mad = numpy.nanmedian(numpy.abs(residual - med))
        scale = 1.4826*mad

        if not numpy.isfinite(scale) or scale == 0.0:
            scale = numpy.nanstd(residual)
        if not numpy.isfinite(scale) or scale == 0.0:
            break

        u = residual/(tuning*scale)
        weights = numpy.ones_like(u)
        large = numpy.abs(u) > 1.0
        weights[large] = 1.0/numpy.abs(u[large])

        coeffs, predicted, residual = my_weighted_least_squares(
            A, d, weights=weights
        )

    regional_vec = A.dot(coeffs)
    residual_vec = d - regional_vec

    if flatten is True:
        regional = regional_vec
        residual_out = residual_vec
    else:
        regional = _restore_shape(regional_vec, shape)
        residual_out = _restore_shape(residual_vec, shape)

    info = {
        "degree": degree,
        "full": full,
        "terms": terms,
        "scale_info": scale_info,
        "iterations": iterations,
        "weights": weights,
    }

    return coeffs, regional, residual_out, info


# ============================================================
# SIMPLE LINEAR REGRESSION AND DIAGNOSTICS
# ============================================================

def my_linear_regression(x, data, include_intercept=True, weights=None):
    '''
    Fit a linear model to data.

    Inputs:
    x - numpy array - predictor vector or matrix
    data - numpy array - observed data
    include_intercept - bool - if True, includes intercept
    weights - numpy array/None - optional weights

    Outputs:
    coeffs - numpy array - regression coefficients
    predicted - numpy array - predicted data
    residual - numpy array - data minus predicted
    '''

    A = my_linear_design_matrix(x, include_intercept=include_intercept)

    if weights is None:
        return my_least_squares(A, data)

    return my_weighted_least_squares(A, data, weights=weights)


def my_regression_statistics(observed, predicted, n_params=None):
    '''
    Compute basic regression statistics.

    Inputs:
    observed - numpy array - observed data
    predicted - numpy array - predicted data
    n_params - int/None - number of estimated parameters

    Output:
    stats - dict - RMSE, MAE, correlation, R2 and variance information
    '''

    observed = _as_array(observed).ravel()
    predicted = _as_array(predicted).ravel()

    if observed.size != predicted.size:
        raise ValueError("observed and predicted must have the same size!")

    mask = _valid_mask(observed, predicted)
    obs = observed[mask]
    pred = predicted[mask]

    residual = obs - pred

    rmse = numpy.sqrt(numpy.mean(residual**2))
    mae = numpy.mean(numpy.abs(residual))
    bias = numpy.mean(residual)

    ss_res = numpy.sum(residual**2)
    ss_tot = numpy.sum((obs - numpy.mean(obs))**2)

    if ss_tot == 0.0:
        r2 = numpy.nan
    else:
        r2 = 1.0 - ss_res/ss_tot

    if obs.size > 1:
        corr = numpy.corrcoef(obs, pred)[0, 1]
    else:
        corr = numpy.nan

    if n_params is None:
        dof = obs.size
    else:
        dof = max(obs.size - int(n_params), 1)

    variance = ss_res/dof

    stats = {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "correlation": corr,
        "r2": r2,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "variance": variance,
        "n_data": obs.size,
        "n_params": n_params,
    }

    return stats


def my_residual(observed, predicted):
    '''
    Return observed minus predicted.
    '''

    observed = _as_array(observed)
    predicted = _as_array(predicted)
    _check_same_shape(observed, predicted)
    return observed - predicted


def my_rmse(observed, predicted):
    '''
    Return root mean square error.
    '''

    r = my_residual(observed, predicted)
    return numpy.sqrt(numpy.nanmean(r**2))


def my_mae(observed, predicted):
    '''
    Return mean absolute error.
    '''

    r = my_residual(observed, predicted)
    return numpy.nanmean(numpy.abs(r))



# ============================================================
# ADVANCED REGIONAL-RESIDUAL SEPARATION
# ============================================================

def my_safe_correlation(a, b):
    '''
    Compute correlation coefficient safely.

    Inputs:
    a, b - numpy arrays - arrays with the same size

    Output:
    corr - float - correlation coefficient
    '''

    a = _as_array(a).ravel()
    b = _as_array(b).ravel()

    if a.size != b.size:
        raise ValueError("a and b must have the same size!")

    mask = _valid_mask(a, b)

    if numpy.sum(mask) < 2:
        return numpy.nan

    av = a[mask]
    bv = b[mask]

    if numpy.nanstd(av) == 0.0 or numpy.nanstd(bv) == 0.0:
        return numpy.nan

    return float(numpy.corrcoef(av, bv)[0, 1])


def my_regional_metrics(reference, estimated):
    '''
    Compute error metrics between a reference field and an estimated field.

    Inputs:
    reference - numpy array - reference/true field
    estimated - numpy array - estimated field

    Output:
    metrics - dict - RMSE, MAE, maximum absolute error, correlation and residual
    '''

    reference = _as_array(reference)
    estimated = _as_array(estimated)

    _check_same_shape(reference, estimated)

    residual = estimated - reference

    metrics = {
        "rmse": float(numpy.sqrt(numpy.nanmean(residual**2))),
        "mae": float(numpy.nanmean(numpy.abs(residual))),
        "max_abs_error": float(numpy.nanmax(numpy.abs(residual))),
        "correlation": my_safe_correlation(reference, estimated),
        "residual": residual,
    }

    return metrics


def my_normalize_xy(x, y, mode="range"):
    '''
    Normalize x and y coordinates to improve numerical stability.

    Inputs:
    x, y - numpy arrays - coordinates
    mode - string - "range" or "std"

    Outputs:
    xn, yn - numpy arrays - normalized coordinates
    scale_info - dict - normalization information
    '''

    x = _as_array(x)
    y = _as_array(y)

    _check_same_shape(x, y)

    x_mean = numpy.nanmean(x)
    y_mean = numpy.nanmean(y)

    if mode == "range":
        x_scale = numpy.nanmax(x) - numpy.nanmin(x)
        y_scale = numpy.nanmax(y) - numpy.nanmin(y)

    elif mode == "std":
        x_scale = numpy.nanstd(x)
        y_scale = numpy.nanstd(y)

    else:
        raise ValueError("mode must be 'range' or 'std'!")

    if not numpy.isfinite(x_scale) or x_scale == 0.0:
        x_scale = 1.0

    if not numpy.isfinite(y_scale) or y_scale == 0.0:
        y_scale = 1.0

    xn = (x - x_mean)/x_scale
    yn = (y - y_mean)/y_scale

    scale_info = {
        "normalize": True,
        "mode": mode,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "x_scale": x_scale,
        "y_scale": y_scale,
    }

    return xn, yn, scale_info


def my_polynomial_terms_2d(degree=2, full=True):
    '''
    Return polynomial exponents for a 2D polynomial model.

    This function is an explicit 2D alias/extension of my_polynomial_terms.
    '''

    return my_polynomial_terms(degree=degree, full=full)


def my_polynomial_design_matrix_2d(x, y, degree=2, full=True,
                                   normalize=True, normalize_mode="range"):
    '''
    Create a 2D polynomial design matrix for regional-residual separation.

    Inputs:
    x, y - numpy arrays - coordinates
    degree - int - polynomial degree
    full - bool - if True, includes cross terms
    normalize - bool - if True, internally normalizes coordinates
    normalize_mode - string - "range" or "std"

    Output:
    result - dict - design matrix and metadata
    '''

    x = _as_array(x)
    y = _as_array(y)

    _check_same_shape(x, y)

    if normalize is True:
        xn, yn, scale_info = my_normalize_xy(x, y, mode=normalize_mode)
        xv = xn.ravel()
        yv = yn.ravel()
    else:
        scale_info = {
            "normalize": False,
            "mode": None,
            "x_mean": 0.0,
            "y_mean": 0.0,
            "x_scale": 1.0,
            "y_scale": 1.0,
        }
        xv = x.ravel()
        yv = y.ravel()

    terms = my_polynomial_terms_2d(degree=degree, full=full)

    G = numpy.zeros((xv.size, len(terms)), dtype=float)

    for col, (i, j) in enumerate(terms):
        G[:, col] = (xv**i)*(yv**j)

    return {
        "G": G,
        "terms": terms,
        "scale_info": scale_info,
        "degree": degree,
        "full": full,
        "normalize": normalize,
        "normalize_mode": normalize_mode,
    }


def my_polynomial_regularization_matrix_2d(terms, kind="damping"):
    '''
    Create a regularization matrix for polynomial coefficients.

    Inputs:
    terms - list of tuples - polynomial powers
    kind - string - "none", "damping", "gradient" or "curvature"

    Output:
    L - numpy array/None - regularization matrix
    '''

    kind = str(kind).lower()

    n_params = len(terms)

    if kind in ["none", "null", "no"]:
        return None

    if kind in ["damping", "identity", "zero"]:
        return numpy.identity(n_params)

    weights = numpy.ones(n_params, dtype=float)

    if kind in ["gradient", "order"]:
        for k, (i, j) in enumerate(terms):
            weights[k] = max(i + j, 1)

        return numpy.diag(weights)

    if kind in ["curvature", "highorder"]:
        for k, (i, j) in enumerate(terms):
            weights[k] = max((i + j)**2, 1)

        return numpy.diag(weights)

    raise ValueError("kind must be 'none', 'damping', 'gradient' or 'curvature'!")


def my_fit_polynomial_surface_2d(x, y, data, degree=2, full=True,
                                 weights=None, mask=None,
                                 normalize=True, normalize_mode="range",
                                 regulator=0.0, regularization="damping",
                                 flatten=False):
    '''
    Fit a 2D polynomial surface to gridded or scattered data.

    This is a complete regional field estimator for potential-field data.
    '''

    x = _as_array(x)
    y = _as_array(y)
    data = _as_array(data)

    _check_same_shape(x, y, data)

    original_shape = data.shape
    d_all = data.ravel()

    design = my_polynomial_design_matrix_2d(
        x=x,
        y=y,
        degree=degree,
        full=full,
        normalize=normalize,
        normalize_mode=normalize_mode
    )

    G_all = design["G"]

    valid = _valid_mask(d_all)
    valid &= numpy.all(numpy.isfinite(G_all), axis=1)

    if mask is not None:
        mask = _as_array(mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError("mask must have the same shape as data!")
        valid &= mask.ravel()

    G = G_all[valid, :]
    d = d_all[valid]

    if weights is not None:
        weights = _as_array(weights).ravel()
        if weights.size != d_all.size:
            raise ValueError("weights must have the same size as data!")
        w = weights[valid]
    else:
        w = None

    if regulator is None or regulator <= 0.0:
        coeffs, predicted_valid, residual_valid = my_weighted_least_squares(
            G, d, weights=w
        )
        solver = "weighted_least_squares" if w is not None else "least_squares"

    else:
        L = my_polynomial_regularization_matrix_2d(
            terms=design["terms"],
            kind=regularization
        )

        coeffs, predicted_valid, residual_valid = my_tikhonov(
            G,
            d,
            regulator=regulator,
            weights=w,
            damping_matrix=L,
            scale_by_trace=True
        )

        solver = "tikhonov"

    regional_vec = G_all.dot(coeffs)
    residual_vec = d_all - regional_vec

    if flatten is True:
        regional = regional_vec
        residual = residual_vec
    else:
        regional = _restore_shape(regional_vec, original_shape)
        residual = _restore_shape(residual_vec, original_shape)

    result = {
        "regional": regional,
        "residual": residual,
        "coefficients": coeffs,
        "terms": design["terms"],
        "degree": degree,
        "full": full,
        "scale_info": design["scale_info"],
        "normalize": normalize,
        "normalize_mode": normalize_mode,
        "regulator": regulator,
        "regularization": regularization,
        "solver": solver,
        "mask": mask,
        "weights": weights,
    }

    result["metrics_data_regional"] = my_regional_metrics(data, regional)

    return result


def my_remove_polynomial_regional_2d(x, y, data, degree=2, full=True,
                                     weights=None, mask=None,
                                     normalize=True, normalize_mode="range",
                                     regulator=0.0, regularization="damping",
                                     flatten=False):
    '''
    Estimate and remove a polynomial regional from potential-field data.
    '''

    return my_fit_polynomial_surface_2d(
        x=x,
        y=y,
        data=data,
        degree=degree,
        full=full,
        weights=weights,
        mask=mask,
        normalize=normalize,
        normalize_mode=normalize_mode,
        regulator=regulator,
        regularization=regularization,
        flatten=flatten
    )


def my_test_polynomial_degrees_2d(x, y, data, degrees=(1, 2, 3, 4),
                                  reference_residual=None,
                                  full=True, weights=None, mask=None,
                                  normalize=True, normalize_mode="range",
                                  regulator=0.0, regularization="damping"):
    '''
    Test different polynomial degrees for regional-residual separation.
    '''

    results_by_degree = {}
    metrics_by_degree = {}

    for degree in degrees:
        result = my_remove_polynomial_regional_2d(
            x=x,
            y=y,
            data=data,
            degree=degree,
            full=full,
            weights=weights,
            mask=mask,
            normalize=normalize,
            normalize_mode=normalize_mode,
            regulator=regulator,
            regularization=regularization,
            flatten=False
        )

        results_by_degree[degree] = result

        if reference_residual is not None:
            metrics = my_regional_metrics(reference_residual, result["residual"])
        else:
            metrics = result["metrics_data_regional"]

        metrics_by_degree[degree] = metrics

    best_degree = min(metrics_by_degree, key=lambda deg: metrics_by_degree[deg]["rmse"])

    return {
        "results_by_degree": results_by_degree,
        "metrics_by_degree": metrics_by_degree,
        "best_degree": best_degree,
        "best_result": results_by_degree[best_degree],
    }


# ============================================================
# LAGRANGE INTERPOLATION
# ============================================================

def my_lagrange_interpolate_1d(x_nodes, y_nodes, x_eval):
    '''
    Evaluate the 1D Lagrange interpolation polynomial.
    '''

    x_nodes = _as_array(x_nodes)
    y_nodes = _as_array(y_nodes)
    x_eval = _as_array(x_eval)

    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same size!")

    n = x_nodes.size
    y_eval = numpy.zeros_like(x_eval, dtype=float)

    for i in range(n):
        Li = numpy.ones_like(x_eval, dtype=float)

        for j in range(n):
            if i != j:
                Li *= (x_eval - x_nodes[j])/(x_nodes[i] - x_nodes[j])

        y_eval += y_nodes[i]*Li

    return y_eval


def my_lagrange_interpolate_2d_tensor(x_nodes, y_nodes, values, x_eval, y_eval):
    '''
    2D tensor-product Lagrange interpolation on a rectangular node grid.
    '''

    x_nodes = _as_array(x_nodes)
    y_nodes = _as_array(y_nodes)
    values = _as_array(values)
    x_eval = _as_array(x_eval)
    y_eval = _as_array(y_eval)

    if values.shape != (y_nodes.size, x_nodes.size):
        raise ValueError("values must have shape (len(y_nodes), len(x_nodes))!")

    _check_same_shape(x_eval, y_eval)

    out = numpy.zeros_like(x_eval, dtype=float)

    for iy in range(y_nodes.size):
        Ly = numpy.ones_like(y_eval, dtype=float)

        for jy in range(y_nodes.size):
            if iy != jy:
                Ly *= (y_eval - y_nodes[jy])/(y_nodes[iy] - y_nodes[jy])

        for ix in range(x_nodes.size):
            Lx = numpy.ones_like(x_eval, dtype=float)

            for jx in range(x_nodes.size):
                if ix != jx:
                    Lx *= (x_eval - x_nodes[jx])/(x_nodes[ix] - x_nodes[jx])

            out += values[iy, ix]*Lx*Ly

    return out


# ============================================================
# SAVING REGIONAL-RESIDUAL RESULTS
# ============================================================

def my_save_regional_result(filename, x, y, observed, regional, residual,
                            fmt="%.10e"):
    '''
    Save regional-residual result as text table.

    Columns:
        x y observed regional residual
    '''

    import os

    folder = os.path.dirname(filename)

    if folder not in ["", "."] and not os.path.exists(folder):
        os.makedirs(folder)

    table = numpy.column_stack([
        _as_array(x).ravel(),
        _as_array(y).ravel(),
        _as_array(observed).ravel(),
        _as_array(regional).ravel(),
        _as_array(residual).ravel(),
    ])

    numpy.savetxt(
        filename,
        table,
        fmt=fmt,
        header="x y observed regional residual"
    )

    return filename


# ============================================================
# BACKWARD-COMPATIBILITY ALIASES
# ============================================================

my_regional_residual = my_polynomial_fit
my_polyfit2d = my_polynomial_fit
my_robust_polyfit2d = my_robust_polynomial_fit
my_lstsq = my_least_squares
my_wlstsq = my_weighted_least_squares
my_tikhonov_solve = my_tikhonov
my_ridge_regression = my_ridge
my_linreg = my_linear_regression
my_stats = my_regression_statistics

my_safe_corr = my_safe_correlation
my_design_matrix_polynomial_2d = my_polynomial_design_matrix_2d
my_fit_polynomial_regional_2d = my_fit_polynomial_surface_2d
my_polynomial_regional_2d = my_fit_polynomial_surface_2d
my_remove_regional_polynomial_2d = my_remove_polynomial_regional_2d
my_regional_residual_polynomial_2d = my_remove_polynomial_regional_2d
my_test_regional_degrees_2d = my_test_polynomial_degrees_2d
my_lagrange_1d = my_lagrange_interpolate_1d
my_lagrange_2d = my_lagrange_interpolate_2d_tensor