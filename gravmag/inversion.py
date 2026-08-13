# -*- coding: utf-8 -*-
"""
inversion.py
============

Numerical fitting, regression and geophysical inversion tools.

Author: Nelson Ribeiro Filho

This module provides practical functions for linear and nonlinear inverse problems:
least squares, weighted least squares, Tikhonov regularization, smoothness
regularization, polynomial fitting, numerical Jacobians, Gauss-Newton,
Levenberg-Marquardt, L-curve support, finite-difference operators and simple
finite-element matrices.

All main inversion functions return dictionaries containing the final result,
parameters, predicted data, residuals and, when applicable, iteration history.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple, Dict, Any, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


# ============================================================
# BASIC UTILITIES
# ============================================================

def _as_1d(a, dtype=float):
    """Convert input to a 1D numpy array."""
    return np.asarray(a, dtype=dtype).ravel()


def _as_2d(a, dtype=float):
    """Convert input to a 2D numpy array."""
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return arr


def _check_matrix_vector(G, d):
    """Check dimensions of matrix G and vector d."""
    G = _as_2d(G, dtype=float)
    d = _as_1d(d, dtype=float)
    if G.shape[0] != d.size:
        raise ValueError("G.shape[0] must be equal to d.size.")
    return G, d


def _safe_norm(x):
    """Return Euclidean norm as float."""
    return float(np.linalg.norm(np.asarray(x).ravel()))


def _safe_corr(a, b):
    """Return correlation coefficient between a and b."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size:
        raise ValueError("a and b must have the same size.")
    if np.nanstd(a) == 0.0 or np.nanstd(b) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def my_residual(observed, predicted):
    """Return residual = observed - predicted."""
    observed = _as_1d(observed)
    predicted = _as_1d(predicted)
    if observed.size != predicted.size:
        raise ValueError("observed and predicted must have the same size.")
    return observed - predicted


def my_rmse(observed, predicted):
    """Return root mean squared error."""
    r = my_residual(observed, predicted)
    return float(np.sqrt(np.nanmean(r**2)))


def my_mae(observed, predicted):
    """Return mean absolute error."""
    r = my_residual(observed, predicted)
    return float(np.nanmean(np.abs(r)))


def my_misfit_report(observed, predicted, model=None, regularizer=None):
    """Return a dictionary with inversion quality metrics."""
    observed = _as_1d(observed)
    predicted = _as_1d(predicted)
    residual = observed - predicted

    report = {
        "ndata": int(observed.size),
        "data_norm": _safe_norm(observed),
        "predicted_norm": _safe_norm(predicted),
        "residual_norm": _safe_norm(residual),
        "rmse": my_rmse(observed, predicted),
        "mae": my_mae(observed, predicted),
        "correlation": _safe_corr(observed, predicted),
        "residual": residual,
    }

    if model is not None:
        model = _as_1d(model)
        report["nparams"] = int(model.size)
        report["model_norm"] = _safe_norm(model)

    if model is not None and regularizer is not None:
        L = np.asarray(regularizer, dtype=float)
        report["regularization_norm"] = _safe_norm(L @ model)

    return report


# ============================================================
# REGULARIZATION MATRICES
# ============================================================

def my_damping_matrix(nparams: int):
    """Return zero-order damping regularization matrix L = I."""
    return np.eye(nparams, dtype=float)


def my_first_difference_matrix(nparams: int):
    """Return 1D first-order finite-difference regularization matrix."""
    if nparams < 2:
        raise ValueError("nparams must be >= 2.")
    L = np.zeros((nparams - 1, nparams), dtype=float)
    for i in range(nparams - 1):
        L[i, i] = -1.0
        L[i, i + 1] = 1.0
    return L


def my_second_difference_matrix(nparams: int):
    """Return 1D second-order finite-difference regularization matrix."""
    if nparams < 3:
        raise ValueError("nparams must be >= 3.")
    L = np.zeros((nparams - 2, nparams), dtype=float)
    for i in range(nparams - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def my_2d_first_difference_matrix(shape: Tuple[int, int]):
    """Return first-order smoothness matrix for a 2D grid model."""
    ny, nx = shape
    n = ny*nx
    rows = []

    def index(i, j):
        return i*nx + j

    for i in range(ny):
        for j in range(nx - 1):
            row = np.zeros(n, dtype=float)
            row[index(i, j)] = -1.0
            row[index(i, j + 1)] = 1.0
            rows.append(row)

    for i in range(ny - 1):
        for j in range(nx):
            row = np.zeros(n, dtype=float)
            row[index(i, j)] = -1.0
            row[index(i + 1, j)] = 1.0
            rows.append(row)

    return np.vstack(rows)


def my_2d_laplacian_matrix(shape: Tuple[int, int]):
    """Return 2D Laplacian regularization matrix for a grid model."""
    ny, nx = shape
    n = ny*nx
    rows = []

    def index(i, j):
        return i*nx + j

    for i in range(ny):
        for j in range(nx):
            row = np.zeros(n, dtype=float)
            row[index(i, j)] = -4.0
            if j > 0:
                row[index(i, j - 1)] = 1.0
            if j < nx - 1:
                row[index(i, j + 1)] = 1.0
            if i > 0:
                row[index(i - 1, j)] = 1.0
            if i < ny - 1:
                row[index(i + 1, j)] = 1.0
            rows.append(row)

    return np.vstack(rows)


def my_regularization_matrix(nparams: int, kind: str = "damping", shape: Optional[Tuple[int, int]] = None):
    """Create a regularization matrix."""
    kind = kind.lower()
    if kind in ["none", "no", "null"]:
        return None
    if kind in ["damping", "zero", "identity", "zeroth"]:
        return my_damping_matrix(nparams)
    if kind in ["first", "first_difference", "gradient", "smoothness1d"]:
        return my_first_difference_matrix(nparams)
    if kind in ["second", "second_difference", "curvature"]:
        return my_second_difference_matrix(nparams)
    if kind in ["smoothness2d", "first2d"]:
        if shape is None:
            raise ValueError("shape must be provided for 2D regularization.")
        return my_2d_first_difference_matrix(shape)
    if kind in ["laplacian2d", "second2d"]:
        if shape is None:
            raise ValueError("shape must be provided for 2D regularization.")
        return my_2d_laplacian_matrix(shape)
    raise ValueError(f"Unknown regularization kind: {kind}")


# ============================================================
# LINEAR INVERSION
# ============================================================

def my_least_squares(G, d, rcond: Optional[float] = None):
    """Solve minimize ||Gm - d||² by ordinary least squares."""
    G, d = _check_matrix_vector(G, d)
    m, residuals, rank, singular_values = np.linalg.lstsq(G, d, rcond=rcond)
    predicted = G @ m
    return {
        "method": "least_squares",
        "model": m,
        "parameters": m,
        "predicted": predicted,
        "residual": d - predicted,
        "rank": int(rank),
        "singular_values": singular_values,
        "residuals_from_lstsq": residuals,
        "report": my_misfit_report(d, predicted, model=m),
    }


def my_weighted_least_squares(G, d, weights=None, data_std=None, rcond: Optional[float] = None):
    """Solve minimize ||Wd(Gm - d)||²."""
    G, d = _check_matrix_vector(G, d)
    if data_std is not None:
        data_std = _as_1d(data_std)
        if data_std.size != d.size:
            raise ValueError("data_std must have the same size as d.")
        weights = 1.0/data_std
    if weights is None:
        weights = np.ones_like(d)
    weights = _as_1d(weights)
    if weights.size != d.size:
        raise ValueError("weights must have the same size as d.")

    Gw = G*weights[:, None]
    dw = d*weights
    m, residuals, rank, singular_values = np.linalg.lstsq(Gw, dw, rcond=rcond)
    predicted = G @ m
    return {
        "method": "weighted_least_squares",
        "model": m,
        "parameters": m,
        "predicted": predicted,
        "residual": d - predicted,
        "weights": weights,
        "rank": int(rank),
        "singular_values": singular_values,
        "residuals_from_lstsq": residuals,
        "report": my_misfit_report(d, predicted, model=m),
    }


def my_tikhonov(G, d, regulator: float = 1.0e-3, regularization: str = "damping",
                L=None, m_ref=None, weights=None, data_std=None,
                model_shape: Optional[Tuple[int, int]] = None,
                use_trace_scaling: bool = True):
    """
    Solve regularized linear inversion:

        minimize ||Wd(Gm - d)||² + λ ||L(m - m_ref)||²
    """
    G, d = _check_matrix_vector(G, d)
    ndata, nparams = G.shape

    if data_std is not None:
        data_std = _as_1d(data_std)
        if data_std.size != d.size:
            raise ValueError("data_std must have the same size as d.")
        weights = 1.0/data_std

    if weights is not None:
        weights = _as_1d(weights)
        if weights.size != d.size:
            raise ValueError("weights must have the same size as d.")
        Gw = G*weights[:, None]
        dw = d*weights
    else:
        Gw = G.copy()
        dw = d.copy()

    if L is None:
        L = my_regularization_matrix(nparams, kind=regularization, shape=model_shape)

    if L is None:
        return my_least_squares(G, d)

    L = np.asarray(L, dtype=float)
    if L.shape[1] != nparams:
        raise ValueError("L.shape[1] must be equal to the number of model parameters.")

    if m_ref is None:
        m_ref = np.zeros(nparams, dtype=float)
    else:
        m_ref = _as_1d(m_ref)
        if m_ref.size != nparams:
            raise ValueError("m_ref must have size equal to the number of parameters.")

    GTG = Gw.T @ Gw
    GTd = Gw.T @ dw
    LTL = L.T @ L

    if use_trace_scaling is True:
        trace_G = np.trace(GTG)/max(nparams, 1)
        trace_L = np.trace(LTL)/max(L.shape[0], 1)
        alpha = regulator*trace_G/trace_L if trace_L > 0.0 else regulator
    else:
        alpha = regulator

    A = GTG + alpha*LTL
    b = GTd + alpha*(LTL @ m_ref)

    try:
        m = np.linalg.solve(A, b)
        solver = "solve"
    except np.linalg.LinAlgError:
        m = np.linalg.lstsq(A, b, rcond=None)[0]
        solver = "lstsq"

    predicted = G @ m
    return {
        "method": "tikhonov",
        "model": m,
        "parameters": m,
        "predicted": predicted,
        "residual": d - predicted,
        "regularization": regularization,
        "regulator": regulator,
        "scaled_regulator": alpha,
        "regularization_matrix": L,
        "reference_model": m_ref,
        "solver": solver,
        "report": my_misfit_report(d, predicted, model=m, regularizer=L),
    }


def my_linear_inversion(G, d, method: str = "tikhonov", regulator: float = 1.0e-3,
                        regularization: str = "damping", **kwargs):
    """General interface for linear inversion."""
    method = method.lower()
    if method in ["least_squares", "lstsq", "ls"]:
        return my_least_squares(G, d, **kwargs)
    if method in ["weighted", "weighted_least_squares", "wls"]:
        return my_weighted_least_squares(G, d, **kwargs)
    if method in ["tikhonov", "ridge", "regularized"]:
        return my_tikhonov(G, d, regulator=regulator, regularization=regularization, **kwargs)
    raise ValueError(f"Unknown linear inversion method: {method}")


# ============================================================
# POLYNOMIAL AND CURVE FITTING
# ============================================================

def my_polynomial_design_matrix(x, degree: int = 1):
    """Create a 1D polynomial design matrix with columns 1, x, x², ..."""
    x = _as_1d(x)
    columns = [x**i for i in range(degree + 1)]
    return np.vstack(columns).T


def my_polynomial_fit(x, y, degree: int = 1, regulator: Optional[float] = None,
                      regularization: str = "damping"):
    """Fit a 1D polynomial model."""
    x = _as_1d(x)
    y = _as_1d(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same size.")
    G = my_polynomial_design_matrix(x, degree=degree)
    if regulator is None:
        result = my_least_squares(G, y)
    else:
        result = my_tikhonov(G, y, regulator=regulator, regularization=regularization)
    result["degree"] = degree
    result["x"] = x
    return result


def my_curve_fit_linearized(basis_functions: Sequence[Callable], x, y,
                            regulator: Optional[float] = None,
                            regularization: str = "damping"):
    """Fit y = m0*f0(x) + m1*f1(x) + ... ."""
    x = _as_1d(x)
    y = _as_1d(y)
    columns = [_as_1d(f(x)) for f in basis_functions]
    G = np.vstack(columns).T
    if G.shape[0] != y.size:
        raise ValueError("Basis functions must return arrays with the same size as y.")
    if regulator is None:
        result = my_least_squares(G, y)
    else:
        result = my_tikhonov(G, y, regulator=regulator, regularization=regularization)
    result["x"] = x
    result["basis_functions"] = basis_functions
    return result


# ============================================================
# NUMERICAL DERIVATIVES AND JACOBIAN
# ============================================================

def my_numerical_jacobian(forward_function: Callable, parameters, args: tuple = (),
                          kwargs: Optional[dict] = None,
                          step: Optional[Union[float, ArrayLike]] = None,
                          relative_step: float = 1.0e-5,
                          method: str = "central"):
    """Compute numerical Jacobian of f(parameters, *args, **kwargs)."""
    if kwargs is None:
        kwargs = {}
    p = _as_1d(parameters)
    nparams = p.size
    f0 = _as_1d(forward_function(p, *args, **kwargs))
    ndata = f0.size
    J = np.zeros((ndata, nparams), dtype=float)

    if step is None:
        h = relative_step*(np.abs(p) + 1.0)
    elif np.isscalar(step):
        h = np.full(nparams, float(step), dtype=float)
    else:
        h = _as_1d(step)
        if h.size != nparams:
            raise ValueError("step must be scalar or have the same size as parameters.")

    method = method.lower()
    for j in range(nparams):
        dp = np.zeros(nparams, dtype=float)
        dp[j] = h[j]
        if method == "central":
            fp = _as_1d(forward_function(p + dp, *args, **kwargs))
            fm = _as_1d(forward_function(p - dp, *args, **kwargs))
            J[:, j] = (fp - fm)/(2.0*h[j])
        elif method == "forward":
            fp = _as_1d(forward_function(p + dp, *args, **kwargs))
            J[:, j] = (fp - f0)/h[j]
        else:
            raise ValueError("method must be 'central' or 'forward'.")
    return J


def my_numerical_gradient(function: Callable, parameters, args: tuple = (),
                          kwargs: Optional[dict] = None,
                          step: Optional[Union[float, ArrayLike]] = None,
                          relative_step: float = 1.0e-5):
    """Compute numerical gradient of a scalar function."""
    if kwargs is None:
        kwargs = {}
    p = _as_1d(parameters)
    nparams = p.size
    if step is None:
        h = relative_step*(np.abs(p) + 1.0)
    elif np.isscalar(step):
        h = np.full(nparams, float(step), dtype=float)
    else:
        h = _as_1d(step)
    grad = np.zeros(nparams, dtype=float)
    for j in range(nparams):
        dp = np.zeros(nparams, dtype=float)
        dp[j] = h[j]
        fp = float(function(p + dp, *args, **kwargs))
        fm = float(function(p - dp, *args, **kwargs))
        grad[j] = (fp - fm)/(2.0*h[j])
    return grad


# ============================================================
# NONLINEAR INVERSION
# ============================================================

def my_objective_function(observed, predicted, parameters=None, regulator: float = 0.0,
                          L=None, m_ref=None, weights=None):
    """Compute ||Wd(d_obs-d_pred)||² + λ||L(m-m_ref)||²."""
    observed = _as_1d(observed)
    predicted = _as_1d(predicted)
    residual = observed - predicted
    residual_w = residual*_as_1d(weights) if weights is not None else residual
    phi_data = float(residual_w @ residual_w)
    phi_reg = 0.0
    if parameters is not None and L is not None and regulator > 0.0:
        parameters = _as_1d(parameters)
        L = np.asarray(L, dtype=float)
        m_ref = np.zeros_like(parameters) if m_ref is None else _as_1d(m_ref)
        reg_res = L @ (parameters - m_ref)
        phi_reg = float(regulator*(reg_res @ reg_res))
    return phi_data + phi_reg


def my_gauss_newton(forward_function: Callable, observed, initial_parameters,
                    args: tuple = (), kwargs: Optional[dict] = None,
                    max_iterations: int = 30, tolerance: float = 1.0e-6,
                    step_tolerance: float = 1.0e-8,
                    regulator: float = 0.0, regularization: str = "damping",
                    L=None, m_ref=None, weights=None,
                    jacobian_function: Optional[Callable] = None,
                    jacobian_step: Optional[Union[float, ArrayLike]] = None,
                    verbose: bool = False):
    """Nonlinear inversion using the Gauss-Newton method."""
    if kwargs is None:
        kwargs = {}
    d = _as_1d(observed)
    m = _as_1d(initial_parameters)
    nparams = m.size
    weights = np.ones_like(d) if weights is None else _as_1d(weights)
    if weights.size != d.size:
        raise ValueError("weights must have the same size as observed.")
    if L is None and regulator > 0.0:
        L = my_regularization_matrix(nparams, kind=regularization)
    if L is not None:
        L = np.asarray(L, dtype=float)
    m_ref = np.zeros_like(m) if m_ref is None else _as_1d(m_ref)
    history = []

    for iteration in range(max_iterations):
        predicted = _as_1d(forward_function(m, *args, **kwargs))
        if predicted.size != d.size:
            raise ValueError("forward_function output must have the same size as observed.")
        residual = d - predicted
        J = np.asarray(jacobian_function(m, *args, **kwargs), dtype=float) if jacobian_function is not None else my_numerical_jacobian(forward_function, m, args=args, kwargs=kwargs, step=jacobian_step)
        WJ = J*weights[:, None]
        Wr = residual*weights
        A = WJ.T @ WJ
        b = WJ.T @ Wr
        if regulator > 0.0 and L is not None:
            LTL = L.T @ L
            A = A + regulator*LTL
            b = b - regulator*(LTL @ (m - m_ref))
        try:
            dm = np.linalg.solve(A, b)
            solver = "solve"
        except np.linalg.LinAlgError:
            dm = np.linalg.lstsq(A, b, rcond=None)[0]
            solver = "lstsq"
        m_new = m + dm
        predicted_new = _as_1d(forward_function(m_new, *args, **kwargs))
        phi = my_objective_function(d, predicted, m, regulator, L, m_ref, weights)
        phi_new = my_objective_function(d, predicted_new, m_new, regulator, L, m_ref, weights)
        history.append({"iteration": iteration, "objective": phi, "objective_new": phi_new, "data_misfit": _safe_norm(residual), "step_norm": _safe_norm(dm), "model_norm": _safe_norm(m), "solver": solver, "parameters": m.copy()})
        if verbose:
            print(f"GN iter {iteration:03d} | phi={phi:.6e} | phi_new={phi_new:.6e} | step={_safe_norm(dm):.6e}")
        m = m_new
        if _safe_norm(dm) <= step_tolerance*(1.0 + _safe_norm(m)):
            break
        if abs(phi - phi_new) <= tolerance*(1.0 + abs(phi)):
            break

    predicted = _as_1d(forward_function(m, *args, **kwargs))
    return {"method": "gauss_newton", "model": m, "parameters": m, "predicted": predicted, "residual": d - predicted, "history": history, "iterations": len(history), "converged": len(history) < max_iterations, "regulator": regulator, "regularization": regularization, "regularization_matrix": L, "reference_model": m_ref, "report": my_misfit_report(d, predicted, model=m, regularizer=L)}


def my_levenberg_marquardt(forward_function: Callable, observed, initial_parameters,
                           args: tuple = (), kwargs: Optional[dict] = None,
                           max_iterations: int = 50, tolerance: float = 1.0e-6,
                           step_tolerance: float = 1.0e-8,
                           damping: float = 1.0e-3, damping_factor: float = 10.0,
                           min_damping: float = 1.0e-12, max_damping: float = 1.0e12,
                           regulator: float = 0.0, regularization: str = "damping",
                           L=None, m_ref=None, weights=None,
                           jacobian_function: Optional[Callable] = None,
                           jacobian_step: Optional[Union[float, ArrayLike]] = None,
                           verbose: bool = False):
    """Nonlinear inversion using the Levenberg-Marquardt method."""
    if kwargs is None:
        kwargs = {}
    d = _as_1d(observed)
    m = _as_1d(initial_parameters)
    nparams = m.size
    weights = np.ones_like(d) if weights is None else _as_1d(weights)
    if weights.size != d.size:
        raise ValueError("weights must have the same size as observed.")
    if L is None and regulator > 0.0:
        L = my_regularization_matrix(nparams, kind=regularization)
    if L is not None:
        L = np.asarray(L, dtype=float)
    m_ref = np.zeros_like(m) if m_ref is None else _as_1d(m_ref)
    mu = float(damping)
    history = []
    predicted = _as_1d(forward_function(m, *args, **kwargs))
    if predicted.size != d.size:
        raise ValueError("forward_function output must have the same size as observed.")
    phi = my_objective_function(d, predicted, m, regulator, L, m_ref, weights)

    for iteration in range(max_iterations):
        residual = d - predicted
        J = np.asarray(jacobian_function(m, *args, **kwargs), dtype=float) if jacobian_function is not None else my_numerical_jacobian(forward_function, m, args=args, kwargs=kwargs, step=jacobian_step)
        WJ = J*weights[:, None]
        Wr = residual*weights
        JTJ = WJ.T @ WJ
        g = WJ.T @ Wr
        diag = np.diag(np.diag(JTJ))
        if np.all(np.diag(diag) == 0.0):
            diag = np.eye(nparams)
        A = JTJ + mu*diag
        b = g.copy()
        if regulator > 0.0 and L is not None:
            LTL = L.T @ L
            A = A + regulator*LTL
            b = b - regulator*(LTL @ (m - m_ref))
        try:
            dm = np.linalg.solve(A, b)
            solver = "solve"
        except np.linalg.LinAlgError:
            dm = np.linalg.lstsq(A, b, rcond=None)[0]
            solver = "lstsq"
        m_trial = m + dm
        predicted_trial = _as_1d(forward_function(m_trial, *args, **kwargs))
        phi_trial = my_objective_function(d, predicted_trial, m_trial, regulator, L, m_ref, weights)
        accepted = phi_trial < phi
        phi_old = phi
        if accepted:
            m = m_trial
            predicted = predicted_trial
            phi = phi_trial
            mu = max(mu/damping_factor, min_damping)
        else:
            mu = min(mu*damping_factor, max_damping)
        history.append({"iteration": iteration, "objective": phi_old, "objective_trial": phi_trial, "objective_current": phi, "accepted": accepted, "damping": mu, "data_misfit": _safe_norm(d - predicted), "step_norm": _safe_norm(dm), "model_norm": _safe_norm(m), "solver": solver, "parameters": m.copy()})
        if verbose:
            status = "accepted" if accepted else "rejected"
            print(f"LM iter {iteration:03d} | phi={phi_old:.6e} | trial={phi_trial:.6e} | mu={mu:.3e} | step={_safe_norm(dm):.3e} | {status}")
        if accepted:
            if _safe_norm(dm) <= step_tolerance*(1.0 + _safe_norm(m)):
                break
            if abs(phi_old - phi) <= tolerance*(1.0 + abs(phi_old)):
                break

    return {"method": "levenberg_marquardt", "model": m, "parameters": m, "predicted": predicted, "residual": d - predicted, "history": history, "iterations": len(history), "converged": len(history) < max_iterations, "damping": mu, "regulator": regulator, "regularization": regularization, "regularization_matrix": L, "reference_model": m_ref, "report": my_misfit_report(d, predicted, model=m, regularizer=L)}


def my_nonlinear_inversion(forward_function: Callable, observed, initial_parameters,
                           method: str = "levenberg_marquardt", **kwargs):
    """General interface for nonlinear inversion."""
    method = method.lower()
    if method in ["levenberg_marquardt", "levenberg", "lm"]:
        return my_levenberg_marquardt(forward_function, observed, initial_parameters, **kwargs)
    if method in ["gauss_newton", "gn"]:
        return my_gauss_newton(forward_function, observed, initial_parameters, **kwargs)
    raise ValueError(f"Unknown nonlinear inversion method: {method}")


# ============================================================
# L-CURVE
# ============================================================

def my_lcurve(G, d, regulators, regularization: str = "damping", L=None, m_ref=None,
              weights=None, data_std=None, model_shape: Optional[Tuple[int, int]] = None):
    """Compute L-curve information for a set of regularization parameters."""
    regulators = _as_1d(regulators)
    data_norms = []
    model_norms = []
    results = []
    for reg in regulators:
        res = my_tikhonov(G, d, regulator=float(reg), regularization=regularization, L=L, m_ref=m_ref, weights=weights, data_std=data_std, model_shape=model_shape)
        data_norms.append(_safe_norm(res["residual"]))
        L_used = res["regularization_matrix"]
        m = res["model"]
        if L_used is not None:
            dm = m if m_ref is None else m - _as_1d(m_ref)
            model_norms.append(_safe_norm(L_used @ dm))
        else:
            model_norms.append(_safe_norm(m))
        results.append(res)
    return {"regulators": regulators, "data_norms": np.asarray(data_norms), "model_norms": np.asarray(model_norms), "results": results}


def my_best_lcurve_corner(lcurve_result):
    """Estimate L-curve corner using maximum discrete curvature in log-log space."""
    regs = np.asarray(lcurve_result["regulators"], dtype=float)
    x = np.log10(np.asarray(lcurve_result["data_norms"], dtype=float))
    y = np.log10(np.asarray(lcurve_result["model_norms"], dtype=float))
    if regs.size < 3:
        raise ValueError("At least 3 regulators are required to estimate curvature.")
    curvature = np.zeros_like(regs)
    for i in range(1, regs.size - 1):
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i + 1], y[i + 1]
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        if a*b*c == 0.0:
            curvature[i] = 0.0
        else:
            area = abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))/2.0
            curvature[i] = 4.0*area/(a*b*c)
    idx = int(np.nanargmax(curvature))
    return {"index": idx, "regulator": regs[idx], "curvature": curvature, "result": lcurve_result["results"][idx]}


# ============================================================
# FINITE DIFFERENCE AND FINITE ELEMENT MATRICES
# ============================================================

def my_first_derivative_matrix_1d(n: int, dx: float = 1.0, scheme: str = "central"):
    """Create a 1D first-derivative finite-difference matrix."""
    if n < 2:
        raise ValueError("n must be >= 2.")
    D = np.zeros((n, n), dtype=float)
    scheme = scheme.lower()
    if scheme == "central":
        for i in range(1, n - 1):
            D[i, i - 1] = -0.5/dx
            D[i, i + 1] = 0.5/dx
        D[0, 0] = -1.0/dx; D[0, 1] = 1.0/dx
        D[-1, -2] = -1.0/dx; D[-1, -1] = 1.0/dx
    elif scheme == "forward":
        for i in range(n - 1):
            D[i, i] = -1.0/dx; D[i, i + 1] = 1.0/dx
        D[-1, -2] = -1.0/dx; D[-1, -1] = 1.0/dx
    elif scheme == "backward":
        for i in range(1, n):
            D[i, i - 1] = -1.0/dx; D[i, i] = 1.0/dx
        D[0, 0] = -1.0/dx; D[0, 1] = 1.0/dx
    else:
        raise ValueError("scheme must be 'central', 'forward' or 'backward'.")
    return D


def my_second_derivative_matrix_1d(n: int, dx: float = 1.0):
    """Create a 1D second-derivative finite-difference matrix."""
    if n < 3:
        raise ValueError("n must be >= 3.")
    D2 = np.zeros((n, n), dtype=float)
    for i in range(1, n - 1):
        D2[i, i - 1] = 1.0/(dx**2)
        D2[i, i] = -2.0/(dx**2)
        D2[i, i + 1] = 1.0/(dx**2)
    D2[0, 0] = 1.0/(dx**2); D2[0, 1] = -2.0/(dx**2); D2[0, 2] = 1.0/(dx**2)
    D2[-1, -3] = 1.0/(dx**2); D2[-1, -2] = -2.0/(dx**2); D2[-1, -1] = 1.0/(dx**2)
    return D2


def my_laplacian_matrix_2d(shape: Tuple[int, int], dx: float = 1.0, dy: float = 1.0):
    """Create a 2D finite-difference Laplacian matrix."""
    ny, nx = shape
    n = ny*nx
    L = np.zeros((n, n), dtype=float)
    def idx(i, j): return i*nx + j
    for i in range(ny):
        for j in range(nx):
            p = idx(i, j)
            L[p, p] = -2.0/(dx**2) - 2.0/(dy**2)
            if j > 0: L[p, idx(i, j - 1)] = 1.0/(dx**2)
            if j < nx - 1: L[p, idx(i, j + 1)] = 1.0/(dx**2)
            if i > 0: L[p, idx(i - 1, j)] = 1.0/(dy**2)
            if i < ny - 1: L[p, idx(i + 1, j)] = 1.0/(dy**2)
    return L


def my_linear_fem_1d_stiffness(n_nodes: int, length: float = 1.0):
    """Create a simple 1D finite-element stiffness matrix for linear elements."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2.")
    h = length/(n_nodes - 1)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    local = (1.0/h)*np.array([[1.0, -1.0], [-1.0, 1.0]])
    for e in range(n_nodes - 1):
        K[e:e+2, e:e+2] += local
    return K


def my_linear_fem_1d_mass(n_nodes: int, length: float = 1.0):
    """Create a simple 1D finite-element mass matrix for linear elements."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2.")
    h = length/(n_nodes - 1)
    M = np.zeros((n_nodes, n_nodes), dtype=float)
    local = (h/6.0)*np.array([[2.0, 1.0], [1.0, 2.0]])
    for e in range(n_nodes - 1):
        M[e:e+2, e:e+2] += local
    return M


# ============================================================
# SAVE RESULTS
# ============================================================

def my_save_inversion_result(filename: str, result: Dict[str, Any]):
    """Save main inversion outputs to a NumPy .npz file."""
    folder = os.path.dirname(filename)
    if folder not in ["", "."] and not os.path.exists(folder):
        os.makedirs(folder)
    save_dict = {}
    for key in ["model", "parameters", "predicted", "residual"]:
        if key in result:
            save_dict[key] = np.asarray(result[key])
    if "history" in result:
        save_dict["history_object"] = np.asarray(result["history"], dtype=object)
    np.savez(filename, **save_dict)
    return filename


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

my_lstsq = my_least_squares
my_wlstsq = my_weighted_least_squares
my_tikhonov_solve = my_tikhonov
my_ridge = my_tikhonov
my_linear_regression = my_least_squares
my_linear_inverse = my_linear_inversion
my_gaussnewton = my_gauss_newton
my_lm = my_levenberg_marquardt
my_jacobian = my_numerical_jacobian
my_gradient = my_numerical_gradient
my_lcurve_corner = my_best_lcurve_corner
my_fd_first_1d = my_first_derivative_matrix_1d
my_fd_second_1d = my_second_derivative_matrix_1d
my_fd_laplacian_2d = my_laplacian_matrix_2d
