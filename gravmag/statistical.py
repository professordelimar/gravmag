# -----------------------------------------------------------------------------------
# Title: Statistical tools
# Description: Statistical utilities for potential-field data, regression and inversion
# Author: Nelson Ribeiro Filho
# Revised with compatibility-preserving improvements
# -----------------------------------------------------------------------------------

from __future__ import division

import warnings
import numpy

try:
    from . import auxiliars
except Exception:  # pragma: no cover - compatibility with local execution
    try:
        import auxiliars
    except Exception:  # pragma: no cover
        auxiliars = None


# ============================================================
# INTERNAL UTILITIES
# ============================================================

def _as_array(data, dtype=float):
    '''
    Convert input data to a numpy array.
    '''
    return numpy.asarray(data, dtype=dtype)


def _valid_mask(*arrays):
    '''
    Return a finite-value mask shared by all arrays.
    '''
    if len(arrays) == 0:
        raise ValueError('At least one array must be provided!')

    shape = numpy.asarray(arrays[0]).shape
    mask = numpy.ones(shape, dtype=bool)

    for arr in arrays:
        arr = numpy.asarray(arr)
        if arr.shape != shape:
            raise ValueError('All arrays must have the same shape!')
        mask &= numpy.isfinite(arr)

    return mask


def _flatten_valid(*arrays):
    '''
    Flatten arrays and keep only common finite values.
    '''
    mask = _valid_mask(*arrays)
    return [numpy.asarray(arr)[mask].ravel() for arr in arrays]


# ============================================================
# BASIC DESCRIPTIVE STATISTICS
# ============================================================

def my_analysis(data, unit='(No Unit)', verbose=True, nan_policy='omit'):
    '''
    A statistical function that calculates the minimum, maximum, mean and
    variation of a dataset. The dataset can be 1D, 2D or N-dimensional.

    This function preserves the original output order:
    datamin, datamax, datamed, datavar

    Inputs:
    data - numpy array - input data
    unit - string - data unit
    verbose - bool - if True, prints the result
    nan_policy - string - 'omit' ignores NaN values; 'propagate' keeps them

    Outputs:
    datamin - float - minimum value
    datamax - float - maximum value
    datamed - float - mean value
    datavar - float - maximum minus minimum
    '''

    data = _as_array(data)

    if data.size <= 1:
        raise ValueError('Data set must have more than one element!')

    if nan_policy == 'omit':
        valid = numpy.isfinite(data)
        if numpy.sum(valid) <= 1:
            raise ValueError('Data set must have more than one finite element!')
        d = data[valid]
    elif nan_policy == 'propagate':
        d = data
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'!")

    datamin = numpy.min(d)
    datamax = numpy.max(d)
    datamed = numpy.mean(d)
    datavar = datamax - datamin

    if verbose is True:
        print('Minimum:    %5.4f %s' % (datamin, unit))
        print('Maximum:    %5.4f %s' % (datamax, unit))
        print('Mean value: %5.4f %s' % (datamed, unit))
        print('Variation:  %5.4f %s' % (datavar, unit))

    return datamin, datamax, datamed, datavar


def my_summary(data, unit='(No Unit)', percentiles=(5, 25, 50, 75, 95),
               verbose=True, nan_policy='omit'):
    '''
    Return a complete statistical summary of a dataset.

    Inputs:
    data - numpy array - input data
    unit - string - data unit
    percentiles - tuple/list - percentiles to compute
    verbose - bool - if True, prints summary
    nan_policy - string - 'omit' or 'propagate'

    Output:
    summary - dict - statistical summary
    '''

    data = _as_array(data)

    if nan_policy == 'omit':
        d = data[numpy.isfinite(data)]
    elif nan_policy == 'propagate':
        d = data.ravel()
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'!")

    if d.size == 0:
        raise ValueError('No valid data available!')

    summary = {
        'size': int(data.size),
        'valid': int(d.size),
        'nan': int(data.size - d.size),
        'min': float(numpy.min(d)),
        'max': float(numpy.max(d)),
        'mean': float(numpy.mean(d)),
        'median': float(numpy.median(d)),
        'std': float(numpy.std(d)),
        'var': float(numpy.var(d)),
        'range': float(numpy.max(d) - numpy.min(d)),
        'rms': float(numpy.sqrt(numpy.mean(d**2))),
        'mad': float(numpy.median(numpy.abs(d - numpy.median(d)))),
    }

    for p in percentiles:
        summary['p%s' % str(p)] = float(numpy.percentile(d, p))

    if verbose is True:
        print('Statistical summary %s' % unit)
        print('Size:       %d' % summary['size'])
        print('Valid:      %d' % summary['valid'])
        print('NaN:        %d' % summary['nan'])
        print('Minimum:    %g' % summary['min'])
        print('Maximum:    %g' % summary['max'])
        print('Mean:       %g' % summary['mean'])
        print('Median:     %g' % summary['median'])
        print('Std:        %g' % summary['std'])
        print('Variance:   %g' % summary['var'])
        print('Range:      %g' % summary['range'])
        print('RMS:        %g' % summary['rms'])
        print('MAD:        %g' % summary['mad'])

    return summary


def my_minmax(data, nan_policy='omit'):
    '''
    Return minimum and maximum values.
    '''
    data = _as_array(data)
    if nan_policy == 'omit':
        data = data[numpy.isfinite(data)]
    return numpy.min(data), numpy.max(data)


def my_variation(data, nan_policy='omit'):
    '''
    Return maximum minus minimum.
    '''
    dmin, dmax = my_minmax(data, nan_policy=nan_policy)
    return dmax - dmin


def my_rms(data, nan_policy='omit'):
    '''
    Return root mean square value.
    '''
    data = _as_array(data)
    if nan_policy == 'omit':
        data = data[numpy.isfinite(data)]
    return numpy.sqrt(numpy.mean(data**2))


# ============================================================
# CORRELATION, COVARIANCE AND SIMILARITY
# ============================================================

def my_correlation_coef(data1, data2, nan_policy='omit'):
    '''
    It returns the simple Pearson correlation coefficient between two data sets.
    Both data sets must have the same shape.

    Inputs:
    data1 - numpy array - first dataset
    data2 - numpy array - second dataset
    nan_policy - string - 'omit' ignores invalid pairs; 'propagate' keeps them

    Output:
    res - float - correlation coefficient
    '''

    data1 = _as_array(data1)
    data2 = _as_array(data2)

    if data1.shape != data2.shape:
        raise ValueError('All inputs must have same shape!')

    if nan_policy == 'omit':
        d1, d2 = _flatten_valid(data1, data2)
    elif nan_policy == 'propagate':
        d1 = data1.ravel()
        d2 = data2.ravel()
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'!")

    if d1.size <= 1:
        raise ValueError('Input arrays must have more than one valid element!')

    den1 = numpy.sum((d1 - numpy.mean(d1))**2)
    den2 = numpy.sum((d2 - numpy.mean(d2))**2)

    if den1 == 0.0 or den2 == 0.0:
        warnings.warn('One input has zero variance. Correlation is undefined.')
        return numpy.nan

    numerator = numpy.sum((d1 - numpy.mean(d1))*(d2 - numpy.mean(d2)))
    res = numerator/numpy.sqrt(den1*den2)

    return res


def my_covariance(data1, data2, ddof=0, nan_policy='omit'):
    '''
    Return covariance between two datasets.
    '''
    data1 = _as_array(data1)
    data2 = _as_array(data2)

    if data1.shape != data2.shape:
        raise ValueError('All inputs must have same shape!')

    if nan_policy == 'omit':
        d1, d2 = _flatten_valid(data1, data2)
    else:
        d1 = data1.ravel()
        d2 = data2.ravel()

    return numpy.sum((d1 - numpy.mean(d1))*(d2 - numpy.mean(d2)))/(d1.size - ddof)


def my_cross_correlation(data1, data2, mode='same'):
    '''
    Return 1D cross-correlation between two arrays.
    '''
    d1, d2 = _flatten_valid(data1, data2)
    d1 = d1 - numpy.mean(d1)
    d2 = d2 - numpy.mean(d2)
    return numpy.correlate(d1, d2, mode=mode)


def my_normalized_cross_correlation(data1, data2):
    '''
    Return normalized cross-correlation coefficient.
    '''
    return my_correlation_coef(data1, data2)


# ============================================================
# ERRORS, MISFIT AND REGRESSION METRICS
# ============================================================

def my_residual(observed, predicted):
    '''
    Return residual = observed - predicted.
    '''
    observed = _as_array(observed)
    predicted = _as_array(predicted)

    if observed.shape != predicted.shape:
        raise ValueError('observed and predicted must have the same shape!')

    return observed - predicted


def my_sse(observed, predicted, nan_policy='omit'):
    '''
    Return sum of squared errors.
    '''
    res = my_residual(observed, predicted)
    if nan_policy == 'omit':
        res = res[numpy.isfinite(res)]
    return numpy.sum(res**2)


def my_mse(observed, predicted, nan_policy='omit'):
    '''
    Return mean squared error.
    '''
    res = my_residual(observed, predicted)
    if nan_policy == 'omit':
        res = res[numpy.isfinite(res)]
    return numpy.mean(res**2)


def my_rmse(observed, predicted, nan_policy='omit'):
    '''
    Return root mean squared error.
    '''
    return numpy.sqrt(my_mse(observed, predicted, nan_policy=nan_policy))


def my_mae(observed, predicted, nan_policy='omit'):
    '''
    Return mean absolute error.
    '''
    res = my_residual(observed, predicted)
    if nan_policy == 'omit':
        res = res[numpy.isfinite(res)]
    return numpy.mean(numpy.abs(res))


def my_mape(observed, predicted, eps=1.0e-12, nan_policy='omit'):
    '''
    Return mean absolute percentage error in percent.
    '''
    observed = _as_array(observed)
    predicted = _as_array(predicted)

    if observed.shape != predicted.shape:
        raise ValueError('observed and predicted must have the same shape!')

    if nan_policy == 'omit':
        observed, predicted = _flatten_valid(observed, predicted)

    return 100.0*numpy.mean(numpy.abs((observed - predicted)/(observed + eps)))


def my_r2_score(observed, predicted, nan_policy='omit'):
    '''
    Return coefficient of determination R2.
    '''
    observed = _as_array(observed)
    predicted = _as_array(predicted)

    if observed.shape != predicted.shape:
        raise ValueError('observed and predicted must have the same shape!')

    if nan_policy == 'omit':
        observed, predicted = _flatten_valid(observed, predicted)
    else:
        observed = observed.ravel()
        predicted = predicted.ravel()

    ss_res = numpy.sum((observed - predicted)**2)
    ss_tot = numpy.sum((observed - numpy.mean(observed))**2)

    if ss_tot == 0.0:
        return numpy.nan

    return 1.0 - ss_res/ss_tot


def my_misfit(observed, predicted, nan_policy='omit'):
    '''
    Return a dictionary with main misfit metrics.
    '''
    return {
        'rmse': float(my_rmse(observed, predicted, nan_policy=nan_policy)),
        'mae': float(my_mae(observed, predicted, nan_policy=nan_policy)),
        'mse': float(my_mse(observed, predicted, nan_policy=nan_policy)),
        'sse': float(my_sse(observed, predicted, nan_policy=nan_policy)),
        'r2': float(my_r2_score(observed, predicted, nan_policy=nan_policy)),
        'correlation': float(my_correlation_coef(observed, predicted, nan_policy=nan_policy)),
    }


# ============================================================
# NORMALIZATION AND STANDARDIZATION
# ============================================================

def my_zscore(data, nan_policy='omit'):
    '''
    Return z-score normalized data.
    '''
    data = _as_array(data)

    if nan_policy == 'omit':
        mean = numpy.nanmean(data)
        std = numpy.nanstd(data)
    else:
        mean = numpy.mean(data)
        std = numpy.std(data)

    if std == 0.0:
        return numpy.zeros_like(data)

    return (data - mean)/std


def my_standardize(data, nan_policy='omit'):
    '''
    Alias for z-score normalization.
    '''
    return my_zscore(data, nan_policy=nan_policy)


def my_minmax_scale(data, feature_range=(0.0, 1.0), nan_policy='omit'):
    '''
    Scale data to a specified range.
    '''
    data = _as_array(data)
    a, b = feature_range

    if nan_policy == 'omit':
        dmin = numpy.nanmin(data)
        dmax = numpy.nanmax(data)
    else:
        dmin = numpy.min(data)
        dmax = numpy.max(data)

    if dmax == dmin:
        return numpy.zeros_like(data) + a

    return a + (data - dmin)*(b - a)/(dmax - dmin)


def my_normalize(data, method='zscore', nan_policy='omit'):
    '''
    Normalize data using 'zscore', 'minmax', 'maxabs' or 'rms'.
    '''
    data = _as_array(data)

    if method == 'zscore':
        return my_zscore(data, nan_policy=nan_policy)

    if method == 'minmax':
        return my_minmax_scale(data, nan_policy=nan_policy)

    if method == 'maxabs':
        scale = numpy.nanmax(numpy.abs(data)) if nan_policy == 'omit' else numpy.max(numpy.abs(data))
        if scale == 0.0:
            return numpy.zeros_like(data)
        return data/scale

    if method == 'rms':
        scale = my_rms(data, nan_policy=nan_policy)
        if scale == 0.0:
            return numpy.zeros_like(data)
        return data/scale

    raise ValueError("method must be 'zscore', 'minmax', 'maxabs' or 'rms'!")


# ============================================================
# OUTLIER DETECTION AND ROBUST STATISTICS
# ============================================================

def my_mad(data, scale=True, nan_policy='omit'):
    '''
    Return median absolute deviation.
    '''
    data = _as_array(data)
    if nan_policy == 'omit':
        data = data[numpy.isfinite(data)]
    med = numpy.median(data)
    mad = numpy.median(numpy.abs(data - med))
    if scale is True:
        mad *= 1.4826
    return mad


def my_iqr(data, nan_policy='omit'):
    '''
    Return interquartile range.
    '''
    data = _as_array(data)
    if nan_policy == 'omit':
        data = data[numpy.isfinite(data)]
    return numpy.percentile(data, 75) - numpy.percentile(data, 25)


def my_outlier_mask_zscore(data, threshold=3.0, nan_policy='omit'):
    '''
    Return boolean mask for outliers using z-score.
    '''
    z = numpy.abs(my_zscore(data, nan_policy=nan_policy))
    return z > threshold


def my_outlier_mask_iqr(data, factor=1.5, nan_policy='omit'):
    '''
    Return boolean mask for outliers using the IQR criterion.
    '''
    data = _as_array(data)
    valid = numpy.isfinite(data) if nan_policy == 'omit' else numpy.ones(data.shape, dtype=bool)
    d = data[valid]

    q1 = numpy.percentile(d, 25)
    q3 = numpy.percentile(d, 75)
    iqr = q3 - q1

    lower = q1 - factor*iqr
    upper = q3 + factor*iqr

    mask = numpy.zeros(data.shape, dtype=bool)
    mask[valid] = (d < lower) | (d > upper)

    return mask


def my_clip_outliers(data, method='iqr', threshold=3.0, factor=1.5,
                     nan_policy='omit'):
    '''
    Clip outliers using z-score or IQR thresholds.
    '''
    data = _as_array(data).copy()

    if method == 'zscore':
        mask = my_outlier_mask_zscore(data, threshold=threshold, nan_policy=nan_policy)
    elif method == 'iqr':
        mask = my_outlier_mask_iqr(data, factor=factor, nan_policy=nan_policy)
    else:
        raise ValueError("method must be 'zscore' or 'iqr'!")

    valid = numpy.isfinite(data)
    d = data[valid & (~mask)]

    if d.size == 0:
        return data

    data[data < numpy.min(d)] = numpy.min(d)
    data[data > numpy.max(d)] = numpy.max(d)

    return data


# ============================================================
# HISTOGRAMS AND DISTRIBUTIONS
# ============================================================

def my_histogram(data, bins=20, range=None, density=False, nan_policy='omit'):
    '''
    Return histogram values and bin centers.
    '''
    data = _as_array(data)
    if nan_policy == 'omit':
        data = data[numpy.isfinite(data)]

    hist, edges = numpy.histogram(data, bins=bins, range=range, density=density)
    centers = 0.5*(edges[:-1] + edges[1:])

    return hist, edges, centers


def my_cumulative_distribution(data, sort=True, nan_policy='omit'):
    '''
    Return empirical cumulative distribution.
    '''
    data = _as_array(data)
    if nan_policy == 'omit':
        data = data[numpy.isfinite(data)]

    if sort is True:
        data = numpy.sort(data.ravel())
    else:
        data = data.ravel()

    cdf = numpy.arange(1, data.size + 1, dtype=float)/data.size

    return data, cdf


# ============================================================
# MOVING-WINDOW STATISTICS FOR GRIDS
# ============================================================

def my_moving_mean(data, window_size=3, mode='nearest'):
    '''
    Return moving-window mean of a 1D or 2D array.
    '''
    try:
        from scipy.ndimage import uniform_filter
    except Exception:
        raise ImportError('scipy is required for my_moving_mean.')

    data = _as_array(data)
    return uniform_filter(data, size=window_size, mode=mode)


def my_moving_std(data, window_size=3, mode='nearest'):
    '''
    Return moving-window standard deviation of a 1D or 2D array.
    '''
    data = _as_array(data)
    mean = my_moving_mean(data, window_size=window_size, mode=mode)
    mean2 = my_moving_mean(data**2, window_size=window_size, mode=mode)
    var = mean2 - mean**2
    var[var < 0.0] = 0.0
    return numpy.sqrt(var)


def my_moving_correlation(data1, data2, window_size=3, mode='nearest', eps=1.0e-12):
    '''
    Return moving-window correlation between two 2D grids.
    '''
    data1 = _as_array(data1)
    data2 = _as_array(data2)

    if data1.shape != data2.shape:
        raise ValueError('All inputs must have same shape!')

    mean1 = my_moving_mean(data1, window_size=window_size, mode=mode)
    mean2 = my_moving_mean(data2, window_size=window_size, mode=mode)

    mean12 = my_moving_mean(data1*data2, window_size=window_size, mode=mode)
    mean11 = my_moving_mean(data1*data1, window_size=window_size, mode=mode)
    mean22 = my_moving_mean(data2*data2, window_size=window_size, mode=mode)

    cov12 = mean12 - mean1*mean2
    var1 = mean11 - mean1**2
    var2 = mean22 - mean2**2

    var1[var1 < 0.0] = 0.0
    var2[var2 < 0.0] = 0.0

    corr = cov12/(numpy.sqrt(var1*var2) + eps)

    return corr


# ============================================================
# GEOPHYSICAL MODEL QUALITY HELPERS
# ============================================================

def my_data_model_report(observed, predicted, unit='(No Unit)', verbose=True):
    '''
    Return and optionally print a report comparing observed and predicted data.
    '''
    metrics = my_misfit(observed, predicted)
    residual = my_residual(observed, predicted)
    residual_summary = my_summary(residual, unit=unit, verbose=False)

    report = {
        'metrics': metrics,
        'residual_summary': residual_summary,
    }

    if verbose is True:
        print('Data-model report %s' % unit)
        print('RMSE:        %g' % metrics['rmse'])
        print('MAE:         %g' % metrics['mae'])
        print('R2:          %g' % metrics['r2'])
        print('Correlation: %g' % metrics['correlation'])
        print('Residual min:  %g' % residual_summary['min'])
        print('Residual max:  %g' % residual_summary['max'])
        print('Residual mean: %g' % residual_summary['mean'])
        print('Residual std:  %g' % residual_summary['std'])

    return report


def my_relative_error(true, estimated, eps=1.0e-12, percent=True, nan_policy='omit'):
    '''
    Return relative error between true and estimated values.
    '''
    true = _as_array(true)
    estimated = _as_array(estimated)

    if true.shape != estimated.shape:
        raise ValueError('true and estimated must have the same shape!')

    error = (estimated - true)/(true + eps)

    if percent is True:
        error *= 100.0

    if nan_policy == 'omit':
        error[~numpy.isfinite(error)] = numpy.nan

    return error


def my_error_statistics(true, estimated, unit='(No Unit)', verbose=True):
    '''
    Return error statistics for synthetic tests.
    '''
    err = my_residual(estimated, true)  # estimated - true
    rel = my_relative_error(true, estimated, percent=True)

    report = {
        'absolute_error': my_summary(err, unit=unit, verbose=False),
        'relative_error_percent': my_summary(rel, unit='%', verbose=False),
        'rmse': float(my_rmse(true, estimated)),
        'mae': float(my_mae(true, estimated)),
        'correlation': float(my_correlation_coef(true, estimated)),
    }

    if verbose is True:
        print('Error statistics')
        print('RMSE:        %g %s' % (report['rmse'], unit))
        print('MAE:         %g %s' % (report['mae'], unit))
        print('Correlation: %g' % report['correlation'])
        print('Mean error:  %g %s' % (report['absolute_error']['mean'], unit))
        print('Max error:   %g %s' % (report['absolute_error']['max'], unit))
        print('Mean relative error: %g %%' % report['relative_error_percent']['mean'])

    return report


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

my_corrcoef = my_correlation_coef
my_corr = my_correlation_coef
my_cov = my_covariance
my_res = my_residual
my_error = my_residual
my_report = my_data_model_report
my_stats = my_summary
my_statistical_summary = my_summary
my_percent_error = my_relative_error
my_ncc = my_normalized_cross_correlation
