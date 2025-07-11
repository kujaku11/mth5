# -*- coding: utf-8 -*-
"""
This code is pulled from the main branch of xr-scipy
https://github.com/fujiisoup/xr-scipy/tree/master

It creates a wrapper for scipy methods for xarray.

"""
# =============================================================================
# Imports
# =============================================================================

import warnings
from fractions import Fraction
from loguru import logger
from typing import Optional, Union
import xarray as xr
import scipy.signal
import numpy as np
import pandas as pd

try:
    from scipy.signal import sosfiltfilt
except ImportError:
    sosfiltfilt = None
# =============================================================================


def _firwin_ba(*args, **kwargs):
    if not kwargs.get("pass_zero"):
        args = (args[0] + 1,) + args[1:]  # numtaps must be odd
    return scipy.signal.firwin(*args, **kwargs), np.array([1])


_BA_FUNCS = {
    "iir": scipy.signal.iirfilter,
    "fir": _firwin_ba,
}

_ORDER_DEFAULTS = {
    "iir": 8,
    "fir": 29,
}


### Warnings
class UnevenSamplingWarning(Warning):
    pass


class FilteringNaNWarning(Warning):
    pass


class DecimationWarning(Warning):
    pass


warnings.filterwarnings("always", category=UnevenSamplingWarning)
warnings.filterwarnings("always", category=FilteringNaNWarning)
warnings.filterwarnings("always", category=DecimationWarning)


def get_maybe_only_dim(darray, dim):
    """
    Check the dimension of the signal.

    Parameters
    ----------
    darray : DataArray
        An xarray DataArray.
    dim : string
        Specifies the dimension.
    """
    if dim is None:
        if len(darray.dims) == 1:
            if isinstance(darray, xr.DataArray):
                return darray.dims[0]
            elif isinstance(darray, xr.Dataset):
                return list(darray.dims.keys())[0]
        else:
            raise ValueError("Specify the dimension")
    else:
        return dim


def get_maybe_last_dim_axis(darray, dim=None):
    if dim is None:
        axis = darray.ndim - 1
        dim = darray.dims[axis]
    else:
        axis = darray.get_axis_num(dim)
    return dim, axis


def get_sampling_step(darray, dim=None, rtol=1e-3):
    """
    Approximate the sample rate in dimension given or first dimension.

    If the dimension is time, need to take into account the data type of
    the time index.  If it is nanoseconds need to divide by 1E9, etc.

    :param darray: DESCRIPTION
    :type darray: TYPE
    :param dim: DESCRIPTION, defaults to None
    :type dim: TYPE, optional
    :param rtol: DESCRIPTION, defaults to 1e-3
    :type rtol: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    dim = get_maybe_only_dim(darray, dim)

    coord = darray.coords[dim]

    if "ns" in coord.dtype.descr[0][1]:
        t_scale = 1e9
    elif "us" in coord.dtype.descr[0][1]:
        t_scale = 1e6
    elif "ms" in coord.dtype.descr[0][1]:
        t_scale = 1e3
    else:
        t_scale = 1
    dt_avg = (
        float(coord[-1] - coord[0]) / (len(coord) - 1)
    ) / t_scale  # N-1 segments
    dt_first = float(coord[1] - coord[0]) / t_scale

    if abs(dt_avg - dt_first) > rtol * min(dt_first, dt_avg):
        # show warning at caller level to see which signal it is related to
        warnings.warn(
            f"Average sampling {dt_avg:.3g} != first sampling step {dt_first:.3g}",
            UnevenSamplingWarning,
            stacklevel=2,
        )
    return dt_avg


def frequency_filter(
    darray,
    f_crit,
    order=None,
    irtype="iir",
    filtfilt=True,
    apply_kwargs=None,
    in_nyq=False,
    dim=None,
    **kwargs,
):
    """
    Applies given frequency filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by `dim`. If `darray` is multi dimensional then axis along which the filter
    is applied has to be specified by `dim` string.

    The type of the filter is chosen by `irtype` and then `filtfilt` states is
    the filter is applied both ways, forward and backward. Additional parameters
    passed to filter function specified by `apply_kwargs`.

    If 'iir' is chosen as `irtype`, then if `filtfilt` is True then the filter
    scipy.signal.filtfilt is used, if False scipy.signal.lfilter applies.

    If 'fir' is chosen as `irtype`, then if `filtfilt` is True then the filter
    scipy.signal.sosfiltfilt is used, if False scipy.signal.sosfilt applies.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_crit : array_like
        A scalar or length-2 sequence giving the critical frequencies.
    order : int, optional
        The order of the filter. If Default then it takes order defaults
        from `_ORDER_DEFAULTS`, which is `irtype` specific.
        Default is None.
    irtype : string, optional
        A string specifying the impulse response of the filter, has to be
        either "fir" then finite impulse response (FIR) is used, or "iir"
        then infinite impulse response (IIR) filter is applied. ValueError
        is raised otherwise.
        Default is "iir".
    filtfilt: bool, optional
        When True the filter is applied both forwards and backwards, otherwise
        only one way, from left to right, is applied.
        Default is True.
    apply_kwargs : dict, optional
        Specifies kwargs, which are passed to the filter function given by
        `irtype` and `filtfilt`.
        Default is None.
    in_nyq : bool, optional
        If True, then the critical frequencies given by `f_crit` are normalized
        by Nyquist frequency.
        Default is False.
    dim : string, optional
        A string specifing the dimension along which the filter is applied.
        If `darray` is 1-d then the dimension is found if not specified by `dim`.
        For multi dimensional `darray` has to be specified, otherwise ValueError
        is raised.
        Default is None.
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed,
        either to scipy.signal.iirfilter if "iir" method for `irtype` is chosen,
        or scipy.signal.firwin.
    """
    if irtype not in _BA_FUNCS:
        raise ValueError(
            "Wrong argument for irtype: {}, must be one of {}".format(
                irtype, _BA_FUNCS.keys()
            )
        )
    if order is None:
        order = _ORDER_DEFAULTS[irtype]
    if apply_kwargs is None:
        apply_kwargs = {}
    dim = get_maybe_only_dim(darray, dim)

    f_crit_norm = np.asarray(f_crit, dtype=np.float)
    if not in_nyq:  # normalize by Nyquist frequency
        f_crit_norm *= 2 * get_sampling_step(darray, dim)
    if np.any(
        np.isnan(np.asarray(darray))
    ):  # only warn since simple forward-filter or FIR is valid
        warnings.warn(
            "data contains NaNs, filter will propagate them",
            FilteringNaNWarning,
            stacklevel=2,
        )
    if sosfiltfilt and irtype == "iir":
        sos = scipy.signal.iirfilter(
            order, f_crit_norm, output="sos", **kwargs
        )
        if filtfilt:
            ret = xr.apply_ufunc(
                sosfiltfilt,
                sos,
                darray,
                input_core_dims=[[], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
        else:
            ret = xr.apply_ufunc(
                scipy.signal.sosfilt,
                sos,
                darray,
                input_core_dims=[[], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
    else:
        b, a = _BA_FUNCS[irtype](order, f_crit_norm, **kwargs)
        if filtfilt:
            ret = xr.apply_ufunc(
                scipy.signal.filtfilt,
                b,
                a,
                darray,
                input_core_dims=[[], [], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
        else:
            ret = xr.apply_ufunc(
                scipy.signal.lfilter,
                b,
                a,
                darray,
                input_core_dims=[[], [], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
    return ret


def _update_ftype_kwargs(kwargs, iirvalue, firvalue):
    if kwargs.get("irtype", "iir") == "iir":
        kwargs.setdefault("btype", iirvalue)
    else:  # fir
        kwargs.setdefault("pass_zero", firvalue)
    return kwargs


def lowpass(darray, f_cutoff, *args, **kwargs):
    """Applies lowpass filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_cutoff : array_like
        A scalar specifying the cut-off frequency for the lowpass filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "lowpass", True)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def highpass(darray, f_cutoff, *args, **kwargs):
    """Applies highpass filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_cutoff: array_like
        A scalar specifying the cut-off frequency for the highpass filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "highpass", False)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def bandpass(darray, f_low, f_high, *args, **kwargs):
    """Applies bandpass filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_low : array_like
        A scalar specifying the lower cut-off frequency for the bandpass filter.
    f_high : array_like
        A scalar specifying the higher cut-off frequency for the bandpass filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "bandpass", False)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


def bandstop(darray, f_low, f_high, *args, **kwargs):
    """Applies bandstop filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_low : array_like
        A scalar specifying the lower cut-off frequency for the bandstop filter.
    f_high : array_like
        A scalar specifying the higher cut-off frequency for the bandstop filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "bandstop", True)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


# def notch(
#     darray,
#     notch_freq,
#     notch_radius=0.5,
#     frequency_radius=0.9,
#     ripple=0.1,
#     db_stop_limit=5.0,
# ):
#     ford, wn = signal.cheb1ord(wp, ws, 1, dbstop)
#     b, a = signal.cheby1(1, 0.5, wn, btype="bandstop")
#     bx = signal.filtfilt(b, a, bx)


def decimate(
    darray: Union[xr.Dataset, xr.DataArray],
    target_sample_rate: float,
    n_order: int = 8,
    dim: Optional[str] = None
):
    """
    Decimate data following the method of scipy.signal.decimate.

    Apply a Cheyshev filter of n_order (default=8), using sosfiltfilt to
    have a zero phase output. Then down sample.

    .. note:: If the darray sample rate / target_sample_rate > 13 decimate
     should be called multiple times.

    :param darray: An xarray to be decimated
    :type darray: :class:`xr.DataArray` or :class:`xr.Dataset`
    :param target_sample_rate: new sample rate in samples per second or samples
     per space
    :type target_sample_rate: float
    :param n_order: Order of the Cheby1 filter, defaults to 8
    :type n_order: integer, optional
    :param dim: dimension to decimate along, defaults to None
    :type dim: string, optional

    :return: DESCRIPTION
    :rtype: TYPE

    """

    dim = get_maybe_only_dim(darray, dim)

    dt = get_sampling_step(darray, dim)
    q = int(np.rint(1 / (dt * target_sample_rate)))

    if q > 13:
        warnings.warn(
            f"Decimation factor is larger than 13 ({q}), the resulting "
            "decimated array maybe incorrect. Suggest calling decimate "
            "multiple times."
        )
    sos = scipy.signal.cheby1(n_order, 0.05, 0.8 / q, output="sos")

    ret = xr.apply_ufunc(
        sosfiltfilt,
        sos,
        darray,
        input_core_dims=[[], [dim]],
        output_core_dims=[[dim]],
        kwargs={},
    )

    return ret.isel(**{dim: slice(None, None, q)})


def resample_poly(darray, new_sample_rate, dim=None, pad_type="mean"):
    """
    Use scipy.signal.resample_poly

    In newer versions of scipy, need to cast data types as floats and returned
    object has data type of float, can change later if desired.

    :param new_sample_rate: The sample rate of the returned data
    :type new_sample_rate: float
    :return: DESCRIPTION
    :rtype: TYPE

    """
    dim = get_maybe_only_dim(darray, dim)
    old_sample_rate = 1.0 / get_sampling_step(darray, dim)

    fraction = Fraction(new_sample_rate / old_sample_rate).limit_denominator()

    # need to resample the time coordinate because it will change and that
    # is illegal in apply_ufunc.
    dim = get_maybe_only_dim(darray, dim)

    ret = xr.apply_ufunc(
        scipy.signal.resample_poly,
        darray.astype(float),
        fraction.numerator,
        fraction.denominator,
        input_core_dims=[[dim], [], []],
        output_core_dims=[[dim]],
        exclude_dims=set([dim]),
        kwargs={"padtype": pad_type},
    )

    dt = get_sampling_step(darray, dim)
    new_step = 1 / (dt * new_sample_rate)
    if new_step % 1 == 0:
        q = int(np.rint(new_step))
        # directly downsample without AAF on dimension
        # this only works if q is an integer, otherwise to
        # the index gets messed up from fractional spacing
        new_dim = darray[dim].values[slice(None, None, q)]

    else:
        logger.warning(
            "New sample rate is not an even number of original sample rate. "
            f"The ratio is {new_step}.  Use the new dimensions with caution."
        )
        # need to reset the end time
        end_time = darray[dim].values[0] + np.timedelta64(int(np.rint(((ret[dim].size -1) / new_sample_rate)*1E9)), "ns")
        if dim in ["time"]:
            new_dim = pd.date_range(
                darray[dim].values[0],
                end_time,
                periods=ret[dim].size,
            )
        else:
            end_index = int(np.rint((ret[dim].size - (darray[dim].size / new_step)))) - 1
            new_dim = np.linspace(
                darray[dim].values[0], darray[dim].values[end_index], ret[dim].size
            )

    # check to make sure the dimension size is the same as the new array
    n_samples_data = len(ret[dim])
    n_samples_axis = len(new_dim)
    if n_samples_data != n_samples_axis:
        logger.warning(
            f"conflicting axes sizes {n_samples_data} data and {n_samples_axis}"
            " axes after resampling"
        )
        logger.info(
            f"trimming {dim} axis from {n_samples_axis} to {n_samples_data}"
        )
        new_dim = new_dim[:n_samples_data]

    ret[dim] = new_dim

    return ret


def savgol_filter(
    darray,
    window_length,
    polyorder,
    deriv=0,
    delta=None,
    dim=None,
    mode="interp",
    cval=0.0,
):
    """Apply a Savitzky-Golay filter to an array.

    This is a 1-d filter.  If `darray` has dimension greater than 1, `dim`
    determines the dimension along which the filter is applied.
    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.  If values of `darray` are not
        a single or double precision floating point array, it will be converted
        to type ``numpy.float64`` before filtering.
    window_length : int
        The length of the filter window (i.e. the number of coefficients).
        `window_length` must be a positive odd integer. If `mode` is 'interp',
        `window_length` must be less than or equal to the size of `darray`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute.  This must be a
        nonnegative integer.  The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.  Default is 1.0.
    dim : string, optional
        Specifies the dimension along which the filter is applied. For 1-d
        darray finds the only dimension, if not specified. For multi
        dimensional darray, the dimension for the filtering has to be
        specified, otherwise raises ValueError.
        Default is None.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    Returns
    -------
    y : DataArray, same shape as `darray`
        The filtered data.
    """
    dim = get_maybe_only_dim(darray, dim)
    if delta is None:
        delta = get_sampling_step(darray, dim)
        window_length = int(np.rint(window_length / delta))
        if window_length % 2 == 0:  # must be odd
            window_length += 1
    return xr.apply_ufunc(
        scipy.signal.savgol_filter,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=dict(
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            mode=mode,
            cval=cval,
        ),
    )


def detrend(darray, dim=None, trend_type="linear"):
    """
    detrend xarray

    :param darray: DESCRIPTION
    :type darray: TYPE
    :param type: DESCRIPTION, defaults to "linear"
    :type type: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """

    dim = get_maybe_only_dim(darray, dim)

    return xr.apply_ufunc(
        scipy.signal.detrend,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs={"type": trend_type},
    )


@xr.register_dataarray_accessor("sps_filters")
@xr.register_dataset_accessor("sps_filters")
class FilterAccessor(object):
    """Accessor exposing common frequency and other filtering methods"""

    def __init__(self, darray):
        self.darray = darray

    @property
    def dt(self):
        """Sampling step of last axis"""
        return get_sampling_step(self.darray)

    @property
    def fs(self):
        """Sampling frequency in inverse units of self.dt"""
        return 1.0 / self.dt

    @property
    def dx(self):
        """Sampling steps for all axes as array"""
        return np.array(
            [get_sampling_step(self.darray, dim) for dim in self.darray.dims]
        )

    # NOTE: the arguments are coded explicitly for tab-completion to work,
    # using a decorator wrapper with *args would not expose them
    def low(self, f_cutoff, *args, **kwargs):
        """Lowpass filter, wraps lowpass"""
        return lowpass(self.darray, f_cutoff, *args, **kwargs)

    def high(self, f_cutoff, *args, **kwargs):
        """Highpass filter, wraps highpass"""
        return highpass(self.darray, f_cutoff, *args, **kwargs)

    def bandpass(self, f_low, f_high, *args, **kwargs):
        """Bandpass filter, wraps bandpass"""
        return bandpass(self.darray, f_low, f_high, *args, **kwargs)

    def bandstop(self, f_low, f_high, *args, **kwargs):
        """Bandstop filter, wraps bandstop"""
        return bandstop(self.darray, f_low, f_high, *args, **kwargs)

    def freq(
        self,
        f_crit,
        order=None,
        irtype="iir",
        filtfilt=True,
        apply_kwargs=None,
        in_nyq=False,
        dim=None,
        **kwargs,
    ):
        """General frequency filter, wraps frequency_filter"""
        return frequency_filter(
            self.darray,
            f_crit,
            order,
            irtype,
            filtfilt,
            apply_kwargs,
            in_nyq,
            dim,
            **kwargs,
        )

    __call__ = freq

    def savgol(
        self,
        window_length,
        polyorder,
        deriv=0,
        delta=None,
        dim=None,
        mode="interp",
        cval=0.0,
    ):
        """Savitzky-Golay filter, wraps savgol_filter"""
        return savgol_filter(
            self.darray,
            window_length,
            polyorder,
            deriv,
            delta,
            dim,
            mode,
            cval,
        )

    def decimate(self, target_sample_rate, n_order=8, dim=None):
        """Decimate signal, wraps decimate"""
        return decimate(self.darray, target_sample_rate, n_order, dim)

    def detrend(self, trend_type="linear", dim=None):
        """Detrend data, wraps detrend"""
        return detrend(self.darray, dim, trend_type)

    def resample_poly(self, target_sample_rate, pad_type="mean", dim=None):
        """Resample using resample_poly"""
        return resample_poly(
            self.darray, target_sample_rate, dim=dim, pad_type=pad_type
        )
