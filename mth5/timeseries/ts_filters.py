#!/usr/bin/env python

"""
time series filters

"""

# =================================================================
import numpy as np
from scipy import signal

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from loguru import logger

# =================================================================


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Butterworth bandpass filter using scipy.signal

    :param lowcut: low cut frequency in Hz
    :type lowcut: float
    :param highcut: high cut frequency in Hz
    :type highcut: float
    :param fs: Sample rate
    :type fs: float
    :param order: Butterworth order, defaults to 5
    :type order: int, optional
    :return: SOS scipy.signal format
    :rtype: scipy.signal.SOS?

    """
    nyq = 0.5 * fs

    if lowcut is not None:
        low = lowcut / nyq
    if highcut is not None:
        high = highcut / nyq
    if lowcut and highcut:
        sos = signal.butter(
            order, [low, high], analog=False, btype="band", output="sos"
        )
    elif highcut is None:
        sos = signal.butter(order, low, analog=False, btype="low", output="sos")
    elif lowcut is None:
        sos = signal.butter(order, high, analog=False, btype="high", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """

    :param data: 1D time series data
    :type data: np.ndarray
    :param lowcut: low cut frequency in Hz
    :type lowcut: float
    :param highcut: high cut frequency in Hz
    :type highcut: float
    :param fs: Sample rate
    :type fs: float
    :param order: Butterworth order, defaults to 5
    :type order: int, optional
    :return: filtered data
    :rtype: np.ndarray

    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y


def low_pass(data, low_pass_freq, cutoff_freq, sampling_rate):
    """

    :param data: 1D time series data
    :type data: np.ndarray
    :param low_pass_freq: low pass frequency in Hz
    :type low_pass_freq: float
    :param cutoff_freq: cut off frequency in Hz
    :type cutoff_freq: float
    :param sampling_rate: Sample rate in samples per second
    :type sampling_rate: float
    :return: lowpass filtered data
    :rtype: np.ndarray

    """
    nyq = 0.5 * sampling_rate
    filt_order, wn = signal.buttord(low_pass_freq / nyq, cutoff_freq / nyq, 3, 40)

    b, a = signal.butter(filt_order, wn, btype="low")
    data_filtered = signal.filtfilt(b, a, data)

    return data_filtered


def zero_pad(input_array, power=2, pad_fill=0):
    """

    :param input_array: 1D array
    :type input_array: np.ndarray
    :param power: base power to used to pad to, defaults to 2 which is optimal
    for the FFT
    :type power: int, optional
    :param pad_fill: fill value for padded values, defaults to 0
    :type pad_fill: float, optional
    :return: zero padded array
    :rtype: np.ndarray

    """

    len_array = input_array.shape[0]
    if power == 2:
        npow = int(np.ceil(np.log2(len_array)))
    if power == 10:
        npow = int(np.ceil(np.log10(len_array)))
    if npow > 32:
        logger.warning(
            "Exceeding memory allocation inherent in your computer 2**32. "
            "Limiting the zero pad to 2**32"
        )
    pad_array = np.zeros(power ** npow)
    if pad_fill != 0:
        pad_array[:] = pad_fill
    pad_array[0:len_array] = input_array

    return pad_array


class RemoveInstrumentResponse:
    """
    Remove instrument response from the given channel response filter

    The order of operations is important (if applied):

        1) detrend
        2) zero mean
        3) zero pad
        4) time window
        5) frequency window
        6) remove response
        7) undo time window
        8) bandpass

    :param ts: time series data to remove response from
    :type ts: np.ndarray((N,) , dtype=float)
    :param time_array: time index that corresponds to the time series
    :type time_array: np.ndarray((N,) , dtype=np.datetime[ns])
    :param sample_interval: seconds per sample (time interval between samples)
    :type sample_interval: float
    :param channel_response: Channel response filter with all filters
    included to convert from counts to physical units
    :type channel_response: `class`:mt_metadata.timeseries.filters.ChannelResponse`

    **kwargs**

    :param plot: to plot the calibration process [ False | True ]
    :type plot: boolean, default True
    :param detrend: Remove linar trend of the time series
    :type detrend: boolean, default True
    :param zero_mean: Remove the mean of the time series
    :type zero_mean: boolean, default True
    :param zero_pad: pad the time series to the next power of 2 for efficiency
    :type zero_pad: boolean, default True
    :param t_window: Time domain windown name see `scipy.signal.windows` for options
    :type t_window: string, default None
    :param t_window_params: Time domain window parameters, parameters can be
    found in `scipy.signal.windows`
    :type t_window_params: dictionary
    :param f_window: Frequency domain windown name see `scipy.signal.windows` for options
    :type f_window: string, defualt None
    :param f_window_params: Frequency window parameters, parameters can be
    found in `scipy.signal.windows`
    :type f_window_params: dictionary
    :param bandpass: bandpass freequency and order {"low":, "high":, "order":,}
    :type bandpass: dictionary


    """

    def __init__(
        self,
        ts,
        time_array,
        sample_interval,
        channel_response,
        **kwargs,
    ):
        """

        :param ts:
        :param time_array:
        :param sample_interval:
        :param channel_response:
        :param filters_to_remove: optional list of specific filters to remove.  If not provided, filters will be
        taken from channel_response.
        """
        self.logger = logger
        self.ts = ts
        self.time_array = time_array
        self.sample_interval = sample_interval
        self.channel_response = channel_response
        self.plot = False
        self.detrend = True
        self.zero_mean = True
        self.zero_pad = True
        self.t_window = None
        self.t_window_params = {}
        self.f_window = None
        self.f_window_params = {}
        self.bandpass = {}
        self.fig = None
        self.nrows = None
        self.subplot_dict = {}
        self.include_decimation = True
        self.include_delay = False

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _subplots(self, x, y, color, num, label):
        """
        helper function to make subplots for if plotting is desired

        :param x: x array
        :type x: np.ndarray
        :param y: y array
        :type y: np.ndarray
        :param color: color of the line
        :type color: tuple
        :param num: subplot number
        :type num: integer
        :param label: legend label
        :type label: string

        """
        ax_t = self.fig.get_axes()[0]
        ax_f = self.fig.get_axes()[1]
        ax = self.fig.add_subplot(self.nrows, 2, num, sharex=ax_t)
        ax2 = self.fig.add_subplot(self.nrows, 2, num + 1, sharex=ax_f)

        # plot time series
        line = Line2D([0], [0], color=color)
        ax.plot(x, y, color=color)
        f = np.fft.rfftfreq(x.size, d=self.sample_interval)

        # plot spectra
        data = np.fft.rfft(y)
        ax2.loglog(f, abs(data), color=color)
        ax.legend(
            [line],
            [label],
            loc="upper left",
            borderaxespad=0.01,
            borderpad=0.1,
            markerscale=0.02,
        ).set_zorder(1000)

    @staticmethod
    def get_window(window, window_params, size):
        """
        Get window from scipy.signal

        :param window: name of the window
        :type window: string
        :param window_params: dictionary of window parameters
        :type window_params: dictionary
        :param size: number of points in the window
        :type size: integer
        :return: window function
        :rtype: class:`scipy.signal`

        """
        return getattr(signal.windows, window)(size, **window_params)

    def apply_detrend(self, ts):
        """
        Detrend time series using scipy.detrend('linear')

        :param ts: input time series
        :type ts: np.ndarray
        :return: detrended time series
        :rtype: np.ndarray

        """
        ts = signal.detrend(np.nan_to_num(ts), type="linear")

        if self.plot:
            self._subplots(
                self.time_array,
                ts,
                (0.45, 0.1, 0.5),
                self.subplot_dict["detrend"],
                "Detrended",
            )
        return ts

    def apply_zero_mean(self, ts):
        """
        Remove the mean from the time series

        :param ts: input time series
        :type ts: np.ndarray
        :return: zero mean time series
        :rtype: np.ndarray

        """
        ts = ts - ts.mean()

        if self.plot:
            self._subplots(
                self.time_array,
                ts,
                (0.55, 0.1, 0.4),
                self.subplot_dict["zero_mean"],
                "Zero Mean",
            )
        return ts

    def apply_zero_pad(self, ts):
        """
        zero pad to power of 2, at the end of the time series to make the
        FFT more efficient

        :param ts: input time series
        :type ts: np.ndarray
        :return: zero padded time series
        :rtype: np.ndarray

        """

        pad_ts = zero_pad(ts)

        if self.plot:
            self._subplots(
                self.time_array,
                pad_ts[0 : self.time_array.size],
                (0.7, 0.1, 0.25),
                self.subplot_dict["pad"],
                "Zero Pad",
            )
        return pad_ts

    def apply_t_window(self, ts):
        """
        Apply a window in the time domain. Get the available windows from
        `scipy.signal.windows`

        :param ts: input time series
        :type ts: np.ndarray
        :return: windowed time series
        :rtype: np.ndarray

        """
        w = self.get_window(self.t_window, self.t_window_params, self.ts.size)
        ts = ts * w

        if self.plot:
            self._subplots(
                self.time_array,
                ts,
                (0.7, 0.1, 0.25),
                self.subplot_dict["t_window"],
                f"t_window {self.t_window.capitalize()}",
            )
        return ts

    def apply_f_window(self, data):
        """
        Apply a frequency domain window. Get the available windows from
        `scipy.signal.windows`

        Need to create a window twice the size of the input because we are
        only taking the rfft which gives just half the spectra
        and then take only half the window

        :param data: input spectra
        :type data: np.ndarray
        :return: windowed spectra
        :rtype: np.ndarray

        """

        w = self.get_window(self.f_window, self.f_window_params, 2 * data.size)[
            data.size :
        ]
        data = data * w
        if self.plot:
            f = np.fft.rfftfreq(2 * data.size, d=self.sample_interval)[1:]
            ax_t = self.fig.get_axes()[0]
            ax_f = self.fig.get_axes()[1]
            ax = self.fig.add_subplot(
                self.nrows, 2, self.subplot_dict["f_window"], sharex=ax_t
            )
            ax2 = self.fig.add_subplot(
                self.nrows, 2, self.subplot_dict["f_window"] + 1, sharex=ax_f
            )

            ax.plot(
                self.time_array,
                abs(np.fft.irfft(data)[0 : self.time_array.size]),
                color=(0.85, 0.1, 0.15),
            )
            ax2.loglog(f, abs(data), color=(0.85, 0.1, 0.15))
            axt2 = ax2.twinx()
            axt2.semilogx(f, w, color=(0.5, 0.5, 0.5), zorder=0)
            line = Line2D([0], [0], color=(0.85, 0.1, 0.15))
            ax.legend(
                [line],
                [f"Freq Window {self.f_window.capitalize()}"],
                loc="upper left",
                borderaxespad=0.01,
                borderpad=0.1,
            )
        return data

    def apply_bandpass(self, ts):
        """
        apply a bandpass filter to the calibrated data


        :param ts: calibrated time series
        :type ts: np.ndarray
        :return: bandpassed time series
        :rtype: np.ndarray

        """
        ts = butter_bandpass_filter(
            ts,
            self.bandpass["low"],
            self.bandpass["high"],
            self.sample_interval,
            order=self.bandpass["order"],
        )

        if self.plot:
            self._subplots(
                self.time_array,
                ts,
                (0.875, 0.1, 0.1),
                self.subplot_dict["bandpass"],
                "Band Pass",
            )
        return ts

    def _get_subplot_count(self):
        """
        helper function to get subplot information

        :return: dictionary of subplot information
        :rtype: dictionary

        """
        order = [
            "detrend",
            "zero_mean",
            "t_window",
            "pad",
            "f_window",
            "bandpass",
        ]
        pdict = {
            "pad": self.zero_pad,
            "zero_mean": self.zero_mean,
            "detrend": self.detrend,
            "t_window": self.t_window,
            "f_window": self.f_window,
            "bandpass": self.bandpass,
        }
        subplot_dict = {}
        count = 3
        nrows = 2
        for key in order:
            value = pdict[key]
            if value:
                subplot_dict[key] = count
                count += 2
                nrows += 1
        self.nrows = nrows

        return subplot_dict

    def remove_instrument_response(self,
                                   operation="divide",
                                   include_decimation=None,
                                   include_delay=None,
                                   filters_to_remove=[]):
        """
        Remove instrument response following the recipe provided

        :return: calibrated time series
        :rtype: np.ndarray

        """
        # if filters to include not specified, get from self
        if not filters_to_remove:
            self.logger.debug("No explicit list of filters was passed to remove")
            self.logger.debug("Will determine filters to remove ... ")
            if include_decimation is None:
                include_decimation = self.include_decimation
            if include_delay is None:
                include_delay = self.include_delay
            filters_to_remove = self.channel_response.get_list_of_filters_to_remove(
                include_decimation=include_decimation, include_delay=include_delay)
            if filters_to_remove is []:
                raise ValueError("There are no filters in channel_response to remove")



        ts = np.copy(self.ts)
        f = np.fft.rfftfreq(ts.size, d=self.sample_interval)
        step = 1
        if self.plot:
            self.subplot_dict = self._get_subplot_count()
            self.fig = plt.figure(figsize=[10, 12])
            self.fig.clf()
            ax = self.fig.add_subplot(self.nrows, 2, 1)
            ax2 = self.fig.add_subplot(self.nrows, 2, 2)
            (l1,) = ax.plot(self.time_array, ts, color="k", lw=2)
            ax.set_xlim((self.time_array[0], self.time_array[-1]))
            (l2,) = ax2.loglog(f, abs(np.fft.rfft(ts)), "k", lw=2)
            ax2.set_xlim((f[0], f[-1]))
            ax.legend(
                [l1],
                ["Original"],
                loc="upper left",
                borderaxespad=0.01,
                borderpad=0.1,
            )
        # detrend
        if self.detrend:
            ts = self.apply_detrend(ts)
            self.logger.debug(f"Step {step}: Applying Linear Detrend")
            step += 1
        # zero mean
        if self.zero_mean:
            ts = self.apply_zero_mean(ts)
            self.logger.debug(f"Step {step}: Removing Mean")
            step += 1
        # filter in time domain
        if self.t_window is not None:
            ts = self.apply_t_window(ts)
            self.logger.debug(f"Step {step}: Applying {self.t_window} Time Window")
            step += 1
            if self.plot:
                wax = self.fig.get_axes()[self.subplot_dict["t_window"] - 1].twinx()
                (tw,) = wax.plot(
                    self.time_array,
                    self.get_window(self.t_window, self.t_window_params, ts.size),
                    color=(0.75, 0.75, 0.75),
                    zorder=0,
                )
        if self.zero_pad:
            # pad the time series to a power of 2, this may be overkill, especially for long time series
            ts = self.apply_zero_pad(ts)
            self.logger.debug(f"Step {step}: Applying Zero Padding")
            step += 1
        # get the real frequencies of the FFT -- zero pad may have changed ts.size
        f = np.fft.rfftfreq(ts.size, d=self.sample_interval)

        # compute the complex response given the frequency range of the FFT
        # the complex response assumes frequencies are in reverse order and flip them on input
        # so we need to flip the complex reponse so it aligns with the fft.
        cr = self.channel_response.complex_response(f,
                                                           filters_list=filters_to_remove,
                                                           )[::-1]
        # remove the DC term at frequency == 0
        cr[-1] = abs(cr[-2]) + 0.0j

        data = np.fft.rfft(ts)
        # remove DC term
        data[-1] = abs(data[-1]) + 0.0j

        # if a window is requested then create it here and mulitply by the data
        # the windows are designed to be symmetrical about frequency = 0
        # here we are taking only the real part of the FFT so we cut the window in half
        if self.f_window is not None:
            data = self.apply_f_window(data)
            self.logger.debug(f"Step {step}: Applying {self.f_window} Frequency Window")
            step += 1
        if self.channel_response.correction_operation == "divide":
        #if operation == "divide":
            # calibrate the time series, compute real part of fft, divide out
            # channel response, inverse fft
            calibrated_ts = np.fft.irfft(data / cr)[0 : self.ts.size]
            self.logger.debug(f"Step {step}: Removing Calibration by {operation}")
            step += 1
        elif self.channel_response.correction_operation == "multiply":
        # elif operation == "multiply":
            logger.warning("It is unusual to apply the instrument response as it should already be in the data")
            # calibrate the time series, compute real part of fft, multiply out
            # channel response, inverse fft
            calibrated_ts = np.fft.irfft(data * cr)[0 : self.ts.size]
            self.logger.debug(f"Step {step}: Removing Calibration  by {operation}")
            step += 1
        # If a time window was applied, need to un-apply it to reconstruct the signal.
        if self.t_window is not None:
            w = self.get_window(self.t_window, self.t_window_params, calibrated_ts.size)
            calibrated_ts = calibrated_ts / w
            self.logger.debug(f"Step {step}: Un-applying Time Window")
            step += 1
        if self.bandpass:
            calibrated_ts = self.apply_bandpass(calibrated_ts)
            self.logger.debug(f"Step {step}: Applying Bandpass Filter")
            step += 1
        if self.plot:
            self._subplots(
                self.time_array[0 : calibrated_ts.size],
                calibrated_ts,
                (1, 0.1, 0.1),
                self.nrows * 2 - 1,
                "Calibrated",
            )
            self.fig.get_axes()[-2].set_ylabel(self.channel_response.units_in)
            if self.t_window is not None:
                wax = self.fig.get_axes()[-2].twinx()
                (tw,) = wax.plot(
                    self.time_array, 1.0 / w, color=(0.5, 0.5, 0.5), zorder=0
                )
            self.fig.tight_layout()
            plt.show()
        return calibrated_ts


def adaptive_notch_filter(
    bx,
    df=100,
    notches=[50, 100],
    notchradius=0.5,
    freqrad=0.9,
    rp=0.1,
    dbstop_limit=5.0,
):
    """

    :param bx: time series to filter
    :type bx: np.ndarray
    :param df: sample rate in samples per second, defaults to 100
    :type df: float, optional
    :param notches: list of frequencies to locate notches at in Hz,
     defaults to [50, 100]
    :type notches: list, optional
    :param notchradius: notch radius, defaults to 0.5
    :type notchradius: float, optional
    :param freqrad: radius to search for a peak at the notch frequency,
     defaults to 0.9
    :type freqrad: float, optional
    :param rp: ripple of Chebyshev type 1 filter, lower numbers means less
     ripples, defaults to 0.1
    :type rp: float, optional
    :param dbstop_limit: limits the difference between the peak at the
     notch and surrounding spectra.  Any difference
     above dbstop_limit will be filtered, anything
     less will not, defaults to 5.0
    :type dbstop_limit: float, optional
    :return: notch filtered data
    :rtype: np.ndarray
    :return: list of notch frequencies
    :rtype: list


    Example
    ---------

        >>> import RemovePeriodicNoise_Kate as rmp
        >>> # make a variable for the file to load in
        >>> fn = r"/home/MT/mt01_20130101_000000.BX"
        >>> # load in file, if the time series is not an ascii file
        >>> # might need to add keywords to np.loadtxt or use another
        >>> # method to read in the file
        >>> bx = np.loadtxt(fn)
        >>> # create a list of frequencies to filter out
        >>> freq_notches = [50, 150, 200]
        >>> # filter data
        >>> bx_filt, filt_lst = rmp.adaptiveNotchFilter(bx, df=100.
        >>> ...                                         notches=freq_notches)
        >>> #save the filtered data into a file
        >>> np.savetxt(r"/home/MT/Filtered/mt01_20130101_000000.BX", bx_filt)

    Notes:

        Most of the time the default parameters work well, the only thing
        you need to change is the notches and perhaps the radius.  I would
        test it out with a few time series to find the optimum parameters.
        Then make a loop over all you time series data. Something like

        >>> import os
        >>> dirpath = r"/home/MT"
        >>> #make a director to save filtered time series
        >>> save_path = r"/home/MT/Filtered"
        >>> if not os.path.exists(save_path):
        >>>     os.mkdir(save_path)
        >>> for fn in os.listdir(dirpath):
        >>>     bx = np.loadtxt(os.path.join(dirpath, fn)
        >>>     bx_filt, filt_lst = rmp.adaptiveNotchFilter(bx, df=100.
        >>>     ...                                         notches=freq_notches)
        >>>     np.savetxt(os.path.join(save_path, fn), bx_filt)

    """

    bx = np.array(bx)

    if type(notches) is list:
        notches = np.array(notches)
    elif type(notches) in [float, int]:
        notches = np.array([notches], dtype=np.float)
    df = float(df)  # make sure df is a float
    dt = 1.0 / df  # sampling rate

    # transform data into frequency domain to find notches
    BX = np.fft.fft(zero_pad(bx))
    n = len(BX)  # length of array
    dfn = df / n  # frequency step
    dfnn = int(freqrad / dfn)  # radius of frequency search
    fn = notchradius  # filter radius
    freq = np.fft.fftfreq(n, dt)

    filtlst = []
    for notch in notches:
        if notch > freq.max():
            break
        else:
            fspot = int(round(notch / dfn))
            nspot = np.where(
                abs(BX) == max(abs(BX[max([fspot - dfnn, 0]) : min([fspot + dfnn, n])]))
            )[0][0]

            med_bx = np.median(
                abs(BX[max([nspot - dfnn * 10, 0]) : min([nspot + dfnn * 10, n])]) ** 2
            )

            # calculate difference between peak and surrounding spectra in dB
            dbstop = 10 * np.log10(abs(BX[nspot]) ** 2 / med_bx)
            if np.nan_to_num(dbstop) == 0.0 or dbstop < dbstop_limit:
                filtlst.append("No need to filter \n")
                pass
            else:
                filtlst.append([freq[nspot], dbstop])
                ws = 2 * np.array([freq[nspot] - fn, freq[nspot] + fn]) / df
                wp = 2 * np.array([freq[nspot] - 2 * fn, freq[nspot] + 2 * fn]) / df
                ford, wn = signal.cheb1ord(wp, ws, 1, dbstop)
                b, a = signal.cheby1(1, 0.5, wn, btype="bandstop")
                bx = signal.filtfilt(b, a, bx)
    return bx, filtlst
