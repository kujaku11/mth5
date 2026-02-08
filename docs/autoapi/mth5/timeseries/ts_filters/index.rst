mth5.timeseries.ts_filters
==========================

.. py:module:: mth5.timeseries.ts_filters

.. autoapi-nested-parse::

   time series filters



Classes
-------

.. autoapisummary::

   mth5.timeseries.ts_filters.RemoveInstrumentResponse


Functions
---------

.. autoapisummary::

   mth5.timeseries.ts_filters.butter_bandpass
   mth5.timeseries.ts_filters.butter_bandpass_filter
   mth5.timeseries.ts_filters.low_pass
   mth5.timeseries.ts_filters.zero_pad
   mth5.timeseries.ts_filters.adaptive_notch_filter


Module Contents
---------------

.. py:function:: butter_bandpass(lowcut, highcut, sample_rate, order=5)

   Butterworth bandpass filter using scipy.signal

   Transforms band corners to angular frequencies

   :param lowcut: low cut frequency in Hz (3dB point)
   :type lowcut: float
   :param highcut: high cut frequency in Hz (3dB point)
   :type highcut: float
   :param sample_rate: Sample rate
   :type sample_rate: float
   :param order: Butterworth order, defaults to 5
   :type order: int, optional
   :return: SOS scipy.signal format
   :rtype: scipy.signal.SOS?



.. py:function:: butter_bandpass_filter(data, lowcut, highcut, fs, order=5)

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



.. py:function:: low_pass(data, low_pass_freq, cutoff_freq, sample_rate)

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



.. py:function:: zero_pad(input_array, power=2, pad_fill=0)

   :param input_array: 1D array
   :type input_array: np.ndarray
   :param power: base power to used to pad to, defaults to 2 which is optimal
   for the FFT
   :type power: int, optional
   :param pad_fill: fill value for padded values, defaults to 0
   :type pad_fill: float, optional
   :return: zero padded array
   :rtype: np.ndarray



.. py:class:: RemoveInstrumentResponse(ts, time_array, sample_interval, channel_response, **kwargs)

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
   :param t_window: Time domain window name see `scipy.signal.windows` for options
   :type t_window: string, default None
   :param t_window_params: Time domain window parameters, parameters can be
   found in `scipy.signal.windows`
   :type t_window_params: dictionary
   :param f_window: Frequency domain window name see `scipy.signal.windows` for options
   :type f_window: string, default None
   :param f_window_params: Frequency window parameters, parameters can be
   found in `scipy.signal.windows`
   :type f_window_params: dictionary
   :param bandpass: bandpass frequency and order {"low":, "high":, "order":,}
   :type bandpass: dictionary




   .. py:attribute:: logger


   .. py:attribute:: ts


   .. py:attribute:: time_array


   .. py:attribute:: sample_interval


   .. py:attribute:: channel_response


   .. py:attribute:: plot
      :value: False



   .. py:attribute:: detrend
      :value: True



   .. py:attribute:: zero_mean
      :value: True



   .. py:attribute:: zero_pad
      :value: True



   .. py:attribute:: t_window
      :value: None



   .. py:attribute:: t_window_params


   .. py:attribute:: f_window
      :value: None



   .. py:attribute:: f_window_params


   .. py:attribute:: bandpass


   .. py:attribute:: fig
      :value: None



   .. py:attribute:: nrows
      :value: None



   .. py:attribute:: subplot_dict


   .. py:attribute:: include_decimation
      :value: False



   .. py:attribute:: include_delay
      :value: False



   .. py:method:: get_window(window, window_params, size)
      :staticmethod:


      Get window from scipy.signal

      :param window: name of the window
      :type window: string
      :param window_params: dictionary of window parameters
      :type window_params: dictionary
      :param size: number of points in the window
      :type size: integer
      :return: window function
      :rtype: class:`scipy.signal`




   .. py:method:: apply_detrend(ts)

      Detrend time series using scipy.detrend('linear')

      :param ts: input time series
      :type ts: np.ndarray
      :return: detrended time series
      :rtype: np.ndarray




   .. py:method:: apply_zero_mean(ts)

      Remove the mean from the time series

      :param ts: input time series
      :type ts: np.ndarray
      :return: zero mean time series
      :rtype: np.ndarray




   .. py:method:: apply_zero_pad(ts)

      zero pad to power of 2, at the end of the time series to make the
      FFT more efficient

      :param ts: input time series
      :type ts: np.ndarray
      :return: zero padded time series
      :rtype: np.ndarray




   .. py:method:: apply_t_window(ts)

      Apply a window in the time domain. Get the available windows from
      `scipy.signal.windows`

      :param ts: input time series
      :type ts: np.ndarray
      :return: windowed time series
      :rtype: np.ndarray




   .. py:method:: apply_f_window(data)

      Apply a frequency domain window. Get the available windows from
      `scipy.signal.windows`

      Need to create a window twice the size of the input because we are
      only taking the rfft which gives just half the spectra
      and then take only half the window

      :param data: input spectra
      :type data: np.ndarray
      :return: windowed spectra
      :rtype: np.ndarray




   .. py:method:: apply_bandpass(ts)

      apply a bandpass filter to the calibrated data


      :param ts: calibrated time series
      :type ts: np.ndarray
      :return: bandpassed time series
      :rtype: np.ndarray




   .. py:method:: remove_instrument_response(operation='divide', include_decimation=None, include_delay=None, filters_to_remove=[])

      Remove instrument response following the recipe provided

      :return: calibrated time series
      :rtype: np.ndarray




.. py:function:: adaptive_notch_filter(bx, df=100, notches=[50, 100], notchradius=0.5, freqrad=0.9, rp=0.1, dbstop_limit=5.0)

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


   .. rubric:: Example

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

   .. rubric:: Notes

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


