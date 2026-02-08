mth5.processing.spectre.frequency_band_helpers
==============================================

.. py:module:: mth5.processing.spectre.frequency_band_helpers

.. autoapi-nested-parse::

   Module for tools for create and manage frequency bands.

   Bands can be defined by explicitly specifying band edges for each band, but here are some convenience
   functions for other ways to specify.



Functions
---------

.. autoapisummary::

   mth5.processing.spectre.frequency_band_helpers.half_octave
   mth5.processing.spectre.frequency_band_helpers.log_spaced_frequencies
   mth5.processing.spectre.frequency_band_helpers.bands_of_constant_q
   mth5.processing.spectre.frequency_band_helpers.partitioned_bands


Module Contents
---------------

.. py:function:: half_octave(target_frequency: float, fft_frequencies: Optional[numpy.ndarray] = None) -> mt_metadata.common.band.Band

   Create a half-octave wide frequency band object centered at target frequency.

   :type target_frequency: float
   :param target_frequency: The center frequency (geometric) of the band
   :type fft_frequencies: Optional[np.ndarray]
   :param fft_frequencies: (array-like) Frequencies associated with an instance of a spectrogram.
    If provided, the indices of the spectrogram associated with the band will be stored in the
    Band object.
   :rtype band:  mt_metadata.common.band.Band
   :return band: FrequencyBand object with lower and upper bounds.



.. py:function:: log_spaced_frequencies(f_lower_bound: float, f_upper_bound: float, num_bands: Optional[int] = None, num_bands_per_decade: Optional[float] = None, num_bands_per_octave: Optional[float] = None)

   Convenience function for generating logarithmically spaced fenceposts running
   from f_lower_bound Hz to f_upper_bound Hz.

   These can be taken, two at a time as band edges, or used as band centers with
   a constant Q scheme.  This is basically the same as np.logspace, but allows
   for specification of frequencies in Hz.

   Note in passing, replacing np.exp with 10** and np.log with np.log10 yields same base.

   :param f_lower_bound: lowest frequency under consideration
   :type f_lower_bound: float
   :param f_upper_bound: highest frequency under consideration
   :type f_upper_bound: float
   :param num_bands:
                     Total number of bands. Note that `num_bands` is one fewer than the number
                      of frequencies returned (gates and fenceposts).
   :type num_bands: int
   :param num_bands_per_decade: number of bands per decade.  8 is a nice choice.
   :type num_bands_per_decade: int (TODO test, float maybe ok also.. need to test)
   :param num_bands: total number of bands.  This supercedes num_bands_per_decade if supplied
   :type num_bands: int

   :returns: **frequencies** -- logarithmically spaced fence posts acoss lowest and highest
             frequencies.  These partition the frequency domain between
             f_lower_bound and f_upper_bound
   :rtype: array


.. py:function:: bands_of_constant_q(band_center_frequencies: numpy.ndarray, q: Optional[float] = None, fractional_bandwidth: Optional[float] = None) -> mt_metadata.processing.aurora.FrequencyBands

   Generate frequency bands centered at band_center_frequencies.
   These bands have Q = f_center/delta_f = constant.
   Normally f_center is defined geometrically, i.e. sqrt(f2*f1) is the center freq between f1 and f2.

   :param band_center_frequencies: The center frequencies for the bands
   :type band_center_frequencies: np.ndarray
   :param q: Q = f_center/delta_f = constant.
             Q is 1/fractional_bandwidth.
             Q is nonsene when less than 1, just as fractional bandwidth is nonsense when greater than 1.
             - Upper case Q is used in the literature
             See
             - https://en.wikipedia.org/wiki/Bandwidth_(signal_processing)#Fractional_bandwidth
             - https://en.wikipedia.org/wiki/Q_factor
   :type q: float

   :returns: **frequency_bands** -- Frequency bands object with bands packed inside.
   :rtype: FrequencyBands


.. py:function:: partitioned_bands(frequencies: Union[numpy.ndarray, list]) -> mt_metadata.processing.aurora.FrequencyBands

       Takes ordered list of frequencies and returns
       a FrequencyBands object

