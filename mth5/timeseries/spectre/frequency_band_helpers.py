"""
    Module for tools for create and manage frequency bands.

    Bands can be defined by explicitly specifying band edges for each band, but here are some convenience
    functions for other ways to specify.
"""
import pandas as pd
from loguru import logger
from mt_metadata.transfer_functions.processing.aurora import Band as FrequencyBand
from mt_metadata.transfer_functions.processing.aurora import FrequencyBands
from typing import Optional, Union

import numpy as np

def half_octave(
        target_frequency: float,
        fft_frequencies: Optional[np.ndarray] = None
) -> FrequencyBand:
    """

    Create a half-octave wide frequency band object centered at target frequency.

    :type target_frequency: float
    :param target_frequency: The center frequency (geometric) of the band
    :type fft_frequencies: Optional[np.ndarray]
    :param fft_frequencies: (array-like) Frequencies associated with an instance of a spectrogram.
     If provided, the indices of the spectrogram associated with the band will be stored in the
     Band object.
    :rtype band:  mt_metadata.transfer_functions.processing.aurora.band.Band
    :return band: FrequencyBand object with lower and upper bounds.

    """
    h = 2**0.25
    f1 = target_frequency / h
    f2 = target_frequency * h
    band = FrequencyBand(frequency_min=f1, frequency_max=f2)

    if fft_frequencies is not None:
        band.set_indices_from_frequencies(fft_frequencies)

    return band

# def log_spaced_frequencies(
#     f_lower_bound: float,
#     f_upper_bound: float,
#     num_bands: Optional[int] = None,
#     num_bands_per_decade: Optional[float] = 8.0,
#     num_bands_per_octave: Optional[float] = None,
# ):
#     """
#     Convenience function for generating logarithmically spaced fenceposts running
#     from f_lower_bound Hz to f_upper_bound Hz.
#
#     These can be taken, two at a time as band edges, or used as band centers with
#     a constant Q scheme.  This is basically the same as np.logspace, but allows
#     for specification of frequencies in Hz.
#
#     Note in passing, replacing np.exp with 10** and np.log with np.log10 yields same base.
#
#     Parameters
#     ----------
#     f_lower_bound : float
#         lowest frequency under consideration
#     f_upper_bound : float
#         highest frequency under consideration
#     num_bands : int
#         Total number of bands.  This supercedes num_bands_per_decade if supplied
#     num_bands_per_decade : int (TODO test, float maybe ok also.. need to test)
#         number of bands per decade
#     num_bands : int
#         total number of bands.  This supercedes num_bands_per_decade if supplied
#
#     Returns
#     -------
#     frequencies : array
#         logarithmically spaced fence posts acoss lowest and highest
#         frequencies.  These partition the frequency domain between
#         f_lower_bound and f_upper_bound
#     """
#     band_spacing_method = None
#     if num_bands:
#         msg = f"generating {num_bands} log-spaced frequencies in range " \
#               f"{f_lower_bound}-{f_upper_bound} Hz"
#         logger.info(msg)
#         band_spacing_method = "geometric"
#
#     if num_bands_per_decade:
#         if band_spacing_method is not None:
#             msg = f"band_spacing_method already set to {band_spacing_method}"
#             msg += "Please specify only one of num_bands_per_decade, num_bands_per_octave, num_bands"
#             logger.error(msg)
#             raise ValueError(msg)
#         else:
#             number_of_decades = np.log10(f_upper_bound / f_lower_bound)
#             num_bands = round(
#                 number_of_decades * num_bands_per_decade
#             )  # floor or ceiling here?
#
#     if num_bands_per_octave:
#         if band_spacing_method is not None:
#             msg = f"band_spacing_method already set to {band_spacing_method}"
#             msg += "Please specify only one of num_bands_per_decade, num_bands_per_octave, num_bands"
#             logger.error(msg)
#             raise ValueError(msg)
#         else:
#             number_of_octaves = np.log2(f_upper_bound / f_lower_bound)
#             num_bands = round(
#                 number_of_octaves * num_bands_per_octave
#             )  # floor or ceiling here?
#
#     base = np.exp((1.0 / num_bands) * np.log(f_upper_bound / f_lower_bound))
#     bases = base * np.ones(num_bands + 1)
#     exponents = np.linspace(0, num_bands, num_bands + 1)
#     frequencies = f_lower_bound * (bases**exponents)
#
#     return frequencies
#
# def bands_of_constant_q(
#     band_center_frequencies: np.ndarray,
#     Q: Optional[float] = None
# ) -> FrequencyBands:
#     """
#         Generate frequency bands centered at band_center_frequencies.
#         These bands have
#         Given The resultant gates would have constant Q.
#          i.e. Q = delta_f/f_center = constant.
#         Normally f_center is defined geometrically, i.e. sqrt(f2*f1) is the center freq
#     between f1 and f2.
#     Parameters
#     ----------
#     frequencies_list
#     Q
#
#     Returns
#     -------
#
#     """
#     num_bands = len(band_center_frequencies)
#     lower_bounds = np.full(num_bands, np.nan)
#     upper_bounds = np.full(num_bands, np.nan)
#     for i, frq in enumerate(band_center_frequencies):
#         delta_f = frq * Q
#         lower_bounds[i] = ff - delta_f
#         upper_bounds[i] = ff + delta_f
#     band_edges_df = pd.DataFrame(
#         data={
#             'lower_bound': lower_bounds,
#             'upper_bound': upper_bounds,
#         }
#     )
#     return FrequencyBands(band_edges=band_edges_df)
#
# def partitioned_bands(frequencies_list: Union[np.ndarray, list]) -> FrequencyBands:
#     """
#         Takes ordered list of frequencies and returns
#         a FrequencyBands object
#     Returns
#     -------
#
#     """
#     num_bands = len(frequencies_list) - 1  # gates and fenceposts
#     lower_bounds = num_bands * [None]
#     upper_bounds = num_bands * [None]
#     lower_bounds = frequencies_list[:-1]
#     upper_bounds = frequencies_list[1:]
#     band_edges_df = pd.DataFrame(
#         data = {
#             'lower_bound': lower_bounds,
#             'upper_bound': upper_bounds,
#         }
#     )
#     return FrequencyBands(band_edges=band_edges_df)
#
# # def bands_at_constant_radius():
# #     pass
#
#
#
# def tst_constant_q(
#     Q: Optional[float] = None,
# ):
#     """
#     See
#     - https://en.wikipedia.org/wiki/Bandwidth_(signal_processing)#Fractional_bandwidth
#     - https://en.wikipedia.org/wiki/Q_factor
#
#
#
#     low q (Q->1 as radius approaches f, and the DC term is included)
#     Returns
#     -------
#
#     """
#     import numpy as np
#     N = 10000
#     sample_rate = 500
#     frequencies = np.fft.rfftfreq(N, 1. / sample_rate)
#     radius = 0.9
#     feature_frequencies = np.logspace(-1, 2, 40)
#     print(f"low freqs = {frequencies[0:2]}")
#     print(f"high freqs = {frequencies[-3:]}")
#     for ff in feature_frequencies:
#         delta_f = radius * ff
#         Q = ff / delta_f
#         lower_bound = ff - delta_f
#         upper_bound = ff + delta_f
#         # print(2*delta_f/ff)
#         print(Q)
#     print("OK")
#
# def measure_q():
#     f_lower_bound = 0.10
#     f_upper_bound = 1000.1
#     edges = log_spaced_frequencies(
#         f_lower_bound,
#         f_upper_bound,
#         num_bands_per_decade=None,
#         num_bands=400,
#         Q = None,
#     )
#     for i in range(len(edges) - 1):
#         band = FrequencyBand(
#             frequency_min = edges[i],
#             frequency_max = edges[i+1],
#         )
#         print(band.Q)
#         #print(band.fractional_bandwidth)
#
#     print(edges)
#     return edges
#
#
# class FrequencyBandsCreator():
#     """
#         This class can generate FrequencyBands objects based on parametric methods.
#
#         The most common will be to make logarithmically spaced bands with constant Q.
#         These may be half-octave
#     """
#     def __init__(self):
#         pass
#
# def main():
#     measure_q()
#     tst_constant_q()
#
#
#
# if __name__ == "__main__":
#     main()
