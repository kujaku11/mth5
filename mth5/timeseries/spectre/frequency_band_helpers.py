"""
    Module for tools for create and manage frequency bands.
"""

from mt_metadata.transfer_functions.processing.aurora import Band as FrequencyBand
from typing import Optional

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
