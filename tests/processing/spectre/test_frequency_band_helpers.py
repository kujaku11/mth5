"""
Proof of concept for issue #209 mulitstation FCs

TODO: Add test that builds FCs

"""
from loguru import logger
from mth5.processing.spectre.frequency_band_helpers import half_octave

import numpy as np
import unittest


class TestHalfOctave(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_initialize(self):
        half_octave_band = half_octave(target_frequency=1.0)
        n_octaves = np.log2(half_octave_band.frequency_max / half_octave_band.frequency_min)
        assert np.isclose(n_octaves, 0.5, atol=1e-2)  # Confirm you have ~0.5 octave.

    def test_fft_frequencies_argument(self):
        """

        :return:
        """
        sample_rate = 500.0  # Hz
        delta_t = 1./sample_rate
        nfft = 128
        fft_freqs = np.fft.rfftfreq(n=nfft, d=delta_t)
        half_octave_band = half_octave(
            target_frequency=sample_rate / 4.0,
            fft_frequencies=fft_freqs
        )
        n_octaves = np.log2(half_octave_band.frequency_max / half_octave_band.frequency_min)
        assert np.isclose(n_octaves, 0.5, atol=1e-2)  # Confirm you have ~0.5 octave.

    def test_fails_with_incompatible_arguments(self):
        """
            Example showing exception raised when an incompatible fftfreqs and target freq are passed
        """
        sample_rate = 500.0  # Hz
        delta_t = 1. / sample_rate
        nfft = 128
        fft_freqs = np.fft.rfftfreq(n=nfft, d=delta_t)
        self.assertRaises(IndexError, half_octave, sample_rate * 4.0, fft_freqs)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()

