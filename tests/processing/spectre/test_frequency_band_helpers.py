"""
Proof of concept for issue #209 mulitstation FCs

TODO: Add test that builds FCs

"""
from loguru import logger
from mth5.processing.spectre.frequency_band_helpers import bands_of_constant_q
from mth5.processing.spectre.frequency_band_helpers import half_octave
from mth5.processing.spectre.frequency_band_helpers import log_spaced_frequencies

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


class TestLogSpacedFrequencies(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        params_dict = {}
        params_dict["geometric"] = {}
        params_dict["geometric"]["f_lower_bound"] = 1e-4
        params_dict["geometric"]["f_upper_bound"] = 1e4
        params_dict["geometric"]["num_bands"] = 42

        params_dict["bands per decade"] = {}
        params_dict["bands per decade"]["f_lower_bound"] = 1e-4
        params_dict["bands per decade"]["f_upper_bound"] = 1e4
        params_dict["bands per decade"]["num_bands_per_decade"] = 8.0

        params_dict["bands per octave"] = {}
        params_dict["bands per octave"]["f_lower_bound"] = 1e-4
        params_dict["bands per octave"]["f_upper_bound"] = 1e4
        params_dict["bands per octave"]["num_bands_per_octave"] = 2.2
        cls.params_dict = params_dict

    def setUp(self) -> None:
        pass

    def test_geometric_spacing(self):
        params = self.params_dict["geometric"]
        freqs = log_spaced_frequencies(
            f_lower_bound=params["f_lower_bound"],
            f_upper_bound=params["f_upper_bound"],
            num_bands=params["num_bands"]
        )
        # check that the ratios of each value to its predescessor are all equal
        assert np.isclose(freqs[1:] / freqs[:-1], freqs[1] / freqs[0]).all()
        assert len(freqs) == params["num_bands"] + 1

    def test_bands_per_decade(self):
        params = self.params_dict["bands per decade"]
        freqs = log_spaced_frequencies(
            f_lower_bound=params["f_lower_bound"],
            f_upper_bound=params["f_upper_bound"],
            num_bands_per_decade=params["num_bands_per_decade"]
        )
        # check that the ratios of each value to its predescessor are all equal
        assert np.isclose(freqs[1:] / freqs[:-1], freqs[1] / freqs[0]).all()

    def test_bands_per_octave(self):
        params = self.params_dict["bands per octave"]
        freqs = log_spaced_frequencies(
            f_lower_bound=params["f_lower_bound"],
            f_upper_bound=params["f_upper_bound"],
            num_bands_per_octave=params["num_bands_per_octave"]
        )
        # check that the ratios of each value to its predescessor are all equal
        assert np.isclose(freqs[1:] / freqs[:-1], freqs[1] / freqs[0]).all()


class TestFrequencyBandsCreation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """ Put stuff you only need to init once in here"""
        params_dict = {}
        params_dict["band_center_frequencies"] = log_spaced_frequencies(
            f_lower_bound=0.01,
            f_upper_bound=100.0,
            num_bands_per_decade=7.7,
        )
        params_dict["fractional bandwidths"] = [0.2, 0.5, 0.8]
        params_dict["Q values"] = [1/x for x in params_dict["fractional bandwidths"]]

        cls.params_dict = params_dict

    def setUp(self) -> None:
        """ Put stuff you want to reset every time in here """
        pass

    def test_bands_of_constant_q(self):
        for Q in self.params_dict["Q values"]:
            frequency_bands = bands_of_constant_q(
                band_center_frequencies=self.params_dict["band_center_frequencies"],
                q=Q
            )
            for i in range(frequency_bands.number_of_bands):
                band = frequency_bands.band(i)
                band.center_averaging_type = "arithmetic"
                assert np.isclose(band.Q, Q)
                band.center_averaging_type = "geometric"
                assert np.isclose(band.Q, Q, rtol=1e-1)




# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()

