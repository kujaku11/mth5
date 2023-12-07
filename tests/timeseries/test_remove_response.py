# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:18:08 2021

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
import numpy as np

from mt_metadata.timeseries.filters import PoleZeroFilter
from mth5.timeseries import ChannelTS

# =============================================================================
#
# =============================================================================


class TestRemoveResponse(unittest.TestCase):
    """
    Test remove response, make a fake signal add some trends,
    ToDo:
    - Add more than one filter to the test.
    - Utilize the is_channel_response attributes
    """

    @classmethod
    def setUpClass(self):
        # pole zero filter
        pz1 = PoleZeroFilter(
            units_in="volts", units_out="nanotesla", name="instrument_response"
        )
        pz1.poles = [
            (-6.283185 + 10.882477j),
            (-6.283185 - 10.882477j),
            (-12.566371 + 0j),
        ]
        pz1.zeros = []
        pz1.normalization_factor = 18244400
        pz2 = PoleZeroFilter(
            units_in="nanotesla", units_out="volts", name="instrument_response2"
        )
        pz2.poles = []
        pz2.zeros = []
        pz2.normalization_factor = 10
        pz2.gain = 12
        # channel properties
        self.channel = ChannelTS()
        self.channel.channel_metadata.filter.applied = [True, ]#False]
        self.channel.channel_metadata.filter.name = ["instrument_response",]# "instrument_response2"]
        self.channel.channel_metadata.component = "hx"
        self.channel.channel_metadata.units = "digital counts"
        self.channel.channel_response_filter.filters_list.append(pz1)
#        self.channel.channel_response_filter.filters_list.append(pz2)
        self.channel.sample_rate = 1
        n_samples = 4096
        self.t = np.arange(n_samples) * self.channel.sample_interval
        # make a random signal
        self.example_ts = np.sum(
            [
                np.cos(2 * np.pi * w * self.t + phi)
                for w, phi in zip(
                    np.logspace(-4, self.channel.sample_rate / 2, 20),
                    np.random.rand(20),
                )
            ],
            axis=0,
        )

        # multiply by filter response
        f = np.fft.rfftfreq(self.t.size, self.channel.sample_interval)
        response_ts = np.fft.irfft(
            np.fft.rfft(self.example_ts) * pz1.complex_response(f)[::-1]
        )

        # add in a linear trend
        self.channel.ts = (0.3 * self.t) + response_ts

        self.calibrated_ts = self.channel.remove_instrument_response()

    def test_return_type(self):
        self.assertIsInstance(self.calibrated_ts, ChannelTS)

    def test_applied(self):
        self.assertTrue(
            (np.array(self.calibrated_ts.channel_metadata.filter.applied) == False).all()
        )

    def test_returned_metadata(self):
        with self.subTest("component"):
            self.assertEqual(self.channel.component, self.calibrated_ts.component)
        with self.subTest("sample rate"):
            self.assertEqual(self.channel.sample_rate, self.calibrated_ts.sample_rate)

    def test_calibration(self):

        original_ts = self.example_ts - self.example_ts.mean()
        std = (self.calibrated_ts.ts - original_ts).std() / original_ts.max()
        self.assertLessEqual(std, 0.075)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
