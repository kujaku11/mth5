# -*- coding: utf-8 -*-
"""
Test time series

Created on Tue Jun 30 16:38:27 2020

:copyright:
    author: Jared Peacock
    
:license:
    MIT
    
"""
# =============================================================================
# imports
# =============================================================================

import unittest

import numpy as np

from mth5.timeseries import ChannelTS, RunTS
from mth5.utils.exceptions import MTTSError

from mt_metadata.utils.mttime import MTime

# =============================================================================
# test run
# =============================================================================
class TestRunTS(unittest.TestCase):
    def setUp(self):
        self.run = RunTS()
        self.maxDiff = None

        self.ex = ChannelTS(
            "electric",
            data=np.random.rand(4096),
            channel_metadata={
                "electric": {
                    "component": "Ex",
                    "sample_rate": 8,
                    "time_period.start": "2015-01-08T19:49:18+00:00",
                }
            },
        )
        self.ey = ChannelTS(
            "electric",
            data=np.random.rand(4096),
            channel_metadata={
                "electric": {
                    "component": "Ey",
                    "sample_rate": 8,
                    "time_period.start": "2015-01-08T19:49:18+00:00",
                }
            },
        )
        self.hx = ChannelTS(
            "magnetic",
            data=np.random.rand(4096),
            channel_metadata={
                "magnetic": {
                    "component": "hx",
                    "sample_rate": 8,
                    "time_period.start": "2015-01-08T19:49:18+00:00",
                }
            },
        )
        self.hy = ChannelTS(
            "magnetic",
            data=np.random.rand(4096),
            channel_metadata={
                "magnetic": {
                    "component": "hy",
                    "sample_rate": 8,
                    "time_period.start": "2015-01-08T19:49:18+00:00",
                }
            },
        )
        self.hz = ChannelTS(
            "magnetic",
            data=np.random.rand(4096),
            channel_metadata={
                "magnetic": {
                    "component": "hz",
                    "sample_rate": 8,
                    "time_period.start": "2015-01-08T19:49:18+00:00",
                }
            },
        )

        self.run.set_dataset([self.ex, self.ey, self.hx, self.hy, self.hz])

    def test_initialize(self):

        self.assertListEqual(["ex", "ey", "hx", "hy", "hz"], self.run.channels)

        self.assertEqual(self.run.sample_rate, 8.0)
        self.assertEqual(self.run.start, MTime("2015-01-08T19:49:18"))
        self.assertEqual(self.run.end, MTime("2015-01-08T19:57:49.875000"))

    def test_sr_fail(self):
        self.hz = ChannelTS(
            "magnetic",
            data=np.random.rand(4096),
            channel_metadata={
                "magnetic": {
                    "component": "hz",
                    "sample_rate": 1,
                    "time_period.start": "2015-01-08T19:49:18+00:00",
                }
            },
        )

        self.assertRaises(
            MTTSError,
            self.run.set_dataset,
            [self.ex, self.ey, self.hx, self.hy, self.hz],
        )

    def test_ex(self):

        self.assertIsInstance(self.run.ex, ChannelTS)
        self.assertEqual(self.ex.sample_rate, self.run.sample_rate)
        self.assertEqual(self.run.start, self.ex.start)
        self.assertEqual(self.run.end, self.ex.end)
        self.assertEqual(self.ex.component, "ex")

    def test_get_channel_fail(self):
        """
        self.run.temperature should return None, because 'temperature' is not in self.channels
        :return:
        """

        self.assertRaises(NameError, getattr, *(self.run, "temperature"))

    def test_wrong_metadata(self):
        self.run.run_metadata.sample_rate = 10
        self.run.validate_metadata()

        self.assertEqual(self.ex.sample_rate, self.run.run_metadata.sample_rate)

        self.run.run_metadata.start = "2020-01-01T00:00:00"
        self.run.validate_metadata()
        self.assertEqual(self.run.start, self.run.run_metadata.time_period.start)

        self.run.run_metadata.end = "2020-01-01T00:00:00"
        self.run.validate_metadata()
        self.assertEqual(self.run.end, self.run.run_metadata.time_period.end)


# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
