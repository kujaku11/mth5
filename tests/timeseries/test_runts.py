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
        self.start = "2015-01-08T19:49:18+00:00"
        self.end = "2015-01-08T19:57:49.875000"
        self.sample_rate = 8
        self.npts = 4096


        self.ex = ChannelTS(
            "electric",
            data=np.random.rand(self.npts),
            channel_metadata={
                "electric": {
                    "component": "Ex",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
        )
        self.ey = ChannelTS(
            "electric",
            data=np.random.rand(self.npts),
            channel_metadata={
                "electric": {
                    "component": "Ey",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
        )
        self.hx = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hx",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
        )
        self.hy = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hy",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
        )
        self.hz = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hz",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
        )

        self.run.set_dataset([self.ex, self.ey, self.hx, self.hy, self.hz])

    def test_initialize(self):

        with self.subTest("channels"):
            self.assertListEqual(["ex", "ey", "hx", "hy", "hz"], self.run.channels)

        with self.subTest("sample rate"):
            self.assertEqual(self.run.sample_rate, self.sample_rate)
        
        with self.subTest("start"):
            self.assertEqual(self.run.start, MTime(self.start))
        
        with self.subTest("end"):
            self.assertEqual(self.run.end, MTime())

    def test_sr_fail(self):
        self.hz = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hz",
                    "sample_rate": 1,
                    "time_period.start": self.start,
                }
            },
        )

        self.assertRaises(
            MTTSError,
            self.run.set_dataset,
            [self.ex, self.ey, self.hx, self.hy, self.hz],
        )

    def test_channels(self):
        
        for comp in ["ex", "ey", "hx", "hy", "hz"]:
            ch = getattr(self, comp)
            
            with self.subTest("isinstance channel"):
                self.assertIsInstance(ch, ChannelTS)
            
            with self.subTest("sample rate"):
                self.assertEqual(ch.sample_rate, self.sample_rate)
            
            with self.subTest("start"):
                self.assertEqual(ch.start, MTime(self.start))
            
            with self.subTest("end"):
                self.assertEqual(ch.end, MTime())
                
            with self.subTest("component"):
                self.assertEqual(ch.component, comp)

    def test_get_channel_fail(self):
        """
        self.run.temperature should return None, because 'temperature' is not in self.channels
        :return:
        """

        self.assertRaises(NameError, getattr, *(self.run, "temperature"))

    def test_wrong_metadata(self):
        self.run.run_metadata.sample_rate = 10
        self.run.validate_metadata()

        with self.subTest("sample rate"):
            self.assertEqual(self.ex.sample_rate, self.run.run_metadata.sample_rate)

        with self.subTest("start"):
            self.run.run_metadata.start = "2020-01-01T00:00:00"
            self.run.validate_metadata()
            self.assertEqual(self.run.start, self.run.run_metadata.time_period.start)

        with self.subTest("end"):
            self.run.run_metadata.end = "2020-01-01T00:00:00"
            self.run.validate_metadata()
            self.assertEqual(self.run.end, self.run.run_metadata.time_period.end)

    def test_get_slice(self):
        
        start = "2015-01-08T19:49:30+00:00"
        npts = 256
        
        r_slice = self.run.get_slice(start, n_samples=npts)
        
        with self.subTest("isinstance runts"):
            self.assertIsInstance(r_slice, RunTS)
        
        with self.subTest("sample rate"):
            self.assertEqual(r_slice.sample_rate, self.sample_rate)
        
        with self.subTest("start"):
            self.assertEqual(r_slice.start, MTime(start))
        
        # with self.subTest("end"):
        #     self.assertEqual(ch.end, MTime())
            
        # with self.subTest("npts"):
        #     self.assertEqual(r_slice., comp)
            
            
        
        

# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
