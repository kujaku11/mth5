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

from mth5 import timeseries
from mth5.utils.exceptions import MTTSError

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime

# =============================================================================
#
# =============================================================================


class TestMTTS(unittest.TestCase):
    def setUp(self):
        self.ts = timeseries.ChannelTS("auxiliary")
        self.maxDiff = None

    def test_input_type_electric(self):
        self.ts = timeseries.ChannelTS("electric")

        electric_meta = metadata.Electric()
        self.assertDictEqual(
            self.ts.metadata.to_dict(), electric_meta.to_dict()
        )

    def test_input_type_magnetic(self):
        self.ts = timeseries.ChannelTS("magnetic")

        magnetic_meta = metadata.Magnetic()
        self.assertDictEqual(
            self.ts.metadata.to_dict(), magnetic_meta.to_dict()
        )

    def test_input_type_auxiliary(self):
        self.ts = timeseries.ChannelTS("auxiliary")

        auxiliary_meta = metadata.Auxiliary()
        self.assertDictEqual(
            self.ts.metadata.to_dict(), auxiliary_meta.to_dict()
        )

    def test_input_type_fail(self):
        self.assertRaises(ValueError, timeseries.ChannelTS, "temperature")

    def test_intialize_with_metadata(self):
        self.ts = timeseries.ChannelTS(
            "electric", channel_metadata={"electric": {"component": "ex"}}
        )
        self.assertEqual(self.ts.metadata.component, "ex")
        self.assertEqual(self.ts.ts.attrs["component"], "ex")

    def test_numpy_input(self):
        self.ts.metadata.sample_rate = 1.0
        self.ts.update_xarray_metadata()

        self.ts.ts = np.random.rand(4096)
        end = self.ts.metadata.time_period._start_dt + (4096 - 1)

        # check to make sure the times align
        self.assertEqual(
            self.ts.ts.coords.to_index()[0].isoformat(),
            self.ts.metadata.time_period._start_dt.iso_no_tz,
        )

        self.assertEqual(
            self.ts.ts.coords.to_index()[-1].isoformat(), end.iso_no_tz
        )

        self.assertEqual(self.ts.n_samples, 4096)

    def test_set_component(self):
        self.ts = timeseries.ChannelTS(
            "electric", channel_metadata={"electric": {"component": "ex"}}
        )

        def set_comp(comp):
            self.ts.component = comp

        self.assertRaises(MTTSError, set_comp, "hx")
        self.assertRaises(MTTSError, set_comp, "bx")
        self.assertRaises(MTTSError, set_comp, "temperature")


# =============================================================================
# test run
# =============================================================================
class TestRunTS(unittest.TestCase):
    def setUp(self):
        self.run = timeseries.RunTS()
        self.maxDiff = None

        self.ex = timeseries.ChannelTS(
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
        self.ey = timeseries.ChannelTS(
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
        self.hx = timeseries.ChannelTS(
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
        self.hy = timeseries.ChannelTS(
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
        self.hz = timeseries.ChannelTS(
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
        self.hz = timeseries.ChannelTS(
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

        self.assertIsInstance(self.run.ex, timeseries.ChannelTS)
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
        self.run.metadata.sample_rate = 10
        self.run.validate_metadata()

        self.assertEqual(self.ex.sample_rate, self.run.metadata.sample_rate)

        self.run.metadata.start = "2020-01-01T00:00:00"
        self.run.validate_metadata()
        self.assertEqual(self.run.start, self.run.metadata.time_period.start)

        self.run.metadata.end = "2020-01-01T00:00:00"
        self.run.validate_metadata()
        self.assertEqual(self.run.end, self.run.metadata.time_period.end)


# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
