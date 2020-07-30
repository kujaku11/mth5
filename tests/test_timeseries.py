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
import pandas as pd
import xarray as xr

from mth5 import timeseries
from mth5 import metadata

# =============================================================================
#
# =============================================================================
class TestMTTS(unittest.TestCase):
    def setUp(self):
        self.ts = timeseries.MTTS("auxiliary")
        self.maxDiff = None

    def test_input_type_electric(self):
        self.ts = timeseries.MTTS("electric")

        electric_meta = metadata.Electric()
        self.assertDictEqual(self.ts.metadata.to_dict(), electric_meta.to_dict())

    def test_input_type_magnetic(self):
        self.ts = timeseries.MTTS("magnetic")

        magnetic_meta = metadata.Magnetic()
        self.assertDictEqual(self.ts.metadata.to_dict(), magnetic_meta.to_dict())

    def test_input_type_auxiliary(self):
        self.ts = timeseries.MTTS("auxiliary")

        auxiliary_meta = metadata.Auxiliary()
        self.assertDictEqual(self.ts.metadata.to_dict(), auxiliary_meta.to_dict())

    def test_input_type_fail(self):
        self.assertRaises(ValueError, timeseries.MTTS, "temperature")

    def test_intialize_with_metadata(self):
        self.ts = timeseries.MTTS(
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

        self.assertEqual(self.ts.ts.coords.to_index()[-1].isoformat(), end.iso_no_tz)

        self.assertEqual(self.ts.n_samples, 4096)


# =============================================================================
# test run
# =============================================================================
class TestRunTS(unittest.TestCase):
    def setUp(self):
        self.run = timeseries.RunTS()
        self.maxDiff = None
                
        self.ex = timeseries.MTTS(
            "electric",
            data=np.random.rand(4096),
            channel_metadata={"electric": {"component": "ex", 
                                           "sample_rate": 8,
                                           'time_period.start': "2015-01-08T19:49:18+00:00"}},
        )
        self.ey = timeseries.MTTS(
            "electric",
            data=np.random.rand(4096),
            channel_metadata={"electric": {"component": "ey",
                                           "sample_rate": 8,
                                           'time_period.start': "2015-01-08T19:49:18+00:00"}},
        )
        self.hx = timeseries.MTTS(
            "magnetic",
            data=np.random.rand(4096),
            channel_metadata={"magnetic": {"component": "hx", 
                                           "sample_rate": 8,
                                           'time_period.start': "2015-01-08T19:49:18+00:00"}},
        )
        self.hy = timeseries.MTTS(
            "magnetic",
            data=np.random.rand(4096),
            channel_metadata={"magnetic": {"component": "hy",
                                           "sample_rate": 8,
                                           'time_period.start': "2015-01-08T19:49:18+00:00"}},
        )
        self.hz = timeseries.MTTS(
            "magnetic",
            data=np.random.rand(4096),
            channel_metadata={"magnetic": {"component": "hz",
                                           "sample_rate": 8,
                                           'time_period.start': "2015-01-08T19:49:18+00:00"}},
        )

    def test_list_input(self):

        self.run.build_dataset([self.ex, self.ey, self.hx, self.hy, self.hz])

        self.assertListEqual(
            ["ex", "ey", "hx", "hy", "hz"], 
            list(self.run.dataset.data_vars.keys())
        )
        
        self.assertEqual(self.run.sample_rate, 8.0)
        self.assertEqual(self.run.start, '2015-01-08T19:49:18')
        self.assertEqual(self.run.end, '2015-01-08T19:57:49.875000')
        
# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
