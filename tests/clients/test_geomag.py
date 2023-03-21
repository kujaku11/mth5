# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:40:28 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
from collections import OrderedDict
import numpy as np
import pandas as pd

from mth5.clients.geomag import USGSGeomag

# =============================================================================


class TestMakeMTH5FromGeomag(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.client = USGSGeomag()

        self.request_df = pd.DataFrame(
            {
                "observatory": ["frn", "frn", "ott", "ott"],
                "type": ["adjusted"] * 4,
                "elements": [["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]],
                "sampling_period": [1, 1, 1, 1],
                "start": [
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                ],
                "end": [
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                ],
            }
        )

    def test_fail_request_df_type(self):
        self.assertRaises(TypeError, self.client.validate_request_df, "a")

    def test_fail_request_df_bad_columns(self):
        self.assertRaises(
            ValueError,
            self.client.validate_request_df,
            pd.DataFrame({"a": [10]}),
        )

    def test_add_run(self):
        rdf = self.client.add_run_id(self.request_df)
        self.assertListEqual(rdf.run.tolist(), ["001", "002", "001", "002"])

    def test_add_run_different_sampling_periods(self):
        request_df = pd.DataFrame(
            {
                "observatory": ["frn", "frn", "ott", "ott"],
                "type": ["adjusted"] * 4,
                "elements": [["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]],
                "sampling_period": [1, 60, 1, 60],
                "start": [
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                ],
                "end": [
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                ],
            }
        )

        rdf = self.client.add_run_id(request_df)
        self.assertListEqual(rdf.run.tolist(), ["001", "001", "001", "001"])


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
