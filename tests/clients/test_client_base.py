# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:48:24 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from pathlib import Path

from mth5.clients.base import ClientBase

# =============================================================================


class TestClientBase(unittest.TestCase):
    def setUp(self):
        self.file_path = Path(__file__)
        self.base = ClientBase(
            self.file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"}
        )

    def test_h5_kwargs(self):
        keys = [
            "compression",
            "compression_opts",
            "data_level",
            "driver",
            "file_version",
            "fletcher32",
            "mode",
            "shuffle",
        ]
        self.assertListEqual(keys, sorted(self.base.h5_kwargs.keys()))

    def test_set_sample_rates_float(self):
        self.base.sample_rates = 10.5
        self.assertListEqual([10.5], self.base.sample_rates)

    def test_set_sample_rates_str(self):
        self.base.sample_rates = "10, 42, 1200"
        self.assertListEqual([10.0, 42.0, 1200.0], self.base.sample_rates)

    def test_set_sample_rate_list(self):
        self.base.sample_rates = [10, 42, 1200]
        self.assertListEqual([10.0, 42.0, 1200.0], self.base.sample_rates)

    def test_set_sample_rate_fail(self):
        def set_sample_rates(value):
            self.base.sample_rates = value

        self.assertRaises(TypeError, set_sample_rates, None)

    def test_set_save_path(self):
        self.base.save_path = self.file_path
        with self.subTest("_save_path"):
            self.assertEqual(self.base._save_path, self.file_path.parent)
        with self.subTest("filename"):
            self.assertEqual(self.base.mth5_filename, self.file_path.name)
        with self.subTest("save_path"):
            self.assertEqual(self.base.save_path, self.file_path)

    def test_initial_fail_None(self):
        self.assertRaises(ValueError, ClientBase, None)

    def test_initial_fail_bad_directory(self):
        self.assertRaises(IOError, ClientBase, r"a:\\")


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
