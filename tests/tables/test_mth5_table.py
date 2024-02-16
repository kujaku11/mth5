# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:11:32 2024

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
import numpy as np
import h5py
from pathlib import Path

from mth5.tables import MTH5Table

fn_path = Path(__file__).parent
# =============================================================================


class TestMTH5Table(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test.h5")
        self.h5 = h5py.File(self.fn, mode="w")
        self.dtype = np.dtype([("a", float), ("b", np.int32)])
        self.data = np.zeros(2, dtype=self.dtype)
        self.dataset = self.h5.create_dataset(
            "table", data=self.data, dtype=self.dtype, maxshape=((None,))
        )
        self.mth5_table = MTH5Table(self.dataset, self.dtype)

    def test_dtype_equal(self):
        self.assertTrue(self.mth5_table.check_dtypes(self.dtype))

    def test_shape(self):
        self.assertEqual(self.data.shape, self.mth5_table.shape)

    def test_nrows(self):
        self.assertEqual(self.data.shape[0], self.mth5_table.nrows)

    @classmethod
    def tearDownClass(self):
        self.h5.flush()
        self.h5.close()
        self.fn.unlink()


class TestMTH5TableChangeDtype(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test_dtype.h5")
        self.h5 = h5py.File(self.fn, mode="a")
        self.dtype = np.dtype([("a", float), ("b", np.int32)])
        self.new_dtype = np.dtype([("a", complex), ("b", np.int64)])
        self.data = np.zeros(2, dtype=self.dtype)
        self.dataset = self.h5.create_dataset(
            "table", data=self.data, dtype=self.dtype, maxshape=((None,))
        )
        self.mth5_table = MTH5Table(self.dataset, self.new_dtype)

    def test_dtype_equal(self):
        self.assertTrue(self.mth5_table.check_dtypes(self.new_dtype))

    def test_shape(self):
        self.assertEqual(self.data.shape, self.mth5_table.shape)

    def test_nrows(self):
        self.assertEqual(self.data.shape[0], self.mth5_table.nrows)

    @classmethod
    def tearDownClass(self):
        self.h5.flush()
        self.h5.close()
        self.fn.unlink()


class TestMTH5TableClearTable(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test_dtype.h5")
        self.h5 = h5py.File(self.fn, mode="a")
        self.dtype = np.dtype([("a", float), ("b", np.int32)])
        self.data = np.zeros(2, dtype=self.dtype)
        self.dataset = self.h5.create_dataset(
            "table", data=self.data, dtype=self.dtype, maxshape=((None,))
        )
        self.mth5_table = MTH5Table(self.dataset, self.dtype)

        self.mth5_table.clear_table()

    def test_dtype_equal(self):
        self.assertTrue(self.mth5_table.check_dtypes(self.dtype))

    def test_shape(self):
        self.assertEqual((1,), self.mth5_table.shape)

    def test_nrows(self):
        self.assertEqual(1, self.mth5_table.nrows)

    @classmethod
    def tearDownClass(self):
        self.h5.flush()
        self.h5.close()
        self.fn.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
