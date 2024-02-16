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
from mth5.utils.exceptions import MTH5TableError

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

    def test_len(self):
        self.assertEqual(self.data.shape[0], len(self.mth5_table))

    def test_mth5_table_input_fail(self):
        self.assertRaises(MTH5TableError, MTH5Table, "a", self.dtype)

    def test_dtype_fail(self):
        self.assertRaises(TypeError, MTH5Table, self.dataset, "a")

    def test_str(self):
        df = self.mth5_table.to_dataframe()
        self.assertEqual(self.mth5_table.__str__(), df.__str__())

    def test_repr(self):
        df = self.mth5_table.to_dataframe()
        self.assertEqual(self.mth5_table.__repr__(), df.__str__())

    def test_hdf5_ref(self):
        self.assertEqual(
            self.dataset.ref.typesize, self.mth5_table.hdf5_reference.typesize
        )

    def test_locate_eq(self):
        self.assertTrue(
            (
                self.mth5_table.locate("a", 0, "eq")
                == np.array([0, 1], dtype=np.int64)
            ).all(),
        )

    def test_locate_le(self):
        self.assertTrue(
            (
                self.mth5_table.locate("a", 0, "le")
                == np.array([0, 1], dtype=np.int64)
            ).all(),
        )

    def test_locate_lt(self):
        self.assertTrue(
            (
                self.mth5_table.locate("a", 0, "lt")
                == np.array([], dtype=np.int64)
            ).all(),
        )

    def test_locate_ge(self):
        self.assertTrue(
            (
                self.mth5_table.locate("a", 0, "ge")
                == np.array([0, 1], dtype=np.int64)
            ).all(),
        )

    def test_locate_gt(self):
        self.assertTrue(
            (
                self.mth5_table.locate("a", 0, "gt")
                == np.array([], dtype=np.int64)
            ).all(),
        )

    def test_locate_be_fail(self):
        self.assertRaises(ValueError, self.mth5_table.locate, "a", 0, "be")

    def test_locate_be(self):
        self.assertTrue(
            (
                self.mth5_table.locate("a", [-1, 1], "be")
                == np.array([0, 1], dtype=np.int64)
            ).all(),
        )

    def test_locate_fail(self):
        self.assertRaises(ValueError, self.mth5_table.locate, "a", 0, "fail")

    def test_set_dtype_fail(self):
        def set_dtype(self, value):
            self.mth5_table.dtype = value

        self.assertRaises(TypeError, set_dtype, "a")

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

    def test_dtype_not_equal(self):
        self.assertFalse(self.mth5_table.check_dtypes(self.dtype))

    def test_shape(self):
        self.assertEqual(self.data.shape, self.mth5_table.shape)

    def test_nrows(self):
        self.assertEqual(self.data.shape[0], self.mth5_table.nrows)

    @classmethod
    def tearDownClass(self):
        self.h5.flush()
        self.h5.close()
        self.fn.unlink()


class TestMTH5TableSetDtype(unittest.TestCase):
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
        self.mth5_table = MTH5Table(self.dataset, self.dtype)
        self.mth5_table.dtype = self.new_dtype

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
