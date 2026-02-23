# -*- coding: utf-8 -*-
"""
Test UoA readers (PR6-24 and Orange Box)

Created on Thu Nov 7 16:00:00 2025

@author: bkay
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.io.uoa import read_uoa, read_orange, UoAReader, OrangeReader

# =============================================================================


class TestUoAReaderImports(unittest.TestCase):
    """Test that UoA readers can be imported"""

    def test_import_read_uoa(self):
        """Test read_uoa function can be imported"""
        self.assertTrue(callable(read_uoa))

    def test_import_read_orange(self):
        """Test read_orange function can be imported"""
        self.assertTrue(callable(read_orange))

    def test_import_uoa_reader_class(self):
        """Test UoAReader class can be imported"""
        self.assertTrue(UoAReader is not None)

    def test_import_orange_reader_class(self):
        """Test OrangeReader class can be imported"""
        self.assertTrue(OrangeReader is not None)


class TestOrangeBoxDetection(unittest.TestCase):
    """Test Orange Box file detection"""

    def test_detect_orange_box_function_exists(self):
        """Test that detect_orange_box function exists in reader module"""
        from mth5.io.reader import detect_orange_box
        self.assertTrue(callable(detect_orange_box))


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
