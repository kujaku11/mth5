"""
Test converters including

- MTH5 -> miniSEED + StationXML
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path
from mth5.io.conversion import MTH5ToMiniSEEDStationXML

# =============================================================================


class TestMTH5ToMiniSEEDStationXMLFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = MTH5ToMiniSEEDStationXML()

    def test_set_mth5_path_none(self):
        self.converter.mth5_path = None
        with self.subTest("mth5_path"):
            self.assertEqual(self.converter.mth5_path, None)
        with self.subTest("save_path"):
            self.assertEqual(self.converter.save_path, Path().cwd())

    def test_set_mth5_path_fail(self):
        def set_path(value):
            self.converter.mth5_path = value

        with self.subTest("bad type"):
            self.assertRaises(TypeError, set_path, 10)
        with self.subTest("non-existant file"):
            self.assertRaises(
                FileExistsError, set_path, Path().cwd().joinpath("fail.h5")
            )

    def test_set_network_code_fail(self):
        def set_code(value):
            self.converter.network_code = value

        with self.subTest("too many characters"):
            self.assertRaises(ValueError, set_code, "abc")
        with self.subTest("bad characters"):
            self.assertRaises(ValueError, set_code, "!a")


# =============================================================================
# Run
if __name__ == "__main__":
    unittest.main()
