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
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.clients import MakeMTH5

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


class TestMTH5ToMiniSEEDStationXML(unittest.TestCase):
    """This just tests if files were created nothing about accuracy

    TODO: need to make a method that checks accuracy of the two files.
    """

    @classmethod
    def setUpClass(cls):
        cls.mth5_path_v1 = create_test1_h5("0.1.0", force_make_mth5=False)
        cls.mth5_path_v2 = create_test1_h5("0.2.0", force_make_mth5=False)

    def convert(self, version):
        if version in ["1", 1, "0.1.0"]:
            h5_path = self.mth5_path_v1
        elif version in ["2", 2, "0.2.0"]:
            h5_path = self.mth5_path_v2
        return MTH5ToMiniSEEDStationXML.convert_mth5_to_ms_stationxml(
            h5_path, network_code="ZU"
        )

    def test_conversion_v1(self):
        stationxml, miniseeds = self.convert("0.1.0")
        with self.subTest("StationXML was written"):
            self.assertTrue(stationxml.exists())
        with self.subTest("miniseeds were written"):
            self.assertEqual(len(miniseeds), 1)

        mth5_file = MakeMTH5.from_fdsn_miniseed_and_stationxml(
            stationxml,
            miniseeds,
            save_path=self.mth5_path_v1.parent,
            **{"mth5_version": "0.1.0"}
        )
        with self.subTest("new MTH5 created"):
            self.assertTrue(mth5_file.exists())

        mth5_file.unlink()
        stationxml.unlink()
        [fn.unlink() for fn in miniseeds]

    def test_conversion_v2(self):
        stationxml, miniseeds = self.convert("0.2.0")
        with self.subTest("StationXML was written"):
            self.assertTrue(stationxml.exists())
        with self.subTest("miniseeds were written"):
            self.assertEqual(len(miniseeds), 1)

        mth5_file = MakeMTH5.from_fdsn_miniseed_and_stationxml(
            stationxml,
            miniseeds,
            save_path=self.mth5_path_v1.parent,
            **{"mth5_version": "0.2.0"}
        )
        with self.subTest("new MTH5 created"):
            self.assertTrue(mth5_file.exists())

        mth5_file.unlink()
        stationxml.unlink()
        [fn.unlink() for fn in miniseeds]

    @classmethod
    def tearDownClass(cls):
        return super().tearDownClass()


# =============================================================================
# Run
if __name__ == "__main__":
    unittest.main()
