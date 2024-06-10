import logging
import mth5
import pandas as pd
import pathlib
import unittest

from mth5.data.make_mth5_from_asc import create_test1_h5
# from mth5.data.make_mth5_from_asc import create_test1_h5_with_nan
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.make_mth5_from_asc import create_test4_h5
from mth5.data.paths import SyntheticTestPaths
from mth5.data.station_config import make_station_03
from mth5.helpers import close_open_files
from mth5.mth5 import MTH5

synthetic_test_paths = SyntheticTestPaths()
synthetic_test_paths.mkdirs()
SOURCE_PATH = synthetic_test_paths.ascii_data_path

# from loguru import logger


class TestDataFolder(unittest.TestCase):
    """ """

    def setUp(self):
        init_file = pathlib.Path(mth5.__file__)
        self.data_folder = init_file.parent.joinpath("data")

    def test_ascii_data_paths(self):
        """
        Make sure that the ascii data are where we think they are.

        """

        self.assertTrue(self.data_folder.exists())
        file_paths = list(self.data_folder.glob("*asc"))
        file_names = [x.name for x in file_paths]

        assert "test1.asc" in file_names
        assert "test2.asc" in file_names

class TestMakeSyntheticMTH5(unittest.TestCase):
    """
    create_test1_h5(file_version=file_version)
    create_test1_h5_with_nan(file_version=file_version)
    create_test2_h5(file_version=file_version)
    create_test12rr_h5(file_version=file_version)
    create_test3_h5(file_version=file_version)
    """

    def test_make_upsampled_mth5(self):
        file_version = "0.2.0"
        create_test4_h5(file_version=file_version, source_folder=SOURCE_PATH)

    def test_make_more_mth5s(self):
        create_test1_h5(file_version="0.1.0", source_folder=SOURCE_PATH)

class TestMetadataValuesSetCorrect(unittest.TestCase):
    """
    Tests setting of start time as per aurora issue #188
    """

    remake_mth5_for_each_test = False

    def setUp(self):
        mth5.mth5.helpers.close_open_files()
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.ticker").disabled = True

    def make_mth5(self):
        close_open_files()
        mth5_path = create_test3_h5(force_make_mth5=self.remake_mth5_for_each_test)
        return mth5_path

    def make_run_summary(self):
        mth5_path = self.make_mth5()
        m = MTH5()
        m.open_mth5(mth5_path)
        station_id = m.station_list[0]  # station should be named "test3"
        self.assertTrue(station_id == "test3")
        station_obj = m.get_station(station_id)
        return station_obj.run_summary

    def test_start_times_correct(self):
        run_summary_df = self.make_run_summary()
        station_03 = make_station_03()
        for run in station_03.runs:
            summary_row = run_summary_df[
                run_summary_df.id == run.run_metadata.id
            ].iloc[0]
            assert summary_row.start == pd.Timestamp(run.start).tz_convert(None)


def main():
#    TestMetadataValuesSetCorrect()
    unittest.main()


if __name__ == "__main__":
    main()
