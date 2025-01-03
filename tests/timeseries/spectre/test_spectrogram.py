# -*- coding: utf-8 -*-
"""
    This modules contains tests for the Spectrogram class.

    TODO: Add a test that accesses the synthetic data, and adds some FC Levels, then reads those FC levels and
    casts them to spectogram objects.  Currently this still depends on methods in aurora.

"""
from loguru import logger
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mth5.helpers import close_open_files
from mth5.timeseries.spectre.helpers import add_fcs_to_mth5
from mth5.timeseries.spectre.helpers import read_back_fcs
from mth5.timeseries.spectre import Spectrogram

import unittest

FORCE_MAKE_MTH5 = True  # Should be True except when debugging locally


class TestAddFourierCoefficientsToSyntheticData(unittest.TestCase):
    """
    Runs several synthetic processing tests from config creation to tf_cls.

    There are two ways to prepare the FC-schema
      a) use the mt_metadata.FCDecimation class explictly
      b) mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel has
      a to_fc_decimation() method that returns mt_metadata.FCDecimation

    Flow is to make some mth5 files from synthetic data, then loop over those files adding fcs.

    Development Notes:
     This test is in process of being moved to mth5 from aurora.  The aurora created spectrograms and then
     proceeded to process the mth5s to make TFs.  It would be nice to make some pytest fixtures that generate
     these mth5 with spectrograms so they could be reused by multiple tests.

    """

    @classmethod
    def setUpClass(self):
        """
        Makes some synthetic h5 files for testing.

        """
        logger.info("Making synthetic data")
        close_open_files()
        self.file_version = "0.1.0"
        mth5_path_1 = create_test1_h5(file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5)
        mth5_path_2 = create_test2_h5(file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5)
        mth5_path_3 = create_test3_h5(file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5)
        mth5_path_12rr = create_test12rr_h5(file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5)
        self.mth5_paths = [
            mth5_path_1,
            mth5_path_2,
            mth5_path_3,
            mth5_path_12rr,
        ]
        self.mth5_path_2 = mth5_path_2

    def test_spectrograms(self):
        """
        This test adds FCs to each of the synthetic files that get built in setUpClass method.
        - This could probably be shortened, it isn't clear that all the h5 files need to have fc added
        and be processed too.

        Development Notes:
         - This is a thinned out version of a test from aurora (test_fourier_coeffcients)
         - setting fc_decimations=None builds them in add_fcs_to_mth5 according to a default recipe

        Returns
        -------

        """
        for mth5_path in self.mth5_paths:
            add_fcs_to_mth5(mth5_path, fc_decimations=None)
            read_back_fcs(mth5_path)

        return

class TestSpectrogram(unittest.TestCase):
    """
    Test Spectrogram class
    """

    @classmethod
    def setUpClass(self):
        pass

    def setUp(self):
        pass

    def test_initialize(self):
        spectrogram = Spectrogram()
        assert isinstance(spectrogram, Spectrogram)

    def test_slice_band(self):
        """
        Place holder
        TODO: Once FCs are added to an mth5, load a spectrogram and extract a Band
        """
        pass


if __name__ == "__main__":
    # tmp = TestSpectrogram()
    # tmp.test_initialize()
    unittest.main()
