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
from mth5.mth5 import MTH5
from mth5.data.station_config import make_station_01
from mth5.data.make_mth5_from_asc import create_mth5_synthetic_file
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from mtpy.processing import RunSummary
from scipy.constants import mu_0

import unittest
import numpy as np
import xarray as xr
import pytest

from mt_metadata.transfer_functions.processing.aurora.frequency_bands import FrequencyBands

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


def test_cross_powers_basic(test1_spectrogram, test_frequency_bands):
    """Test basic cross power computation."""
    # Define channel pairs to test, including a conjugate pair
    channel_pairs = [("ex", "ey"), ("hx", "hy"), ("ex", "hx"), ("hx", "ex")]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=channel_pairs
    )
    xpowers = xpowers.to_dataset(dim='variable')

    # Test structure
    assert len(xpowers.data_vars) == len(channel_pairs)
    for pair in channel_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers
        assert np.iscomplexobj(xpowers[key].values)

    # Test conjugate property
    ex_hx = xpowers["ex_hx"].values
    hx_ex = xpowers["hx_ex"].values
    assert np.allclose(ex_hx, np.conj(hx_ex), rtol=1e-10, atol=1e-15)

def test_cross_powers_specific_pairs(test1_spectrogram, test_frequency_bands):
    """Test cross power computation with specific channel pairs."""
    # Define channel pairs to test
    channel_pairs = [("ex", "hx"), ("ey", "hy")]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=channel_pairs
    )
    xpowers = xpowers.to_dataset(dim='variable')

    # Test structure
    assert len(xpowers.data_vars) == len(channel_pairs)
    for pair in channel_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers
        assert np.iscomplexobj(xpowers[key].values)

def test_integrated_cross_powers(test1_spectrogram, test_frequency_bands):
    """
    Test cross power computation using synthetic data.

    This tests:
    1. E-field cross powers (ex-ey)
    2. H-field cross powers (hx-hy, hx-hz, hy-hz)
    3. E-H cross powers (ex-hx, ex-hy, ey-hx, ey-hy)
    """
    # Define channel pairs to test
    e_pairs = [("ex", "ey")]
    h_pairs = [("hx", "hy"), ("hx", "hz"), ("hy", "hz")]
    eh_pairs = [("ex", "hx"), ("ey", "hx"), ("ey", "hy")]  # Remove ex-hy since it's in conjugate_pairs

    # For conjugate test, add both forward and reverse pairs
    conjugate_pairs = [("ex", "hy"), ("hy", "ex")]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=e_pairs + h_pairs + eh_pairs + conjugate_pairs
    )
    xpowers = xpowers.to_dataset(dim='variable')

    # Test structure
    assert len(xpowers.data_vars) == len(e_pairs + h_pairs + eh_pairs + conjugate_pairs)

    # Test E-field cross powers
    for pair in e_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers
        assert np.iscomplexobj(xpowers[key].values)

    # Test H-field cross powers
    for pair in h_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers
        assert np.iscomplexobj(xpowers[key].values)

    # Test E-H cross powers (transfer functions)
    for pair in eh_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers
        assert np.iscomplexobj(xpowers[key].values)

    # Test conjugate property
    ex_hy = xpowers["ex_hy"].values
    hy_ex = xpowers["hy_ex"].values
    assert np.allclose(ex_hy, np.conj(hy_ex), rtol=1e-10, atol=1e-15)

def test_impedance_from_synthetic_data(test1_spectrogram, test_frequency_bands):
    """Test impedance tensor calculation using synthetic data."""
    # Define channel pairs for impedance calculation
    channel_pairs = [
        ("ex", "hx"), ("ex", "hy"),
        ("ey", "hx"), ("ey", "hy"),
        ("hx", "hx"), ("hy", "hy"),
        ("hx", "hy")
    ]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=channel_pairs
    )
    xpowers = xpowers.to_dataset(dim='variable')

    # Calculate Zxy for each frequency band
    ex_hy = xpowers["ex_hy"].values
    ex_hx = xpowers["ex_hx"].values
    hx_hx = xpowers["hx_hx"].values
    hy_hy = xpowers["hy_hy"].values
    hx_hy = xpowers["hx_hy"].values

    # Compute determinant of H-field auto-spectra matrix
    det_h = hx_hx * hy_hy - hx_hy * np.conj(hx_hy)

    # Calculate Zxy using matrix operations
    zxy = (ex_hy * hx_hx - ex_hx * hx_hy) / det_h

    # Unit conversion factors
    # Input data is in nT and mV/km
    # Convert to SI units (T and V/m) for impedance calculation
    zxy *= 1e3

    # Calculate apparent resistivity
    freq = test_frequency_bands.band_centers()
    omega = 2 * np.pi * freq
    zxy2 = np.abs(zxy) ** 2
    rho_xy = (mu_0 / omega) * zxy2.T
    rho_xy_mean = np.mean(rho_xy, axis=0)

    # Test that apparent resistivity is approximately 100 Ohm-m
    assert (rho_xy_mean > 75.).all()
    assert (rho_xy_mean < 125.).all()

@pytest.fixture
def test1_spectrogram():
    """Create a test spectrogram from synthetic data."""
    from mth5.data.station_config import make_station_01
    from mth5.data.make_mth5_from_asc import create_mth5_synthetic_file
    from mth5.timeseries.spectre.helpers import add_fcs_to_mth5
    from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
    from mtpy.processing import RunSummary, KernelDataset

    # Create test configuration
    station_cfg = make_station_01()

    # Create test h5 file with Fourier coefficients
    mth5_path = create_mth5_synthetic_file(
        station_cfgs=[station_cfg],
        mth5_name="test1.h5",
        force_make_mth5=True
    )

    # Create run summary and dataset
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1")

    # Get processing config and FC decimations
    processing_config = create_test_run_config("test1", tfk_dataset)
    fc_decimations = [x.to_fc_decimation() for x in processing_config.decimations]

    # Add Fourier coefficients to MTH5 file
    add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)

    # Read back the Fourier coefficients directly
    with MTH5() as m:
        m.open_mth5(mth5_path, mode="r")
        station_obj = m.get_station("test1", "EMTF Synthetic")
        fc_group = station_obj.fourier_coefficients_group.get_fc_group("001")
        dec_level = fc_group.get_decimation_level("0")  # Get first decimation level
        ds = dec_level.to_xarray()
        return Spectrogram(dataset=ds)

@pytest.fixture
def test_frequency_bands():
    """Create frequency bands for testing"""
    edges = np.array([
        [0.01, 0.1],   # Low band
        [0.1, 0.2],    # Mid band
        [0.2, 0.49]     # High band
    ])
    return FrequencyBands(edges)

if __name__ == "__main__":
    unittest.main()
