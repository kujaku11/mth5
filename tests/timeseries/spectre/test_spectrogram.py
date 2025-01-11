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
from mtpy.processing import RunSummary, KernelDataset

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


@pytest.fixture
def sample_spectrogram():
    """
    Create a sample spectrogram with synthetic data.
    
    For a 1D earth with Zxy ≈ 100:
    - Ex = 100 * Hy  (primary relationship)
    - Hx and Hy are independent
    """
    # Create time and frequency axes
    times = np.linspace(0, 10, 100)
    freqs = np.linspace(0.1, 1.0, 10)
    
    # Create synthetic magnetic field data
    data_hx = np.random.randn(len(times), len(freqs)) + 1j * np.random.randn(len(times), len(freqs))
    data_hy = np.random.randn(len(times), len(freqs)) + 1j * np.random.randn(len(times), len(freqs))
    
    # Create electric field data with known relationship to magnetic field
    # Ex = 100 * Hy for Zxy ≈ 100
    data_ex = 100 * data_hy
    data_ey = np.random.randn(len(times), len(freqs)) + 1j * np.random.randn(len(times), len(freqs))
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={
            'ex': (('time', 'frequency'), data_ex),
            'ey': (('time', 'frequency'), data_ey),
            'hx': (('time', 'frequency'), data_hx),
            'hy': (('time', 'frequency'), data_hy)
        },
        coords={
            'time': times,
            'frequency': freqs
        }
    )
    
    return Spectrogram(dataset=ds)


@pytest.fixture
def sample_frequency_bands():
    """Create sample frequency bands"""
    edges = np.array([
        [0.1, 0.4],
        [0.4, 0.7],
        [0.7, 1.0]
    ])
    return FrequencyBands(edges)


def test_cross_powers_basic(sample_spectrogram, sample_frequency_bands):
    """Test basic cross power computation"""
    # Compute cross powers
    xpowers = sample_spectrogram.cross_powers(sample_frequency_bands)
    
    # Check structure
    assert len(xpowers.channel_pairs) == 1  # Only ex-ey pair
    assert len(xpowers.features['ex_ey']) == 3  # Three frequency bands
    
    # Check values are complex
    assert np.iscomplexobj(xpowers.features['ex_ey'])
    
    # Check time dimension is preserved
    assert xpowers.features['ex_ey'].dims == ('band', 'time')
    assert len(xpowers.features['ex_ey'].time) == len(sample_spectrogram.dataset.time)


def test_cross_powers_specific_pairs(sample_spectrogram, sample_frequency_bands):
    """Test cross powers with specific channel pairs"""
    channel_pairs = [('ex', 'ey')]
    xpowers = sample_spectrogram.cross_powers(
        sample_frequency_bands,
        channel_pairs=channel_pairs
    )
    
    # Check structure
    assert xpowers.channel_pairs == channel_pairs
    assert list(xpowers.features.keys()) == ['ex_ey']
    
    # Check time dimension is preserved
    assert xpowers.features['ex_ey'].dims == ('band', 'time')
    assert len(xpowers.features['ex_ey'].time) == len(sample_spectrogram.dataset.time)


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


@pytest.fixture
def test1_spectrogram():
    """
    Create a test spectrogram from synthetic data.
    Uses test1.h5 which has channels: ex, ey, hx, hy, hz
    """
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
        [0.1, 0.5],    # Mid band
        [0.5, 1.0]     # High band
    ])
    return FrequencyBands(edges)


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
    eh_pairs = [("ex", "hx"), ("ex", "hy"), ("ey", "hx"), ("ey", "hy")]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=e_pairs + h_pairs + eh_pairs
    )

    # Test structure
    assert len(xpowers.channel_pairs) == len(e_pairs + h_pairs + eh_pairs)
    assert len(xpowers.features["ex_ey"]) == test_frequency_bands.number_of_bands

    # Test E-field cross powers
    for pair in e_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers.features
        assert np.iscomplexobj(xpowers.features[key].values)

    # Test H-field cross powers
    for pair in h_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers.features
        assert np.iscomplexobj(xpowers.features[key].values)

    # Test E-H cross powers (transfer functions)
    for pair in eh_pairs:
        key = f"{pair[0]}_{pair[1]}"
        assert key in xpowers.features
        assert np.iscomplexobj(xpowers.features[key].values)

    # Test conjugate property
    ex_hy = xpowers.features["ex_hy"].values
    hy_ex_conj = xpowers.get_conjugate("hy", "ex").values
    assert np.allclose(ex_hy, np.conj(hy_ex_conj))


def test_impedance_from_cross_powers(test1_spectrogram, test_frequency_bands):
    """
    Test impedance tensor calculation from cross powers using synthetic data.

    For synthetic data in test1.h5, we expect:
    - Zxy ≈ 100 for all frequency bands
    - Zyx should be related to Zxy by a known relationship

    This test validates that:
    1. Cross powers are computed correctly
    2. Using these cross powers to calculate impedances gives expected values
    """
    # Compute all needed cross powers
    channel_pairs = [
        ("ex", "hy"), ("ex", "hx"),  # E-H pairs needed for Zxy
        ("hx", "hx"), ("hy", "hy"), ("hx", "hy")  # H-H pairs for denominator
    ]

    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=channel_pairs
    )

    # For each frequency band, calculate Zxy
    for i, band in enumerate(test_frequency_bands.bands()):
        # Get cross and auto powers for this band
        ex_hy = xpowers.features["ex_hy"][i].values
        ex_hx = xpowers.features["ex_hx"][i].values
        hx_hx = xpowers.features["hx_hx"][i].values
        hy_hy = xpowers.features["hy_hy"][i].values
        hx_hy = xpowers.features["hx_hy"][i].values
        hy_hx = np.conj(hx_hy)  # Using conjugate property

        # Calculate Zxy using equation from tf_calculation.ipynb
        numerator = ex_hy * hx_hx - ex_hx * hx_hy
        denominator = hx_hx * hy_hy - hx_hy * hy_hx

        zxy = numerator / denominator

        # Test that magnitude is approximately 100
        assert np.abs(zxy).mean() == pytest.approx(100.0, rel=0.1), \
            f"Expected |Zxy| ≈ 100 for band {i}, got {np.abs(zxy).mean()}"


def test_impedance_from_synthetic_data(test1_spectrogram, test_frequency_bands):
    """
    Test impedance tensor calculation from cross powers using the synthetic data.

    The synthetic data in test1.asc represents a 100 Ohm-m halfspace model, which should show:
    1. Flat apparent resistivity at 100 Ohm-m
    2. Constant 45 degrees phase
    3. Data is organized as [hx, hy, hz, ex, ey] with units in nT and mV/km

    This test validates that the cross powers computed from the Fourier coefficients
    yield the expected impedance values.
    """
    # Compute all needed cross powers
    channel_pairs = [
        ("ex", "hy"), ("ex", "hx"),  # For Zxy
        ("ey", "hx"), ("ey", "hy"),  # For Zyx
        ("hx", "hx"), ("hy", "hy"),  # Auto powers
        ("hx", "hy")                 # Cross power
    ]

    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=channel_pairs
    )

    # Verify cross powers have time dimension
    for key in xpowers.features:
        assert xpowers.features[key].dims == ('band', 'time')
        assert len(xpowers.features[key].time) == len(test1_spectrogram.dataset.time)

    # Constants for converting to apparent resistivity
    MU0 = 4 * np.pi * 1e-7  # H/m

    # For each frequency band, calculate impedance tensor elements
    for i, band in enumerate(test_frequency_bands.bands()):
        # Center frequency for this band
        freq = (band.lower_bound + band.upper_bound) / 2
        omega = 2 * np.pi * freq

        # Get cross and auto powers for this band and average over time
        ex_hy = xpowers.features["ex_hy"][i].mean().values
        ex_hx = xpowers.features["ex_hx"][i].mean().values
        ey_hx = xpowers.features["ey_hx"][i].mean().values
        ey_hy = xpowers.features["ey_hy"][i].mean().values
        hx_hx = xpowers.features["hx_hx"][i].mean().values
        hy_hy = xpowers.features["hy_hy"][i].mean().values
        hx_hy = xpowers.features["hx_hy"][i].mean().values
        hy_hx = np.conj(hx_hy)

        # Calculate Zxy
        numerator_xy = ex_hy * hx_hx - ex_hx * hx_hy
        denominator = hx_hx * hy_hy - hx_hy * hy_hx
        zxy = numerator_xy / denominator

        # Calculate Zyx
        numerator_yx = ey_hx * hy_hy - ey_hy * hy_hx
        zyx = numerator_yx / denominator

        # Calculate apparent resistivity and phase for Zxy
        rho_xy = np.abs(zxy)**2 / (omega * MU0)
        phase_xy = np.angle(zxy, deg=True)

        # Calculate apparent resistivity and phase for Zyx
        rho_yx = np.abs(zyx)**2 / (omega * MU0)
        phase_yx = np.angle(zyx, deg=True)

        # Test that apparent resistivity is approximately 100 Ohm-m
        assert np.abs(rho_xy) == pytest.approx(100.0, rel=0.1), \
            f"Expected rho_xy ≈ 100 Ohm-m for band {i}, got {np.abs(rho_xy):.1f}"

        # Test that phase is approximately 45 degrees
        assert phase_xy == pytest.approx(45.0, abs=5.0), \
            f"Expected phase_xy ≈ 45° for band {i}, got {phase_xy:.1f}°"


if __name__ == "__main__":
    # tmp = TestSpectrogram()
    # tmp.test_initialize()
    unittest.main()
