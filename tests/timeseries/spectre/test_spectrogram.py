# -*- coding: utf-8 -*-
"""
This modules contains tests for the Spectrogram class.

WIP: There are also some tests of cross-power features in here, that should move to either
test_cross_powers or test_features.
- Most of the original cross power tests only considered single station data, keyed simply by
channel or component, such as `ex`, `hy`, etc.  These are largely only useful for tests and for
single station processing, as for most practical applications we need to at least include cross
powers for the reference station.  This is often handled by renaming remote channels `hx`, `hy`
to `rx`, `ry`.  An alternative is to bind the station and component into a unique channel name
and use a MultivariateDataset.
    -
TODO: Add a test_cross_powers that operates on a multivariate dataset.

"""
import mth5.mth5
from loguru import logger
from mt_metadata.processing.aurora import FrequencyBands
from mth5.data.make_mth5_from_asc import create_mth5_synthetic_file
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mth5.data.station_config import make_station_01
from mth5.helpers import close_open_files
from mth5.mth5 import MTH5
from mth5.timeseries.spectre import MultivariateDataset
from mth5.timeseries.spectre import Spectrogram
from mth5.timeseries.spectre.helpers import add_fcs_to_mth5
from mth5.timeseries.spectre.helpers import read_back_fcs
from mth5.processing.spectre.frequency_band_helpers import half_octave
from scipy.constants import mu_0

import unittest
import numpy as np
import xarray as xr
import pytest

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
     This test was moved to mth5 from aurora.  The aurora created spectrograms and then proceeded to process
     the mth5s to make TFs.  It would be nice to make some pytest fixtures that generate these mth5 with
     spectrograms so they could be reused by multiple tests. (there is an issue in MTH5 for this, #synthetic
     #fixture #pytest)
     TODO: This function (19 Feb 2025) does noet actually use the Spectrogram class -- check add_fcs_to_mth5
     if it should be using that.

    """

    @classmethod
    def setUpClass(self):
        """
        Makes some synthetic h5 files for testing.

        """
        logger.info("Making synthetic data")
        close_open_files()
        self.file_version = "0.1.0"
        mth5_path_1 = create_test1_h5(
            file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5
        )
        mth5_path_2 = create_test2_h5(
            file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5
        )
        mth5_path_3 = create_test3_h5(
            file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5
        )
        mth5_path_12rr = create_test12rr_h5(
            file_version=self.file_version, force_make_mth5=FORCE_MAKE_MTH5
        )
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
    """Test basic cross power computation for a single station."""
    # Define channel pairs to test, including a conjugate pair
    channel_pairs = [("ex", "ey"), ("hx", "hy"), ("ex", "hx"), ("hx", "ex")]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands, channel_pairs=channel_pairs
    )
    xpowers = xpowers.to_dataset(dim="variable")

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
        test_frequency_bands, channel_pairs=channel_pairs
    )
    xpowers = xpowers.to_dataset(dim="variable")

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
    eh_pairs = [
        ("ex", "hx"),
        ("ey", "hx"),
        ("ey", "hy"),
    ]  # Skip ex-hy (it's in conjugate_pairs below)

    # For conjugate test, add both forward and reverse pairs
    conjugate_pairs = [("ex", "hy"), ("hy", "ex")]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands,
        channel_pairs=e_pairs + h_pairs + eh_pairs + conjugate_pairs,
    )
    xpowers = xpowers.to_dataset(dim="variable")

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
        ("ex", "hx"),
        ("ex", "hy"),
        ("ey", "hx"),
        ("ey", "hy"),
        ("hx", "hx"),
        ("hy", "hy"),
        ("hx", "hy"),
    ]

    # Compute cross powers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands, channel_pairs=channel_pairs
    )
    xpowers = xpowers.to_dataset(dim="variable")

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
    assert (rho_xy_mean > 70.0).all()
    assert (rho_xy_mean < 130.0).all()


def test_store_and_read_cross_power_features(test1_spectrogram, test_frequency_bands):
    """
        Generate Cross powers as above, but this time:
            - store them as features
            - close the h5
            - open the h5
            - read them from file
            - assert they are equal to the computed values

    Parameters
    ----------
    test1_spectrogram
    test_frequency_bands

    Returns
    -------

    """
    # Define channel pairs to test
    channel_pairs = None  #  Setting to None will make all cross powers
    # channel_pairs = [("ex", "hx"), ("ey", "hy")]

    # Compute cross powers
    # TODO: add dcimation level contexst to makign these xpowers
    xpowers = test1_spectrogram.cross_powers(
        test_frequency_bands, channel_pairs=channel_pairs
    )
    xpowers = xpowers.to_dataset(dim="variable")

    # slightly hacky workaround for getting the mth5 path
    station_cfg = make_station_01()
    mth5_path = create_mth5_synthetic_file(
        station_cfgs=[station_cfg], mth5_name="test1.h5", force_make_mth5=False
    )

    # Now try storing the cross power features in the mth5 and close the file
    # - logic here follows tests/version_2/test_features.py
    m = MTH5(filename=mth5_path)
    m.open_mth5(mode="a")
    m.channel_summary.to_dataframe()
    station_group = m.get_station("test1", "EMTF Synthetic")
    features_group = station_group.features_group
    from mt_metadata.features.cross_powers import CrossPowers

    feature_md = CrossPowers()
    feature_md.name = "cross_powers"
    feature_md.id = "cross_powers"
    feature_fc = features_group.add_feature_group(
        feature_name="cross_powers", feature_metadata=feature_md
    )
    fc_run = feature_fc.add_feature_run_group("cross_powers", domain="frequency")
    dl = fc_run.add_decimation_level("0")
    sample_interval_in_nano_seconds = xpowers.time.diff(dim="time")[0].item()
    sample_rate = 1e9 / sample_interval_in_nano_seconds  # 0.010416666666666
    for key, value in xpowers.items():
        feature_ch = dl.add_channel(key)
        feature_ch.from_xarray(data=value, sample_rate_decimation_level=sample_rate)

    m.close_mth5()

    # Above shows we are OK storing a multiple channels one at a time.
    # TODO: Can we add them en-masse? e.g. dl.from_xarray(xpowers, sample_rate) ?
    # Will this cause performance issues if not?

    # open the mth5 and access the data
    # note slightly hacky _2 added to var names to make sure we are not interacting
    # with ones declared above
    # TODO: clean that up - maybe def _read_features()
    m.open_mth5(mode="r")
    station_group_2 = m.get_station("test1", survey="EMTF Synthetic")
    features_group_2 = station_group_2.features_group
    feature_fc_2 = features_group_2.get_feature_group("cross_powers")
    feature_run = feature_fc_2.get_feature_run_group("cross powers", domain="frequency")
    dl_2 = feature_run.get_decimation_level("0")
    for key, value in xpowers.items():
        stored_feature = dl_2.get_channel(key)
        accessed_data = stored_feature.to_numpy()  # this works
        assert np.isclose(accessed_data - xpowers[key].data, 0).all()
        # accessed_data.to_xarray()  # TODO: FIXME this doesn't work yet, needs work on metadata
    m.close_mth5()


def test_can_mvds(
    test_multivariate_spectrogram,  # pytest.Fixture -- TODO typehint this properly
):
    """
        Placeholder:
         Currently yields a multivariate spectrogram from fixture

         TODO make this into a test that generates mv crosspowers and coherences.

    Parameters
    ----------
    test_multivariate_spectrogram

    Returns
    -------

    """
    spectrogram = test_multivariate_spectrogram
    #  Pick a target frequency, say 1/4 way along the axis
    target_frequency = spectrogram.frequency_axis.data.mean() / 2
    frequency_band = half_octave(
        target_frequency=target_frequency, fft_frequencies=spectrogram.frequency_axis
    )
    band_spectrogram = spectrogram.extract_band(
        frequency_band=frequency_band, epsilon=0.0
    )
    xrds = band_spectrogram.flatten()
    mvds = MultivariateDataset(dataset=xrds)
    mvds.cross_power()

    return


# ======================== pytest fixtures ======================== #
# TODO: Move fixtures to conftest.py
@pytest.fixture
def test1_spectrogram():
    """
    Create a test spectrogram from synthetic data for the station named 'test1'.
    """

    # Create test configuration
    station_cfg = make_station_01()

    # Create test h5 file with Fourier coefficients
    mth5_path = create_mth5_synthetic_file(
        station_cfgs=[station_cfg], mth5_name="test1.h5", force_make_mth5=True
    )

    # Add Fourier coefficients to MTH5 file
    add_fcs_to_mth5(mth5_path, fc_decimations=None)

    # Read back the Fourier coefficients directly
    with MTH5() as m:
        m.open_mth5(mth5_path, mode="r")
        station_obj = m.get_station("test1", "EMTF Synthetic")
        fc_group = station_obj.fourier_coefficients_group.get_fc_group("001")
        dec_level = fc_group.get_decimation_level("0")  # Get first decimation level
        ds = dec_level.to_xarray()
        return Spectrogram(dataset=ds)


@pytest.fixture
def test_multivariate_spectrogram():
    """
        Create a MultiVariate dataset object with spectrograms.

        In this example, all runs are assumed to be within a single h5 archive.
        TODO: A more general workflow for this could involve runs from multiple h5 files.

        Development Notes:
            - In this example, we are supposing that we already know which runs from which stations will be merged
            into a multivariate dataset.
            - The runs are packed into a list of FCRunChunk objects, and a


    Returns
    -------

    """
    from mth5.mth5 import MTH5
    from mth5.timeseries.spectre.multiple_station import make_multistation_spectrogram
    from mth5.timeseries.spectre.multiple_station import FCRunChunk

    # get the path to an mth5 file that has multiple stations
    mth5_path_12rr = create_test12rr_h5(
        file_version="0.1.0",
        force_make_mth5=False,  # normally FORCE_MAKE_MTH5=False here
    )

    # Careful here with hard-coded station and run ids
    # Also, it is assumed that these runs span the same time interval
    fc_run_chunk_1 = FCRunChunk(
        station_id="test1",
        run_id="001",
        decimation_level_id="0",
        start="",
        end="",
        channels=(),
    )

    fc_run_chunk_2 = FCRunChunk(
        station_id="test2",
        run_id="001",
        decimation_level_id="0",
        start="",
        end="",
        channels=(),
    )

    with MTH5().open_mth5(mth5_path_12rr, mode="r") as m:
        mvds = make_multistation_spectrogram(
            m, [fc_run_chunk_1, fc_run_chunk_2], rtype=None
        )
    assert len(mvds.channels) == 10

    spectrogram = Spectrogram(dataset=mvds.dataset)

    return spectrogram


@pytest.fixture
def test_frequency_bands():
    """Create frequency bands for testing"""
    edges = np.array(
        [[0.01, 0.1], [0.1, 0.2], [0.2, 0.49]]  # Low band  # Mid band  # High band
    )
    return FrequencyBands(edges)


# ======================== helper functions ======================== #
# This is a placeholder - delete if unneeded.


if __name__ == "__main__":
    test_store_and_read_cross_power_features()
    unittest.main()
