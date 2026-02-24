# -*- coding: utf-8 -*-
"""Tests for :class:`mth5.clients.nims.NIMSClient` using real NIMS data."""
from pathlib import Path

import pytest


# Ensure subtests fixture is available; skip module if plugin is missing.
pytest.importorskip("pytest_subtests")

from mth5.clients.nims import NIMSClient
from mth5.mth5 import MTH5


try:
    import mth5_test_data

    NIMS_DATA_PATH = mth5_test_data.get_test_data_path("nims")
except ImportError:
    NIMS_DATA_PATH = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def nims_data_path():
    """Path to the packaged NIMS test data."""
    if NIMS_DATA_PATH is None:
        pytest.skip("mth5_test_data not available")

    path = Path(NIMS_DATA_PATH)
    if not path.exists():
        pytest.skip(f"NIMS test data path not found: {path}")

    return path


@pytest.fixture(scope="function")
def client_factory(tmp_path_factory, nims_data_path):
    """Factory to build isolated NIMSClient instances per test."""

    def _factory(**kwargs):
        save_dir = tmp_path_factory.mktemp("nims_client")
        return NIMSClient(data_path=nims_data_path, save_path=save_dir, **kwargs)

    return _factory


@pytest.fixture(scope="function")
def nims_client(client_factory):
    """Default NIMSClient instance with isolated save path."""

    return client_factory()


# =============================================================================
# Tests
# =============================================================================


def test_client_initialization_sets_defaults(nims_client, nims_data_path):
    """Client should wire defaults and paths correctly."""

    assert nims_client.data_path == nims_data_path
    assert nims_client.save_path.parent.exists()
    assert nims_client.sample_rates == [1.0, 8.0]
    assert nims_client.calibration_path is None
    assert nims_client.collection.file_path == nims_data_path


@pytest.mark.parametrize(
    "sample_rates, expected",
    [
        (8, [8.0]),
        ("8,1", [8.0, 1.0]),
        ([1, 8], [1.0, 8.0]),
    ],
)
def test_sample_rate_coercion(client_factory, sample_rates, expected):
    """Sample rates should be coerced to a float list regardless of input type."""

    client = client_factory(sample_rates=sample_rates)
    assert client.sample_rates == expected


def test_calibration_path_validation(client_factory, nims_data_path):
    """Invalid calibration paths should raise an error."""

    invalid_path = nims_data_path / "does_not_exist.cal"
    with pytest.raises(IOError):
        client_factory(calibration_path=invalid_path)


def test_get_run_dict_structure(nims_client, subtests):
    """Run dictionary should contain stations, runs, and usable DataFrames."""

    run_dict = nims_client.get_run_dict()
    assert run_dict, "Expected at least one station"

    for station_id, runs in run_dict.items():
        with subtests.test(msg="station-runs", station=station_id):
            assert runs, f"No runs found for station {station_id}"

            for run_id, run_df in runs.items():
                with subtests.test(msg="run-dataframe", station=station_id, run=run_id):
                    assert not run_df.empty
                    assert set(["station", "run", "fn"]).issubset(run_df.columns)
                    assert (run_df["station"].unique() == station_id).all()
                    assert run_df["fn"].apply(Path.exists).all()


def test_get_survey_from_runs(nims_client):
    """Survey name should be pulled from the run metadata."""

    run_dict = nims_client.get_run_dict()
    station_runs = next(iter(run_dict.values()))
    survey_name = nims_client.get_survey(station_runs)
    assert survey_name == nims_client.collection.survey_id


def test_make_mth5_from_nims_creates_runs(nims_client, subtests):
    """End-to-end creation should write an MTH5 with surveys, stations, and runs."""

    output_path = nims_client.make_mth5_from_nims(survey_id="nims_survey")
    assert output_path.exists()

    with MTH5() as mth5_obj:
        mth5_obj = mth5_obj.open_mth5(output_path, mode="r")
        survey_ids = mth5_obj.surveys_group.groups_list
        assert "nims_survey" in survey_ids

        survey_group = mth5_obj.get_survey("nims_survey")
        assert survey_group.metadata.id == "nims_survey"

        station_ids = survey_group.stations_group.groups_list
        assert station_ids, "Survey should contain stations"

        for station_id in station_ids:
            with subtests.test(msg="station-has-runs", station=station_id):
                station_group = survey_group.stations_group.get_station(station_id)
                assert station_group.metadata.id == station_id

                run_summary = station_group.run_summary
                assert (
                    not run_summary.empty
                ), f"Station {station_id} should contain runs"

                # Verify run_summary has expected columns
                expected_cols = ["id", "start", "end", "sample_rate"]
                with subtests.test(msg="run-summary-columns", station=station_id):
                    assert all(
                        col in run_summary.columns for col in expected_cols
                    ), f"run_summary missing columns: {expected_cols}"

                # Verify run_summary has valid data
                with subtests.test(msg="run-summary-data", station=station_id):
                    assert run_summary["sample_rate"].notna().all()
                    assert (run_summary["sample_rate"] > 0).all()
                    assert run_summary["start"].notna().all()
                    assert run_summary["end"].notna().all()
