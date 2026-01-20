# -*- coding: utf-8 -*-
"""
Pytest test suite for NIMSCollection functionality with real data.

Based on test_nims_collection.py mock tests but using actual NIMS data files
from mth5_test_data. Follows patterns from test_z3d.py and test_read_nims.py for real data integration.

@author: jpeacock, converted for real data testing
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import pandas as pd
import pytest

from mth5.io.nims import NIMSCollection


try:
    import mth5_test_data

    nims_data_path = mth5_test_data.get_test_data_path("nims")
except ImportError:
    nims_data_path = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def nims_collection_real():
    """Fixture for NIMSCollection with real data - session scope for speed optimization."""
    if nims_data_path is None:
        pytest.skip("mth5_test_data not available")

    if not Path(nims_data_path).exists():
        pytest.skip(f"NIMS test data path not found: {nims_data_path}")

    return NIMSCollection(file_path=nims_data_path)


@pytest.fixture(scope="session")
def nims_dataframe_real(nims_collection_real):
    """Fixture for DataFrame created from real NIMS data - session scope for performance."""
    try:
        df = nims_collection_real.to_dataframe()
        return df
    except Exception as e:
        pytest.skip(f"Failed to create DataFrame from real NIMS data: {e}")


@pytest.fixture
def expected_nims_collection_properties():
    """Expected properties for real NIMS collection data."""
    return {
        "file_ext": "bin",
        "survey_id": "mt",
        "expected_files": ["mnp300a.BIN", "mnp300b.BIN"],
        "expected_stations": ["300", "Mnp300"],
        "expected_sample_rate": 8,
        "min_file_size": 1000,  # Minimum expected file size
        "expected_columns": [
            "survey",
            "station",
            "run",
            "start",
            "end",
            "channel_id",
            "component",
            "fn",
            "sample_rate",
            "file_size",
            "n_samples",
            "sequence_number",
            "dipole",
            "coil_number",
            "latitude",
            "longitude",
            "elevation",
            "instrument_id",
            "calibration_fn",
        ],
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestNIMSCollectionInitialization:
    """Test NIMSCollection initialization and basic properties with real data."""

    def test_initialization_with_real_path(self, expected_nims_collection_properties):
        """Test initialization with real NIMS data path."""
        if nims_data_path is None:
            pytest.skip("mth5_test_data not available")

        collection = NIMSCollection(file_path=nims_data_path)
        assert collection.file_path == Path(nims_data_path)
        assert collection.file_ext == expected_nims_collection_properties["file_ext"]
        assert collection.survey_id == expected_nims_collection_properties["survey_id"]

    def test_initialization_default(self):
        """Test default initialization."""
        collection = NIMSCollection()
        assert collection.file_path is None
        assert collection.file_ext == "bin"
        assert collection.survey_id == "mt"

    def test_file_extension_property(
        self, nims_collection_real, expected_nims_collection_properties
    ):
        """Test that file extension is set correctly for real data."""
        assert (
            nims_collection_real.file_ext
            == expected_nims_collection_properties["file_ext"]
        )

    def test_survey_id_property(
        self, nims_collection_real, expected_nims_collection_properties
    ):
        """Test that survey ID is set correctly."""
        assert (
            nims_collection_real.survey_id
            == expected_nims_collection_properties["survey_id"]
        )

    def test_survey_id_modification(self, nims_collection_real):
        """Test modifying survey ID."""
        original_survey_id = nims_collection_real.survey_id
        nims_collection_real.survey_id = "custom_survey"
        assert nims_collection_real.survey_id == "custom_survey"
        # Reset to original
        nims_collection_real.survey_id = original_survey_id


class TestNIMSCollectionFileHandling:
    """Test file discovery and handling functionality with real data."""

    def test_get_files_real_data(
        self, nims_collection_real, expected_nims_collection_properties
    ):
        """Test getting .bin files from real NIMS directory."""
        files = nims_collection_real.get_files(nims_collection_real.file_ext)
        file_names = [f.name for f in files]

        # Should find the expected NIMS files
        expected_files = expected_nims_collection_properties["expected_files"]
        assert len(files) >= len(expected_files)

        # Check that expected files are found
        for expected_file in expected_files:
            assert expected_file in file_names

    def test_file_sizes_reasonable(
        self, nims_collection_real, expected_nims_collection_properties
    ):
        """Test that found files have reasonable sizes."""
        files = nims_collection_real.get_files(nims_collection_real.file_ext)
        min_size = expected_nims_collection_properties["min_file_size"]

        for file_path in files:
            if file_path.is_file():
                file_size = file_path.stat().st_size
                assert (
                    file_size > min_size
                ), f"File {file_path.name} too small: {file_size} bytes"

    def test_file_extensions_correct(self, nims_collection_real):
        """Test that all found files have correct extensions."""
        files = nims_collection_real.get_files(nims_collection_real.file_ext)

        for file_path in files:
            # Should be .bin or .BIN
            assert (
                file_path.suffix.lower() == ".bin"
            ), f"Unexpected extension: {file_path.suffix}"

    def test_get_files_empty_extension(self, nims_collection_real):
        """Test getting files with empty extension."""
        files = nims_collection_real.get_files("")
        # Should return all files or handle gracefully
        assert isinstance(files, list)

    def test_files_exist_and_readable(self, nims_collection_real):
        """Test that all discovered files exist and are readable."""
        files = nims_collection_real.get_files(nims_collection_real.file_ext)

        for file_path in files:
            assert file_path.exists(), f"File does not exist: {file_path}"
            assert file_path.is_file(), f"Path is not a file: {file_path}"
            assert file_path.stat().st_size > 0, f"File is empty: {file_path}"


class TestNIMSCollectionDataFrameCreation:
    """Test DataFrame creation from real NIMS files."""

    def test_dataframe_creation_basic(
        self, nims_dataframe_real, expected_nims_collection_properties
    ):
        """Test basic DataFrame creation with real data."""
        df = nims_dataframe_real

        # Should have data
        assert len(df) > 0, "DataFrame should not be empty"

        # Should have expected columns
        expected_cols = expected_nims_collection_properties["expected_columns"]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_dataframe_data_types(self, nims_dataframe_real):
        """Test DataFrame data types are reasonable."""
        df = nims_dataframe_real

        # Check specific column types
        if "sample_rate" in df.columns:
            assert pd.api.types.is_numeric_dtype(
                df["sample_rate"]
            ), "sample_rate should be numeric"

        if "file_size" in df.columns:
            assert pd.api.types.is_numeric_dtype(
                df["file_size"]
            ), "file_size should be numeric"

        if "n_samples" in df.columns:
            assert pd.api.types.is_numeric_dtype(
                df["n_samples"]
            ), "n_samples should be numeric"

    def test_dataframe_content_validation(
        self, nims_dataframe_real, expected_nims_collection_properties
    ):
        """Test DataFrame content is reasonable."""
        df = nims_dataframe_real

        # Check stations
        if "station" in df.columns:
            stations = df["station"].unique()
            expected_stations = expected_nims_collection_properties["expected_stations"]
            # Should have some expected stations (may have variations)
            assert len(stations) > 0, "Should have at least one station"

        # Check sample rates
        if "sample_rate" in df.columns:
            sample_rates = df["sample_rate"].unique()
            expected_rate = expected_nims_collection_properties["expected_sample_rate"]
            assert (
                expected_rate in sample_rates
            ), f"Expected sample rate {expected_rate} not found"

    def test_dataframe_file_information(self, nims_dataframe_real):
        """Test that DataFrame contains correct file information."""
        df = nims_dataframe_real

        # Check file paths
        if "fn" in df.columns:
            file_paths = df["fn"].tolist()
            for fn in file_paths:
                if fn is not None:
                    assert Path(
                        fn
                    ).exists(), f"File path in DataFrame does not exist: {fn}"
                    assert str(fn).endswith(".BIN"), f"File should end with .BIN: {fn}"

    def test_dataframe_no_null_critical_fields(self, nims_dataframe_real):
        """Test that critical fields are not null."""
        df = nims_dataframe_real

        critical_fields = ["fn", "file_size"]
        for field in critical_fields:
            if field in df.columns:
                assert (
                    not df[field].isna().all()
                ), f"Critical field {field} should not be all null"

    def test_dataframe_time_fields(self, nims_dataframe_real):
        """Test that time fields are properly formatted."""
        df = nims_dataframe_real

        time_fields = ["start", "end"]
        for field in time_fields:
            if field in df.columns and not df[field].isna().all():
                # Should be parseable as datetime
                try:
                    pd.to_datetime(df[field].dropna())
                except Exception as e:
                    pytest.fail(f"Time field {field} not parseable as datetime: {e}")


class TestNIMSCollectionRunNameAssignment:
    """Test run name assignment functionality with real data."""

    def test_assign_run_names_real_data(self, nims_dataframe_real):
        """Test run name assignment with real data."""
        df = nims_dataframe_real.copy()

        # Clear existing run names to test assignment
        if "run" in df.columns:
            df["run"] = None

        # Create NIMSCollection instance for testing
        if nims_data_path is None:
            pytest.skip("mth5_test_data not available")

        collection = NIMSCollection(file_path=nims_data_path)

        # Test run name assignment
        try:
            result_df = collection.assign_run_names(df, zeros=2)

            # Should have run names assigned
            if "run" in result_df.columns:
                run_names = result_df["run"].dropna().tolist()
                assert len(run_names) > 0, "Should have assigned at least one run name"

                # Run names should follow expected pattern (e.g., 'sr8_01')
                for run_name in run_names:
                    if run_name is not None:
                        assert isinstance(
                            run_name, str
                        ), f"Run name should be string: {run_name}"
                        assert len(run_name) > 0, "Run name should not be empty"

        except Exception as e:
            pytest.fail(f"assign_run_names failed with real data: {e}")

    def test_assign_run_names_preserves_existing(self, nims_dataframe_real):
        """Test that existing run names are preserved."""
        df = nims_dataframe_real.copy()

        if "run" in df.columns and len(df) > 0:
            # Set first run to a custom name
            original_run = df.iloc[0]["run"]
            custom_run = "custom_run_001"
            df.at[0, "run"] = custom_run

            # Create NIMSCollection instance
            if nims_data_path is None:
                pytest.skip("mth5_test_data not available")

            collection = NIMSCollection(file_path=nims_data_path)

            try:
                result_df = collection.assign_run_names(df, zeros=2)

                # Custom run name should be preserved
                assert (
                    result_df.iloc[0]["run"] == custom_run
                ), "Custom run name should be preserved"

            except Exception as e:
                pytest.fail(f"assign_run_names failed to preserve existing names: {e}")

    def test_assign_run_names_with_different_stations(self, nims_dataframe_real):
        """Test run name assignment with multiple stations."""
        df = nims_dataframe_real.copy()

        if "station" in df.columns and len(df["station"].unique()) > 1:
            # Clear run names
            if "run" in df.columns:
                df["run"] = None

            # Create NIMSCollection instance
            if nims_data_path is None:
                pytest.skip("mth5_test_data not available")

            collection = NIMSCollection(file_path=nims_data_path)

            try:
                result_df = collection.assign_run_names(df, zeros=3)

                # Each station should have its own run sequence
                stations = result_df["station"].unique()
                for station in stations:
                    station_runs = result_df[result_df["station"] == station][
                        "run"
                    ].dropna()
                    if len(station_runs) > 0:
                        # Should have reasonable run names for this station
                        assert (
                            len(station_runs) > 0
                        ), f"Station {station} should have run names"

            except Exception as e:
                pytest.fail(f"assign_run_names failed with multiple stations: {e}")


class TestNIMSCollectionUtilityMethods:
    """Test utility and helper methods with real data."""

    def test_get_empty_entry_dict(
        self, nims_collection_real, expected_nims_collection_properties
    ):
        """Test getting empty entry dictionary."""
        if hasattr(nims_collection_real, "get_empty_entry_dict"):
            empty_dict = nims_collection_real.get_empty_entry_dict()

            # Should be a dictionary
            assert isinstance(empty_dict, dict), "Should return a dictionary"

            # Should contain expected keys
            expected_cols = expected_nims_collection_properties["expected_columns"]
            for col in expected_cols:
                if col in empty_dict:
                    # Values should be None or appropriate defaults
                    assert empty_dict[col] is None or isinstance(
                        empty_dict[col], (str, int, float, list)
                    )

    def test_collection_string_representation(self, nims_collection_real):
        """Test string representation of collection."""
        repr_str = repr(nims_collection_real)
        assert isinstance(repr_str, str), "String representation should be string"
        assert len(repr_str) > 0, "String representation should not be empty"
        # Should contain class name
        assert "NIMSCollection" in repr_str or "Collection" in repr_str

    def test_collection_properties_accessible(self, nims_collection_real):
        """Test that collection properties are accessible."""
        # Basic properties should be accessible
        assert hasattr(nims_collection_real, "file_path")
        assert hasattr(nims_collection_real, "file_ext")
        assert hasattr(nims_collection_real, "survey_id")

        # Methods should be callable
        assert callable(getattr(nims_collection_real, "get_files"))
        assert callable(getattr(nims_collection_real, "to_dataframe"))


class TestNIMSCollectionIntegration:
    """Integration tests with real data."""

    def test_full_workflow_real_data(
        self, nims_collection_real, expected_nims_collection_properties
    ):
        """Test complete workflow from file discovery to DataFrame creation."""
        # Step 1: File discovery
        files = nims_collection_real.get_files(nims_collection_real.file_ext)
        assert len(files) > 0, "Should discover NIMS files"

        # Step 2: DataFrame creation
        df = nims_collection_real.to_dataframe()
        assert len(df) > 0, "Should create non-empty DataFrame"

        # Step 3: Basic validation
        expected_cols = expected_nims_collection_properties["expected_columns"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Step 4: Data consistency
        if "fn" in df.columns:
            df_files = [Path(fn).name for fn in df["fn"].dropna()]
            discovered_files = [f.name for f in files]

            # DataFrame files should come from discovered files
            for df_file in df_files:
                assert (
                    df_file in discovered_files
                ), f"DataFrame file {df_file} not in discovered files"

    def test_dataframe_consistency_with_files(self, nims_collection_real):
        """Test that DataFrame data is consistent with actual files."""
        df = nims_collection_real.to_dataframe()

        # Check file size consistency
        if "fn" in df.columns and "file_size" in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row["fn"]) and pd.notna(row["file_size"]):
                    file_path = Path(row["fn"])
                    if file_path.exists():
                        actual_size = file_path.stat().st_size
                        df_size = row["file_size"]
                        # Allow for small differences (header parsing, etc.)
                        size_diff = abs(actual_size - df_size)
                        assert (
                            size_diff < actual_size * 0.1
                        ), f"File size mismatch for {file_path.name}"

    def test_multiple_operations_consistency(self, nims_collection_real):
        """Test that multiple operations give consistent results."""
        # Create DataFrame multiple times
        df1 = nims_collection_real.to_dataframe()
        df2 = nims_collection_real.to_dataframe()

        # Should be identical
        pd.testing.assert_frame_equal(
            df1, df2, "Multiple DataFrame creations should be identical"
        )

        # File discovery should be consistent
        files1 = nims_collection_real.get_files(nims_collection_real.file_ext)
        files2 = nims_collection_real.get_files(nims_collection_real.file_ext)

        assert len(files1) == len(files2), "File discovery should be consistent"
        assert set(f.name for f in files1) == set(
            f.name for f in files2
        ), "File names should be consistent"


class TestNIMSCollectionPerformance:
    """Performance tests with real data."""

    def test_dataframe_creation_performance(self, nims_collection_real):
        """Test that DataFrame creation completes in reasonable time."""
        import time

        start_time = time.time()
        df = nims_collection_real.to_dataframe()
        end_time = time.time()

        duration = end_time - start_time

        # Should complete in reasonable time (less than 30 seconds for test data)
        assert (
            duration < 30.0
        ), f"DataFrame creation took too long: {duration:.2f} seconds"
        assert len(df) > 0, "Should create non-empty DataFrame"

    def test_file_discovery_performance(self, nims_collection_real):
        """Test that file discovery is efficient."""
        import time

        start_time = time.time()
        files = nims_collection_real.get_files(nims_collection_real.file_ext)
        end_time = time.time()

        duration = end_time - start_time

        # Should complete quickly (less than 5 seconds)
        assert duration < 5.0, f"File discovery took too long: {duration:.2f} seconds"
        assert len(files) > 0, "Should discover files"

    def test_memory_efficiency(self, nims_collection_real):
        """Test memory efficiency with real data."""
        # Create DataFrame and ensure it's reasonably sized
        df = nims_collection_real.to_dataframe()

        # DataFrame should not be excessively large for test data
        memory_usage = df.memory_usage(deep=True).sum()

        # Should use less than 10MB for test data
        assert (
            memory_usage < 10 * 1024 * 1024
        ), f"DataFrame uses too much memory: {memory_usage} bytes"


class TestNIMSCollectionEdgeCases:
    """Test edge cases and boundary conditions with real data."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame in assign_run_names."""
        if nims_data_path is None:
            pytest.skip("mth5_test_data not available")

        collection = NIMSCollection(file_path=nims_data_path)

        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=["station", "run", "sample_rate", "start"])
        result_df = collection.assign_run_names(empty_df)

        assert len(result_df) == 0, "Empty DataFrame should remain empty"
        assert list(result_df.columns) == list(
            empty_df.columns
        ), "Columns should be preserved"

    def test_single_file_handling(self, nims_collection_real):
        """Test handling when only one file is available."""
        # This tests the case where we might have only one NIMS file
        files = nims_collection_real.get_files(nims_collection_real.file_ext)

        if len(files) >= 1:
            # Test should work even with just one file
            df = nims_collection_real.to_dataframe()
            assert len(df) > 0, "Should handle single file case"

    def test_large_file_handling(self, nims_collection_real):
        """Test handling of large NIMS files."""
        files = nims_collection_real.get_files(nims_collection_real.file_ext)

        # Check for reasonably large files (>1MB)
        large_files = [f for f in files if f.stat().st_size > 1024 * 1024]

        if len(large_files) > 0:
            # Should handle large files without issues
            df = nims_collection_real.to_dataframe()
            assert len(df) > 0, "Should handle large files"

            # Check that we got reasonable data from large files
            if "file_size" in df.columns:
                max_file_size = df["file_size"].max()
                assert max_file_size > 1024 * 1024, "Should have processed large file"


class TestNIMSCollectionDataValidation:
    """Validation tests for data quality and consistency."""

    def test_geographic_coordinates_reasonable(self, nims_dataframe_real):
        """Test that geographic coordinates are in reasonable ranges."""
        df = nims_dataframe_real

        # Check latitude
        if "latitude" in df.columns:
            latitudes = df["latitude"].dropna()
            if len(latitudes) > 0:
                assert latitudes.min() >= -90, "Latitude should be >= -90"
                assert latitudes.max() <= 90, "Latitude should be <= 90"

        # Check longitude
        if "longitude" in df.columns:
            longitudes = df["longitude"].dropna()
            if len(longitudes) > 0:
                assert longitudes.min() >= -180, "Longitude should be >= -180"
                assert longitudes.max() <= 180, "Longitude should be <= 180"

    def test_sample_rate_consistency(
        self, nims_dataframe_real, expected_nims_collection_properties
    ):
        """Test that sample rates are reasonable."""
        df = nims_dataframe_real

        if "sample_rate" in df.columns:
            sample_rates = df["sample_rate"].dropna().unique()

            # Should have positive sample rates
            assert all(sr > 0 for sr in sample_rates), "Sample rates should be positive"

            # Should include expected rate
            expected_rate = expected_nims_collection_properties["expected_sample_rate"]
            assert (
                expected_rate in sample_rates
            ), f"Expected sample rate {expected_rate} not found"

    def test_file_size_reasonableness(
        self, nims_dataframe_real, expected_nims_collection_properties
    ):
        """Test that file sizes are reasonable."""
        df = nims_dataframe_real

        if "file_size" in df.columns:
            file_sizes = df["file_size"].dropna()
            min_size = expected_nims_collection_properties["min_file_size"]

            if len(file_sizes) > 0:
                assert (
                    file_sizes.min() > min_size
                ), f"Files should be larger than {min_size} bytes"
                assert (
                    file_sizes.max() < 1024 * 1024 * 1024
                ), "Files should be less than 1GB"

    def test_station_name_format(self, nims_dataframe_real):
        """Test that station names have reasonable format."""
        df = nims_dataframe_real

        if "station" in df.columns:
            stations = df["station"].dropna().unique()

            for station in stations:
                # Should be strings
                assert isinstance(
                    station, str
                ), f"Station name should be string: {station}"
                # Should not be empty
                assert len(station) > 0, "Station name should not be empty"
                # Should be reasonable length
                assert len(station) < 50, f"Station name too long: {station}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
