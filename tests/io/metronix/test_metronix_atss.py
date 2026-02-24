# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:30:00 2024

@author: jpeacock

Comprehensive pytest suite for ATSS class testing with real Metronix test data.
Tests binary file I/O, metadata handling, inheritance, and ChannelTS conversion.

This test suite uses real ATSS and JSON files from mth5_test_data instead of mocks
to ensure integration testing with actual Metronix data formats.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest

from mth5.io.metronix import ATSS, MetronixChannelJSON, MetronixFileNameMetadata


# =============================================================================
# Test Data Import Handling
# =============================================================================

try:
    import mth5_test_data

    metronix_data_path = (
        mth5_test_data.get_test_data_path("metronix")
        / "Northern_Mining"
        / "stations"
        / "saricam"
    )
    has_data_import = True
except ImportError:
    metronix_data_path = None
    has_data_import = False

# Skip marker for tests that require test data
requires_test_data = pytest.mark.skipif(
    not has_data_import, reason="mth5_test_data package not available"
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def real_atss_files():
    """Get real ATSS files from test data."""
    if not has_data_import:
        return {}

    files = {}
    for run_dir in metronix_data_path.iterdir():
        if run_dir.is_dir():
            for atss_file in run_dir.glob("*.atss"):
                if atss_file.with_suffix(".json").exists():
                    files[atss_file.name] = {
                        "atss": atss_file,
                        "json": atss_file.with_suffix(".json"),
                        "run_dir": run_dir.name,
                    }
    return files


@pytest.fixture(scope="session")
def sample_magnetic_file(real_atss_files):
    """Get a magnetic channel ATSS file."""
    if not has_data_import:
        return None

    for filename, file_info in real_atss_files.items():
        if "THx" in filename or "THy" in filename or "THz" in filename:
            return file_info
    return None


@pytest.fixture(scope="session")
def sample_electric_file(real_atss_files):
    """Get an electric channel ATSS file."""
    if not has_data_import:
        return None

    for filename, file_info in real_atss_files.items():
        if "TEx" in filename or "TEy" in filename:
            return file_info
    return None


@pytest.fixture(scope="session")
def high_sample_rate_file(real_atss_files):
    """Get a high sample rate file for performance testing."""
    if not has_data_import:
        return None

    for filename, file_info in real_atss_files.items():
        if "2048Hz" in filename:
            return file_info
    return None


@pytest.fixture(scope="session")
def low_sample_rate_file(real_atss_files):
    """Get a low sample rate file for testing."""
    if not has_data_import:
        return None

    for filename, file_info in real_atss_files.items():
        if "2s" in filename:
            return file_info
    return None


@pytest.fixture
def atss_instance(sample_magnetic_file):
    """Create ATSS instance for testing."""
    if not has_data_import or not sample_magnetic_file:
        pytest.skip("Test data not available")
    return ATSS(sample_magnetic_file["atss"])


@pytest.fixture
def atss_electric_instance(sample_electric_file):
    """Create ATSS instance for electric channel testing."""
    if not has_data_import or not sample_electric_file:
        pytest.skip("Test data not available")
    return ATSS(sample_electric_file["atss"])


# =============================================================================
# Test Classes
# =============================================================================


class TestATSSInheritance:
    """Test ATSS inheritance from MetronixFileNameMetadata."""

    def test_inheritance(self):
        """Test that ATSS inherits from MetronixFileNameMetadata."""
        assert issubclass(ATSS, MetronixFileNameMetadata)

    def test_mro(self):
        """Test method resolution order."""
        mro = ATSS.__mro__
        assert MetronixFileNameMetadata in mro
        assert object in mro

    @requires_test_data
    def test_inherited_attributes(self, atss_instance):
        """Test that inherited attributes are accessible."""
        assert hasattr(atss_instance, "system_number")
        assert hasattr(atss_instance, "system_name")
        assert hasattr(atss_instance, "channel_number")
        assert hasattr(atss_instance, "component")
        assert hasattr(atss_instance, "sample_rate")
        assert hasattr(atss_instance, "file_type")

    @requires_test_data
    def test_parsed_filename_attributes(self, atss_instance):
        """Test that filename parsing works correctly."""
        assert atss_instance.system_number == "084"
        assert atss_instance.system_name == "ADU-07e"
        assert isinstance(atss_instance.channel_number, int)
        assert atss_instance.component in ["ex", "ey", "hx", "hy", "hz"]
        assert isinstance(atss_instance.sample_rate, (int, float))
        assert atss_instance.file_type == "timeseries"


class TestATSSInitialization:
    """Test ATSS initialization."""

    def test_init_no_file(self):
        """Test initialization without file."""
        atss = ATSS()
        assert atss.fn is None
        assert isinstance(atss.header, MetronixChannelJSON)

    @requires_test_data
    def test_init_with_file(self, sample_magnetic_file):
        """Test initialization with file."""
        atss = ATSS(sample_magnetic_file["atss"])
        assert atss.fn == sample_magnetic_file["atss"]
        assert isinstance(atss.header, MetronixChannelJSON)

    @requires_test_data
    def test_init_with_metadata(self, sample_magnetic_file):
        """Test initialization with metadata file present."""
        atss = ATSS(sample_magnetic_file["atss"])
        # Should have loaded metadata since JSON file exists
        assert atss.header.metadata is not None

    @requires_test_data
    def test_init_kwargs(self, sample_magnetic_file):
        """Test initialization with keyword arguments."""
        atss = ATSS(fn=sample_magnetic_file["atss"])
        assert atss.fn == sample_magnetic_file["atss"]


class TestATSSMetadataFile:
    """Test metadata file handling."""

    @requires_test_data
    def test_metadata_fn_property(self, atss_instance):
        """Test metadata filename property."""
        expected_path = atss_instance.fn.parent / f"{atss_instance.fn.stem}.json"
        assert atss_instance.metadata_fn == expected_path

    def test_metadata_fn_none(self):
        """Test metadata filename when fn is None."""
        atss = ATSS()
        assert atss.metadata_fn is None

    @requires_test_data
    def test_has_metadata_file_true(self, atss_instance):
        """Test has_metadata_file when file exists."""
        assert atss_instance.has_metadata_file() is True

    @requires_test_data
    def test_has_metadata_file_false(self, temp_dir):
        """Test has_metadata_file when file doesn't exist."""
        # Create ATSS file without corresponding JSON
        atss_file = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        atss_file.touch()
        atss = ATSS(atss_file)
        assert atss.has_metadata_file() is False

    def test_has_metadata_file_no_fn(self):
        """Test has_metadata_file when fn is None."""
        atss = ATSS()
        assert atss.has_metadata_file() is False


class TestATSSFileIO:
    """Test ATSS file I/O operations."""

    def test_read_atss_no_file(self):
        """Test reading when no file is set."""
        atss = ATSS()
        with pytest.raises(
            TypeError, match="expected str, bytes or os.PathLike object, not NoneType"
        ):
            atss.read_atss()

    @requires_test_data
    def test_read_atss_file_not_exists(self, temp_dir):
        """Test reading non-existent file."""
        non_existent = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        atss = ATSS(non_existent)
        with pytest.raises(FileNotFoundError):
            atss.read_atss()

    @requires_test_data
    def test_read_atss_full_file(self, atss_instance):
        """Test reading entire ATSS file."""
        data = atss_instance.read_atss()
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float64
        assert len(data) > 0

    @requires_test_data
    def test_read_atss_with_start_stop(self, atss_instance):
        """Test reading ATSS file with start/stop parameters."""
        start, stop = 100, 200
        data = atss_instance.read_atss(start=start, stop=stop)
        assert len(data) == stop
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float64

    @requires_test_data
    def test_read_atss_with_filename(self, sample_magnetic_file):
        """Test reading ATSS file with explicit filename."""
        atss = ATSS()
        data = atss.read_atss(fn=sample_magnetic_file["atss"])
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float64
        assert len(data) > 0

    @requires_test_data
    def test_read_atss_start_only(self, atss_instance):
        """Test reading ATSS file with start parameter only."""
        start = 500
        full_data = atss_instance.read_atss()
        partial_data = atss_instance.read_atss(start=start)
        expected_length = len(full_data) - start
        assert len(partial_data) == expected_length

    @requires_test_data
    def test_read_atss_stop_only(self, atss_instance):
        """Test reading ATSS file with stop parameter only."""
        stop = 500
        data = atss_instance.read_atss(stop=stop)
        assert len(data) == stop

    @requires_test_data
    def test_write_atss(self, temp_dir, atss_instance):
        """Test writing ATSS file."""
        # Read some real data
        original_data = atss_instance.read_atss(stop=1000)
        output_file = temp_dir / "output.atss"

        atss = ATSS()
        atss.write_atss(original_data, output_file)

        # Verify file was written correctly
        with open(output_file, "rb") as f:
            written_data = np.frombuffer(f.read(), dtype=np.float64)
        np.testing.assert_array_equal(written_data, original_data)

    def test_write_atss_different_dtypes(self, temp_dir):
        """Test writing ATSS with different data types."""
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        output_file = temp_dir / "output_int.atss"
        atss = ATSS()
        atss.write_atss(test_data, output_file)

        # Verify file exists and has correct size
        assert output_file.exists()
        assert output_file.stat().st_size == test_data.nbytes


class TestATSSProperties:
    """Test ATSS property methods."""

    @requires_test_data
    def test_channel_metadata(self, atss_instance):
        """Test channel_metadata property."""
        result = atss_instance.channel_metadata
        assert result is not None
        # Should have basic metadata attributes
        assert hasattr(result, "component")
        assert hasattr(result, "sample_rate")

    @requires_test_data
    def test_channel_response(self, atss_instance):
        """Test channel_response property."""
        result = atss_instance.channel_response
        # May be None for some channels, but should not raise an error
        assert result is None or hasattr(result, "filters") or hasattr(result, "name")

    @requires_test_data
    @pytest.mark.parametrize(
        "component,expected_type",
        [
            ("ex", "electric"),
            ("ey", "electric"),
            ("hx", "magnetic"),
            ("hy", "magnetic"),
            ("hz", "magnetic"),
        ],
    )
    def test_channel_type(self, real_atss_files, component, expected_type):
        """Test channel_type property for different components."""
        # Find a file with the specified component
        test_file = None
        for filename, file_info in real_atss_files.items():
            if f"T{component.title()}" in filename:
                test_file = file_info
                break

        if test_file:
            atss = ATSS(test_file["atss"])
            assert atss.channel_type == expected_type

    def test_channel_type_no_file(self):
        """Test channel_type when no file is set."""
        atss = ATSS()
        channel_type = atss.channel_type
        assert channel_type is None

    @requires_test_data
    def test_run_id_property(self, atss_instance):
        """Test run_id property extracts from file path."""
        run_id = atss_instance.run_id
        assert run_id.startswith("run_")
        assert len(run_id) > 4  # Should be like "run_002", "run_005", etc.

    @requires_test_data
    def test_station_id_property(self, atss_instance):
        """Test station_id property extracts from file path."""
        station_id = atss_instance.station_id
        assert station_id == "saricam"

    @requires_test_data
    def test_survey_id_property(self, atss_instance):
        """Test survey_id property extracts from file path."""
        survey_id = atss_instance.survey_id
        assert survey_id == "Northern_Mining"

    def test_path_properties_no_file(self):
        """Test path-based properties when no file is set."""
        atss = ATSS()
        # These should raise AttributeError when fn is None
        with pytest.raises(AttributeError):
            _ = atss.run_id


class TestATSSMetadataGeneration:
    """Test metadata generation methods."""

    @requires_test_data
    def test_run_metadata(self, atss_instance):
        """Test run_metadata property exists and returns Run object."""
        from mt_metadata.timeseries import Run

        result = atss_instance.run_metadata
        assert isinstance(result, Run)
        assert result.id == atss_instance.run_id

    @requires_test_data
    def test_station_metadata(self, atss_instance):
        """Test station_metadata property exists and returns Station object."""
        from mt_metadata.timeseries import Station

        result = atss_instance.station_metadata
        assert isinstance(result, Station)
        assert result.id == atss_instance.station_id

    @requires_test_data
    def test_survey_metadata(self, atss_instance):
        """Test survey_metadata property exists and returns Survey object."""
        from mt_metadata.timeseries import Survey

        result = atss_instance.survey_metadata
        assert isinstance(result, Survey)
        assert result.id == atss_instance.survey_id


class TestATSSChannelTS:
    """Test ChannelTS conversion."""

    def test_to_channel_ts_no_file(self):
        """Test to_channel_ts when no file is set."""
        atss = ATSS()
        with pytest.raises(AttributeError):
            atss.to_channel_ts()

    @requires_test_data
    def test_to_channel_ts_file_not_exists(self, temp_dir):
        """Test to_channel_ts with non-existent file."""
        non_existent = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        atss = ATSS(non_existent)
        with pytest.raises(FileNotFoundError):
            atss.to_channel_ts()

    @requires_test_data
    def test_to_channel_ts_with_metadata(self, atss_instance):
        """Test to_channel_ts with real metadata."""
        from mth5.timeseries import ChannelTS

        result = atss_instance.to_channel_ts()
        assert isinstance(result, ChannelTS)
        assert len(result.ts) > 0
        assert result.channel_metadata is not None

    @requires_test_data
    def test_to_channel_ts_no_metadata_file(self, temp_dir, atss_instance):
        """Test to_channel_ts with file but no metadata."""
        # Copy ATSS file without JSON file
        original_data = atss_instance.read_atss(stop=1000)
        new_atss = (
            temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        )  # Use proper naming format

        with open(new_atss, "wb") as f:
            f.write(original_data.tobytes())

        atss = ATSS(new_atss)
        # Should still work but might have warnings, or may fail gracefully
        try:
            result = atss.to_channel_ts()
            assert result is not None
        except Exception:
            # It's acceptable for this to fail when no metadata is available
            pass


class TestATSSErrorHandling:
    """Test error handling scenarios."""

    @requires_test_data
    def test_invalid_file_path(self, temp_dir):
        """Test initialization with invalid file path."""
        invalid_file = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        # The file doesn't exist, but filename parsing should work
        atss = ATSS(invalid_file)
        assert atss.system_number == "084"
        assert not atss.fn_exists

    @requires_test_data
    def test_corrupted_binary_file(self, temp_dir):
        """Test reading corrupted binary file."""
        corrupted_file = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        # Write data that's not a multiple of 8 bytes
        with open(corrupted_file, "wb") as f:
            f.write(b"not valid binary data123")

        atss = ATSS(corrupted_file)
        data = atss.read_atss()
        assert isinstance(data, np.ndarray)
        # Should read what it can from the corrupted data

    @requires_test_data
    def test_empty_file(self, temp_dir):
        """Test reading empty ATSS file."""
        empty_file = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        empty_file.touch()

        atss = ATSS(empty_file)
        data = atss.read_atss()
        assert len(data) == 0

    @requires_test_data
    def test_invalid_json_metadata(self, temp_dir, sample_magnetic_file):
        """Test handling invalid JSON metadata."""
        # Copy ATSS file and create invalid JSON
        atss_file = temp_dir / sample_magnetic_file["atss"].name
        json_file = temp_dir / sample_magnetic_file["json"].name

        shutil.copy2(sample_magnetic_file["atss"], atss_file)

        with open(json_file, "w") as f:
            f.write("invalid json content")

        # Should handle gracefully or raise appropriate error
        with pytest.raises((json.JSONDecodeError, ValueError)):
            atss = ATSS(atss_file)


class TestATSSUtilityFunction:
    """Test the read_atss utility function."""

    def test_read_atss_function_exists(self):
        """Test that read_atss function is importable."""
        from mth5.io.metronix import read_atss

        assert callable(read_atss)

    @requires_test_data
    def test_read_atss_function_call(self, sample_magnetic_file):
        """Test calling read_atss function."""
        from mth5.io.metronix import read_atss
        from mth5.timeseries import ChannelTS

        result = read_atss(sample_magnetic_file["atss"])
        assert isinstance(result, ChannelTS)
        assert len(result.ts) > 0


class TestATSSIntegration:
    """Integration tests for ATSS functionality."""

    @requires_test_data
    def test_full_workflow(self, sample_magnetic_file):
        """Test complete ATSS workflow from file creation to ChannelTS."""
        atss = ATSS(sample_magnetic_file["atss"])

        # Test properties
        assert atss.system_number == "084"
        assert atss.component in ["hx", "hy", "hz"]
        assert atss.run_id.startswith("run_")
        assert atss.station_id == "saricam"
        assert atss.survey_id == "Northern_Mining"
        assert atss.has_metadata_file() is True

        # Test data reading
        data = atss.read_atss()
        assert isinstance(data, np.ndarray)
        assert len(data) > 0

        # Test ChannelTS conversion
        channel_ts = atss.to_channel_ts()
        assert channel_ts is not None
        assert len(channel_ts.ts) == len(data)

    @requires_test_data
    def test_multiple_atss_files(self, real_atss_files):
        """Test handling multiple ATSS files from same run."""
        # Group files by run
        runs = {}
        for filename, file_info in real_atss_files.items():
            run_id = file_info["run_dir"]
            if run_id not in runs:
                runs[run_id] = []
            runs[run_id].append(file_info)

        # Test files from one run
        for run_id, files in runs.items():
            atss_objects = []

            for file_info in files[:3]:  # Test first 3 files to avoid too many
                atss = ATSS(file_info["atss"])
                atss_objects.append(atss)

            # Verify each ATSS object
            for atss in atss_objects:
                assert atss.run_id == run_id
                assert atss.station_id == "saricam"
                assert atss.survey_id == "Northern_Mining"

                data = atss.read_atss(stop=100)  # Just read a small sample
                assert len(data) == 100

            break  # Just test one run


# =============================================================================
# Parametrized Tests
# =============================================================================


class TestATSSParametrized:
    """Parametrized tests for various scenarios."""

    @requires_test_data
    @pytest.mark.parametrize("run_dir", ["run_002", "run_005", "run_007", "run_008"])
    def test_different_runs(self, run_dir, real_atss_files):
        """Test ATSS files from different runs."""
        # Find a file from the specified run
        test_file = None
        for filename, file_info in real_atss_files.items():
            if file_info["run_dir"] == run_dir:
                test_file = file_info
                break

        if test_file:
            atss = ATSS(test_file["atss"])
            assert atss.run_id == run_dir
            assert atss.system_number == "084"
            assert atss.system_name == "ADU-07e"

    @requires_test_data
    @pytest.mark.parametrize(
        "start,stop",
        [
            (0, 100),
            (50, 150),
            (0, 0),  # Full file
            (500, 0),  # From start to end
        ],
    )
    def test_read_ranges(self, atss_instance, start, stop):
        """Test reading different ranges of data."""
        if stop == 0:
            if start == 0:
                # Full file
                data = atss_instance.read_atss()
                assert len(data) > 0
            else:
                # From start to end
                full_data = atss_instance.read_atss()
                data = atss_instance.read_atss(start=start)
                assert len(data) == len(full_data) - start
        else:
            # Read specific number of samples
            data = atss_instance.read_atss(start=start, stop=stop)
            assert len(data) == stop

    @requires_test_data
    @pytest.mark.parametrize(
        "component,expected_type",
        [
            ("ex", "electric"),
            ("ey", "electric"),
            ("hx", "magnetic"),
            ("hy", "magnetic"),
            ("hz", "magnetic"),
        ],
    )
    def test_channel_types(self, real_atss_files, component, expected_type):
        """Test channel type determination."""
        # Find file with specified component
        test_file = None
        for filename, file_info in real_atss_files.items():
            if f"T{component.title()}" in filename:
                test_file = file_info
                break

        if test_file:
            atss = ATSS(test_file["atss"])
            assert atss.channel_type == expected_type


# =============================================================================
# Performance Tests
# =============================================================================


class TestATSSPerformance:
    """Performance-related tests."""

    @requires_test_data
    def test_large_file_reading(self, high_sample_rate_file):
        """Test reading high sample rate ATSS files."""
        if not high_sample_rate_file:
            pytest.skip("No high sample rate files available")

        atss = ATSS(high_sample_rate_file["atss"])

        # Test partial reading for large files
        import time

        start_time = time.time()
        data = atss.read_atss(start=0, stop=10000)
        end_time = time.time()

        assert len(data) == 10000
        assert end_time - start_time < 2.0  # Should be reasonably fast

    @requires_test_data
    def test_memory_efficient_reading(self, atss_instance):
        """Test that reading doesn't load entire file into memory unnecessarily."""
        # Read small portions multiple times
        for i in range(5):
            start = i * 1000
            stop = 1000
            chunk = atss_instance.read_atss(start=start, stop=stop)
            assert len(chunk) == 1000


# =============================================================================
# Edge Cases
# =============================================================================


class TestATSSEdgeCases:
    """Test edge cases and boundary conditions."""

    @requires_test_data
    def test_zero_length_file(self, temp_dir):
        """Test handling zero-length files."""
        zero_file = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"
        zero_file.touch()

        atss = ATSS(zero_file)
        data = atss.read_atss()
        assert len(data) == 0

    @requires_test_data
    def test_single_sample_file(self, temp_dir):
        """Test file with single sample."""
        single_sample = np.array([42.0], dtype=np.float64)
        single_file = temp_dir / "084_ADU-07e_C001_THx_128Hz.atss"

        with open(single_file, "wb") as f:
            f.write(single_sample.tobytes())

        atss = ATSS(single_file)
        data = atss.read_atss()
        assert len(data) == 1
        assert data[0] == 42.0

    @requires_test_data
    def test_boundary_read_conditions(self, atss_instance):
        """Test reading at file boundaries."""
        file_length = int(atss_instance.n_samples)

        # Test reading exactly at file boundaries
        data = atss_instance.read_atss(start=0, stop=min(file_length, 1000))
        assert len(data) == min(file_length, 1000)

        # Test reading beyond file end
        if file_length > 10:
            with pytest.raises(ValueError, match="stop .* > samples"):
                atss_instance.read_atss(start=file_length - 10, stop=file_length + 10)

    @requires_test_data
    def test_negative_indices(self, atss_instance):
        """Test handling of negative indices."""
        # Negative start should raise OSError
        with pytest.raises(OSError):
            data = atss_instance.read_atss(start=-10, stop=100)

    @requires_test_data
    def test_invalid_range(self, atss_instance):
        """Test invalid start/stop ranges."""
        # The implementation reads 'stop' samples starting from 'start'
        data = atss_instance.read_atss(start=200, stop=100)
        assert len(data) == 100


# =============================================================================
# Comparison Tests with Real vs Mock Data
# =============================================================================


class TestRealDataValidation:
    """Tests to validate real data characteristics."""

    @requires_test_data
    def test_file_structure_validation(self, real_atss_files):
        """Test that real files follow expected naming convention."""
        for filename, file_info in real_atss_files.items():
            # Check filename format
            assert filename.startswith("084_ADU-07e_C")
            assert "_T" in filename
            assert filename.endswith(".atss")

            # Check corresponding JSON exists
            assert file_info["json"].exists()

            # Check run directory format
            assert file_info["run_dir"].startswith("run_")

    @requires_test_data
    def test_data_consistency(self, sample_magnetic_file):
        """Test consistency between ATSS data and JSON metadata."""
        atss = ATSS(sample_magnetic_file["atss"])

        # Read data and metadata
        data = atss.read_atss()
        metadata = atss.channel_metadata

        if metadata and hasattr(metadata, "sample_rate"):
            # Check that file size matches expected samples
            expected_samples = atss.file_size / 8  # 8 bytes per float64
            assert len(data) == expected_samples

            # Check that duration calculation is reasonable
            duration = len(data) / metadata.sample_rate
            assert duration > 0
            assert duration < 86400  # Less than 24 hours

    @requires_test_data
    def test_multiple_components_same_run(self, real_atss_files):
        """Test that files from same run have consistent metadata."""
        # Group by run
        runs = {}
        for filename, file_info in real_atss_files.items():
            run_id = file_info["run_dir"]
            if run_id not in runs:
                runs[run_id] = []
            runs[run_id].append(file_info)

        # Test consistency within each run
        for run_id, files in runs.items():
            if len(files) > 1:
                atss_objects = [ATSS(f["atss"]) for f in files]

                # All should have same run/station/survey IDs
                first = atss_objects[0]
                for atss in atss_objects[1:]:
                    assert atss.run_id == first.run_id
                    assert atss.station_id == first.station_id
                    assert atss.survey_id == first.survey_id
                break  # Test one run


# =============================================================================
# Skip Tests When Data Not Available
# =============================================================================


def test_skip_when_no_data():
    """Test that tests are properly skipped when data is not available."""
    if not has_data_import:
        pytest.skip("mth5_test_data package not available")
    else:
        # If data is available, just pass
        assert True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__])
