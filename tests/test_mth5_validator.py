"""
Tests for MTH5 Validator
=========================

Test suite for the MTH5 file validator.

Author: MTH5 Development Team
Date: February 7, 2026
"""

import tempfile
from pathlib import Path

import h5py
import pytest

from mth5.mth5 import MTH5
from mth5.utils.mth5_validator import MTH5Validator, validate_mth5_file


class TestMTH5ValidatorBasic:
    """Test basic validator functionality."""

    def test_validator_creation(self):
        """Test creating a validator instance."""
        # Create a temp file path (doesn't need to exist for this test)
        temp_path = Path("test.mth5")
        validator = MTH5Validator(temp_path)

        assert validator.file_path == temp_path
        assert validator.validate_metadata == True
        assert validator.check_data == False
        assert validator.verbose == False

    def test_validator_options(self):
        """Test validator with different options."""
        validator = MTH5Validator(
            "test.mth5", verbose=True, validate_metadata=False, check_data=True
        )

        assert validator.verbose == True
        assert validator.validate_metadata == False
        assert validator.check_data == True

    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = MTH5Validator("nonexistent.mth5")
        results = validator.validate()

        assert not results.is_valid
        assert results.error_count > 0


class TestMTH5ValidatorWithFile:
    """Test validator with actual MTH5 files."""

    @pytest.fixture
    def temp_mth5_v02(self):
        """Create a temporary valid v0.2.0 MTH5 file."""
        with tempfile.NamedTemporaryFile(suffix=".mth5", delete=False) as tmp:
            temp_path = Path(tmp.name)

        # Create a valid MTH5 file
        with MTH5(file_version="0.2.0") as m:
            m.open_mth5(temp_path, "w")
            survey = m.add_survey("test_survey")
            station = m.add_station("TEST001", survey="test_survey")

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def temp_mth5_v01(self):
        """Create a temporary valid v0.1.0 MTH5 file."""
        with tempfile.NamedTemporaryFile(suffix=".mth5", delete=False) as tmp:
            temp_path = Path(tmp.name)

        # Create a valid MTH5 file
        with MTH5(file_version="0.1.0") as m:
            m.open_mth5(temp_path, "w")
            station = m.add_station("TEST001")

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def temp_invalid_mth5(self):
        """Create an invalid MTH5 file (missing required groups)."""
        with tempfile.NamedTemporaryFile(suffix=".mth5", delete=False) as tmp:
            temp_path = Path(tmp.name)

        # Create an HDF5 file but don't follow MTH5 structure
        with h5py.File(temp_path, "w") as f:
            f.attrs["file.type"] = "MTH5"
            f.attrs["file.version"] = "0.2.0"
            # Missing data_level and groups

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_validate_v02_file(self, temp_mth5_v02):
        """Test validation of v0.2.0 file."""
        results = validate_mth5_file(temp_mth5_v02)

        assert results.is_valid
        assert results.error_count == 0
        assert "file_version" in results.checked_items
        assert "root_group" in results.checked_items

    def test_validate_v01_file(self, temp_mth5_v01):
        """Test validation of v0.1.0 file."""
        validator = MTH5Validator(temp_mth5_v01)
        results = validator.validate()

        assert results.is_valid
        assert results.error_count == 0
        assert "file_version" in results.checked_items

    def test_validate_invalid_file(self, temp_invalid_mth5):
        """Test validation of invalid file."""
        results = validate_mth5_file(temp_invalid_mth5)

        assert not results.is_valid
        assert results.error_count > 0

    def test_validator_verbose(self, temp_mth5_v02):
        """Test verbose validation."""
        validator = MTH5Validator(temp_mth5_v02, verbose=True)
        results = validator.validate()

        assert results.is_valid
        # Check we have info messages
        assert results.info_count > 0

    def test_validator_with_metadata_check(self, temp_mth5_v02):
        """Test with metadata validation."""
        validator = MTH5Validator(temp_mth5_v02, validate_metadata=True)
        results = validator.validate()

        assert results.is_valid
        assert "metadata_validation" in results.checked_items

    def test_validator_skip_metadata(self, temp_mth5_v02):
        """Test skipping metadata validation."""
        validator = MTH5Validator(temp_mth5_v02, validate_metadata=False)
        results = validator.validate()

        assert results.is_valid
        assert "metadata_validation" not in results.checked_items


class TestValidationResults:
    """Test ValidationResults class."""

    def test_results_creation(self):
        """Test creating ValidationResults."""
        from mth5.utils.mth5_validator import ValidationResults

        results = ValidationResults(file_path=Path("test.mth5"))

        assert results.is_valid
        assert results.error_count == 0
        assert results.warning_count == 0
        assert results.info_count == 0

    def test_add_messages(self):
        """Test adding different message types."""
        from mth5.utils.mth5_validator import ValidationResults

        results = ValidationResults(file_path=Path("test.mth5"))

        results.add_error("Test", "Error message")
        results.add_warning("Test", "Warning message")
        results.add_info("Test", "Info message")

        assert results.error_count == 1
        assert results.warning_count == 1
        assert results.info_count == 1
        assert not results.is_valid  # Has errors

    def test_results_to_dict(self):
        """Test converting results to dictionary."""
        from mth5.utils.mth5_validator import ValidationResults

        results = ValidationResults(file_path=Path("test.mth5"))
        results.add_error("Test", "Test error")

        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert "file_path" in result_dict
        assert "is_valid" in result_dict
        assert "messages" in result_dict

    def test_results_to_json(self):
        """Test converting results to JSON."""
        from mth5.utils.mth5_validator import ValidationResults

        results = ValidationResults(file_path=Path("test.mth5"))
        results.add_warning("Test", "Test warning")

        json_str = results.to_json()

        assert isinstance(json_str, str)
        assert "test.mth5" in json_str
        assert "Test warning" in json_str


class TestValidatorIntegration:
    """Integration tests for validator."""

    @pytest.fixture
    def complete_mth5_file(self):
        """Create a complete MTH5 file with data."""
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mth5", delete=False) as tmp:
            temp_path = Path(tmp.name)

        # Create a complete file
        with MTH5(file_version="0.2.0") as m:
            m.open_mth5(temp_path, "w")

            # Add survey
            survey = m.add_survey("test_survey")

            # Add station with metadata
            station_metadata = survey.stations_group.metadata.Station()
            station_metadata.id = "TEST001"
            station_metadata.location.latitude = 40.0
            station_metadata.location.longitude = -120.0

            station = m.add_station(
                "TEST001", survey="test_survey", station_metadata=station_metadata
            )

            # Add run
            run = station.add_run("TEST001a")

            # Add channel with data
            data = np.random.random(1000)
            run.add_channel("Ex", "electric", data)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_full_validation(self, complete_mth5_file):
        """Test full validation with all checks."""
        validator = MTH5Validator(
            complete_mth5_file,
            verbose=True,
            validate_metadata=True,
            check_data=True,
        )

        results = validator.validate()

        assert results.is_valid
        assert results.error_count == 0
        # Should have some info messages
        assert results.info_count > 0

    def test_validation_report(self, complete_mth5_file, capsys):
        """Test printing validation report."""
        results = validate_mth5_file(complete_mth5_file)
        results.print_report(include_info=True)

        captured = capsys.readouterr()
        assert "MTH5 Validation Report" in captured.out
        assert "VALID" in captured.out or "INVALID" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
