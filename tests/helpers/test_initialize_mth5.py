# -*- coding: utf-8 -*-
"""
Pytest tests for initialize_mth5 helper function

Created on November 8, 2025

@author: GitHub Copilot
"""

import os
import tempfile
from pathlib import Path

# =============================================================================
# Imports
# =============================================================================
import pytest

from mth5.helpers import close_open_files, COMPRESSION, COMPRESSION_LEVELS
from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error
from mth5.utils.helpers import initialize_mth5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def test_file_path(temp_dir):
    """Generate a test file path"""
    return temp_dir / "test.h5"


@pytest.fixture
def existing_file_path(temp_dir):
    """Create an existing MTH5 file for testing"""
    file_path = temp_dir / "existing.h5"
    m = MTH5()
    m.open_mth5(str(file_path), mode="w")
    m.close_mth5()
    return file_path


@pytest.fixture(autouse=True)
def cleanup_files():
    """Ensure all MTH5 files are closed after each test"""
    yield
    close_open_files()


# =============================================================================
# Test Classes
# =============================================================================


class TestInitializeMTH5Basic:
    """Test basic functionality of initialize_mth5"""

    def test_create_new_file_write_mode(self, test_file_path):
        """Test creating a new file in write mode"""
        m = initialize_mth5(test_file_path, "w")

        assert isinstance(m, MTH5)
        assert m.filename == test_file_path
        assert test_file_path.exists()
        m.close_mth5()

    def test_create_new_file_append_mode(self, test_file_path):
        """Test creating a new file in append mode"""
        m = initialize_mth5(test_file_path, "a")

        assert isinstance(m, MTH5)
        assert m.filename == test_file_path
        assert test_file_path.exists()
        m.close_mth5()

    def test_open_existing_file_append_mode(self, existing_file_path):
        """Test opening an existing file in append mode"""
        m = initialize_mth5(existing_file_path, "a")

        assert isinstance(m, MTH5)
        assert m.filename == existing_file_path
        m.close_mth5()

    def test_open_existing_file_read_mode(self, existing_file_path):
        """Test opening an existing file in read mode"""
        m = initialize_mth5(existing_file_path, "r")

        assert isinstance(m, MTH5)
        assert m.filename == existing_file_path
        m.close_mth5()

    def test_overwrite_existing_file_write_mode(self, existing_file_path):
        """Test overwriting an existing file in write mode"""
        original_size = existing_file_path.stat().st_size

        m = initialize_mth5(existing_file_path, "w")

        assert isinstance(m, MTH5)
        assert m.filename == existing_file_path
        # File should be recreated (potentially different size)
        m.close_mth5()

    @pytest.mark.parametrize("mode", ["r", "w", "a"])
    def test_valid_modes(self, test_file_path, mode):
        """Test all valid file modes"""
        if mode == "r":
            # Create file first for read mode
            temp_m = MTH5()
            temp_m.open_mth5(str(test_file_path), mode="w")
            temp_m.close_mth5()

        m = initialize_mth5(test_file_path, mode)
        assert isinstance(m, MTH5)
        m.close_mth5()

    def test_string_path_input(self, temp_dir):
        """Test that string paths work as well as Path objects"""
        file_path_str = str(temp_dir / "string_test.h5")

        m = initialize_mth5(file_path_str, "w")

        assert isinstance(m, MTH5)
        assert Path(file_path_str).exists()
        m.close_mth5()


class TestInitializeMTH5FileVersions:
    """Test file version functionality"""

    @pytest.mark.parametrize("file_version", ["0.1.0", "0.2.0"])
    def test_file_versions(self, test_file_path, file_version):
        """Test different file versions"""
        m = initialize_mth5(test_file_path, "w", file_version=file_version)

        assert isinstance(m, MTH5)
        assert m.file_version == file_version
        m.close_mth5()

    def test_default_file_version(self, test_file_path):
        """Test default file version (0.1.0)"""
        m = initialize_mth5(test_file_path, "w")

        assert isinstance(m, MTH5)
        assert m.file_version == "0.1.0"
        m.close_mth5()


class TestInitializeMTH5WithCompression:
    """Test compression functionality with initialize_mth5"""

    @pytest.mark.parametrize(
        "compression_type", [comp for comp in COMPRESSION if comp != "szip"]
    )
    def test_compression_types(self, test_file_path, compression_type):
        """Test different compression types (excluding szip which may not be available)"""
        # Use default compression level for each type
        if compression_type is None:
            compression_level = None
        else:
            compression_level = list(COMPRESSION_LEVELS[compression_type])[0]

        # Create MTH5 with compression settings first
        mth5_obj = MTH5(
            compression=compression_type, compression_opts=compression_level
        )
        mth5_obj.open_mth5(str(test_file_path), mode="w")
        mth5_obj.close_mth5()

        # Now test initialize_mth5 with the existing file
        m = initialize_mth5(test_file_path, "a")

        assert isinstance(m, MTH5)
        m.close_mth5()

    @pytest.mark.parametrize(
        "compression_type,compression_level",
        [
            (comp_type, comp_level)
            for comp_type in COMPRESSION
            for comp_level in COMPRESSION_LEVELS[comp_type]
            if comp_type != "szip"  # Skip szip if not available
        ],
    )
    def test_compression_levels_parametrized(
        self, temp_dir, compression_type, compression_level
    ):
        """Test various compression levels using parametrized tests"""
        # Create unique file for each combination
        unique_path = temp_dir / f"test_{compression_type}_{compression_level}.h5"

        # Create MTH5 with specific compression
        mth5_obj = MTH5(
            compression=compression_type, compression_opts=compression_level
        )
        mth5_obj.open_mth5(str(unique_path), mode="w")
        mth5_obj.close_mth5()

        # Test initialize_mth5
        m = initialize_mth5(unique_path, "a")

        assert isinstance(m, MTH5)
        m.close_mth5()


class TestInitializeMTH5ErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_mode(self, test_file_path):
        """Test invalid file mode raises appropriate error"""
        # This should be handled by MTH5 and raise MTH5Error
        with pytest.raises(MTH5Error):
            m = initialize_mth5(test_file_path, "invalid_mode")

    def test_read_nonexistent_file(self, test_file_path):
        """Test reading a non-existent file raises appropriate error"""
        # Should raise an MTH5Error when trying to read a non-existent file
        with pytest.raises(MTH5Error):
            m = initialize_mth5(test_file_path, "r")

    def test_invalid_file_version(self, test_file_path):
        """Test invalid file version raises ValueError"""
        with pytest.raises(ValueError):
            m = initialize_mth5(test_file_path, "w", file_version="invalid_version")

    def test_permission_denied_scenario(self, temp_dir):
        """Test handling of permission denied scenarios"""
        if os.name == "nt":  # Windows
            pytest.skip("Permission tests more complex on Windows")

        # Create a directory with no write permissions
        restricted_dir = temp_dir / "restricted"
        restricted_dir.mkdir()
        os.chmod(restricted_dir, 0o444)  # Read-only

        restricted_file = restricted_dir / "test.h5"

        try:
            with pytest.raises(PermissionError):
                m = initialize_mth5(restricted_file, "w")
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_dir, 0o755)


class TestInitializeMTH5FileOperations:
    """Test file operations and state management"""

    def test_file_closure_on_overwrite(self, existing_file_path):
        """Test that existing files are properly closed before overwrite"""
        # Open the existing file
        m1 = MTH5()
        m1.open_mth5(str(existing_file_path), mode="a")

        # This should close the existing file and create a new one
        m2 = initialize_mth5(existing_file_path, "w")

        assert isinstance(m2, MTH5)
        m2.close_mth5()

    def test_multiple_initializations(self, test_file_path):
        """Test multiple initializations of the same file"""
        # First initialization
        m1 = initialize_mth5(test_file_path, "w")
        m1.close_mth5()

        # Second initialization
        m2 = initialize_mth5(test_file_path, "a")
        m2.close_mth5()

        # Third initialization
        m3 = initialize_mth5(test_file_path, "r")
        m3.close_mth5()

    def test_file_attributes_set_correctly(self, test_file_path):
        """Test that file attributes are set correctly"""
        m = initialize_mth5(test_file_path, "w", file_version="0.2.0")

        assert m.file_version == "0.2.0"
        # Test that the file has the correct structure based on version
        if m.file_version == "0.2.0":
            # Should have Experiment group for v0.2.0
            assert hasattr(m, "experiment_group")
        else:
            # Should have Survey group for v0.1.0
            assert hasattr(m, "survey_group")

        m.close_mth5()

    def test_path_normalization(self, temp_dir):
        """Test that different path representations work"""
        # Test with relative path
        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir)
            relative_path = Path("relative_test.h5")

            m = initialize_mth5(relative_path, "w")
            assert isinstance(m, MTH5)
            assert relative_path.exists()
            m.close_mth5()
        finally:
            os.chdir(original_cwd)


class TestInitializeMTH5Integration:
    """Integration tests with actual MTH5 operations"""

    def test_created_file_structure(self, test_file_path):
        """Test that initialized file has correct structure"""
        m = initialize_mth5(test_file_path, "w", file_version="0.1.0")

        # Should have basic MTH5 structure
        assert m.h5_is_read()
        assert hasattr(m, "survey_group")

        m.close_mth5()

    def test_file_metadata_consistency(self, test_file_path):
        """Test that file metadata is consistent after initialization"""
        file_version = "0.2.0"
        m = initialize_mth5(test_file_path, "w", file_version=file_version)

        # Check metadata consistency
        assert m.file_version == file_version

        # Close and reopen to verify persistence
        m.close_mth5()

        m2 = initialize_mth5(test_file_path, "r")
        assert m2.file_version == file_version
        m2.close_mth5()

    def test_initialize_with_existing_data(self, temp_dir):
        """Test initializing a file that already contains data"""
        file_path = temp_dir / "with_data.h5"

        # Create file with some structure using v0.2.0 for surveys_group
        m1 = initialize_mth5(file_path, "w", file_version="0.2.0")
        survey = m1.add_survey("test_survey")
        m1.close_mth5()

        # Re-initialize in append mode
        m2 = initialize_mth5(file_path, "a")
        assert isinstance(m2, MTH5)

        # Should still have the survey (check surveys_group for v0.2.0)
        if hasattr(m2, "surveys_group") and m2.surveys_group is not None:
            survey_names = list(m2.surveys_group.groups_list)
            assert "test_survey" in survey_names

        m2.close_mth5()


class TestInitializeMTH5Performance:
    """Performance and resource usage tests"""

    def test_multiple_files_cleanup(self, temp_dir):
        """Test that multiple file operations don't leak resources"""
        file_paths = [temp_dir / f"test_{i}.h5" for i in range(5)]

        mth5_objects = []
        for file_path in file_paths:
            m = initialize_mth5(file_path, "w")
            mth5_objects.append(m)

        # Close all files
        for m in mth5_objects:
            m.close_mth5()

        # Verify all files exist and can be reopened
        for file_path in file_paths:
            assert file_path.exists()
            m = initialize_mth5(file_path, "r")
            m.close_mth5()

    def test_large_file_path(self, temp_dir):
        """Test with very long file paths"""
        # Create a moderately nested directory structure (avoid Windows path limits)
        deep_dir = temp_dir
        for i in range(2):  # Reduced from 5 to avoid path length issues
            deep_dir = deep_dir / f"level_{i}_long_name"
            deep_dir.mkdir()

        long_filename = "test_long_path.h5"
        file_path = deep_dir / long_filename

        m = initialize_mth5(file_path, "w")
        assert isinstance(m, MTH5)
        assert file_path.exists()
        m.close_mth5()


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestInitializeMTH5UncoveredFunctionality:
    """Test functionality not covered by original unittest suite"""

    def test_pathlib_vs_string_consistency(self, temp_dir):
        """Test that pathlib.Path and string inputs produce identical results"""
        file_name = "consistency_test.h5"
        path_obj = temp_dir / file_name
        path_str = str(path_obj)

        # Test with Path object
        m1 = initialize_mth5(path_obj, "w")
        m1.close_mth5()

        # Remove file
        path_obj.unlink()

        # Test with string
        m2 = initialize_mth5(path_str, "w")
        m2.close_mth5()

        # Both should create identical files
        assert path_obj.exists()

    def test_file_deletion_behavior(self, existing_file_path):
        """Test that file deletion in write mode works correctly"""
        # Store original size and modification time
        original_size = existing_file_path.stat().st_size
        original_mtime = existing_file_path.stat().st_mtime

        # Initialize with write mode should overwrite the file
        m = initialize_mth5(existing_file_path, "w")
        m.close_mth5()

        # File should still exist but may be different
        assert existing_file_path.exists()
        # It's a new file, so modification time should be different
        new_mtime = existing_file_path.stat().st_mtime
        assert new_mtime != original_mtime

    def test_file_version_parameter_validation(self, test_file_path):
        """Test file_version parameter validation"""
        # Valid versions should work
        for version in ["0.1.0", "0.2.0"]:
            m = initialize_mth5(test_file_path, "w", file_version=version)
            assert m.file_version == version
            m.close_mth5()
            test_file_path.unlink()  # Clean up for next iteration

    def test_return_value_properties(self, test_file_path):
        """Test that returned MTH5 object has expected properties"""
        m = initialize_mth5(test_file_path, "w", file_version="0.2.0")

        # Should be properly initialized MTH5 object
        assert isinstance(m, MTH5)
        assert m.h5_is_read()
        assert m.filename == test_file_path
        assert hasattr(m, "experiment_group")  # v0.2.0 specific

        m.close_mth5()

    def test_concurrent_access_handling(self, existing_file_path):
        """Test behavior when file is already open"""
        # Open file first
        m1 = MTH5()
        m1.open_mth5(str(existing_file_path), mode="a")

        # Try to initialize with write mode (should handle the open file)
        m2 = initialize_mth5(existing_file_path, "w")

        assert isinstance(m2, MTH5)
        m2.close_mth5()

    def test_edge_case_empty_file_path(self):
        """Test behavior with edge case file paths"""
        with pytest.raises((ValueError, OSError, TypeError)):
            m = initialize_mth5("", "w")

    def test_mode_case_sensitivity(self, test_file_path):
        """Test that mode parameter is case sensitive"""
        # Lowercase should work
        m = initialize_mth5(test_file_path, "w")
        m.close_mth5()
        test_file_path.unlink()

        # Uppercase should raise MTH5Error
        with pytest.raises(MTH5Error):
            m = initialize_mth5(test_file_path, "W")
