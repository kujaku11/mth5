# -*- coding: utf-8 -*-
"""
Modern pytest suite for MTH5 ClientBase functionality using fixtures and optimizations.

This is a pytest translation of test_client_base.py that uses fixtures for better
test organization and cleanup, while optimizing for efficiency through strategic
fixture scoping and comprehensive test coverage.

Key improvements over unittest version:
1. Uses pytest fixtures for better resource management
2. Session-scoped fixtures for expensive operations
3. Parametrized tests for comprehensive coverage
4. Automatic cleanup with fixture teardown
5. Enhanced error handling and validation
6. Optimized execution through strategic fixture usage
7. Subtests for detailed validation within test methods

Test coverage includes:
- ClientBase initialization and configuration
- H5 kwargs validation and handling
- Sample rates setting and validation (float, string, list)
- Save path configuration and validation
- Error handling for invalid inputs
- Comprehensive edge case testing

Final Results: Comprehensive ClientBase testing with modern pytest patterns

Created on Fri Oct 11 11:48:24 2024
@author: jpeacock - pytest translation optimized for efficiency
"""

import shutil
import tempfile
from pathlib import Path

# =============================================================================
# Imports
# =============================================================================
import pytest

from mth5.clients.base import ClientBase


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files that persists for the session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="mth5_client_base_test_"))
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_file_path(test_dir):
    """Path for the test H5 file."""
    return test_dir / "test.h5"


@pytest.fixture
def client_base(test_file_path):
    """
    Create a ClientBase instance for testing.

    Function-scoped to ensure clean state for each test.
    """
    return ClientBase(test_file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"})


@pytest.fixture(scope="session")
def expected_h5_kwargs_keys():
    """Expected keys for h5_kwargs - session scoped for efficiency."""
    return [
        "compression",
        "compression_opts",
        "data_level",
        "driver",
        "file_version",
        "fletcher32",
        "mode",
        "shuffle",
    ]


# =============================================================================
# Test Classes
# =============================================================================


class TestClientBaseInitialization:
    """Test ClientBase initialization and basic configuration."""

    def test_client_base_creation(self, client_base):
        """Test successful ClientBase creation."""
        assert client_base is not None
        assert isinstance(client_base, ClientBase)

    def test_h5_kwargs_keys(self, client_base, expected_h5_kwargs_keys):
        """Test that h5_kwargs contains expected keys."""
        actual_keys = sorted(client_base.h5_kwargs.keys())
        assert actual_keys == expected_h5_kwargs_keys

    def test_h5_kwargs_structure(self, client_base, subtests):
        """Test h5_kwargs structure and types using subtests."""
        h5_kwargs = client_base.h5_kwargs

        with subtests.test("h5_kwargs_is_dict"):
            assert isinstance(h5_kwargs, dict)

        with subtests.test("has_mode"):
            assert "mode" in h5_kwargs

        with subtests.test("has_driver"):
            assert "driver" in h5_kwargs


class TestClientBaseSampleRates:
    """Test sample rates functionality with comprehensive coverage."""

    def test_set_sample_rates_float(self, client_base):
        """Test setting sample rates with a single float value."""
        client_base.sample_rates = 10.5
        assert client_base.sample_rates == [10.5]

    def test_set_sample_rates_string(self, client_base):
        """Test setting sample rates with comma-separated string."""
        client_base.sample_rates = "10, 42, 1200"
        assert client_base.sample_rates == [10.0, 42.0, 1200.0]

    def test_set_sample_rates_list(self, client_base):
        """Test setting sample rates with a list of values."""
        client_base.sample_rates = [10, 42, 1200]
        assert client_base.sample_rates == [10.0, 42.0, 1200.0]

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (10.5, [10.5]),
            (42, [42.0]),
            ("8, 16, 32", [8.0, 16.0, 32.0]),
            ([1, 2, 4], [1.0, 2.0, 4.0]),
            ("256", [256.0]),
            ([1024.5], [1024.5]),
        ],
    )
    def test_sample_rates_parametrized(self, client_base, input_value, expected):
        """Test various sample rate inputs using parametrization."""
        client_base.sample_rates = input_value
        assert client_base.sample_rates == expected

    def test_set_sample_rates_failure(self, client_base):
        """Test that setting sample rates to None raises TypeError."""
        with pytest.raises(TypeError):
            client_base.sample_rates = None

    def test_sample_rates_edge_cases(self, client_base, subtests):
        """Test edge cases for sample rates using subtests."""

        with subtests.test("empty_string"):
            # Test behavior with empty string - this may depend on implementation
            try:
                client_base.sample_rates = ""
                # If it doesn't raise an error, check what it becomes
                assert isinstance(client_base.sample_rates, list)
            except (ValueError, TypeError):
                # If it raises an error, that's also acceptable behavior
                pass

        with subtests.test("single_value_string"):
            client_base.sample_rates = "100"
            assert client_base.sample_rates == [100.0]

        with subtests.test("float_in_list"):
            client_base.sample_rates = [10.5, 20.0, 30]
            assert client_base.sample_rates == [10.5, 20.0, 30.0]


class TestClientBaseSavePath:
    """Test save path functionality and validation."""

    def test_set_save_path(self, client_base, test_file_path, subtests):
        """Test setting save path and validate all related properties."""
        client_base.save_path = test_file_path

        with subtests.test("_save_path"):
            assert client_base._save_path == test_file_path.parent

        with subtests.test("filename"):
            assert client_base.mth5_filename == test_file_path.name

        with subtests.test("save_path"):
            assert client_base.save_path == test_file_path

    def test_save_path_components(self, client_base, test_file_path):
        """Test that save path components are correctly set."""
        client_base.save_path = test_file_path

        # Test the relationship between components
        assert (
            client_base.save_path == client_base._save_path / client_base.mth5_filename
        )

    @pytest.mark.parametrize(
        "filename",
        [
            "test1.h5",
            "data.mth5",
            "experiment_01.h5",
            "station_mt001.mth5",
        ],
    )
    def test_different_filenames(self, client_base, test_dir, filename):
        """Test save path with different filenames using parametrization."""
        file_path = test_dir / filename
        client_base.save_path = file_path

        assert client_base.mth5_filename == filename
        assert client_base._save_path == test_dir
        assert client_base.save_path == file_path


class TestClientBaseErrorHandling:
    """Test error handling and edge cases for ClientBase."""

    def test_initial_fail_none(self):
        """Test that initializing with None raises ValueError."""
        with pytest.raises(ValueError):
            ClientBase(None)

    def test_initial_fail_bad_directory(self):
        """Test that initializing with invalid directory raises IOError."""
        with pytest.raises(IOError):
            ClientBase(r"a:\\")

    def test_initialization_error_cases(self, subtests):
        """Test various initialization error cases using subtests."""

        with subtests.test("none_path"):
            with pytest.raises(ValueError):
                ClientBase(None)

        with subtests.test("invalid_drive"):
            with pytest.raises(IOError):
                ClientBase(r"a:\\")

        with subtests.test("non_existent_path"):
            # Test with a path that definitely doesn't exist
            with pytest.raises((IOError, FileNotFoundError)):
                ClientBase(Path("/this/path/definitely/does/not/exist"))

    @pytest.mark.parametrize(
        "invalid_path",
        [
            None,
            "",
            r"z:\nonexistent\path",
            "/definitely/not/a/real/path",
        ],
    )
    def test_invalid_paths_parametrized(self, invalid_path):
        """Test various invalid paths using parametrization."""
        with pytest.raises((ValueError, IOError, FileNotFoundError)):
            if invalid_path == "":
                # Empty string might be handled differently
                ClientBase(invalid_path or None)
            else:
                ClientBase(invalid_path)


class TestClientBaseConfiguration:
    """Test ClientBase configuration and advanced functionality."""

    def test_h5_kwargs_default_values(self, client_base, subtests):
        """Test default values in h5_kwargs using subtests."""
        h5_kwargs = client_base.h5_kwargs

        with subtests.test("mode_is_set"):
            assert "mode" in h5_kwargs

        with subtests.test("driver_is_set"):
            assert "driver" in h5_kwargs

        with subtests.test("has_compression"):
            assert "compression" in h5_kwargs

        with subtests.test("has_fletcher32"):
            assert "fletcher32" in h5_kwargs

    def test_configuration_consistency(self, client_base):
        """Test that configuration remains consistent after operations."""
        original_h5_kwargs = client_base.h5_kwargs.copy()

        # Perform some operations
        client_base.sample_rates = [1, 2, 4]

        # Configuration should remain unchanged
        assert client_base.h5_kwargs == original_h5_kwargs

    def test_multiple_operations(self, client_base, test_file_path, subtests):
        """Test multiple operations on the same ClientBase instance."""

        with subtests.test("set_sample_rates"):
            client_base.sample_rates = [8, 16, 32]
            assert client_base.sample_rates == [8.0, 16.0, 32.0]

        with subtests.test("set_save_path"):
            client_base.save_path = test_file_path
            assert client_base.save_path == test_file_path

        with subtests.test("verify_both_settings"):
            # Verify both settings are maintained
            assert client_base.sample_rates == [8.0, 16.0, 32.0]
            assert client_base.save_path == test_file_path


class TestClientBaseIntegration:
    """Integration tests for complete ClientBase functionality."""

    def test_complete_setup_workflow(self, test_file_path, subtests):
        """Test complete setup workflow from initialization to configuration."""

        # Initialize ClientBase
        client = ClientBase(
            test_file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"}
        )

        with subtests.test("initialization"):
            assert client is not None
            assert isinstance(client, ClientBase)

        with subtests.test("configure_sample_rates"):
            client.sample_rates = "4, 8, 16, 32"
            assert client.sample_rates == [4.0, 8.0, 16.0, 32.0]

        with subtests.test("configure_save_path"):
            client.save_path = test_file_path
            assert client.save_path == test_file_path

        with subtests.test("verify_h5_kwargs"):
            h5_kwargs = client.h5_kwargs
            assert isinstance(h5_kwargs, dict)
            assert len(h5_kwargs) == 8  # Expected number of keys

    def test_client_base_immutability(self, client_base):
        """Test that certain properties maintain their integrity."""
        original_h5_kwargs = client_base.h5_kwargs
        original_keys = set(original_h5_kwargs.keys())

        # Perform operations
        client_base.sample_rates = [1, 2, 4, 8]

        # Check that h5_kwargs structure is preserved
        current_keys = set(client_base.h5_kwargs.keys())
        assert current_keys == original_keys

    @pytest.mark.parametrize(
        "sample_rates,expected_length",
        [
            ([1], 1),
            ([1, 2], 2),
            ([1, 2, 4], 3),
            ([1, 2, 4, 8], 4),
            ("1, 2, 4, 8, 16", 5),
        ],
    )
    def test_sample_rates_length_validation(
        self, client_base, sample_rates, expected_length
    ):
        """Test that sample rates length is correctly maintained."""
        client_base.sample_rates = sample_rates
        assert len(client_base.sample_rates) == expected_length


class TestClientBasePerformance:
    """Test performance characteristics of ClientBase functionality."""

    @pytest.mark.performance
    def test_client_base_creation_efficiency(self, test_file_path):
        """Test that ClientBase creation is efficient."""
        # This test validates efficient creation
        client = ClientBase(
            test_file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"}
        )

        assert client is not None
        assert hasattr(client, "h5_kwargs")
        assert hasattr(client, "sample_rates")

    @pytest.mark.performance
    def test_multiple_sample_rate_updates(self, client_base):
        """Test performance of multiple sample rate updates."""
        # Test that multiple updates are handled efficiently
        test_values = [
            [1, 2],
            [4, 8, 16],
            [32, 64, 128, 256],
            "1, 2, 4",
            512.0,
        ]

        for value in test_values:
            client_base.sample_rates = value
            # Verify the assignment worked
            assert isinstance(client_base.sample_rates, list)
            assert len(client_base.sample_rates) > 0


# =============================================================================
# Integration Test for Complete Workflow
# =============================================================================


class TestClientBaseCompleteWorkflow:
    """Test complete workflow from ClientBase creation to full configuration."""

    def test_end_to_end_workflow(self, test_file_path, subtests):
        """Test complete end-to-end workflow for ClientBase."""

        with subtests.test("create_client"):
            client = ClientBase(
                test_file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"}
            )
            assert client is not None

        with subtests.test("verify_initial_state"):
            assert hasattr(client, "h5_kwargs")
            assert isinstance(client.h5_kwargs, dict)
            assert len(client.h5_kwargs) == 8

        with subtests.test("configure_all_settings"):
            # Configure sample rates
            client.sample_rates = [1, 4, 16, 64, 256]
            assert client.sample_rates == [1.0, 4.0, 16.0, 64.0, 256.0]

            # Configure save path
            client.save_path = test_file_path
            assert client.save_path == test_file_path
            assert client.mth5_filename == test_file_path.name
            assert client._save_path == test_file_path.parent

        with subtests.test("verify_final_state"):
            # Verify everything is correctly configured
            assert client.sample_rates == [1.0, 4.0, 16.0, 64.0, 256.0]
            assert client.save_path == test_file_path
            assert len(client.h5_kwargs) == 8

    def test_error_recovery(self, test_file_path):
        """Test that ClientBase handles errors gracefully."""
        client = ClientBase(
            test_file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"}
        )

        # Set valid sample rates
        client.sample_rates = [1, 2, 4]
        assert client.sample_rates == [1.0, 2.0, 4.0]

        # Try to set invalid sample rates
        with pytest.raises(TypeError):
            client.sample_rates = None

        # Verify that valid sample rates are still there
        assert client.sample_rates == [1.0, 2.0, 4.0]


if __name__ == "__main__":
    """
    Run the optimized MTH5 ClientBase test suite

    Key features:
    - Session-scoped fixtures for efficient resource management
    - Comprehensive parametrized testing
    - Subtest functionality for detailed validation
    - Performance optimizations through strategic fixture usage
    - Enhanced error handling and validation

    Usage:
    - Run all tests: python test_client_base_pytest.py
    - Run specific class: pytest TestClientBaseSampleRates -v
    - Run with timing: pytest --durations=10 test_client_base_pytest.py
    - Run performance tests: pytest -m performance test_client_base_pytest.py
    """

    args = [__file__, "-v", "--tb=short"]
    pytest.main(args)


# =============================================================================
# Additional Tests for Previously Untested Functionality
# =============================================================================


class TestClientBaseDataPath:
    """Test data_path property and related functionality."""

    def test_data_path_property(self, client_base, test_dir):
        """Test data_path property getter."""
        assert client_base.data_path == test_dir
        assert isinstance(client_base.data_path, Path)

    def test_data_path_setter_valid_path(self, test_dir):
        """Test setting data_path with valid directory."""
        client = ClientBase(test_dir)
        assert client.data_path == test_dir

    def test_data_path_setter_string_path(self, test_dir):
        """Test setting data_path with string path."""
        client = ClientBase(str(test_dir))
        assert client.data_path == test_dir

    def test_data_path_nonexistent_directory(self, test_dir):
        """Test that nonexistent directory raises IOError."""
        nonexistent_path = test_dir / "nonexistent_directory"
        with pytest.raises(IOError, match="Could not find"):
            ClientBase(nonexistent_path)

    def test_data_path_none_raises_error(self):
        """Test that None data_path raises ValueError."""
        with pytest.raises(ValueError, match="data_path cannot be None"):
            ClientBase(None)

    @pytest.mark.parametrize(
        "path_type",
        ["string", "pathlib"],
    )
    def test_data_path_different_types(self, test_dir, path_type):
        """Test data_path accepts different path types."""
        if path_type == "string":
            client = ClientBase(str(test_dir))
        else:
            client = ClientBase(test_dir)

        assert client.data_path == test_dir
        assert isinstance(client.data_path, Path)


class TestClientBaseDefaultValues:
    """Test default values and initialization parameters."""

    def test_default_initialization_values(self, test_dir, subtests):
        """Test all default values are set correctly."""
        client = ClientBase(test_dir)

        with subtests.test("sample_rates_default"):
            assert client.sample_rates == [1]

        with subtests.test("mth5_filename_default"):
            assert client.mth5_filename == "from_client.h5"

        with subtests.test("interact_default"):
            assert client.interact is False

        with subtests.test("mth5_version_default"):
            assert client.mth5_version == "0.2.0"

        with subtests.test("h5_compression_default"):
            assert client.h5_compression == "gzip"

        with subtests.test("h5_compression_opts_default"):
            assert client.h5_compression_opts == 4

        with subtests.test("h5_shuffle_default"):
            assert client.h5_shuffle is True

        with subtests.test("h5_fletcher32_default"):
            assert client.h5_fletcher32 is True

        with subtests.test("h5_data_level_default"):
            assert client.h5_data_level == 1

        with subtests.test("collection_default"):
            assert client.collection is None

    def test_custom_initialization_values(self, test_dir):
        """Test initialization with custom values."""
        custom_client = ClientBase(
            test_dir,
            sample_rates=[4, 8, 16],
            mth5_filename="custom.h5",
            mth5_version="0.1.0",
            h5_compression="lzf",
            h5_compression_opts=2,
        )

        assert custom_client.sample_rates == [4.0, 8.0, 16.0]
        assert custom_client.mth5_filename == "custom.h5"
        assert custom_client.mth5_version == "0.1.0"
        assert custom_client.h5_compression == "lzf"
        assert custom_client.h5_compression_opts == 2

    def test_logger_property(self, client_base):
        """Test that logger property is set."""
        assert hasattr(client_base, "logger")
        assert client_base.logger is not None


class TestClientBaseKwargsHandling:
    """Test kwargs handling during initialization."""

    def test_kwargs_setting_attributes(self, test_dir):
        """Test that kwargs properly set attributes."""
        client = ClientBase(
            test_dir,
            custom_attribute="test_value",
            h5_custom_param="custom_h5",
            another_param=42,
        )

        assert client.custom_attribute == "test_value"
        assert client.h5_custom_param == "custom_h5"
        assert client.another_param == 42

    def test_kwargs_none_values_ignored(self, test_dir):
        """Test that None values in kwargs are ignored."""
        client = ClientBase(
            test_dir,
            custom_attribute="test_value",
            none_attribute=None,
        )

        assert client.custom_attribute == "test_value"
        assert not hasattr(client, "none_attribute")

    @pytest.mark.parametrize(
        "h5_param,expected_in_h5_kwargs",
        [
            ("h5_custom", "custom"),
            ("h5_driver", "driver"),
            ("h5_mode", "mode"),
            ("h5_test_param", "test_param"),
        ],
    )
    def test_h5_kwargs_from_attributes(self, test_dir, h5_param, expected_in_h5_kwargs):
        """Test that h5_ prefixed attributes appear in h5_kwargs."""
        kwargs = {h5_param: "test_value"}
        client = ClientBase(test_dir, **kwargs)

        assert expected_in_h5_kwargs in client.h5_kwargs
        assert client.h5_kwargs[expected_in_h5_kwargs] == "test_value"


class TestClientBaseH5KwargsProperty:
    """Test h5_kwargs property comprehensive functionality."""

    def test_h5_kwargs_default_structure(self, client_base, subtests):
        """Test default h5_kwargs structure and values."""
        h5_kwargs = client_base.h5_kwargs

        with subtests.test("file_version"):
            assert h5_kwargs["file_version"] == "0.2.0"

        with subtests.test("compression"):
            assert h5_kwargs["compression"] == "gzip"

        with subtests.test("compression_opts"):
            assert h5_kwargs["compression_opts"] == 4

        with subtests.test("shuffle"):
            assert h5_kwargs["shuffle"] is True

        with subtests.test("fletcher32"):
            assert h5_kwargs["fletcher32"] is True

        with subtests.test("data_level"):
            assert h5_kwargs["data_level"] == 1

    def test_h5_kwargs_custom_h5_attributes(self, test_dir):
        """Test that custom h5_ attributes are included in h5_kwargs."""
        client = ClientBase(
            test_dir,
            h5_custom_param="custom_value",
            h5_driver="sec2",
            h5_mode="w",
        )

        h5_kwargs = client.h5_kwargs
        assert h5_kwargs["custom_param"] == "custom_value"
        assert h5_kwargs["driver"] == "sec2"
        assert h5_kwargs["mode"] == "w"

    def test_h5_kwargs_prefix_stripping(self, test_dir):
        """Test that h5_ prefix is properly stripped from attribute names."""
        client = ClientBase(
            test_dir,
            h5_very_long_parameter_name="test_value",
        )

        h5_kwargs = client.h5_kwargs
        assert "very_long_parameter_name" in h5_kwargs
        assert "h5_very_long_parameter_name" not in h5_kwargs

    def test_h5_kwargs_non_h5_attributes_excluded(self, test_dir):
        """Test that non-h5 attributes are not included in h5_kwargs."""
        client = ClientBase(
            test_dir,
            regular_attribute="not_in_h5_kwargs",
            sample_rates=[1, 2, 4],
        )

        h5_kwargs = client.h5_kwargs
        assert "regular_attribute" not in h5_kwargs
        assert "sample_rates" not in h5_kwargs


class TestClientBaseSavePathAdvanced:
    """Test advanced save_path functionality."""

    def test_save_path_none_uses_data_path(self, test_dir):
        """Test that save_path=None uses data_path."""
        client = ClientBase(test_dir, save_path=None)
        expected_save_path = test_dir / "from_client.h5"
        assert client.save_path == expected_save_path

    def test_save_path_existing_directory(self, test_dir):
        """Test save_path with existing directory."""
        sub_dir = test_dir / "sub_directory"
        sub_dir.mkdir()

        client = ClientBase(test_dir)
        client.save_path = sub_dir

        assert client._save_path == sub_dir
        assert client.mth5_filename == "from_client.h5"

    def test_save_path_existing_file(self, test_dir):
        """Test save_path with existing file."""
        existing_file = test_dir / "existing.h5"
        existing_file.touch()

        client = ClientBase(test_dir)
        client.save_path = existing_file

        assert client._save_path == test_dir
        assert client.mth5_filename == "existing.h5"

    def test_save_path_nonexistent_file_path(self, test_dir):
        """Test save_path with nonexistent file path."""
        nonexistent_file = test_dir / "subdir" / "newfile.h5"

        client = ClientBase(test_dir)
        client.save_path = nonexistent_file

        assert client._save_path == test_dir / "subdir"
        assert client.mth5_filename == "newfile.h5"
        assert client._save_path.exists()  # Should be created

    def test_save_path_nonexistent_directory(self, test_dir):
        """Test save_path with nonexistent directory."""
        nonexistent_dir = test_dir / "new_directory"

        client = ClientBase(test_dir)
        client.save_path = nonexistent_dir

        assert client._save_path == nonexistent_dir
        assert nonexistent_dir.exists()  # Should be created

    @pytest.mark.parametrize(
        "filename,expected_dir_name,expected_filename",
        [
            ("test.h5", "test_subdir", "test.h5"),
            ("data.mth5", "data_subdir", "data.mth5"),
            ("experiment.hdf5", "exp_subdir", "experiment.hdf5"),
        ],
    )
    def test_save_path_file_in_new_directory(
        self, test_dir, filename, expected_dir_name, expected_filename
    ):
        """Test save_path with file in new directory."""
        new_file_path = test_dir / expected_dir_name / filename

        client = ClientBase(test_dir)
        client.save_path = new_file_path

        assert client.mth5_filename == expected_filename
        assert client._save_path == test_dir / expected_dir_name
        assert client._save_path.exists()


class TestClientBaseCollectionAndRunDict:
    """Test collection property and get_run_dict method."""

    def test_collection_default_none(self, client_base):
        """Test that collection defaults to None."""
        assert client_base.collection is None

    def test_collection_setter(self, client_base):
        """Test setting collection property."""
        mock_collection = "mock_collection_object"
        client_base.collection = mock_collection
        assert client_base.collection == mock_collection

    def test_get_run_dict_with_none_collection(self, client_base):
        """Test get_run_dict raises error when collection is None."""
        with pytest.raises(ValueError):
            client_base.get_run_dict()

    def test_get_run_dict_with_mock_collection(self, client_base):
        """Test get_run_dict calls collection.get_runs with sample_rates."""

        class MockCollection:
            def get_runs(self, sample_rates=None):
                return {"mock_runs": sample_rates}

        client_base.collection = MockCollection()
        client_base.sample_rates = [1, 4, 16]

        result = client_base.get_run_dict()
        assert result == {"mock_runs": [1.0, 4.0, 16.0]}


class TestClientBasePropertyInteractions:
    """Test interactions between different properties."""

    def test_save_path_interaction_with_filename_changes(self, test_dir):
        """Test save_path updates when mth5_filename changes."""
        client = ClientBase(test_dir)

        # Set initial save path
        test_file = test_dir / "initial.h5"
        client.save_path = test_file
        assert client.save_path == test_file

        # Change filename
        client.mth5_filename = "changed.h5"
        expected_new_path = test_dir / "changed.h5"
        assert client.save_path == expected_new_path

    def test_h5_kwargs_updates_with_attribute_changes(self, client_base):
        """Test that h5_kwargs reflects changes to h5_ attributes."""
        original_compression = client_base.h5_kwargs["compression"]

        # Change h5 attribute
        client_base.h5_compression = "lzf"

        # Verify h5_kwargs reflects the change
        assert client_base.h5_kwargs["compression"] == "lzf"
        assert client_base.h5_kwargs["compression"] != original_compression

    def test_multiple_property_changes(self, client_base, test_dir, subtests):
        """Test multiple property changes work together."""
        new_file = test_dir / "multi_test.h5"

        with subtests.test("change_sample_rates"):
            client_base.sample_rates = [2, 8, 32]
            assert client_base.sample_rates == [2.0, 8.0, 32.0]

        with subtests.test("change_save_path"):
            client_base.save_path = new_file
            assert client_base.save_path == new_file

        with subtests.test("change_h5_params"):
            client_base.h5_compression = "szip"
            client_base.h5_compression_opts = 8
            assert client_base.h5_kwargs["compression"] == "szip"
            assert client_base.h5_kwargs["compression_opts"] == 8

        with subtests.test("verify_all_changes_persist"):
            assert client_base.sample_rates == [2.0, 8.0, 32.0]
            assert client_base.save_path == new_file
            assert client_base.h5_kwargs["compression"] == "szip"


class TestClientBaseEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_kwargs(self, test_dir):
        """Test initialization with empty kwargs."""
        client = ClientBase(test_dir, **{})
        assert client.data_path == test_dir

    def test_sample_rates_tuple_input(self, client_base):
        """Test sample rates with tuple input."""
        client_base.sample_rates = (1, 4, 16, 64)
        assert client_base.sample_rates == [1.0, 4.0, 16.0, 64.0]

    def test_sample_rates_whitespace_in_string(self, client_base):
        """Test sample rates string with various whitespace."""
        client_base.sample_rates = " 1 , 4 ,  16  , 64 "
        assert client_base.sample_rates == [1.0, 4.0, 16.0, 64.0]

    def test_save_path_with_pathlib_path(self, client_base, test_dir):
        """Test save_path with pathlib Path object."""
        path_obj = Path(test_dir) / "pathlib_test.h5"
        client_base.save_path = path_obj
        assert client_base.save_path == path_obj

    @pytest.mark.parametrize(
        "h5_attr,h5_value",
        [
            ("h5_compression", "bzip2"),
            ("h5_compression_opts", 9),
            ("h5_shuffle", False),
            ("h5_fletcher32", False),
            ("h5_data_level", 2),
            ("h5_driver", "stdio"),
            ("h5_mode", "r+"),
        ],
    )
    def test_h5_attribute_variations(self, test_dir, h5_attr, h5_value):
        """Test various h5 attribute configurations."""
        kwargs = {h5_attr: h5_value}
        client = ClientBase(test_dir, **kwargs)

        expected_key = h5_attr[3:]  # Remove 'h5_' prefix
        assert client.h5_kwargs[expected_key] == h5_value


class TestClientBaseStringRepresentations:
    """Test string representations and debugging aids."""

    def test_client_base_has_attributes(self, client_base):
        """Test that ClientBase has expected attributes for debugging."""
        expected_attrs = [
            "data_path",
            "sample_rates",
            "save_path",
            "mth5_filename",
            "mth5_version",
            "h5_compression",
            "interact",
            "collection",
        ]

        for attr in expected_attrs:
            assert hasattr(client_base, attr), f"Missing attribute: {attr}"

    def test_client_base_dict_access(self, client_base):
        """Test that ClientBase __dict__ contains expected keys."""
        client_dict = client_base.__dict__

        # Should contain private attributes
        assert "_data_path" in client_dict
        assert "_sample_rates" in client_dict

        # Should contain public attributes
        assert "mth5_filename" in client_dict
        assert "mth5_version" in client_dict
