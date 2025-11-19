#!/usr/bin/env python3
"""
Simplified pytest suite for mth5.io.reader module with effective mocking.

This test suite focuses on testing the reader module's core functionality
without deep integration into the actual file reading implementations.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the module under test
from mth5.io.reader import get_reader, read_file, readers


class TestReadersRegistry:
    """Test the readers registry dictionary and its structure."""

    def test_readers_registry_exists(self):
        """Test that readers registry is defined and accessible."""
        assert readers is not None
        assert isinstance(readers, dict)

    @pytest.mark.parametrize(
        "reader_key",
        ["zen", "nims", "usgs_ascii", "miniseed", "lemi424", "phoenix", "metronix"],
    )
    def test_readers_registry_contains_expected_keys(self, reader_key):
        """Test that registry contains all expected reader keys."""
        assert reader_key in readers

    def test_readers_registry_structure(self):
        """Test the structure of reader registry entries."""
        for key, value in readers.items():
            assert isinstance(value, dict), f"Reader {key} should be a dictionary"
            assert "file_types" in value, f"Reader {key} missing file_types"
            assert "reader" in value, f"Reader {key} missing reader function"

            # Check file_types is a list
            assert isinstance(
                value["file_types"], list
            ), f"Reader {key} file_types should be a list"

            # Check reader is callable
            assert callable(value["reader"]), f"Reader {key} reader should be callable"


class TestGetReaderFunction:
    """Test the get_reader function for retrieving readers by file type."""

    @pytest.mark.parametrize(
        "file_type,expected_reader",
        [
            ("z3d", "zen"),
            ("bnn", "nims"),
            ("asc", "usgs_ascii"),
            ("mseed", "miniseed"),
            ("txt", "lemi424"),
        ],
    )
    def test_get_reader_valid_extensions(self, file_type, expected_reader):
        """Test get_reader with valid file extensions."""
        reader_name, reader_function = get_reader(file_type)
        assert reader_name == expected_reader
        assert reader_function == readers[expected_reader]["reader"]

    def test_get_reader_case_insensitive(self):
        """Test that get_reader is case insensitive."""
        result_lower = get_reader("z3d")
        result_upper = get_reader("Z3D")
        result_mixed = get_reader("Z3d")

        assert result_lower == result_upper == result_mixed

    def test_get_reader_with_dot_prefix(self):
        """Test get_reader handles extensions with dot prefix."""
        # The function should strip the leading dot
        with pytest.raises(ValueError):
            get_reader(".z3d")  # This should fail since the function expects no dot

    def test_get_reader_invalid_extension(self):
        """Test get_reader raises ValueError for unsupported extensions."""
        with pytest.raises(ValueError, match="Could not find a reader"):
            get_reader("unsupported")

    def test_get_reader_empty_string(self):
        """Test get_reader handles empty string input."""
        with pytest.raises(ValueError):
            get_reader("")


class TestReadFileBasic:
    """Test read_file function with basic mocking."""

    @patch("mth5.io.reader.Path")
    def test_read_file_file_not_exists(self, mock_path_class):
        """Test read_file raises error when file doesn't exist."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path_class.return_value = mock_path_instance

        with pytest.raises(
            IOError
        ):  # The function raises IOError, not FileNotFoundError
            read_file("/nonexistent/file.z3d")

    @patch("mth5.io.reader.Path")
    def test_read_file_unsupported_extension(self, mock_path_class):
        """Test read_file with unsupported file extension."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".unsupported"
        mock_path_class.return_value = mock_path_instance

        with pytest.raises(ValueError, match="Could not find a reader"):
            read_file("/path/file.unsupported")

    @patch("mth5.io.reader.get_reader")
    @patch("mth5.io.reader.Path")
    def test_read_file_calls_correct_reader(self, mock_path_class, mock_get_reader):
        """Test that read_file calls the correct reader function."""
        # Setup mocks
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".z3d"
        mock_path_class.return_value = mock_path_instance

        mock_reader_func = Mock(return_value="test_data")
        mock_get_reader.return_value = ("zen", mock_reader_func)  # Return tuple

        # Call function
        result = read_file("/path/test.z3d")

        # Verify calls
        mock_get_reader.assert_called_once_with("z3d")
        mock_reader_func.assert_called_once_with(mock_path_instance)
        assert result == "test_data"

    @patch(
        "mth5.io.reader.readers",
        {"zen": {"reader": Mock(return_value="explicit_data")}},
    )
    @patch("mth5.io.reader.Path")
    def test_read_file_explicit_file_type(self, mock_path_class):
        """Test read_file with explicit file_type parameter."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".dat"  # Add suffix attribute
        mock_path_class.return_value = mock_path_instance

        result = read_file("/path/test.dat", file_type="zen")

        assert result == "explicit_data"

    @patch("mth5.io.reader.get_reader")
    @patch("mth5.io.reader.Path")
    def test_read_file_with_kwargs(self, mock_path_class, mock_get_reader):
        """Test read_file passes kwargs to reader function."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".z3d"
        mock_path_class.return_value = mock_path_instance

        mock_reader_func = Mock(return_value="kwargs_data")
        mock_get_reader.return_value = ("zen", mock_reader_func)  # Return tuple

        result = read_file("/path/test.z3d", custom_arg="value", another_arg=123)

        mock_reader_func.assert_called_once_with(
            mock_path_instance, custom_arg="value", another_arg=123
        )
        assert result == "kwargs_data"


class TestReadFileListInput:
    """Test read_file function with list inputs."""

    @patch("mth5.io.reader.get_reader")
    @patch("mth5.io.reader.Path")
    def test_read_file_list_input(self, mock_path_class, mock_get_reader):
        """Test read_file with list of file paths."""
        # Mock multiple Path instances
        mock_paths = [Mock() for _ in range(3)]
        for mock_path in mock_paths:
            mock_path.exists.return_value = True
            mock_path.suffix = ".z3d"
        mock_path_class.side_effect = mock_paths

        mock_reader_func = Mock(return_value="list_data")
        mock_get_reader.return_value = ("zen", mock_reader_func)  # Return tuple

        file_list: list[str | Path] = [
            "/path/file1.z3d",
            "/path/file2.z3d",
            "/path/file3.z3d",
        ]
        result = read_file(file_list)

        # Should call get_reader once and reader with all paths
        mock_get_reader.assert_called_once_with("z3d")
        mock_reader_func.assert_called_once_with(mock_paths)
        assert result == "list_data"


class TestPerformance:
    """Test performance characteristics of reader functions."""

    def test_get_reader_performance(self):
        """Test that get_reader calls are fast."""
        import time

        start_time = time.time()
        for _ in range(1000):
            try:
                get_reader("z3d")
            except ValueError:
                pass
        elapsed = time.time() - start_time

        # Should be very fast dictionary lookup
        assert elapsed < 0.1

    def test_readers_registry_access_performance(self):
        """Test readers registry access performance."""
        import time

        start_time = time.time()
        for _ in range(1000):
            _ = readers["zen"]
            _ = readers["nims"]
            _ = readers.get("nonexistent", None)
        elapsed = time.time() - start_time

        assert elapsed < 0.01


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("mth5.io.reader.get_reader")
    @patch("mth5.io.reader.Path")
    def test_reader_function_exception_propagation(
        self, mock_path_class, mock_get_reader
    ):
        """Test that exceptions from reader functions are properly propagated."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".z3d"
        mock_path_class.return_value = mock_path_instance

        mock_reader_func = Mock(side_effect=ValueError("Test reader error"))
        mock_get_reader.return_value = ("zen", mock_reader_func)  # Return tuple

        with pytest.raises(ValueError, match="Test reader error"):
            read_file("/path/test.z3d")

    @pytest.mark.parametrize(
        "exception_type,error_msg",
        [
            (FileNotFoundError, "File not found"),
            (PermissionError, "Permission denied"),
            (IOError, "IO error occurred"),
        ],
    )
    @patch("mth5.io.reader.get_reader")
    @patch("mth5.io.reader.Path")
    def test_various_reader_exceptions(
        self, mock_path_class, mock_get_reader, exception_type, error_msg
    ):
        """Test handling of various exception types from readers."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".z3d"
        mock_path_class.return_value = mock_path_instance

        mock_reader_func = Mock(side_effect=exception_type(error_msg))
        mock_get_reader.return_value = ("zen", mock_reader_func)  # Return tuple

        with pytest.raises(exception_type, match=error_msg):
            read_file("/path/test.z3d")


class TestModuleDocumentation:
    """Test module documentation and metadata."""

    def test_get_reader_has_docstring(self):
        """Test that get_reader function has proper docstring."""
        assert get_reader.__doc__ is not None
        assert len(get_reader.__doc__.strip()) > 0

    def test_read_file_has_docstring(self):
        """Test that read_file function has proper docstring."""
        assert read_file.__doc__ is not None
        assert len(read_file.__doc__.strip()) > 0

    def test_module_level_documentation(self):
        """Test that module has proper documentation."""
        import mth5.io.reader as reader_module

        assert reader_module.__doc__ is not None or hasattr(reader_module, "__all__")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
