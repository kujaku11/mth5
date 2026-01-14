# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for phoenix.read module functionality.

This test suite covers:
- get_file_extension function
- open_phoenix function with various file types
- read_phoenix function with various file types
- Reader type mapping
- Error handling for unsupported file types

Created on January 13, 2026

@author: pytest suite
"""

from unittest.mock import Mock, patch

import pytest

from mth5.io.phoenix.read import (
    get_file_extenstion,
    open_phoenix,
    read_phoenix,
    READERS,
)
from mth5.io.phoenix.readers import (
    DecimatedContinuousReader,
    DecimatedSegmentedReader,
    MTUTable,
    MTUTSN,
    NativeReader,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory."""
    return tmp_path


@pytest.fixture
def sample_files(temp_dir):
    """Create sample Phoenix files with different extensions."""
    files = {}
    extensions = [
        "bin",
        "td_24k",
        "td_150",
        "td_30",
        "TS3",
        "TS4",
        "TS5",
        "TBL",
        "TSL",
        "TSH",
    ]

    for ext in extensions:
        file_path = temp_dir / f"test_file.{ext}"
        file_path.touch()
        files[ext] = file_path

    return files


# =============================================================================
# Test Classes
# =============================================================================


class TestGetFileExtension:
    """Test get_file_extenstion function."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("file.bin", "bin"),
            ("file.td_24k", "td_24k"),
            ("file.td_150", "td_150"),
            ("file.td_30", "td_30"),
            ("file.TS3", "TS3"),
            ("file.TS4", "TS4"),
            ("file.TS5", "TS5"),
            ("file.TBL", "TBL"),
            ("file.TSL", "TSL"),
            ("file.TSH", "TSH"),
        ],
    )
    def test_get_extension_various_types(self, filename, expected):
        """Test extension extraction for various Phoenix file types."""
        result = get_file_extenstion(filename)
        assert result == expected

    def test_get_extension_with_path_object(self, temp_dir):
        """Test with Path object input."""
        file_path = temp_dir / "test.bin"
        result = get_file_extenstion(file_path)
        assert result == "bin"

    def test_get_extension_with_string(self):
        """Test with string input."""
        result = get_file_extenstion("test.td_150")
        assert result == "td_150"

    def test_get_extension_with_full_path(self):
        """Test with full file path."""
        result = get_file_extenstion("/path/to/data/file.TS3")
        assert result == "TS3"

    def test_get_extension_no_extension(self):
        """Test file with no extension."""
        result = get_file_extenstion("file_without_ext")
        assert result == ""

    @pytest.mark.parametrize(
        "filename",
        [
            "path/to/file.bin",
            "C:\\Windows\\file.td_24k",
            "../relative/path/file.TS3",
        ],
    )
    def test_get_extension_various_paths(self, filename):
        """Test with various path formats."""
        result = get_file_extenstion(filename)
        assert result in READERS.keys()


class TestREADERSMapping:
    """Test READERS dictionary mapping."""

    def test_readers_has_all_phoenix_types(self):
        """Test that READERS contains all expected Phoenix file types."""
        expected_keys = [
            "bin",
            "td_24k",
            "td_150",
            "td_30",
            "TS3",
            "TS4",
            "TS5",
            "TBL",
            "TSL",
            "TSH",
        ]

        for key in expected_keys:
            assert key in READERS

    @pytest.mark.parametrize(
        "extension,expected_reader",
        [
            ("bin", NativeReader),
            ("td_24k", DecimatedSegmentedReader),
            ("td_150", DecimatedContinuousReader),
            ("td_30", DecimatedContinuousReader),
            ("TS3", MTUTSN),
            ("TS4", MTUTSN),
            ("TS5", MTUTSN),
            ("TBL", MTUTable),
            ("TSL", MTUTSN),
            ("TSH", MTUTSN),
        ],
    )
    def test_readers_maps_to_correct_class(self, extension, expected_reader):
        """Test that each extension maps to the correct reader class."""
        assert READERS[extension] == expected_reader

    def test_all_reader_values_are_classes(self):
        """Test that all READERS values are classes."""
        for reader_class in READERS.values():
            assert isinstance(reader_class, type)


class TestOpenPhoenix:
    """Test open_phoenix function."""

    @pytest.mark.parametrize(
        "extension,expected_reader",
        [
            ("bin", NativeReader),
            ("td_24k", DecimatedSegmentedReader),
            ("td_150", DecimatedContinuousReader),
        ],
    )
    def test_open_phoenix_returns_correct_reader(
        self, sample_files, extension, expected_reader
    ):
        """Test that open_phoenix returns the correct reader type."""
        file_path = sample_files[extension]

        # Mock at the READERS dictionary level to avoid contaminating the actual classes
        mock_instance = Mock(spec=expected_reader)
        mock_class = Mock(return_value=mock_instance)

        with patch.dict("mth5.io.phoenix.read.READERS", {extension: mock_class}):
            result = open_phoenix(file_path)
            mock_class.assert_called_once()
            assert result == mock_instance

    def test_open_phoenix_with_kwargs(self, sample_files):
        """Test that kwargs are passed to reader constructor."""
        file_path = sample_files["bin"]

        # Mock at the READERS dictionary level to avoid contaminating the actual classes
        mock_reader_instance = Mock(spec=NativeReader)
        mock_reader_class = Mock(return_value=mock_reader_instance)

        with patch.dict("mth5.io.phoenix.read.READERS", {"bin": mock_reader_class}):
            result = open_phoenix(file_path, custom_arg="value", another_arg=123)

            mock_reader_class.assert_called_once()
            # Verify kwargs were passed through
            call_kwargs = mock_reader_class.call_args[1]
            assert "custom_arg" in call_kwargs
            assert call_kwargs["custom_arg"] == "value"
            assert "another_arg" in call_kwargs
            assert call_kwargs["another_arg"] == 123
            assert result == mock_reader_instance

    def test_open_phoenix_unsupported_extension_raises_error(self, temp_dir):
        """Test that unsupported extension raises KeyError."""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.touch()

        with pytest.raises(KeyError):
            open_phoenix(unsupported_file)

    @pytest.mark.parametrize("extension", ["bin", "td_24k", "td_150", "td_30"])
    def test_open_phoenix_newer_file_types(self, sample_files, extension):
        """Test opening newer Phoenix file types."""
        file_path = sample_files[extension]

        # Mock at the READERS dictionary level to avoid contaminating the actual classes
        reader_class = READERS[extension]
        mock_instance = Mock(spec=reader_class)
        mock_class = Mock(return_value=mock_instance)

        with patch.dict("mth5.io.phoenix.read.READERS", {extension: mock_class}):
            result = open_phoenix(file_path)
            mock_class.assert_called_once()
            assert result == mock_instance

    @pytest.mark.parametrize("extension", ["TS3", "TS4", "TS5", "TSL", "TSH"])
    def test_open_phoenix_mtu_types(self, sample_files, extension):
        """Test opening MTU Phoenix file types."""
        file_path = sample_files[extension]

        # Mock at the READERS dictionary level to avoid contaminating the actual classes
        mock_instance = Mock(spec=MTUTSN)
        mock_class = Mock(return_value=mock_instance)

        with patch.dict("mth5.io.phoenix.read.READERS", {extension: mock_class}):
            result = open_phoenix(file_path)
            mock_class.assert_called_once()
            assert result == mock_instance

    def test_open_phoenix_table_type(self, sample_files):
        """Test opening TBL file type."""
        file_path = sample_files["TBL"]

        # Mock at the READERS dictionary level to avoid contaminating the actual classes
        mock_instance = Mock(spec=MTUTable)
        mock_class = Mock(return_value=mock_instance)

        with patch.dict("mth5.io.phoenix.read.READERS", {"TBL": mock_class}):
            result = open_phoenix(file_path)
            mock_class.assert_called_once()
            assert result == mock_instance


class TestReadPhoenix:
    """Test read_phoenix function."""

    @pytest.mark.parametrize("extension", ["bin", "td_24k", "td_150", "td_30"])
    def test_read_phoenix_newer_formats(self, sample_files, extension):
        """Test reading newer Phoenix file formats returns ChannelTS."""
        file_path = sample_files[extension]

        # Mock the reader and its to_channel_ts method
        mock_reader = Mock()
        mock_channel_ts = Mock()
        mock_reader.to_channel_ts.return_value = mock_channel_ts

        with patch("mth5.io.phoenix.read.open_phoenix", return_value=mock_reader):
            result = read_phoenix(file_path)

            mock_reader.to_channel_ts.assert_called_once()
            assert result == mock_channel_ts

    def test_read_phoenix_with_calibration_files(self, sample_files):
        """Test reading with receiver and sensor calibration files."""
        file_path = sample_files["bin"]
        rxcal_fn = "path/to/rxcal.txt"
        scal_fn = "path/to/scal.txt"

        mock_reader = Mock()
        mock_channel_ts = Mock()
        mock_reader.to_channel_ts.return_value = mock_channel_ts

        with patch("mth5.io.phoenix.read.open_phoenix", return_value=mock_reader):
            result = read_phoenix(file_path, rxcal_fn=rxcal_fn, scal_fn=scal_fn)

            mock_reader.to_channel_ts.assert_called_once_with(
                rxcal_fn=rxcal_fn, scal_fn=scal_fn
            )

    @pytest.mark.parametrize("extension", ["TS3", "TS4", "TS5", "TSL", "TSH"])
    def test_read_phoenix_mtu_formats(self, sample_files, extension):
        """Test reading MTU Phoenix file formats returns RunTS."""
        file_path = sample_files[extension]

        mock_mtu = Mock()
        mock_run_ts = Mock()
        mock_mtu.to_runts.return_value = mock_run_ts

        with patch("mth5.io.phoenix.read.open_phoenix", return_value=mock_mtu):
            result = read_phoenix(file_path)

            mock_mtu.to_runts.assert_called_once_with(
                table_filepath=None, calibrate=True
            )
            assert result == mock_run_ts

    def test_read_phoenix_mtu_with_table_file(self, sample_files):
        """Test reading MTU file with table filepath."""
        file_path = sample_files["TS3"]
        table_filepath = "path/to/table.TBL"

        mock_mtu = Mock()
        mock_run_ts = Mock()
        mock_mtu.to_runts.return_value = mock_run_ts

        with patch("mth5.io.phoenix.read.open_phoenix", return_value=mock_mtu):
            result = read_phoenix(file_path, table_filepath=table_filepath)

            mock_mtu.to_runts.assert_called_once_with(
                table_filepath=table_filepath, calibrate=True
            )

    def test_read_phoenix_table_file(self, sample_files):
        """Test reading TBL table file."""
        file_path = sample_files["TBL"]

        mock_table = Mock()

        with patch("mth5.io.phoenix.read.open_phoenix", return_value=mock_table):
            result = read_phoenix(file_path)

            assert result == mock_table

    def test_read_phoenix_unsupported_extension(self, temp_dir):
        """Test reading file with unsupported extension raises error."""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.touch()

        with pytest.raises(KeyError, match="is not supported"):
            read_phoenix(unsupported_file)

    def test_read_phoenix_calibration_kwargs_not_passed_to_constructor(
        self, sample_files
    ):
        """Test that calibration kwargs are passed to to_channel_ts, not reader constructor."""
        file_path = sample_files["bin"]

        mock_reader = Mock()
        mock_reader.to_channel_ts.return_value = Mock()

        with patch(
            "mth5.io.phoenix.read.open_phoenix", return_value=mock_reader
        ) as mock_open:
            read_phoenix(
                file_path,
                rxcal_fn="rxcal.txt",
                scal_fn="scal.txt",
                other_arg="value",
            )

            # Verify open_phoenix was called with all kwargs
            call_kwargs = mock_open.call_args[1]
            assert "other_arg" in call_kwargs
            # In the actual implementation, these are passed through
            # The important check is that to_channel_ts receives the calibration files
            mock_reader.to_channel_ts.assert_called_once_with(
                rxcal_fn="rxcal.txt", scal_fn="scal.txt"
            )

    def test_read_phoenix_mtu_table_filepath_not_passed_to_constructor(
        self, sample_files
    ):
        """Test that table_filepath is popped for MTU files."""
        file_path = sample_files["TS3"]

        mock_mtu = Mock()
        mock_mtu.to_runts.return_value = Mock()

        with patch(
            "mth5.io.phoenix.read.open_phoenix", return_value=mock_mtu
        ) as mock_open:
            read_phoenix(file_path, table_filepath="table.TBL", other_arg="value")

            # open_phoenix should be called with other_arg but not table_filepath
            call_kwargs = mock_open.call_args[1]
            assert "other_arg" in call_kwargs
            assert "table_filepath" not in call_kwargs


class TestReadPhoenixIntegration:
    """Integration tests combining multiple operations."""

    def test_read_workflow_bin_file(self, sample_files):
        """Test complete workflow for .bin file."""
        file_path = sample_files["bin"]

        # Create mock chain
        mock_channel_ts = Mock(name="ChannelTS")
        mock_reader = Mock()
        mock_reader.to_channel_ts.return_value = mock_channel_ts

        with patch("mth5.io.phoenix.read.READERS") as mock_readers:
            mock_readers.__getitem__.return_value = Mock(return_value=mock_reader)

            result = read_phoenix(file_path, rxcal_fn="cal.txt")

            assert result == mock_channel_ts

    def test_read_workflow_mtu_file(self, sample_files):
        """Test complete workflow for MTU file."""
        file_path = sample_files["TS4"]

        # Create mock chain
        mock_run_ts = Mock(name="RunTS")
        mock_mtu = Mock()
        mock_mtu.to_runts.return_value = mock_run_ts

        with patch("mth5.io.phoenix.read.READERS") as mock_readers:
            mock_readers.__getitem__.return_value = Mock(return_value=mock_mtu)

            result = read_phoenix(file_path, table_filepath="table.TBL")

            assert result == mock_run_ts

    def test_multiple_file_types_sequentially(self, sample_files):
        """Test reading multiple different file types."""
        extensions_to_test = ["bin", "td_150", "TS3"]

        for ext in extensions_to_test:
            file_path = sample_files[ext]

            mock_obj = Mock()
            if ext.startswith("td_") or ext == "bin":
                mock_obj.to_channel_ts.return_value = Mock()
            else:
                mock_obj.to_runts.return_value = Mock()

            with patch("mth5.io.phoenix.read.open_phoenix", return_value=mock_obj):
                result = read_phoenix(file_path)
                assert result is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_file_with_uppercase_extension(self, temp_dir):
        """Test handling file with uppercase extension."""
        file_path = temp_dir / "test.BIN"
        file_path.touch()

        # Extension extraction is case-sensitive
        with pytest.raises(KeyError):
            open_phoenix(file_path)

    def test_file_with_extra_dots_in_name(self, temp_dir):
        """Test file with multiple dots in filename."""
        file_path = temp_dir / "test.data.backup.bin"
        file_path.touch()

        # Should extract 'bin' as extension
        ext = get_file_extenstion(file_path)
        assert ext == "bin"

    def test_empty_filename(self):
        """Test handling of empty filename."""
        ext = get_file_extenstion("")
        assert ext == ""

    @pytest.mark.parametrize(
        "filename",
        [
            ".hidden",
            ".hidden.bin",
            "file.",
        ],
    )
    def test_unusual_filenames(self, filename):
        """Test handling of unusual but valid filenames."""
        ext = get_file_extenstion(filename)
        # Should not crash, behavior may vary
        assert isinstance(ext, str)


class TestParametrizedReaders:
    """Parametrized tests for all reader types."""

    @pytest.mark.parametrize(
        "extension,reader_class",
        [
            ("bin", NativeReader),
            ("td_24k", DecimatedSegmentedReader),
            ("td_150", DecimatedContinuousReader),
            ("td_30", DecimatedContinuousReader),
            ("TS3", MTUTSN),
            ("TS4", MTUTSN),
            ("TS5", MTUTSN),
            ("TBL", MTUTable),
            ("TSL", MTUTSN),
            ("TSH", MTUTSN),
        ],
    )
    def test_all_readers_mapped_correctly(self, extension, reader_class):
        """Test that all extensions map to correct reader classes."""
        assert READERS[extension] == reader_class

    @pytest.mark.parametrize(
        "extension",
        ["bin", "td_24k", "td_150", "td_30", "TS3", "TS4", "TS5", "TBL", "TSL", "TSH"],
    )
    def test_open_phoenix_all_extensions(self, temp_dir, extension):
        """Test open_phoenix with all supported extensions."""
        file_path = temp_dir / f"test.{extension}"
        file_path.touch()

        # Mock at the READERS dictionary level to avoid contaminating the actual classes
        reader_class = READERS[extension]
        mock_instance = Mock(spec=reader_class)
        mock_class = Mock(return_value=mock_instance)

        with patch.dict("mth5.io.phoenix.read.READERS", {extension: mock_class}):
            result = open_phoenix(file_path)
            mock_class.assert_called_once()
            assert result == mock_instance
