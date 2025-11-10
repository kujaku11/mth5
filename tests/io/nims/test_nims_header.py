# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for NIMS Header class testing.

Created on Fri Nov 22 18:15:00 2024

@author: jpeacock

Pytest suite for NIMSHeader class with comprehensive coverage including:
- Header parsing and initialization
- File I/O operations
- GPS coordinate conversion
- Error handling and edge cases
- Property validation
"""

import shutil
import tempfile
from pathlib import Path

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata.common.mttime import MTime

from mth5.io.nims import NIMSHeader
from mth5.io.nims.header import NIMSError


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_header_string():
    """Sample NIMS header string based on the example in the docstring."""
    return (
        b">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\r"
        b">>>user field>>>>>>>>>>>>>>>>>>>>>>>>>>>>\r"
        b"SITE NAME: Budwieser Spring\r"
        b"STATE/PROVINCE: CA\r"
        b"COUNTRY: USA\r"
        b">>> The following code in double quotes is REQUIRED to start the NIMS <<\r"
        b">>> The next 3 lines contain values required for processing <<<<<<<<<<<<\r"
        b">>> The lines after that are optional <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\r"
        b'"300b"  <-- 2CHAR EXPERIMENT CODE + 3 CHAR SITE CODE + RUN LETTER\r'
        b"1105-3; 1305-3  <-- SYSTEM BOX I.D.; MAG HEAD ID (if different)\r"
        b"106  0 <-- N-S Ex WIRE LENGTH (m); HEADING (deg E mag N)\r"
        b"109  90 <-- E-W Ey WIRE LENGTH (m); HEADING (deg E mag N)\r"
        b"1         <-- N ELECTRODE ID\r"
        b"3         <-- E ELECTRODE ID\r"
        b"2         <-- S ELECTRODE ID\r"
        b"4         <-- W ELECTRODE ID\r"
        b"Cu        <-- GROUND ELECTRODE INFO\r"
        b"GPS INFO: 26/09/19 18:29:29 34.7268 N 115.7350 W 939.8\r"
        b"OPERATOR: KP\r"
        b"COMMENT: N/S CRS: .95/.96 DCV: 3.5 ACV:1\r"
        b"E/W CRS: .85/.86 DCV: 1.5 ACV: 1\r"
        b"Redeployed site for run b b/c possible animal disturbance\r"
        b"\r"
        b"g12345"  # Start of binary data
    )


@pytest.fixture
def expected_parsed_data():
    """Expected parsed data from the sample header."""
    return {
        "site_name": "Budwieser Spring",
        "state_province": "CA",
        "country": "USA",
        "run_id": "300b",
        "box_id": "1105-3",
        "mag_id": "1305-3",
        "ex_length": 106.0,
        "ex_azimuth": 0.0,
        "ey_length": 109.0,
        "ey_azimuth": 90.0,
        "n_electrode_id": "1",
        "e_electrode_id": "3",
        "s_electrode_id": "2",
        "w_electrode_id": "4",
        "ground_electrode_info": "Cu",
        "header_gps_latitude": 34.7268,
        "header_gps_longitude": -115.7350,  # W means negative
        "header_gps_elevation": 939.8,
        "operator": "KP",
        "station": "300",  # run_id without last character
    }


@pytest.fixture
def temp_nims_file(sample_header_string):
    """Create a temporary NIMS file for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / "test_nims.bnn"

    # Write header + some binary data
    with open(temp_file, "wb") as f:
        f.write(sample_header_string)
        f.write(b"binary_data_continues_here" * 100)  # Some binary content

    yield temp_file

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def nims_header():
    """Create a default NIMSHeader instance."""
    return NIMSHeader()


@pytest.fixture
def populated_header_dict():
    """Header dictionary with all expected fields."""
    return {
        "site name": "Budwieser Spring",
        "state/province": "CA",
        "country": "USA",
        "run letter": '"300b"',
        "system box i.d.; mag head id (if different)": "1105-3; 1305-3",
        "n-s ex wire length (m); heading (deg e mag n)": "106  0",
        "e-w ey wire length (m); heading (deg e mag n)": "109  90",
        "n electrode id": "1",
        "e electrode id": "3",
        "s electrode id": "2",
        "w electrode id": "4",
        "ground electrode info": "Cu",
        "gps info": "26/09/19 18:29:29 34.7268 N 115.7350 W 939.8",
        "operator": "KP",
        "comment": "N/S CRS: .95/.96 DCV: 3.5 ACV:1",
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestNIMSHeaderInitialization:
    """Test NIMSHeader initialization and basic properties."""

    def test_default_initialization(self, nims_header):
        """Test default initialization of NIMSHeader."""
        assert nims_header.fn is None
        assert nims_header._max_header_length == 1000
        assert nims_header.header_dict is None
        assert nims_header.data_start_seek == 0

        # Test all attributes are initialized to None
        none_attributes = [
            "site_name",
            "state_province",
            "country",
            "box_id",
            "mag_id",
            "ex_length",
            "ex_azimuth",
            "ey_length",
            "ey_azimuth",
            "n_electrode_id",
            "s_electrode_id",
            "e_electrode_id",
            "w_electrode_id",
            "ground_electrode_info",
            "header_gps_stamp",
            "header_gps_longitude",
            "header_gps_elevation",
            "operator",
            "comments",
            "run_id",
        ]

        for attr in none_attributes:
            assert getattr(nims_header, attr) is None

    def test_initialization_with_filename(self, temp_nims_file):
        """Test initialization with filename."""
        header = NIMSHeader(fn=temp_nims_file)
        assert header.fn == temp_nims_file
        assert isinstance(header.fn, Path)

    def test_filename_property_setter(self, nims_header, temp_nims_file):
        """Test filename property setter."""
        nims_header.fn = temp_nims_file
        assert nims_header.fn == temp_nims_file
        assert isinstance(nims_header.fn, Path)

        # Test setting to None
        nims_header.fn = None
        assert nims_header.fn is None

    def test_station_property(self, nims_header):
        """Test station property derivation from run_id."""
        # Test with None run_id
        assert nims_header.station is None

        # Test with valid run_id
        nims_header.run_id = "300b"
        assert nims_header.station == "300"

        nims_header.run_id = "123a"
        assert nims_header.station == "123"

    def test_file_size_property(self, nims_header, temp_nims_file):
        """Test file size property."""
        # Test with no file
        assert nims_header.file_size is None

        # Test with actual file
        nims_header.fn = temp_nims_file
        file_size = nims_header.file_size
        assert isinstance(file_size, int)
        assert file_size > 0


class TestNIMSHeaderFileOperations:
    """Test file reading and error handling operations."""

    def test_read_header_success(
        self, nims_header, temp_nims_file, expected_parsed_data
    ):
        """Test successful header reading."""
        nims_header.read_header(temp_nims_file)

        # Verify basic parsing worked
        assert nims_header.header_dict is not None
        assert isinstance(nims_header.header_dict, dict)
        assert nims_header.data_start_seek > 0

        # Check key parsed values
        assert nims_header.site_name == expected_parsed_data["site_name"]
        assert nims_header.state_province == expected_parsed_data["state_province"]
        assert nims_header.country == expected_parsed_data["country"]
        assert nims_header.run_id == expected_parsed_data["run_id"]
        assert nims_header.station == expected_parsed_data["station"]

    def test_read_header_file_not_found(self, nims_header):
        """Test error handling for non-existent file."""
        non_existent_file = Path("/non/existent/path/file.bnn")

        with pytest.raises(NIMSError, match="Could not find nims file"):
            nims_header.read_header(non_existent_file)

    def test_read_header_with_filename_parameter(self, nims_header, temp_nims_file):
        """Test read_header with filename parameter."""
        nims_header.read_header(fn=temp_nims_file)
        assert nims_header.fn == temp_nims_file
        assert nims_header.header_dict is not None

    def test_read_header_data_start_detection(self, sample_header_string):
        """Test detection of data start position."""
        header = NIMSHeader()

        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "test_data_start.bnn"

        try:
            with open(temp_file, "wb") as f:
                f.write(sample_header_string)

            header.read_header(temp_file)

            # Data should start at the 'g' character
            assert header.data_start_seek > 0

            # Verify we can find the data start byte
            with open(temp_file, "rb") as f:
                f.seek(header.data_start_seek)
                data_byte = f.read(1)
                assert data_byte in [b"g", b"1"]  # Should be start of binary data

        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.parametrize(
        "data_prefix,expected_handling",
        [
            (b"$g12345", "handle_dollar_prefix"),
            (b" g12345", "handle_space_prefix"),
            (b"g12345", "direct_detection"),
            (b"12345", "numeric_start"),
        ],
    )
    def test_data_start_edge_cases(self, data_prefix, expected_handling):
        """Test edge cases in data start detection."""
        header_content = (
            b"SITE NAME: Test Site\r"
            b"OPERATOR: Test\r"
            b"COMMENT: Test comment\r"
            b"\r"
        )

        full_content = header_content + data_prefix

        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "test_edge.bnn"

        try:
            with open(temp_file, "wb") as f:
                f.write(full_content)

            header = NIMSHeader()
            header.read_header(temp_file)

            # Should successfully detect data start
            assert header.data_start_seek > 0
            assert header.data_start_seek < len(full_content)

        finally:
            shutil.rmtree(temp_dir)


class TestNIMSHeaderParsing:
    """Test header dictionary parsing functionality."""

    def test_parse_header_dict_basic(
        self, nims_header, populated_header_dict, expected_parsed_data
    ):
        """Test basic header dictionary parsing."""
        nims_header.parse_header_dict(populated_header_dict)

        # Check parsed values
        assert nims_header.site_name == expected_parsed_data["site_name"]
        assert nims_header.state_province == expected_parsed_data["state_province"]
        assert nims_header.country == expected_parsed_data["country"]
        assert nims_header.operator == expected_parsed_data["operator"]

    def test_parse_wire_information(self, nims_header):
        """Test parsing of wire length and azimuth information."""
        wire_dict = {
            "n-s ex wire length (m); heading (deg e mag n)": "106  0",
            "e-w ey wire length (m); heading (deg e mag n)": "109  90",
        }

        nims_header.parse_header_dict(wire_dict)

        assert nims_header.ex_length == 106.0
        assert nims_header.ex_azimuth == 0.0
        assert nims_header.ey_length == 109.0
        assert nims_header.ey_azimuth == 90.0

    def test_parse_system_information(self, nims_header):
        """Test parsing of system box and mag head IDs."""
        system_dict = {
            "system box i.d.; mag head id (if different)": "1105-3; 1305-3",
        }

        nims_header.parse_header_dict(system_dict)

        assert nims_header.box_id == "1105-3"
        assert nims_header.mag_id == "1305-3"

    def test_parse_gps_information(self, nims_header):
        """Test parsing of GPS information."""
        gps_dict = {"gps info": "26/09/19 18:29:29 34.7268 N 115.7350 W 939.8"}

        nims_header.parse_header_dict(gps_dict)

        assert isinstance(nims_header.header_gps_stamp, MTime)
        assert nims_header.header_gps_latitude == 34.7268
        assert nims_header.header_gps_longitude == -115.7350  # W = negative
        assert nims_header.header_gps_elevation == 939.8

    def test_parse_run_information(self, nims_header):
        """Test parsing of run ID with quote removal."""
        run_dict = {
            "run letter": '"300b"',
        }

        nims_header.parse_header_dict(run_dict)

        assert nims_header.run_id == "300b"  # Quotes should be removed

    def test_parse_electrode_ids(self, nims_header):
        """Test parsing of electrode IDs."""
        electrode_dict = {
            "n electrode id": "1",
            "e electrode id": "3",
            "s electrode id": "2",
            "w electrode id": "4",
            "ground electrode info": "Cu",
        }

        nims_header.parse_header_dict(electrode_dict)

        assert nims_header.n_electrode_id == "1"
        assert nims_header.e_electrode_id == "3"
        assert nims_header.s_electrode_id == "2"
        assert nims_header.w_electrode_id == "4"
        assert nims_header.ground_electrode_info == "Cu"

    def test_parse_generic_attributes(self, nims_header):
        """Test parsing of generic attributes with space/slash replacement."""
        generic_dict = {
            "test field": "test value",
            "field with spaces": "value with spaces",
            "field/with/slashes": "value with slashes",
        }

        nims_header.parse_header_dict(generic_dict)

        assert nims_header.test_field == "test value"
        assert nims_header.field_with_spaces == "value with spaces"
        assert nims_header.field_with_slashes == "value with slashes"

    def test_parse_header_dict_assertion_error(self, nims_header):
        """Test assertion error for non-dict input."""
        with pytest.raises(AssertionError):
            nims_header.parse_header_dict("not a dict")


class TestGPSCoordinateConversion:
    """Test GPS coordinate conversion methods."""

    @pytest.mark.parametrize(
        "latitude,hemisphere,expected",
        [
            (34.5678, "N", 34.5678),
            (34.5678, "n", 34.5678),
            (34.5678, "S", -34.5678),
            (34.5678, "s", -34.5678),
            ("34.5678", "N", 34.5678),
            ("34.5678", "S", -34.5678),
        ],
    )
    def test_get_latitude(self, nims_header, latitude, hemisphere, expected):
        """Test latitude conversion with different hemispheres."""
        result = nims_header._get_latitude(latitude, hemisphere)
        assert result == expected

    @pytest.mark.parametrize(
        "longitude,hemisphere,expected",
        [
            (115.7350, "E", 115.7350),
            (115.7350, "e", 115.7350),
            (115.7350, "W", -115.7350),
            (115.7350, "w", -115.7350),
            ("115.7350", "E", 115.7350),
            ("115.7350", "W", -115.7350),
        ],
    )
    def test_get_longitude(self, nims_header, longitude, hemisphere, expected):
        """Test longitude conversion with different hemispheres."""
        result = nims_header._get_longitude(longitude, hemisphere)
        assert result == expected


class TestHeaderLineParsing:
    """Test parsing of different header line formats."""

    def test_header_parsing_colon_format(self, temp_nims_file):
        """Test parsing lines with colon format (KEY: value)."""
        header_content = (
            b"SITE NAME: Test Site\r"
            b"OPERATOR: Test Operator\r"
            b"COUNTRY: USA\r"
            b"\r"
            b"g12345"
        )

        # Write to temp file
        with open(temp_nims_file, "wb") as f:
            f.write(header_content)

        header = NIMSHeader()
        header.read_header(temp_nims_file)

        assert "site name" in header.header_dict
        assert header.header_dict["site name"] == "Test Site"
        assert "operator" in header.header_dict
        assert header.header_dict["operator"] == "Test Operator"
        assert "country" in header.header_dict
        assert header.header_dict["country"] == "USA"

    def test_header_parsing_arrow_format(self, temp_nims_file):
        """Test parsing lines with arrow format (value <-- KEY)."""
        header_content = (
            b"106  0 <-- N-S Ex WIRE LENGTH (m); HEADING (deg E mag N)\r"
            b"109  90 <-- E-W Ey WIRE LENGTH (m); HEADING (deg E mag N)\r"
            b"Cu        <-- GROUND ELECTRODE INFO\r"
            b"\r"
            b"g12345"
        )

        # Write to temp file
        with open(temp_nims_file, "wb") as f:
            f.write(header_content)

        header = NIMSHeader()
        header.read_header(temp_nims_file)

        assert "n-s ex wire length (m); heading (deg e mag n)" in header.header_dict
        assert (
            header.header_dict["n-s ex wire length (m); heading (deg e mag n)"]
            == "106  0"
        )
        assert "ground electrode info" in header.header_dict
        assert header.header_dict["ground electrode info"] == "Cu"

    def test_header_parsing_ignore_comments(self, temp_nims_file):
        """Test that comment lines starting with '>' are ignored."""
        header_content = (
            b">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\r"
            b">>>user field>>>>>>>>>>>>>>>>>>>>>>>>>>>>\r"
            b">>> This is a comment line <<<<<<<<<<<<<<<\r"
            b"SITE NAME: Test Site\r"
            b"> Another comment\r"
            b"OPERATOR: Test\r"
            b"\r"
            b"g12345"
        )

        # Write to temp file
        with open(temp_nims_file, "wb") as f:
            f.write(header_content)

        header = NIMSHeader()
        header.read_header(temp_nims_file)

        # Comments should not be in the header dict
        comment_found = any(
            key for key in header.header_dict.keys() if key.startswith(">")
        )
        assert not comment_found

        # Valid fields should be present
        assert "site name" in header.header_dict
        assert header.header_dict["site name"] == "Test Site"

    def test_header_parsing_comments_section(self, temp_nims_file):
        """Test handling of comments section detection."""
        header_content = (
            b"SITE NAME: Test Site\r"
            b"OPERATOR: Test\r"
            b"COMMENT: This is the start of comments\r"
            b"Additional comment line\r"
            b"More comments here\r"
            b"\r"
            b"g12345"
        )

        # Write to temp file
        with open(temp_nims_file, "wb") as f:
            f.write(header_content)

        header = NIMSHeader()
        header.read_header(temp_nims_file)

        assert "comment" in header.header_dict
        assert header.header_dict["comment"] == "This is the start of comments"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_header_file(self, temp_nims_file):
        """Test handling of empty or very small files."""
        # Write minimal content
        with open(temp_nims_file, "wb") as f:
            f.write(b"g")

        header = NIMSHeader()
        header.read_header(temp_nims_file)

        assert header.header_dict == {}
        assert header.data_start_seek >= 0

    def test_malformed_wire_data(self, nims_header):
        """Test handling of malformed wire length data."""
        # Test with insufficient data
        malformed_dict = {
            "n-s ex wire length (m); heading (deg e mag n)": "106",  # Missing azimuth
        }

        # This should handle the error gracefully or raise appropriate exception
        with pytest.raises((IndexError, ValueError)):
            nims_header.parse_header_dict(malformed_dict)

    def test_malformed_system_data(self, nims_header):
        """Test handling of malformed system box data."""
        malformed_dict = {
            "system box i.d.; mag head id (if different)": "1105-3",  # Missing mag ID
        }

        with pytest.raises(IndexError):
            nims_header.parse_header_dict(malformed_dict)

    def test_malformed_gps_data(self, nims_header):
        """Test handling of malformed GPS data."""
        malformed_dict = {
            "gps info": "26/09/19 18:29:29",  # Incomplete GPS data
        }

        with pytest.raises(IndexError):
            nims_header.parse_header_dict(malformed_dict)

    def test_invalid_coordinate_values(self, nims_header):
        """Test handling of invalid coordinate values."""
        # Test invalid latitude hemisphere - methods return None for unrecognized hemispheres
        result_lat = nims_header._get_latitude(34.5, "X")  # Invalid hemisphere
        assert result_lat is None  # Should return None if hemisphere not recognized

        # Test invalid longitude hemisphere
        result_lon = nims_header._get_longitude(115.7, "Y")  # Invalid hemisphere
        assert result_lon is None  # Should return None if hemisphere not recognized

    def test_very_long_header(self):
        """Test handling of headers longer than max_header_length."""
        header = NIMSHeader()

        # Create a header longer than _max_header_length (1000 bytes)
        long_content = b"SITE NAME: Test\r" + b"COMMENT: " + (b"x" * 2000) + b"\rg12345"

        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "test_long.bnn"

        try:
            with open(temp_file, "wb") as f:
                f.write(long_content)

            # Should still work, just truncate at max length
            header.read_header(temp_file)
            assert header.header_dict is not None

        finally:
            shutil.rmtree(temp_dir)


class TestIntegration:
    """Integration tests combining multiple functionality."""

    def test_full_workflow_realistic_header(self, nims_header):
        """Test complete workflow with realistic header data."""
        realistic_header = (
            b">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\r"
            b">>>user field>>>>>>>>>>>>>>>>>>>>>>>>>>>>\r"
            b"SITE NAME: Mountain Lake Survey\r"
            b"STATE/PROVINCE: Colorado\r"
            b"COUNTRY: USA\r"
            b'"MT001a"  <-- 2CHAR EXPERIMENT CODE + 3 CHAR SITE CODE + RUN LETTER\r'
            b"2210-1; 2310-1  <-- SYSTEM BOX I.D.; MAG HEAD ID (if different)\r"
            b"100  15 <-- N-S Ex WIRE LENGTH (m); HEADING (deg E mag N)\r"
            b"95  105 <-- E-W Ey WIRE LENGTH (m); HEADING (deg E mag N)\r"
            b"A1         <-- N ELECTRODE ID\r"
            b"A3         <-- E ELECTRODE ID\r"
            b"A2         <-- S ELECTRODE ID\r"
            b"A4         <-- W ELECTRODE ID\r"
            b"Cu/CuSO4        <-- GROUND ELECTRODE INFO\r"
            b"OPERATOR: Field Team Alpha\r"
            b"COMMENT: Clear weather conditions\r"
            b"Second comment line\r"
            b"Third comment line\r"
            b"\r"
            b"binary_data_starts_here"
        )

        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "realistic_test.bnn"

        try:
            with open(temp_file, "wb") as f:
                f.write(realistic_header)

            nims_header.read_header(temp_file)

            # Verify all major components parsed correctly (excluding GPS)
            assert nims_header.site_name == "Mountain Lake Survey"
            assert nims_header.state_province == "Colorado"
            assert nims_header.country == "USA"
            assert nims_header.run_id == "MT001a"
            assert nims_header.station == "MT001"
            assert nims_header.box_id == "2210-1"
            assert nims_header.mag_id == "2310-1"
            assert nims_header.ex_length == 100.0
            assert nims_header.ex_azimuth == 15.0
            assert nims_header.ey_length == 95.0
            assert nims_header.ey_azimuth == 105.0
            assert nims_header.operator == "Field Team Alpha"
            assert nims_header.ground_electrode_info == "Cu/CuSO4"

            # Verify file properties
            assert nims_header.fn == temp_file
            assert nims_header.file_size > 0
            assert nims_header.data_start_seek > 0

        finally:
            shutil.rmtree(temp_dir)


class TestPerformance:
    """Test performance-related aspects."""

    def test_multiple_header_parsing(self, sample_header_string):
        """Test parsing multiple headers efficiently."""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create multiple test files
            test_files = []
            for i in range(5):
                temp_file = Path(temp_dir) / f"test_{i}.bnn"
                with open(temp_file, "wb") as f:
                    f.write(sample_header_string)
                test_files.append(temp_file)

            # Parse all headers
            headers = []
            for test_file in test_files:
                header = NIMSHeader()
                header.read_header(test_file)
                headers.append(header)

            # Verify all parsed correctly
            assert len(headers) == 5
            for header in headers:
                assert header.site_name == "Budwieser Spring"
                assert header.run_id == "300b"
                assert header.station == "300"

        finally:
            shutil.rmtree(temp_dir)

    def test_large_file_handling(self):
        """Test handling of large files with normal headers."""
        header = NIMSHeader()

        # Create a file with normal header but large binary section
        normal_header = b"SITE NAME: Large File Test\r" b"OPERATOR: Test\r" b"\r"
        large_data = b"x" * 10000  # 10KB of data

        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "large_test.bnn"

        try:
            with open(temp_file, "wb") as f:
                f.write(normal_header)
                f.write(large_data)

            header.read_header(temp_file)

            # Should parse header correctly despite large file
            assert "site name" in header.header_dict
            assert header.header_dict["site name"] == "Large File Test"
            assert header.file_size > 10000

        finally:
            shutil.rmtree(temp_dir)


# =============================================================================
# Run Tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
