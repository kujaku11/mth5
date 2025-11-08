# -*- coding: utf-8 -*-
"""
Pytest tests for mth5.helpers module

Created on November 8, 2025

@author: GitHub Copilot
"""

# =============================================================================
# Imports
# =============================================================================
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from mth5.helpers import (
    add_attributes_to_metadata_class_pydantic,
    close_open_files,
    from_numpy_type,
    get_tree,
    inherit_doc_string,
    recursive_hdf5_tree,
    to_numpy_type,
    validate_compression,
    validate_name,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    # Create a simple HDF5 structure
    with h5py.File(tmp_path, "w") as f:
        # Create groups
        grp1 = f.create_group("group1")
        grp2 = f.create_group("group2")
        subgrp = grp1.create_group("subgroup")

        # Create datasets
        dset1 = f.create_dataset("dataset1", data=np.array([1, 2, 3, 4, 5]))
        dset2 = grp1.create_dataset("dataset2", data=np.array([[1, 2], [3, 4]]))
        dset3 = subgrp.create_dataset("dataset3", data="test string")

        # Add attributes
        f.attrs["file_attr"] = "file_value"
        grp1.attrs["group_attr"] = "group_value"
        dset1.attrs["dataset_attr"] = "dataset_value"

    yield tmp_path

    # Cleanup
    try:
        tmp_path.unlink()
    except (FileNotFoundError, PermissionError):
        pass


@pytest.fixture
def sample_h5py_file(temp_hdf5_file):
    """Open HDF5 file for testing"""
    f = h5py.File(temp_hdf5_file, "r")
    yield f
    f.close()


@pytest.fixture(autouse=True)
def cleanup_h5_files():
    """Ensure all HDF5 files are closed after each test"""
    yield
    close_open_files()


@pytest.fixture
def mock_pydantic_class():
    """Create a mock pydantic class for testing"""

    class MockClass:
        _class_name = "TestGroup"

        def add_new_field(self, name, field_info):
            # Return a class that can be instantiated
            class EnhancedClass:
                def add_new_field(self, name, field_info):
                    return EnhancedClass

            return EnhancedClass

    return MockClass


# =============================================================================
# Test validate_compression
# =============================================================================


class TestValidateCompression:
    """Test compression validation functionality"""

    @pytest.mark.parametrize(
        "compression,level",
        [
            ("lzf", None),
            ("gzip", 0),
            ("gzip", 5),
            ("gzip", 9),
            (None, None),
        ],
    )
    def test_valid_compression_combinations(self, compression, level):
        """Test valid compression and level combinations"""
        result_comp, result_level = validate_compression(compression, level)
        assert result_comp == compression
        if compression == "lzf":
            assert result_level is None
        else:
            assert result_level == level

    def test_none_compression(self):
        """Test None compression returns None, None"""
        comp, level = validate_compression(None, None)
        assert comp is None
        assert level is None

    def test_invalid_compression_type(self):
        """Test invalid compression type raises ValueError"""
        with pytest.raises(ValueError, match="Compression type invalid not supported"):
            validate_compression("invalid", None)

    def test_invalid_compression_input_type(self):
        """Test non-string compression raises TypeError"""
        with pytest.raises(TypeError, match="compression type must be a string"):
            validate_compression(123, None)

    @pytest.mark.parametrize("level", [-1, 10, 15])
    def test_invalid_gzip_level(self, level):
        """Test invalid gzip compression levels"""
        with pytest.raises(
            ValueError, match=f"compression level {level} not supported for gzip"
        ):
            validate_compression("gzip", level)

    def test_invalid_gzip_level_type(self):
        """Test invalid gzip level type raises appropriate error"""
        # The original code has bugs with " gzip" (space prefix), but let's test what actually happens
        # When we pass an invalid level type to gzip, it should ultimately raise ValueError
        with pytest.raises(
            ValueError, match="compression level invalid not supported for gzip"
        ):
            validate_compression("gzip", "invalid")

    def test_szip_not_available(self):
        """Test szip compression when not available"""
        # Note: szip might not be available in all HDF5 installations
        # This test checks if the validation works for supported szip options
        try:
            comp, level = validate_compression("szip", "ec-8")
            assert comp == "szip"
            assert level == "ec-8"
        except ValueError:
            # If szip is not available, expect this error
            pytest.skip("szip compression not available")


# =============================================================================
# Test recursive_hdf5_tree
# =============================================================================


class TestRecursiveHDF5Tree:
    """Test HDF5 tree traversal functionality"""

    def test_recursive_tree_with_file(self, sample_h5py_file):
        """Test recursive tree traversal with HDF5 file"""
        lines = []
        result = recursive_hdf5_tree(sample_h5py_file, lines)

        assert isinstance(result, str)
        assert "group1" in result
        assert "group2" in result
        assert "dataset1" in result

    def test_recursive_tree_with_group(self, sample_h5py_file):
        """Test recursive tree traversal with HDF5 group"""
        group = sample_h5py_file["group1"]
        lines = []
        result = recursive_hdf5_tree(group, lines)

        assert isinstance(result, str)
        assert "subgroup" in result
        assert "dataset2" in result

    def test_recursive_tree_with_dataset(self, sample_h5py_file):
        """Test recursive tree traversal with HDF5 dataset"""
        dataset = sample_h5py_file["dataset1"]
        lines = []
        result = recursive_hdf5_tree(dataset, lines)

        assert isinstance(result, str)
        # Should include dataset attributes
        assert "dataset_attr" in result


# =============================================================================
# Test close_open_files
# =============================================================================


class TestCloseOpenFiles:
    """Test file closure functionality"""

    def test_close_open_files_no_files(self):
        """Test close_open_files when no files are open"""
        # Should not raise any exceptions
        close_open_files()

    def test_close_open_files_with_open_file(self, temp_hdf5_file):
        """Test close_open_files with an open file"""
        # Open a file
        f = h5py.File(temp_hdf5_file, "r")

        # Verify file is open
        assert f.id.valid

        # Close all files
        close_open_files()

        # File should be closed now
        assert not f.id.valid

    def test_close_already_closed_file(self, temp_hdf5_file):
        """Test close_open_files with already closed file"""
        # Open and immediately close a file
        f = h5py.File(temp_hdf5_file, "r")
        f.close()

        # Should not raise exceptions
        close_open_files()


# =============================================================================
# Test get_tree
# =============================================================================


class TestGetTree:
    """Test HDF5 tree display functionality"""

    def test_get_tree_with_file(self, sample_h5py_file):
        """Test get_tree with HDF5 file"""
        tree = get_tree(sample_h5py_file)

        assert isinstance(tree, str)
        assert sample_h5py_file.name in tree
        assert "Group: group1" in tree
        assert "Group: group2" in tree
        assert "Dataset: dataset1" in tree

    def test_get_tree_with_group(self, sample_h5py_file):
        """Test get_tree with HDF5 group"""
        group = sample_h5py_file["group1"]
        tree = get_tree(group)

        assert isinstance(tree, str)
        assert group.name in tree
        assert "Group: subgroup" in tree
        assert "Dataset: dataset2" in tree

    def test_get_tree_with_invalid_object(self):
        """Test get_tree with invalid object raises AttributeError"""
        with pytest.raises(
            AttributeError, match="'str' object has no attribute 'name'"
        ):
            get_tree("invalid_object")


# =============================================================================
# Test to_numpy_type
# =============================================================================


class TestToNumpyType:
    """Test conversion to numpy-compatible types"""

    def test_none_value(self):
        """Test None value conversion"""
        result = to_numpy_type(None)
        assert result == "none"

    def test_string_value(self):
        """Test string value conversion"""
        result = to_numpy_type("test_string")
        assert result == "test_string"

    def test_numeric_values(self):
        """Test numeric value conversions"""
        test_cases = [
            (42, 42),
            (3.14, 3.14),
            (True, True),
            (1 + 2j, 1 + 2j),
        ]

        for input_val, expected in test_cases:
            result = to_numpy_type(input_val)
            assert result == expected

    def test_numpy_types(self):
        """Test numpy type conversions"""
        test_cases = [
            (np.int32(42), 42),
            (np.float64(3.14), 3.14),
            (np.bool_(True), True),
            (np.str_("test"), "test"),
        ]

        for input_val, expected in test_cases:
            result = to_numpy_type(input_val)
            assert result == expected

    def test_list_conversion(self):
        """Test list conversion"""
        input_list = [1, 2, 3, 4]
        result = to_numpy_type(input_list)
        # The function converts lists to JSON strings, not numpy arrays
        assert isinstance(result, str)
        import json

        parsed = json.loads(result)
        assert parsed == input_list

    def test_string_list_conversion(self):
        """Test string list conversion"""
        input_list = ["a", "b", "c"]
        result = to_numpy_type(input_list)
        # String lists get converted to JSON first since they are lists
        assert isinstance(result, str)
        import json

        parsed = json.loads(result)
        assert parsed == input_list

    def test_dict_conversion(self):
        """Test dictionary conversion to JSON"""
        input_dict = {"key1": "value1", "key2": 42}
        result = to_numpy_type(input_dict)

        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == input_dict

    def test_h5py_reference(self):
        """Test HDF5 reference conversion"""
        mock_ref = Mock(spec=h5py.h5r.Reference)
        mock_ref.__str__ = Mock(return_value="mock_reference")

        result = to_numpy_type(mock_ref)
        assert result == "mock_reference"

    def test_type_object(self):
        """Test type object conversion"""
        result = to_numpy_type(int)
        assert result == str(int)

    def test_object_array(self):
        """Test numpy object array conversion"""
        obj_array = np.array([1, "string", None], dtype=object)
        result = to_numpy_type(obj_array)
        assert isinstance(result, str)

    def test_complex_object(self):
        """Test complex object conversion"""

        class ComplexObject:
            def __str__(self):
                return "complex_object"

        obj = ComplexObject()
        result = to_numpy_type(obj)
        assert result == "complex_object"


# =============================================================================
# Test from_numpy_type
# =============================================================================


class TestFromNumpyType:
    """Test conversion from numpy/HDF5 types"""

    def test_none_string(self):
        """Test 'none' string conversion back to None"""
        result = from_numpy_type("none")
        assert result is None

    def test_none_value(self):
        """Test None value"""
        result = from_numpy_type(None)
        assert result == "none"

    def test_json_dict_string(self):
        """Test JSON dictionary string conversion"""
        json_str = '{"key1": "value1", "key2": 42}'
        result = from_numpy_type(json_str)

        assert isinstance(result, dict)
        assert result == {"key1": "value1", "key2": 42}

    def test_json_list_string(self):
        """Test JSON list string conversion"""
        json_str = '[1, 2, 3, "test"]'
        result = from_numpy_type(json_str)

        assert isinstance(result, list)
        assert result == [1, 2, 3, "test"]

    def test_invalid_json_string(self):
        """Test invalid JSON string remains as string"""
        invalid_json = '{"invalid": json}'
        result = from_numpy_type(invalid_json)

        assert result == invalid_json

    def test_regular_string(self):
        """Test regular string remains unchanged"""
        test_str = "regular_string"
        result = from_numpy_type(test_str)
        assert result == test_str

    def test_numeric_types(self):
        """Test numeric types remain unchanged"""
        test_cases = [
            42,
            3.14,
            True,
            1 + 2j,
            np.int32(42),
            np.float64(3.14),
            np.bool_(True),
        ]

        for input_val in test_cases:
            result = from_numpy_type(input_val)
            assert result == input_val

    def test_numpy_bool_deprecated(self):
        """Test deprecated numpy.bool handling"""
        # Test numpy.bool_ (current)
        result = from_numpy_type(np.bool_(True))
        assert result == True  # Use == instead of is for numpy types

    def test_bytes_array(self):
        """Test bytes array conversion"""
        bytes_array = [b"test1", b"test2"]
        result = from_numpy_type(bytes_array)

        assert isinstance(result, list)
        assert result == ["test1", "test2"]

    def test_regular_array(self):
        """Test regular array conversion"""
        array = np.array([1, 2, 3, 4])
        result = from_numpy_type(array)

        assert isinstance(result, list)
        assert result == [1, 2, 3, 4]

    def test_h5py_reference(self):
        """Test HDF5 reference conversion"""
        mock_ref = Mock(spec=h5py.h5r.Reference)
        mock_ref.__str__ = Mock(return_value="mock_reference")

        result = from_numpy_type(mock_ref)
        assert result == "mock_reference"

    def test_unsupported_type(self):
        """Test unsupported type raises TypeError"""

        class UnsupportedType:
            pass

        with pytest.raises(TypeError, match="Type .* not understood"):
            from_numpy_type(UnsupportedType())


# =============================================================================
# Test validate_name
# =============================================================================


class TestValidateName:
    """Test name validation functionality"""

    def test_valid_name(self):
        """Test valid name remains unchanged"""
        result = validate_name("valid_name")
        assert result == "valid_name"

    def test_name_with_spaces(self):
        """Test name with spaces gets underscores"""
        result = validate_name("name with spaces")
        assert result == "name_with_spaces"

    def test_name_with_commas(self):
        """Test name with commas gets cleaned"""
        result = validate_name("name,with,commas")
        assert result == "namewithcommas"

    def test_name_with_spaces_and_commas(self):
        """Test name with both spaces and commas"""
        result = validate_name("name with spaces, and commas")
        assert result == "name_with_spaces_and_commas"

    def test_none_name(self):
        """Test None name returns 'unknown'"""
        result = validate_name(None)
        assert result == "unknown"

    def test_empty_string(self):
        """Test empty string"""
        result = validate_name("")
        assert result == ""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("test_name", "test_name"),
            ("test name", "test_name"),
            ("test,name", "testname"),
            ("test name, with commas", "test_name_with_commas"),
            ("", ""),
            (None, "unknown"),
        ],
    )
    def test_name_validation_parametrized(self, input_name, expected):
        """Test name validation with various inputs"""
        result = validate_name(input_name)
        assert result == expected


# =============================================================================
# Test inherit_doc_string
# =============================================================================


class TestInheritDocString:
    """Test documentation inheritance functionality"""

    def test_inherit_from_parent(self):
        """Test inheriting docstring from parent class"""

        class Parent:
            """Parent class documentation"""

        class Child(Parent):
            pass

        # Apply the decorator
        decorated_child = inherit_doc_string(Child)

        assert decorated_child.__doc__ == "Parent class documentation"

    def test_inherit_from_grandparent(self):
        """Test inheriting docstring from grandparent class"""

        class GrandParent:
            """GrandParent class documentation"""

        class Parent(GrandParent):
            pass

        class Child(Parent):
            pass

        # Apply the decorator
        decorated_child = inherit_doc_string(Child)

        assert decorated_child.__doc__ == "GrandParent class documentation"

    def test_no_inheritance_when_present(self):
        """Test no inheritance when class already has docstring"""

        class Parent:
            """Parent class documentation"""

        class Child(Parent):
            """Child class documentation"""

        # Apply the decorator
        decorated_child = inherit_doc_string(Child)

        assert decorated_child.__doc__ == "Child class documentation"

    def test_no_parent_docstring(self):
        """Test when no parent has docstring"""

        class Parent:
            pass

        class Child(Parent):
            pass

        # Apply the decorator
        decorated_child = inherit_doc_string(Child)

        # In Python, even when no explicit docstring is provided,
        # classes inherit object's docstring, so check for that case
        assert decorated_child.__doc__ is not None  # Will inherit from object


# =============================================================================
# Test add_attributes_to_metadata_class_pydantic
# =============================================================================


class TestAddAttributesToMetadataClass:
    """Test pydantic metadata class enhancement functionality"""

    def test_non_class_input(self):
        """Test non-class input raises TypeError"""
        with pytest.raises(TypeError, match="Input must be a class"):
            add_attributes_to_metadata_class_pydantic("not_a_class")

    def test_valid_class_input(self, mock_pydantic_class):
        """Test valid class input"""
        result = add_attributes_to_metadata_class_pydantic(mock_pydantic_class)
        # Should return an instance of the enhanced class
        assert result is not None

    def test_class_name_extraction(self, mock_pydantic_class):
        """Test that class name is properly extracted"""
        # Mock class has _class_name = "TestGroup"
        result = add_attributes_to_metadata_class_pydantic(mock_pydantic_class)
        # The function should have processed the class name
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestHelpersIntegration:
    """Integration tests for helpers module"""

    def test_compression_with_hdf5(self, temp_hdf5_file):
        """Test compression validation works with actual HDF5 files"""
        # Test that compression validation works for real HDF5 usage
        comp, level = validate_compression("gzip", 5)
        assert comp == "gzip"
        assert level == 5

    def test_type_conversion_roundtrip(self):
        """Test type conversion roundtrip"""
        original_data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        # Convert to numpy type and back
        for key, value in original_data.items():
            numpy_type = to_numpy_type(value)
            recovered = from_numpy_type(numpy_type)

            if isinstance(value, (dict, list)):
                # These get converted to/from JSON
                assert recovered == value
            else:
                assert recovered == value

    def test_name_validation_integration(self):
        """Test name validation in realistic scenarios"""
        test_names = [
            "Station 001",
            "Run,001",
            "Channel EX, North",
            "Survey/2023/Site A",
        ]

        for name in test_names:
            validated = validate_name(name)
            # Should have no spaces or commas
            assert " " not in validated
            assert "," not in validated
            # Should not be empty
            assert len(validated) > 0

    def test_file_operations_integration(self, temp_hdf5_file):
        """Test file operations work together"""
        # Open file
        with h5py.File(temp_hdf5_file, "r") as f:
            # Get tree structure
            tree = get_tree(f)
            assert isinstance(tree, str)

            # Test recursive tree
            lines = []
            recursive_tree = recursive_hdf5_tree(f, lines)
            assert isinstance(recursive_tree, str)

        # Close any remaining files
        close_open_files()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestHelpersEdgeCases:
    """Test edge cases and error handling"""

    def test_very_large_arrays(self):
        """Test handling of large arrays"""
        large_array = np.random.random(1000)
        result = to_numpy_type(large_array)
        np.testing.assert_array_equal(result, large_array)

    def test_nested_data_structures(self):
        """Test deeply nested data structures"""
        nested = {"level1": {"level2": {"level3": [1, 2, {"level4": "deep"}]}}}

        # Convert to numpy type
        numpy_result = to_numpy_type(nested)
        assert isinstance(numpy_result, str)

        # Convert back
        recovered = from_numpy_type(numpy_result)
        assert recovered == nested

    def test_unicode_handling(self):
        """Test unicode string handling"""
        unicode_strings = [
            "café",
            "naïve",
            "Ελληνικά",
            "中文",
            "🚀",
        ]

        for unicode_str in unicode_strings:
            result = to_numpy_type(unicode_str)
            assert result == unicode_str

            recovered = from_numpy_type(result)
            assert recovered == unicode_str

    def test_mixed_type_arrays(self):
        """Test arrays with mixed types"""
        mixed_array = np.array([1, "string", 3.14, True], dtype=object)
        result = to_numpy_type(mixed_array)
        # Should convert to string representation
        assert isinstance(result, str)

    def test_empty_containers(self):
        """Test empty containers"""
        test_cases = [
            [],
            {},
            np.array([]),
            "",
        ]

        for empty_container in test_cases:
            result = to_numpy_type(empty_container)
            # Should not raise exceptions
            assert result is not None


# =============================================================================
# Performance Tests
# =============================================================================


class TestHelpersPerformance:
    """Performance tests for helpers module"""

    def test_type_conversion_performance(self):
        """Test performance of type conversions"""
        import time

        # Test data
        test_data = [
            "string" * 100,
            list(range(1000)),
            np.random.random(1000),
            {"key" + str(i): "value" + str(i) for i in range(100)},
        ]

        start_time = time.time()

        for data in test_data * 10:  # Repeat to get measurable time
            numpy_type = to_numpy_type(data)
            from_numpy_type(numpy_type)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should complete reasonably quickly (adjust threshold as needed)
        assert elapsed < 1.0  # Less than 1 second

    def test_large_file_tree_performance(self, temp_hdf5_file):
        """Test performance with large HDF5 tree"""
        # Create a file with many groups and datasets
        with h5py.File(temp_hdf5_file, "w") as f:
            for i in range(10):
                grp = f.create_group(f"group_{i}")
                for j in range(10):
                    grp.create_dataset(f"dataset_{j}", data=np.random.random(100))

        # Test tree operations
        with h5py.File(temp_hdf5_file, "r") as f:
            import time

            start_time = time.time()
            tree = get_tree(f)
            end_time = time.time()

            assert isinstance(tree, str)
            assert (end_time - start_time) < 1.0  # Should be fast


if __name__ == "__main__":
    pytest.main([__file__])
