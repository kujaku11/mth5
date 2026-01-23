# -*- coding: utf-8 -*-
"""
Helper functions for HDF5

Created on Tue Jun  2 12:37:50 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT

"""
import gc
import inspect

# =============================================================================
# Imports
# =============================================================================
from collections.abc import Iterable
from typing import Any, Type

import h5py
import numpy as np
from loguru import logger
from mt_metadata.base import MetadataBase
from pydantic.fields import FieldInfo


# =============================================================================
# Acceptable compressions
# =============================================================================
COMPRESSION = ["lzf", "gzip", "szip", None]
COMPRESSION_LEVELS = {
    "lzf": [None],
    "gzip": range(10),
    "szip": ["ec-8", "ee-10", "nn-8", "nn-10"],
    None: [None],
}


def validate_compression(
    compression: str | None, level: int | str | None
) -> tuple[str | None, int | str | None]:
    """
    Validate that the input compression is supported.

    Parameters
    ----------
    compression : str or None
        Type of lossless compression. Options are 'lzf', 'gzip', 'szip', or None.
    level : int, str, or None
        Compression level if supported.
        - int for 'gzip' (0-9)
        - str for 'szip' ('ec-8', 'ee-10', 'nn-8', 'nn-10')
        - None for 'lzf' or None compression

    Returns
    -------
    compression : str or None
        Validated compression type
    level : int, str, or None
        Validated compression level

    Raises
    ------
    ValueError
        If compression or level are not supported
    TypeError
        If compression is not a string or None, or if compression level
        type is incorrect for the specified compression type

    """
    if compression is None:
        return None, None
    if not isinstance(compression, (str, type(None))):
        msg = f"compression type must be a string, not {type(compression)}"
        logger.error(msg)
        raise TypeError(msg)
    if not compression in COMPRESSION:
        msg = (
            f"Compression type {compression} not supported. "
            f"Supported options are {COMPRESSION}"
        )
        logger.error(msg)
        raise ValueError(msg)
    if compression == "lzf":
        level = COMPRESSION_LEVELS["lzf"][0]
    elif compression == " gzip":
        if not isinstance(level, (int)):
            msg = (
                f"Level type for gzip must be an int, not {type(level)}. "
                f"Options are {COMPRESSION_LEVELS['gzip']}"
            )
            logger.error(msg)
            raise TypeError(msg)
    elif compression == " szip":
        if not isinstance(level, (str)):
            msg = (
                f"Level type for szip must be an str, not {type(level)}. "
                f"Options are {COMPRESSION_LEVELS['szip']}"
            )
            logger.error(msg)
            raise TypeError(msg)
    if not level in COMPRESSION_LEVELS[compression]:
        msg = (
            f"compression level {level} not supported for {compression}. "
            f"Options are {COMPRESSION_LEVELS[compression]}"
        )

        logger.error(msg)
        raise ValueError(msg)
    return compression, level


def recursive_hdf5_tree(
    group: h5py.Group | h5py.File | h5py.Dataset, lines: list[str] | None = None
) -> str:
    """
    Recursively traverse an HDF5 group and return a string representation of its structure.

    Parameters
    ----------
    group : h5py.Group, h5py.File, or h5py.Dataset
        HDF5 object to traverse
    lines : list of str, optional
        List to accumulate the tree representation lines. If None, an empty list is used.

    Returns
    -------
    str
        String representation of the HDF5 tree structure

    Notes
    -----
    This function recursively traverses HDF5 groups and files, building a text
    representation of the structure including groups, datasets, and attributes.
    """
    if lines is None:
        lines = []
    if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
        for key, value in group.items():
            lines.append(f"-{key}: {value}")
            recursive_hdf5_tree(value, lines)
    elif isinstance(group, h5py._hl.dataset.Dataset):
        for key, value in group.attrs.items():
            lines.append(f"\t-{key}: {value}")
    return "\n".join(lines)


def close_open_files() -> None:
    """
    Close all open HDF5 files found in memory.

    This function searches through all objects in memory using garbage collection
    to find and close any open HDF5 files. This is useful for cleanup operations
    to ensure no files are left open.

    Notes
    -----
    This function iterates through all objects in memory and attempts to close
    any h5py.File objects that are found. If a file is already closed, it will
    log that information. Any exceptions during the process are caught and logged.
    """
    for obj in gc.get_objects():
        try:
            if isinstance(obj, h5py.File):
                msg = "Found HDF5 File object "
                logger.debug(msg)
                try:
                    msg = f"{obj.filename}, "
                    obj.flush()
                    obj.close()
                    msg += "Closed File"
                    logger.info(msg)
                except:
                    msg += f"{obj.filename} file already closed."
                    logger.info(msg)
        except:
            logger.debug(f"Object {type(obj)} does not have __class__")


def get_tree(parent: h5py.Group | h5py.File) -> str:
    """
    Recursively print the contents of an HDF5 group in a formatted tree structure.

    Parameters
    ----------
    parent : h5py.Group or h5py.File
        HDF5 (sub-)tree to print

    Returns
    -------
    str
        Formatted string representation of the HDF5 tree structure

    Raises
    ------
    TypeError
        If the provided object is not an h5py.File or h5py.Group object

    Notes
    -----
    This function creates a hierarchical text representation of an HDF5 file
    or group structure, showing groups and datasets with appropriate indentation
    and formatting.
    """
    lines = ["{0}:".format(parent.name), "=" * 20]
    if not isinstance(parent, (h5py.File, h5py.Group)):
        raise TypeError("Provided object is not a h5py.File or h5py.Group " "object")

    def fancy_print(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        # lines.append(name)
        spacing = " " * 4 * (name.count("/") + 1)
        group_name = name[name.rfind("/") + 1 :]

        if isinstance(obj, h5py.Group):
            lines.append(f"{spacing}|- Group: {group_name}")
            lines.append("{0}{1}".format(spacing, (len(group_name) + 10) * "-"))
        elif isinstance(obj, h5py.Dataset):
            lines.append(f"{spacing}--> Dataset: {group_name}")
            lines.append("{0}{1}".format(spacing, (len(group_name) + 15) * "."))

    # lines.append(parent.name)
    parent.visititems(fancy_print)
    return "\n".join(lines)


def to_numpy_type(value: Any) -> Any:
    """
    Convert a value to a numpy/HDF5 compatible type.

    This function handles the conversion of various Python data types to formats
    that are compatible with both NumPy and HDF5. For numbers and booleans, this
    is straightforward as they are automatically mapped to numpy types. For strings
    and complex data structures, special handling is required.

    Parameters
    ----------
    value : any
        The value to convert to a numpy/HDF5 compatible type

    Returns
    -------
    various
        The converted value in a numpy/HDF5 compatible format:
        - None becomes "none" string
        - Dictionaries and lists become JSON strings
        - Type objects become string representations
        - h5py References become strings
        - Object arrays become string representations
        - Iterables with strings become numpy byte arrays
        - Other iterables become numpy arrays
        - Basic types (str, int, float, bool, complex) are returned as-is

    Notes
    -----
    HDF5 should only deal with ASCII characters or Unicode. No binary data
    is allowed. This function ensures compatibility by converting complex
    Python objects to appropriate string or array representations.

    Lists and dictionaries are converted to JSON strings for storage in HDF5,
    which can be reconstructed using `from_numpy_type`.
    """

    if value is None:
        return "none"
    # For now turn references into a generic string
    if isinstance(value, h5py.h5r.Reference):
        value = str(value)

    # Handle type objects and classes that might come from pydantic serialization
    if isinstance(value, type):
        # Use a stable, fully-qualified type name rather than the raw repr
        type_str = f"{value.__module__}.{value.__qualname__}"
        logger.warning(
            f"Converting type object {value!r} to its fully qualified name "
            f"{type_str!r} for HDF5 metadata storage. "
            "This may indicate that a type object was passed where a value was expected."
        )
        return type_str

    # Handle dictionaries and lists by converting to JSON
    if isinstance(value, (dict, list)):
        try:
            import json

            return json.dumps(value)
        except (TypeError, ValueError):
            # If JSON serialization fails, convert to string
            return str(value)

    # Handle numpy arrays with object dtype
    if isinstance(value, np.ndarray) and value.dtype == np.dtype("O"):
        # Try to convert to string representation
        return str(value)

    if isinstance(
        value,
        (
            str,
            np.str_,
            int,
            float,
            bool,
            complex,
            np.int_,
            np.float64,
            np.bool_,
            np.complex128,
        ),
    ):
        return value
    if isinstance(value, Iterable):
        if np.any([type(x) in [str, bytes, np.str_] for x in value]):
            return np.array(value, dtype="S")
        else:
            try:
                converted_array = np.array(value)
                # Check if the resulting array has object dtype
                if converted_array.dtype == np.dtype("O"):
                    return str(value)
                return converted_array
            except (ValueError, TypeError):
                # If we can't convert to numpy array, convert to string representation
                return str(value)
    else:
        # For pydantic models and other complex objects, convert to string
        try:
            # First try to convert directly
            converted_array = np.array(value)
            # Check if the resulting array has object dtype
            if converted_array.dtype == np.dtype("O"):
                return str(value)
            return converted_array
        except (ValueError, TypeError):
            # If that fails, convert to string representation
            return str(value)


def validate_name(name: str) -> str:
    """
    Clean a name by replacing spaces and slashes with underscores.

    Parameters
    ----------
    name : str
        The name to validate and clean

    Returns
    -------
    str
        The cleaned name with spaces and slashes replaced by underscores

    Notes
    -----
    This function ensures that names are compatible with HDF5 naming conventions
    by removing problematic characters.
    """

    return name.replace(" ", "_").replace("/", "_")


def from_numpy_type(value: Any) -> Any:
    """
    Convert a value from numpy/HDF5 format back to standard Python types.

    This function handles the reverse conversion from numpy/HDF5 compatible types
    back to standard Python data types. It's the counterpart to `to_numpy_type`.

    Parameters
    ----------
    value : any
        The value to convert from numpy/HDF5 format

    Returns
    -------
    various
        The converted value in standard Python format:
        - "none" string becomes None
        - JSON strings become dictionaries or lists
        - h5py References become strings
        - Numpy types become standard Python types
        - Byte arrays become string lists
        - Other arrays become Python lists

    Raises
    ------
    TypeError
        If the value type is not understood or supported

    Notes
    -----
    This function reverses the conversions made by `to_numpy_type`, including:
    - Converting JSON strings back to dictionaries and lists
    - Converting "none" strings back to None
    - Converting numpy arrays back to Python lists
    - Handling deprecated numpy.bool types

    For numbers and booleans, they are automatically mapped from h5py to numpy types.
    For strings, especially lists of strings, special handling is required.
    HDF5 deals with ASCII characters or Unicode, no binary data is allowed.
    """

    if value is None:
        return "none"

    # Convert "none" string back to None when reading from HDF5
    if isinstance(value, str) and value.lower() == "none":
        return None

    # Handle JSON-like strings that represent dictionaries or lists from HDF5
    if isinstance(value, str):
        # Check if it looks like a JSON dictionary or list
        if (value.startswith("{") and value.endswith("}")) or (
            value.startswith("[") and value.endswith("]")
        ):
            try:
                import json

                parsed = json.loads(value)
                return parsed
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, just return the string
                pass

    # For now turn references into a generic string
    if isinstance(value, h5py.h5r.Reference):
        value = str(value)
    if isinstance(
        value,
        (
            str,
            np.str_,
            int,
            float,
            bool,
            complex,
            np.int32,
            np.float64,
            np.complex128,
            np.intp,
            np.bool_,  # Add support for numpy.bool_
        ),
    ):
        return value

    # Handle deprecated numpy.bool (numpy >=1.20 deprecates numpy.bool)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    # if isinstance(
    #     value,
    #     (
    #         np.int32,
    #     )
    # ):
    #     return np.int64(value)
    if isinstance(value, Iterable):
        if np.any([type(x) in [bytes, np.bytes_] for x in value]):
            return np.array(value, dtype="U").tolist()
        else:
            return np.array(value).tolist()
    else:
        raise TypeError("Type {0} not understood".format(type(value)))


def coerce_value_to_expected_type(
    self, key: str, value: Any, expected_type: Any
) -> Any:
    """
    Coerce a value to the expected type based on metadata field definitions.

    This method handles type conversions for older MTH5 files that may have
    stored metadata with less strict type enforcement. Uses the metadata's
    attribute_information method to get expected types.

    Parameters
    ----------
    key : str
        Metadata field name (may include dots for nested attributes).
    value : Any
        Value to coerce.
    expected_type : Any
        Expected value type (can be a type object or string representation).
    Returns
    -------
    Any
        Coerced value matching expected type, or original value if coercion fails.

    Examples
    --------
    >>> coerced = channel._coerce_value_to_expected_type('sample_rate', '256.0', float)
    >>> print(type(coerced), coerced)
    <class 'float'> 256.0

    >>> coerced = channel._coerce_value_to_expected_type('channel_number', 1.0, int)
    >>> print(type(coerced), coerced)
    <class 'int'> 1
    """
    # Return None values as-is
    if value is None:
        return value

    try:
        if expected_type is None:
            return value

        # Convert string representation to type if needed
        if isinstance(expected_type, str):
            try:
                expected_type = get_data_type(expected_type)
            except ValueError:
                # Can't convert, return original value
                return value

        # Already the correct type
        if isinstance(value, expected_type):
            return value

        # Handle common type coercions
        if expected_type == float:
            if isinstance(value, (int, str, np.integer, np.floating)):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    self.logger.debug(f"Could not coerce {key}={value} to float")
                    return value
            elif isinstance(value, list):
                if len(value) == 1:
                    try:
                        return float(value[0])
                    except (ValueError, TypeError):
                        self.logger.debug(f"Could not coerce {key}={value} to float")
                        return value

        elif expected_type == int:
            if isinstance(value, (float, str, np.integer, np.floating)):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    self.logger.debug(f"Could not coerce {key}={value} to int")
                    return value
            elif isinstance(value, list):
                if len(value) == 1:
                    try:
                        return int(value[0])
                    except (ValueError, TypeError):
                        self.logger.debug(f"Could not coerce {key}={value} to int")
                        return value

        elif expected_type == str:
            if isinstance(value, list):
                if len(value) == 1:
                    try:
                        return str(value[0])
                    except (ValueError, TypeError):
                        self.logger.debug(f"Could not coerce {key}={value} to str")
                        return value
            elif not isinstance(value, str):
                try:
                    return str(value)
                except (ValueError, TypeError):
                    self.logger.debug(f"Could not coerce {key}={value} to str")
                    return value

        elif expected_type == bool:
            if isinstance(value, (int, float, str, np.integer, np.floating)):
                try:
                    # Handle string representations
                    if isinstance(value, str):
                        return value.lower() in ("true", "1", "yes", "y")
                    # Handle numeric representations
                    return bool(value)
                except (ValueError, TypeError):
                    self.logger.debug(f"Could not coerce {key}={value} to bool")
                    return value
            elif isinstance(value, list):
                if len(value) == 1:
                    try:
                        val = value[0]
                        if isinstance(val, str):
                            return val.lower() in ("true", "1", "yes", "y")
                        return bool(val)
                    except (ValueError, TypeError):
                        self.logger.debug(f"Could not coerce {key}={value} to bool")
                        return value

        elif expected_type == list:
            if isinstance(value, str):
                # Handle string representations of lists
                try:
                    import json

                    return json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Try comma-separated values
                    if "," in value:
                        return [v.strip() for v in value.split(",")]
                    self.logger.debug(f"Could not coerce {key}={value} to list")
                    return value
            elif not isinstance(value, list):
                # Try to convert to list
                try:
                    return list(value)
                except (ValueError, TypeError):
                    self.logger.debug(f"Could not coerce {key}={value} to list")
                    return value

    except Exception as e:
        # If anything goes wrong, log and return original value
        self.logger.debug(f"Exception during type coercion for {key}: {e}")
        return value

    # Return original value if no coercion applied
    return value


def get_metadata_type_dict(metadata_class: MetadataBase) -> dict[str, Type[Any]]:
    """
    get dictionary of expected data types from the metadata object.

    Parameters
    ----------
    metadata_class : MetadataBase
        Metadata class to extract data types from

    Returns
    -------
    dict[str, Type[Any]]
        Dictionary mapping metadata field names to their expected data types.
    """
    type_dict = {}
    for key, field_info in metadata_class.get_all_fields().items():
        type_str = field_info.get("type")
        if isinstance(type_str, type):
            # Already a type object
            type_dict[key] = type_str
        elif isinstance(type_str, str):
            # Convert string representation to type
            try:
                type_dict[key] = get_data_type(type_str)
            except ValueError:
                # If conversion fails, store the string
                type_dict[key] = type_str
        else:
            type_dict[key] = type_str
    return type_dict


def get_data_type(string_representation: str) -> Type[Any]:
    """
    Get the Python data type from its string representation.

    Parameters
    ----------
    string_representation : str
        String representation of the data type (e.g., 'int', 'float', 'str').

    Returns
    -------
    type
        Corresponding Python data type.

    Raises
    ------
    ValueError
        If the string representation does not correspond to a known data type.

    Notes
    -----
    This function maps common string representations of data types to their
    corresponding Python types. It supports basic types like int, float, str,
    bool, list, and dict.
    """
    type_mapping = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "complex": complex,
    }

    dtype = (
        string_representation.replace("'<class", "")
        .replace("'>", "")
        .replace("<class '", "")
        .replace("'>", "")
        .replace("<class", "")
        .replace("'", "")
        .replace(">", "")
        .split("|")[0]
        .strip()
    )
    if "[" in dtype and "]" in dtype:
        dtype = dtype[: dtype.find("[")].strip()
    try:
        return type_mapping[dtype.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown data type string representation: {string_representation}"
        )


def read_attrs_to_dict(
    attrs_dict: dict[str, Any], metadata_object: MetadataBase
) -> dict[str, Any]:
    """
    Read HDF5 attributes from a group or dataset into a dictionary.

    Parameters
    ----------
    attrs_dict : dict[str, Any]
        Dictionary of attributes to read and convert.
    metadata_object : MetadataBase
        Metadata object to use for type information.

    Returns
    -------
    dict[str, Any]
        Dictionary containing attribute names and their corresponding values.
    """
    data_types = get_metadata_type_dict(metadata_object)

    for key, value in attrs_dict.items():
        # First convert from numpy types
        value = from_numpy_type(value)
        # Then coerce to expected type based on metadata schema
        attrs_dict[key] = coerce_value_to_expected_type(
            key, value, get_data_type(data_types[key])
        )
    return attrs_dict


# =============================================================================
#
# =============================================================================
def inherit_doc_string(cls: Type[Any]) -> Type[Any]:
    """
    Class decorator to inherit docstring from parent classes.

    This decorator searches through the method resolution order (MRO) of a class
    to find the first parent class with a docstring and applies it to the current class.

    Parameters
    ----------
    cls : type
        The class to apply docstring inheritance to

    Returns
    -------
    type
        The same class with inherited docstring if found

    Notes
    -----
    This is useful for subclasses that should inherit documentation from their
    parent classes when they don't have their own docstring defined.
    """
    for base in inspect.getmro(cls):
        if base.__doc__ is not None:
            cls.__doc__ = base.__doc__
            break
    return cls


def validate_name(name: str | None, pattern: str | None = None) -> str:
    """
    Validate and clean a name for HDF5 compatibility.

    Parameters
    ----------
    name : str or None
        The name to validate and clean
    pattern : str, optional
        Pattern for validation (currently not used but reserved for future use)

    Returns
    -------
    str
        The cleaned name with spaces replaced by underscores and commas removed.
        Returns "unknown" if input name is None.

    Notes
    -----
    This function ensures that names are compatible with HDF5 naming conventions
    by removing problematic characters. If the input name is None, it returns
    "unknown" as a default value.
    """
    if name is None:
        return "unknown"
    return name.replace(" ", "_").replace(",", "")


def add_attributes_to_metadata_class_pydantic(obj: Type[Any]) -> Type[Any]:
    """
    Add MTH5-specific attributes to a pydantic metadata class.

    This function enhances a pydantic class by adding two important fields:
    - mth5_type: derived from the class name, indicates the type of MTH5 group
    - hdf5_reference: stores the HDF5 internal reference

    Parameters
    ----------
    obj : type
        A pydantic class to enhance with MTH5 attributes

    Returns
    -------
    object
        An instance of the enhanced class with added MTH5-specific fields

    Raises
    ------
    TypeError
        If the input is not a class

    Notes
    -----
    This function is used to dynamically add metadata fields that are required
    for MTH5 group management. The mth5_type field is derived from the class
    name by removing "Group" suffix, and the hdf5_reference field is initialized
    to None but will be set when the object is associated with an HDF5 group.
    """
    if not inspect.isclass(obj):
        raise TypeError("Input must be a class")

    # Create an instance of the class
    obj = obj()
    # Create FieldInfo for mth5_type
    mth5_type_field = FieldInfo(
        annotation=str,
        default=obj._class_name.split("Group")[0],
        description="type of group",
        json_schema_extra={
            "required": True,
            "units": None,
            "examples": ["group_name"],
        },
    )

    # Use add_new_field to add mth5_type - this returns a class, not an instance
    enhanced_class = obj.add_new_field("mth5_type", mth5_type_field)()

    # Create FieldInfo for hdf5_reference
    # Use a plain type for annotation (object) because FieldInfo.annotation expects a concrete type,
    # not a typing.Union; the default None and json_schema_extra still indicate optionality.
    hdf5_ref_field = FieldInfo(
        annotation=object,
        default=None,  # Will be set later
        description="hdf5 internal reference",
        json_schema_extra={
            "required": True,
            "units": None,
            "examples": ["<HDF5 Group Reference>"],
        },
    )

    # Create an instance of the enhanced class to add the second field
    return enhanced_class.add_new_field("hdf5_reference", hdf5_ref_field)()
