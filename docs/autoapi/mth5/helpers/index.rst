mth5.helpers
============

.. py:module:: mth5.helpers

.. autoapi-nested-parse::

   Helper functions for HDF5

   Created on Tue Jun  2 12:37:50 2020

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license:
       MIT



Attributes
----------

.. autoapisummary::

   mth5.helpers.COMPRESSION
   mth5.helpers.COMPRESSION_LEVELS


Functions
---------

.. autoapisummary::

   mth5.helpers.validate_compression
   mth5.helpers.recursive_hdf5_tree
   mth5.helpers.close_open_files
   mth5.helpers.get_tree
   mth5.helpers.to_numpy_type
   mth5.helpers.validate_name
   mth5.helpers.from_numpy_type
   mth5.helpers.coerce_value_to_expected_type
   mth5.helpers.get_metadata_type_dict
   mth5.helpers.get_data_type
   mth5.helpers.read_attrs_to_dict
   mth5.helpers.inherit_doc_string
   mth5.helpers.validate_name
   mth5.helpers.add_attributes_to_metadata_class_pydantic


Module Contents
---------------

.. py:data:: COMPRESSION
   :value: ['lzf', 'gzip', 'szip', None]


.. py:data:: COMPRESSION_LEVELS

.. py:function:: validate_compression(compression: str | None, level: int | str | None) -> tuple[str | None, int | str | None]

   Validate that the input compression is supported.

   :param compression: Type of lossless compression. Options are 'lzf', 'gzip', 'szip', or None.
   :type compression: str or None
   :param level: Compression level if supported.
                 - int for 'gzip' (0-9)
                 - str for 'szip' ('ec-8', 'ee-10', 'nn-8', 'nn-10')
                 - None for 'lzf' or None compression
   :type level: int, str, or None

   :returns: * **compression** (*str or None*) -- Validated compression type
             * **level** (*int, str, or None*) -- Validated compression level

   :raises ValueError: If compression or level are not supported
   :raises TypeError: If compression is not a string or None, or if compression level
       type is incorrect for the specified compression type


.. py:function:: recursive_hdf5_tree(group: h5py.Group | h5py.File | h5py.Dataset, lines: list[str] | None = None) -> str

   Recursively traverse an HDF5 group and return a string representation of its structure.

   :param group: HDF5 object to traverse
   :type group: h5py.Group, h5py.File, or h5py.Dataset
   :param lines: List to accumulate the tree representation lines. If None, an empty list is used.
   :type lines: list of str, optional

   :returns: String representation of the HDF5 tree structure
   :rtype: str

   .. rubric:: Notes

   This function recursively traverses HDF5 groups and files, building a text
   representation of the structure including groups, datasets, and attributes.


.. py:function:: close_open_files() -> None

   Close all open HDF5 files found in memory.

   This function searches through all objects in memory using garbage collection
   to find and close any open HDF5 files. This is useful for cleanup operations
   to ensure no files are left open.

   .. rubric:: Notes

   This function iterates through all objects in memory and attempts to close
   any h5py.File objects that are found. If a file is already closed, it will
   log that information. Any exceptions during the process are caught and logged.


.. py:function:: get_tree(parent: h5py.Group | h5py.File) -> str

   Recursively print the contents of an HDF5 group in a formatted tree structure.

   :param parent: HDF5 (sub-)tree to print
   :type parent: h5py.Group or h5py.File

   :returns: Formatted string representation of the HDF5 tree structure
   :rtype: str

   :raises TypeError: If the provided object is not an h5py.File or h5py.Group object

   .. rubric:: Notes

   This function creates a hierarchical text representation of an HDF5 file
   or group structure, showing groups and datasets with appropriate indentation
   and formatting.


.. py:function:: to_numpy_type(value: Any) -> Any

   Convert a value to a numpy/HDF5 compatible type.

   This function handles the conversion of various Python data types to formats
   that are compatible with both NumPy and HDF5. For numbers and booleans, this
   is straightforward as they are automatically mapped to numpy types. For strings
   and complex data structures, special handling is required.

   :param value: The value to convert to a numpy/HDF5 compatible type
   :type value: any

   :returns: The converted value in a numpy/HDF5 compatible format:
             - None becomes "none" string
             - Dictionaries and lists become JSON strings
             - Type objects become string representations
             - h5py References become strings
             - Object arrays become string representations
             - Iterables with strings become numpy byte arrays
             - Other iterables become numpy arrays
             - Basic types (str, int, float, bool, complex) are returned as-is
   :rtype: various

   .. rubric:: Notes

   HDF5 should only deal with ASCII characters or Unicode. No binary data
   is allowed. This function ensures compatibility by converting complex
   Python objects to appropriate string or array representations.

   Lists and dictionaries are converted to JSON strings for storage in HDF5,
   which can be reconstructed using `from_numpy_type`.


.. py:function:: validate_name(name: str) -> str

   Clean a name by replacing spaces and slashes with underscores.

   :param name: The name to validate and clean
   :type name: str

   :returns: The cleaned name with spaces and slashes replaced by underscores
   :rtype: str

   .. rubric:: Notes

   This function ensures that names are compatible with HDF5 naming conventions
   by removing problematic characters.


.. py:function:: from_numpy_type(value: Any) -> Any

   Convert a value from numpy/HDF5 format back to standard Python types.

   This function handles the reverse conversion from numpy/HDF5 compatible types
   back to standard Python data types. It's the counterpart to `to_numpy_type`.

   :param value: The value to convert from numpy/HDF5 format
   :type value: any

   :returns: The converted value in standard Python format:
             - "none" string becomes None
             - JSON strings become dictionaries or lists
             - h5py References become strings
             - Numpy types become standard Python types
             - Byte arrays become string lists
             - Other arrays become Python lists
   :rtype: various

   :raises TypeError: If the value type is not understood or supported

   .. rubric:: Notes

   This function reverses the conversions made by `to_numpy_type`, including:
   - Converting JSON strings back to dictionaries and lists
   - Converting "none" strings back to None
   - Converting numpy arrays back to Python lists
   - Handling deprecated numpy.bool types

   For numbers and booleans, they are automatically mapped from h5py to numpy types.
   For strings, especially lists of strings, special handling is required.
   HDF5 deals with ASCII characters or Unicode, no binary data is allowed.


.. py:function:: coerce_value_to_expected_type(key: str, value: Any, expected_type: Any) -> Any

   Coerce a value to the expected type based on metadata field definitions.

   This method handles type conversions for older MTH5 files that may have
   stored metadata with less strict type enforcement. Uses the metadata's
   attribute_information method to get expected types.

   :param key: Metadata field name (may include dots for nested attributes).
   :type key: str
   :param value: Value to coerce.
   :type value: Any
   :param expected_type: Expected value type (can be a type object or string representation).
   :type expected_type: Any

   :returns: Coerced value matching expected type, or original value if coercion fails.
   :rtype: Any

   .. rubric:: Examples

   >>> coerced = channel._coerce_value_to_expected_type('sample_rate', '256.0', float)
   >>> print(type(coerced), coerced)
   <class 'float'> 256.0

   >>> coerced = channel._coerce_value_to_expected_type('channel_number', 1.0, int)
   >>> print(type(coerced), coerced)
   <class 'int'> 1


.. py:function:: get_metadata_type_dict(metadata_class: mt_metadata.base.MetadataBase) -> dict[str, Type[Any]]

   get dictionary of expected data types from the metadata object.

   :param metadata_class: Metadata class to extract data types from
   :type metadata_class: MetadataBase

   :returns: Dictionary mapping metadata field names to their expected data types.
   :rtype: dict[str, Type[Any]]


.. py:function:: get_data_type(string_representation: str) -> Type[Any]

   Get the Python data type from its string representation.

   :param string_representation: String representation of the data type (e.g., 'int', 'float', 'str').
   :type string_representation: str

   :returns: Corresponding Python data type.
   :rtype: type

   :raises ValueError: If the string representation does not correspond to a known data type.

   .. rubric:: Notes

   This function maps common string representations of data types to their
   corresponding Python types. It supports basic types like int, float, str,
   bool, list, and dict.


.. py:function:: read_attrs_to_dict(attrs_dict: dict[str, Any], metadata_object: mt_metadata.base.MetadataBase) -> dict[str, Any]

   Read HDF5 attributes from a group or dataset into a dictionary.

   :param attrs_dict: Dictionary of attributes to read and convert.
   :type attrs_dict: dict[str, Any]
   :param metadata_object: Metadata object to use for type information.
   :type metadata_object: MetadataBase

   :returns: Dictionary containing attribute names and their corresponding values.
   :rtype: dict[str, Any]


.. py:function:: inherit_doc_string(cls: Type[Any]) -> Type[Any]

   Class decorator to inherit docstring from parent classes.

   This decorator searches through the method resolution order (MRO) of a class
   to find the first parent class with a docstring and applies it to the current class.

   :param cls: The class to apply docstring inheritance to
   :type cls: type

   :returns: The same class with inherited docstring if found
   :rtype: type

   .. rubric:: Notes

   This is useful for subclasses that should inherit documentation from their
   parent classes when they don't have their own docstring defined.


.. py:function:: validate_name(name: str | None, pattern: str | None = None) -> str

   Validate and clean a name for HDF5 compatibility.

   :param name: The name to validate and clean
   :type name: str or None
   :param pattern: Pattern for validation (currently not used but reserved for future use)
   :type pattern: str, optional

   :returns: The cleaned name with spaces replaced by underscores and commas removed.
             Returns "unknown" if input name is None.
   :rtype: str

   .. rubric:: Notes

   This function ensures that names are compatible with HDF5 naming conventions
   by removing problematic characters. If the input name is None, it returns
   "unknown" as a default value.


.. py:function:: add_attributes_to_metadata_class_pydantic(obj: Type[Any]) -> Type[Any]

   Add MTH5-specific attributes to a pydantic metadata class.

   This function enhances a pydantic class by adding two important fields:
   - mth5_type: derived from the class name, indicates the type of MTH5 group
   - hdf5_reference: stores the HDF5 internal reference

   :param obj: A pydantic class to enhance with MTH5 attributes
   :type obj: type

   :returns: An instance of the enhanced class with added MTH5-specific fields
   :rtype: object

   :raises TypeError: If the input is not a class

   .. rubric:: Notes

   This function is used to dynamically add metadata fields that are required
   for MTH5 group management. The mth5_type field is derived from the class
   name by removing "Group" suffix, and the hdf5_reference field is initialized
   to None but will be set when the object is associated with an HDF5 group.


