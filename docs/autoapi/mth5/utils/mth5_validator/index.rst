mth5.utils.mth5_validator
=========================

.. py:module:: mth5.utils.mth5_validator

.. autoapi-nested-parse::

   MTH5 File Validator
   ===================

   Validates MTH5 files for structural integrity and metadata compliance.

   This module provides comprehensive validation of MTH5 files including:
   - File format and version checks
   - Group structure validation
   - Metadata schema validation
   - Summary table validation

   Created on February 7, 2026

   :copyright: MTH5 Development Team
   :license: MIT

   .. rubric:: Examples

   Validate a file programmatically:

   >>> from mth5.utils.mth5_validator import MTH5Validator
   >>> validator = MTH5Validator('data.mth5')
   >>> results = validator.validate()
   >>> print(results.is_valid)
   True

   Validate with detailed reporting:

   >>> validator = MTH5Validator('data.mth5', verbose=True)
   >>> results = validator.validate()
   >>> results.print_report()



Classes
-------

.. autoapisummary::

   mth5.utils.mth5_validator.ValidationLevel
   mth5.utils.mth5_validator.ValidationMessage
   mth5.utils.mth5_validator.ValidationResults
   mth5.utils.mth5_validator.MTH5Validator


Functions
---------

.. autoapisummary::

   mth5.utils.mth5_validator.validate_mth5_file


Module Contents
---------------

.. py:class:: ValidationLevel(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Validation severity levels.


   .. py:attribute:: ERROR
      :value: 'ERROR'



   .. py:attribute:: WARNING
      :value: 'WARNING'



   .. py:attribute:: INFO
      :value: 'INFO'



.. py:class:: ValidationMessage

   Container for a single validation message.


   .. py:attribute:: level
      :type:  ValidationLevel


   .. py:attribute:: category
      :type:  str


   .. py:attribute:: message
      :type:  str


   .. py:attribute:: path
      :type:  str | None
      :value: None



   .. py:attribute:: details
      :type:  dict[str, Any]


.. py:class:: ValidationResults

   Container for validation results.


   .. py:attribute:: file_path
      :type:  pathlib.Path


   .. py:attribute:: messages
      :type:  list[ValidationMessage]
      :value: []



   .. py:attribute:: checked_items
      :type:  dict[str, bool]


   .. py:property:: is_valid
      :type: bool


      Check if file passed validation (no errors).


   .. py:property:: error_count
      :type: int


      Count of error messages.


   .. py:property:: warning_count
      :type: int


      Count of warning messages.


   .. py:property:: info_count
      :type: int


      Count of info messages.


   .. py:method:: add_error(category: str, message: str, path: str | None = None, **details) -> None

      Add an error message.



   .. py:method:: add_warning(category: str, message: str, path: str | None = None, **details) -> None

      Add a warning message.



   .. py:method:: add_info(category: str, message: str, path: str | None = None, **details) -> None

      Add an info message.



   .. py:method:: print_report(include_info: bool = False) -> None

      Print a formatted validation report.



   .. py:method:: to_dict() -> dict

      Convert results to dictionary.



   .. py:method:: to_json(**kwargs) -> str

      Convert results to JSON string.



.. py:class:: MTH5Validator(file_path: str | pathlib.Path, verbose: bool = False, validate_metadata: bool = True, check_data: bool = False)

   MTH5 file validator.

   Performs comprehensive validation of MTH5 files including file format,
   group structure, and metadata validation.

   :param file_path: Path to the MTH5 file to validate.
   :type file_path: str | Path
   :param verbose: Enable verbose logging during validation.
   :type verbose: bool, default False
   :param validate_metadata: Enable metadata validation using mt_metadata schemas.
   :type validate_metadata: bool, default True
   :param check_data: Check that channels contain data (can be slow for large files).
   :type check_data: bool, default False

   .. attribute:: results

      Validation results after running validate().

      :type: ValidationResults

   .. rubric:: Examples

   Basic validation:

   >>> validator = MTH5Validator('data.mth5')
   >>> results = validator.validate()
   >>> if results.is_valid:
   ...     print("File is valid!")

   Detailed validation with report:

   >>> validator = MTH5Validator('data.mth5', verbose=True, check_data=True)
   >>> results = validator.validate()
   >>> results.print_report(include_info=True)


   .. py:attribute:: file_path


   .. py:attribute:: verbose
      :value: False



   .. py:attribute:: validate_metadata
      :value: True



   .. py:attribute:: check_data
      :value: False



   .. py:attribute:: results


   .. py:attribute:: h5_file
      :value: None



   .. py:method:: validate() -> ValidationResults

      Run full validation suite.

      :returns: Complete validation results with all messages.
      :rtype: ValidationResults

      .. rubric:: Examples

      >>> validator = MTH5Validator('data.mth5')
      >>> results = validator.validate()
      >>> print(f"Valid: {results.is_valid}")



.. py:function:: validate_mth5_file(file_path: str | pathlib.Path, verbose: bool = False, **kwargs) -> ValidationResults

   Convenience function to validate an MTH5 file.

   :param file_path: Path to MTH5 file to validate.
   :type file_path: str | Path
   :param verbose: Enable verbose output.
   :type verbose: bool, default False
   :param \*\*kwargs: Additional arguments passed to MTH5Validator.

   :returns: Validation results.
   :rtype: ValidationResults

   .. rubric:: Examples

   >>> from mth5.utils.mth5_validator import validate_mth5_file
   >>> results = validate_mth5_file('data.mth5')
   >>> if not results.is_valid:
   ...     results.print_report()


