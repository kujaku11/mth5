mth5.groups.standards
=====================

.. py:module:: mth5.groups.standards

.. autoapi-nested-parse::

   Created on Wed Dec 23 17:05:33 2020

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license: MIT



Attributes
----------

.. autoapisummary::

   mth5.groups.standards.ts_classes
   mth5.groups.standards.flt_classes


Classes
-------

.. autoapisummary::

   mth5.groups.standards.StandardsGroup


Functions
---------

.. autoapisummary::

   mth5.groups.standards.summarize_metadata_standards


Module Contents
---------------

.. py:data:: ts_classes

.. py:data:: flt_classes

.. py:function:: summarize_metadata_standards() -> mt_metadata.base.BaseDict

   Summarize metadata standards into a dictionary.

   Aggregates metadata standard definitions from timeseries and filter
   classes, creating a flattened dictionary suitable for storage in
   the standards summary table.

   :returns: Flattened dictionary containing metadata standards for all supported
             classes (Survey, Station, Run, Electric, Magnetic, Auxiliary,
             and various Filter types).
   :rtype: BaseDict

   .. rubric:: Notes

   Creates copies of attribute dictionaries to avoid mutations to the
   original class definitions.

   .. rubric:: Examples

   >>> standards = summarize_metadata_standards()
   >>> 'survey' in standards
   True
   >>> 'electric' in standards
   True


.. py:class:: StandardsGroup(group: Any, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for metadata standards documentation stored in the HDF5 file.

   Stores metadata standards used throughout the survey in a standardized
   summary table. This enables users to understand metadata directly from
   the file without requiring external documentation.

   The standards are organized in a summary table at ``/Survey/Standards/summary``
   with columns for attribute name, type, requirements, style, units, and
   descriptions.

   .. attribute:: summary_table

      The standards summary table with metadata definitions.

      :type: MTH5Table

   .. rubric:: Notes

   Standards include definitions for:

   - Survey, Station, Run, Electric, Magnetic, Auxiliary metadata
   - Filter types: Coefficient, FIR, FrequencyResponseTable, PoleZero, TimeDelay
   - Processing standards from aurora and fourier_coefficients modules

   .. rubric:: Examples

   >>> with MTH5('survey.mth5') as mth5_obj:
   ...     standards = mth5_obj.standards_group
   ...     summary = standards.summary_table
   ...     print(summary.array.dtype.names)
   ('attribute', 'type', 'required', 'style', 'units', 'description', ...)

   Get information about a specific attribute:

   >>> standards.get_attribute_information('survey.release_license')
   survey.release_license
   --------------------------
           type          : string
           required      : True
           style         : controlled vocabulary
           ...


   .. py:property:: summary_table
      :type: mth5.tables.MTH5Table



   .. py:method:: get_attribute_information(attribute_name: str) -> None

      Print detailed information about a metadata attribute.

      Retrieves and displays all metadata standards information for
      the specified attribute from the standards summary table.

      :param attribute_name: Name of the attribute to describe (e.g., 'survey.release_license').
      :type attribute_name: str

      :raises MTH5TableError: If the attribute is not found in the standards summary table.

      .. rubric:: Notes

      Prints formatted output including:

      - Data type
      - Whether attribute is required
      - Style (e.g., controlled vocabulary)
      - Units
      - Description
      - Valid options
      - Aliases
      - Example values
      - Default value

      .. rubric:: Examples

      >>> standards = mth5_obj.standards_group
      >>> standards.get_attribute_information('survey.release_license')
      survey.release_license
      --------------------------
              type          : string
              required      : True
              style         : controlled vocabulary
              units         :
              description   : How the data can be used. The options are based on
                       Creative Commons licenses.
              options       : CC-0,CC-BY,CC-BY-SA,CC-BY-ND,CC-BY-NC-SA
              alias         :
              example       : CC-0
              default       : CC-0



   .. py:method:: summary_table_from_dict(summary_dict: dict[str, Any]) -> None

      Populate summary table from a dictionary of metadata standards.

      Converts a flattened dictionary of metadata standards into rows
      in the HDF5 summary table.

      :param summary_dict: Flattened dictionary of all metadata standards. Keys are
                           attribute names, values are dictionaries with type, required,
                           style, units, description, etc.
      :type summary_dict: dict[str, Any]

      .. rubric:: Notes

      Processes dictionary values:

      - Lists are converted to comma-separated strings
      - None values become empty strings
      - Bytes are decoded to UTF-8

      .. todo:: Adapt method to accept pandas.DataFrame as alternative input.

      .. rubric:: Examples

      >>> standards = StandardsGroup(group)
      >>> metadata = summarize_metadata_standards()
      >>> standards.summary_table_from_dict(metadata)



   .. py:method:: get_standards_summary(modules: Optional[list[str]] = None) -> numpy.ndarray

      Get standards for specified metadata modules.

      Retrieves and concatenates standards arrays from one or more
      metadata modules for inclusion in the standards table.

      :param modules: List of module names to include (e.g., 'timeseries', 'filters').
                      If None, uses default modules: common, timeseries, timeseries.filters,
                      transfer_functions.tf, features, features.weights, processing,
                      processing.fourier_coefficients, processing.aurora.
                      Default is None.
      :type modules: list[str], optional

      :returns: Concatenated numpy structured array containing standards for all
                requested modules with dtype matching STANDARDS_DTYPE.
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> standards = StandardsGroup(group)
      >>> ts_standards = standards.get_standards_summary(['timeseries'])
      >>> print(ts_standards.shape)
      (45,)

      Get all default modules:

      >>> all_standards = standards.get_standards_summary()



   .. py:method:: summary_table_from_array(array: numpy.ndarray) -> None

      Populate summary table from a numpy structured array.

      Converts a structured numpy array into rows in the HDF5 summary table.

      :param array: Structured numpy array with dtype matching STANDARDS_DTYPE.
                    Each row represents one metadata attribute definition.
      :type array: np.ndarray

      .. rubric:: Notes

      Iterates through all rows of the structured array and adds them
      sequentially to the summary table using add_row().

      .. rubric:: Examples

      >>> standards = StandardsGroup(group)
      >>> standards_array = standards.get_standards_summary()
      >>> standards.summary_table_from_array(standards_array)



   .. py:method:: initialize_group() -> None

      Initialize the standards group and create the summary table.

      Creates the summary table dataset in the HDF5 file and populates it
      with metadata standards from all default modules. Sets appropriate
      HDF5 attributes and writes the group metadata.

      .. rubric:: Notes

      Initialization process:

      1. Creates HDF5 dataset for summary table with maximum expandable shape
      2. Applies compression if configured in dataset_options
      3. Sets HDF5 attributes: type, last_updated, reference
      4. Populates table with standards from all default modules
      5. Writes group metadata to HDF5

      The summary table uses STANDARDS_DTYPE and supports up to 1000 rows.

      .. rubric:: Examples

      >>> mth5_obj.initialize_group()
      >>> summary_table = mth5_obj.standards_group.summary_table
      >>> print(summary_table.array.shape)
      (342,)



