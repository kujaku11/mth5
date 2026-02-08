mth5.tables.mth5_table
======================

.. py:module:: mth5.tables.mth5_table

.. autoapi-nested-parse::

   MTH5 table utilities.

   This module provides the `MTH5Table` base class which wraps an HDF5 dataset
   and offers convenience methods for row management, locating entries, and
   exporting to `pandas.DataFrame`.

   .. rubric:: Notes

   - Designed as a thin layer on top of NumPy/HDF5; for complex querying, prefer
       converting to a DataFrame via `to_dataframe()`.
   - Datatypes are validated and kept consistent with the underlying dataset.



Classes
-------

.. autoapisummary::

   mth5.tables.mth5_table.MTH5Table


Module Contents
---------------

.. py:class:: MTH5Table(hdf5_dataset: h5py.Dataset, default_dtype: numpy.dtype)

   Base wrapper around an HDF5 dataset representing a typed table.

   Provides simple NumPy-based operations including row insertion/removal,
   basic locating utilities, and conversion to `pandas.DataFrame`.

   :param hdf5_dataset: The HDF5 dataset that stores the table.
   :type hdf5_dataset: h5py.Dataset
   :param default_dtype: The default dtype schema for the table entries.
   :type default_dtype: numpy.dtype

   :raises MTH5TableError: If `hdf5_dataset` is not an instance of `h5py.Dataset`.

   .. rubric:: Examples

   Create a simple table and add a row::

       >>> import h5py, numpy as np
       >>> f = h5py.File('example.h5', 'w')
       >>> dtype = np.dtype([('name', 'S16'), ('value', 'f8')])
       >>> ds = f.create_dataset('table', (1,), maxshape=(None,), dtype=dtype)
       >>> from mth5.tables.mth5_table import MTH5Table
       >>> t = MTH5Table(ds, dtype)
       >>> row = np.array([('alpha'.encode('utf-8'), 1.23)], dtype=dtype)
       >>> t.add_row(row)
       1
       >>> df = t.to_dataframe()
       >>> df.head()


   .. py:attribute:: logger


   .. py:property:: hdf5_reference
      :type: object



   .. py:property:: dtype
      :type: numpy.dtype



   .. py:method:: check_dtypes(other_dtype: numpy.dtype) -> bool

      Check that dtypes match the table's dtype (including field names).

      :param other_dtype: The dtype to compare against the table's dtype.
      :type other_dtype: numpy.dtype

      :returns: True if the dtypes match; otherwise False.
      :rtype: bool



   .. py:property:: shape
      :type: tuple[int, Ellipsis]



   .. py:property:: nrows
      :type: int



   .. py:method:: locate(column: str, value: Any, test: Literal['eq', 'lt', 'le', 'gt', 'ge', 'be', 'bt'] = 'eq') -> numpy.ndarray

      Locate row indices where a column satisfies a comparison.

      :param column: Name of the column to test.
      :type column: str
      :param value: Value to compare against. For string columns, a `str` is converted
                    to a `numpy.bytes_`. For time columns (`start`, `end`,
                    `start_date`, `end_date`), values are coerced to `numpy.datetime64`.
      :type value: Any
      :param test: Type of comparison to perform.
                   - 'eq': equals
                   - 'lt': less than
                   - 'le': less than or equal to
                   - 'gt': greater than
                   - 'ge': greater than or equal to
                   - 'be': strictly between
                   - 'bt': alias for 'be'
      :type test: {'eq','lt','le','gt','ge','be','bt'}, default 'eq'

      :returns: Array of matching row indices.
      :rtype: numpy.ndarray

      :raises ValueError: If `test` is 'be'/'bt' and `value` is not a 2-length iterable.

      .. rubric:: Examples

      Find rows with value greater than 10::

          >>> idx = t.locate('value', 10, test='gt')



   .. py:method:: add_row(row: numpy.ndarray, index: int | None = None) -> int

      Add a row to the table.

      :param row: Row to insert. Must have the same dtype (or same field names,
                  allowing safe casting) as the table.
      :type row: numpy.ndarray
      :param index: Index at which to insert the row. If None, appends to the end.
      :type index: int, optional

      :returns: Index of the inserted row.
      :rtype: int

      :raises TypeError: If `row` is not a `numpy.ndarray`.
      :raises ValueError: If the dtype is incompatible with the table.



   .. py:method:: update_row(entry: numpy.ndarray) -> int

      Update a row by locating its index and rewriting the entry.

      :param entry: Entry to update, with the same dtype as the table.
      :type entry: numpy.ndarray

      :returns: Row index that was updated, or the new row index if not found.
      :rtype: int

      .. rubric:: Notes

      Matching by `hdf5_reference` is not reliable; this uses `add_row`
      and will append if the original row cannot be located.



   .. py:method:: remove_row(index: int) -> int

      Remove a row by replacing it with a null entry.

      :param index: Index of the row to remove.
      :type index: int

      :returns: Index that was updated with a null row.
      :rtype: int

      :raises IndexError: If the index is out of bounds for the current shape.

      .. rubric:: Notes

      - There is no intrinsic index stored within the array; indexing is
        on-the-fly. Prefer using the HDF5 reference column for robust
        identification.
      - The current approach inserts a null row at the specified index.



   .. py:method:: to_dataframe() -> pandas.DataFrame

      Convert the table into a `pandas.DataFrame`.

      :returns: DataFrame with decoded string columns where applicable.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      Convert and preview::

          >>> df = t.to_dataframe()
          >>> df.head()



   .. py:method:: clear_table() -> None

      Reset the table by recreating the dataset with a single null row.

      .. rubric:: Notes

      Deletes the current dataset and replaces it with a new dataset with
      the same compression/options and `dtype`, but shape `(1,)`.



   .. py:method:: update_dtype(new_dtype: numpy.dtype) -> None

      Update the dataset's dtype while preserving data and field names.

      :param new_dtype: New dtype to apply. Must have identical field names.
      :type new_dtype: numpy.dtype

      .. rubric:: Notes

      Performs a manual copy into a new array to avoid unsafe casting
      errors, then recreates the dataset with the new dtype and same
      dataset options.



