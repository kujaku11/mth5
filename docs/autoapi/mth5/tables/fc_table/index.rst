mth5.tables.fc_table
====================

.. py:module:: mth5.tables.fc_table

.. autoapi-nested-parse::

   Tabulate Fourier coefficients stored in an MTH5 file.

   This module provides a small utility for summarizing Fourier-coefficient
   datasets (e.g., `FCChannel`) into a structured table and exporting
   to a convenient `pandas.DataFrame` for querying and analysis.

   .. rubric:: Notes

   - A basic test for this module exists under
       ``mth5/tests/version_1/test_fcs.py``.
   - The table is populated by traversing the HDF5 hierarchy and collecting
       entries for datasets labeled with the attribute ``mth5_type='FCChannel'``.



Classes
-------

.. autoapisummary::

   mth5.tables.fc_table.FCSummaryTable


Module Contents
---------------

.. py:class:: FCSummaryTable(hdf5_dataset: h5py.Dataset)

   Bases: :py:obj:`mth5.tables.MTH5Table`


   Summary table for Fourier coefficients.

   This class wraps an HDF5 dataset that stores a summary of Fourier
   coefficient datasets and provides convenience functions such as
   `summarize()` (to populate the table) and `to_dataframe()` (to export
   entries).

   .. rubric:: Examples

   Populate and export a summary from an existing MTH5 file::

       >>> import h5py
       >>> from mth5.tables.fc_table import FCSummaryTable
       >>> f = h5py.File('example.mth5', 'r')
       >>> # Assume the summary dataset already exists at this path
       >>> table_ds = f['Exchange']['FC_Summary']
       >>> fc_table = FCSummaryTable(table_ds)
       >>> fc_table.summarize()  # walk the file and fill entries
       >>> df = fc_table.to_dataframe()
       >>> df.head()


   .. py:method:: to_dataframe() -> pandas.DataFrame

      Convert the table to a `pandas.DataFrame` for easier querying.

      :returns: A dataframe with decoded string columns and parsed start/end
                timestamps.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      Export to a dataframe and filter by component::

          >>> df = fc_table.to_dataframe()
          >>> df[df.component == 'ex']



   .. py:method:: summarize() -> None

      Populate the summary table by traversing the HDF5 hierarchy.

      The traversal searches for datasets with attribute
      ``mth5_type == 'FCChannel'`` and adds a corresponding summary row
      for each.

      :rtype: None

      .. rubric:: Notes

      - If the table contains rows from a different OS/encoding,
        row insertion can raise a `ValueError`. A warning is logged and
        processing continues for subsequent rows.

      .. rubric:: Examples

      Refresh the table entries::

          >>> fc_table.clear_table()
          >>> fc_table.summarize()



