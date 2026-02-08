mth5.tables.tf_table
====================

.. py:module:: mth5.tables.tf_table

.. autoapi-nested-parse::

   Transfer function summary table utilities.

   Summarize `TransferFunction` groups stored in an MTH5 file into a structured
   table and provide a convenient `pandas.DataFrame` view for querying.

   .. rubric:: Notes

   - Traversal searches for groups with attribute ``mth5_type='transferfunction'``
       and collects basic availability flags (impedance, tipper, covariance) along
       with period range and references.



Classes
-------

.. autoapisummary::

   mth5.tables.tf_table.TFSummaryTable


Module Contents
---------------

.. py:class:: TFSummaryTable(hdf5_dataset: h5py.Dataset)

   Bases: :py:obj:`mth5.tables.MTH5Table`


   Summary table for `TransferFunction` groups.

   Provides convenience functions to populate the table (`summarize`) and
   export to `pandas.DataFrame` (`to_dataframe`).

   .. rubric:: Examples

   Build and export a TF summary::

       >>> import h5py
       >>> from mth5.tables.tf_table import TFSummaryTable
       >>> f = h5py.File('example.mth5', 'r')
       >>> tf_summary_ds = f['Exchange']['TF_Summary']
       >>> tf_table = TFSummaryTable(tf_summary_ds)
       >>> tf_table.summarize()
       >>> df = tf_table.to_dataframe()
       >>> df.head()


   .. py:method:: to_dataframe() -> pandas.DataFrame

      Convert the table to a `pandas.DataFrame` for easier querying.

      :returns: A dataframe with decoded string columns.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      Filter transfer functions that include tipper::

          >>> df = tf_table.to_dataframe()
          >>> df[df.has_tipper]



   .. py:method:: summarize() -> None

      Populate the summary table by traversing the HDF5 hierarchy.

      Searches for groups where ``mth5_type`` equals ``'transferfunction'``
      and adds a row indicating available datasets (impedance, tipper,
      covariance), period min/max, and relevant references.

      :rtype: None

      .. rubric:: Examples

      Refresh the TF summary::

          >>> tf_table.clear_table()
          >>> tf_table.summarize()



