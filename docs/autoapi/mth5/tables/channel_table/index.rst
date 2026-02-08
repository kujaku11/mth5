mth5.tables.channel_table
=========================

.. py:module:: mth5.tables.channel_table


Classes
-------

.. autoapisummary::

   mth5.tables.channel_table.ChannelSummaryTable


Module Contents
---------------

.. py:class:: ChannelSummaryTable(hdf5_dataset: h5py.Dataset)

   Bases: :py:obj:`mth5.tables.MTH5Table`


   Convenience wrapper around the channel summary dataset.

   Provides helpers to summarize channels, convert to pandas, and derive
   run-level summaries.

   .. rubric:: Examples

   >>> ch_table = ChannelSummaryTable(hdf5_dataset)
   >>> df = ch_table.to_dataframe()  # doctest: +SKIP
   >>> run_df = ch_table.to_run_summary()  # doctest: +SKIP


   .. py:method:: to_dataframe() -> pandas.DataFrame

      Convert the channel summary to a pandas DataFrame.

      :returns: Channel summary with decoded string columns and parsed datetimes.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      >>> df = ch_table.to_dataframe()  # doctest: +SKIP
      >>> df.head()  # doctest: +SKIP



   .. py:method:: summarize() -> None

      Populate the summary table from channel datasets in the file.



   .. py:method:: to_run_summary(allowed_input_channels: Iterable[str] = ALLOWED_INPUT_CHANNELS, allowed_output_channels: Iterable[str] = ALLOWED_OUTPUT_CHANNELS, sortby: list[str] | None = None) -> pandas.DataFrame

      Compress channel summary into a run-level summary (one row per run).

      :param allowed_input_channels: Allowed input channel names, by default ``ALLOWED_INPUT_CHANNELS``.
      :type allowed_input_channels: Iterable[str], optional
      :param allowed_output_channels: Allowed output channel names, by default ``ALLOWED_OUTPUT_CHANNELS``.
      :type allowed_output_channels: Iterable[str], optional
      :param sortby: Columns to sort by; defaults to ``["station", "start"]`` when ``None``.
      :type sortby: list of str or None, optional

      :returns: Run-level summary including channels, durations, and references.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      >>> run_df = ch_table.to_run_summary()  # doctest: +SKIP
      >>> run_df.columns[:4].tolist()  # doctest: +SKIP
      ['survey', 'station', 'run', 'start']



