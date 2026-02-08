mth5.utils.extract_subset_mth5
==============================

.. py:module:: mth5.utils.extract_subset_mth5


Functions
---------

.. autoapisummary::

   mth5.utils.extract_subset_mth5.extract_subset


Module Contents
---------------

.. py:function:: extract_subset(source_file: pathlib.Path, target_file: pathlib.Path, subset_df: pandas.DataFrame, filters: str = 'all')

   This function is a proof-of-concept of issue 219: exporting a subset

   TODO: add check that subset_df is a subset of source_file
   TODO: add tests for source/target v0.1.0
   TODO: add tests for source/target v0.2.0
   TODO: Consider add tests for source v0.1.0/target v0.2.0
   TODO: Consider add tests for source v0.2.0/target v0.1.0

   :param source_file: Where the data will be extracted from
   :param target_file: Where the data will be exported to
   :param subset_df: description of the data to extract
   :param filters: whether to bring all the filters or only those that are needed to describe the data.
   Right now this is "all", but
   TODO: support "required_only" filters, meaning that we only bring the filters from the selected channels.

   :return:



