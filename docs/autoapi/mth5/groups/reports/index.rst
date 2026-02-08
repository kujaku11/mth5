mth5.groups.reports
===================

.. py:module:: mth5.groups.reports


Classes
-------

.. autoapisummary::

   mth5.groups.reports.ReportsGroup


Module Contents
---------------

.. py:class:: ReportsGroup(group: h5py.Group, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Store report files (PDF/text) and images under ``/Survey/Reports``.

   Files are embedded into HDF5 datasets with basic metadata preserved.

   .. rubric:: Examples

   >>> reports = survey.reports_group
   >>> _ = reports.add_report("site_report", filename="/tmp/report.pdf")
   >>> _ = reports.get_report("site_report")  # doctest: +SKIP


   .. py:method:: add_report(report_name: str, report_metadata: dict[str, Any] | None = None, filename: str | pathlib.Path | None = None) -> None

      Add a report or image file to the group.

      :param report_name: Dataset name to store the file under.
      :type report_name: str
      :param report_metadata: Additional attributes to attach to the dataset.
      :type report_metadata: dict, optional
      :param filename: Path to the file to embed; supported types: PDF/TXT/MD and common images.
      :type filename: str or Path, optional

      :raises FileNotFoundError: If ``filename`` does not exist.

      .. rubric:: Examples

      >>> reports.add_report("manual", filename="docs/manual.pdf")  # doctest: +SKIP



   .. py:method:: get_report(report_name: str, write=True) -> pathlib.Path

      Extract a stored report or image to the current working directory.

      :param report_name: Name of the stored dataset.
      :type report_name: str

      :returns: Path to the materialized file on disk.
      :rtype: pathlib.Path

      :raises ValueError: If the stored file type is unsupported.

      .. rubric:: Examples

      >>> path = reports.get_report("site_report")  # doctest: +SKIP
      >>> path.exists()
      True



   .. py:method:: list_reports() -> list[str]

      List all stored reports and images in the group.

      :returns: Names of all stored datasets in the reports group.
      :rtype: list of str

      .. rubric:: Examples

      >>> report_names = reports.list_reports()  # doctest: +SKIP
      >>> print(report_names)
      ['site_report', 'manual', 'overview_image']



   .. py:method:: remove_report(report_name: str) -> None

      Remove a stored report or image from the group.

      :param report_name: Name of the stored dataset to remove.
      :type report_name: str

      .. rubric:: Examples

      >>> reports.remove_report("manual")  # doctest: +SKIP



