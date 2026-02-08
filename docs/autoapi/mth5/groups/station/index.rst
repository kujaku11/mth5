mth5.groups.station
===================

.. py:module:: mth5.groups.station


Attributes
----------

.. autoapisummary::

   mth5.groups.station.meta_classes


Classes
-------

.. autoapisummary::

   mth5.groups.station.MasterStationGroup
   mth5.groups.station.StationGroup


Module Contents
---------------

.. py:data:: meta_classes

.. py:class:: MasterStationGroup(group: h5py.Group, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Collection helper for all stations in a survey.

   The group lives at ``/Survey/Stations`` and offers convenience accessors
   to add, fetch, or remove stations along with a summary table.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5()
   >>> _ = mth5_obj.open_mth5("/tmp/example.mth5", mode="a")
   >>> stations = mth5_obj.stations_group
   >>> _ = stations.add_station("MT001")
   >>> stations.station_summary.head()  # doctest: +SKIP


   .. py:property:: station_summary
      :type: pandas.DataFrame


      Return a summary DataFrame of all stations in the file.

      :returns: Columns include ``station``, ``start``, ``end``, ``latitude``,
                and ``longitude``. Empty if no stations are present.
      :rtype: pandas.DataFrame

      .. rubric:: Notes

      Timestamps are parsed to pandas ``datetime64[ns]`` when possible.

      .. rubric:: Examples

      >>> summary = stations.station_summary
      >>> list(summary.columns)
      ['station', 'start', 'end', 'latitude', 'longitude']


   .. py:method:: add_station(station_name: str, station_metadata: mt_metadata.timeseries.Station | None = None) -> StationGroup

      Add or fetch a station group at ``/Survey/Stations/<name>``.

      :param station_name: Station identifier, typically matches ``metadata.id``.
      :type station_name: str
      :param station_metadata: Metadata container to seed the station attributes.
      :type station_metadata: mt_metadata.timeseries.Station, optional

      :returns: Convenience wrapper for the created or existing station.
      :rtype: StationGroup

      :raises ValueError: If ``station_name`` is empty.

      .. rubric:: Examples

      >>> station = stations.add_station("MT001")
      >>> station.metadata.id
      'MT001'



   .. py:method:: get_station(station_name: str) -> StationGroup

      Return an existing station by name.

      :param station_name: Name of the station to retrieve.
      :type station_name: str

      :returns: Wrapper for the requested station.
      :rtype: StationGroup

      :raises MTH5Error: If the station does not exist.

      .. rubric:: Examples

      >>> existing = stations.get_station("MT001")
      >>> existing.name
      'MT001'



   .. py:method:: remove_station(station_name: str) -> None

      Delete a station group reference from the file.

      :param station_name: Existing station name.
      :type station_name: str

      .. rubric:: Notes

      HDF5 deletion removes the reference only; underlying storage is not
      reclaimed.

      .. rubric:: Examples

      >>> stations.remove_station("MT001")



.. py:class:: StationGroup(group: h5py.Group, station_metadata: mt_metadata.timeseries.Station | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Utility wrapper for a single station at ``/Survey/Stations/<id>``.

   Station groups manage run collections, metadata propagation, and provide
   summary utilities for quick inspection.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> m5 = mth5.MTH5()
   >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
   >>> station = m5.stations_group.add_station("MT001")
   >>> _ = station.add_run("MT001a")
   >>> station.run_summary.shape[0] >= 1
   True


   .. py:method:: initialize_group(**kwargs: Any) -> None

      Create default subgroups and write metadata.

      :param \*\*kwargs: Additional attributes to set on the instance before initialization.

      .. rubric:: Examples

      >>> station.initialize_group()



   .. py:property:: master_station_group
      :type: MasterStationGroup


      Shortcut to the containing master station group.


   .. py:property:: transfer_functions_group
      :type: mth5.groups.TransferFunctionsGroup


      Convenience accessor for ``/Station/Transfer_Functions``.


   .. py:property:: fourier_coefficients_group
      :type: mth5.groups.MasterFCGroup


      Convenience accessor for ``/Station/Fourier_Coefficients``.


   .. py:property:: features_group
      :type: mth5.groups.MasterFeaturesGroup


      Convenience accessor for ``/Station/Features``.


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Return survey metadata with this station appended.


   .. py:method:: metadata() -> mt_metadata.timeseries.Station

      Station metadata enriched with run information.



   .. py:property:: name
      :type: str



   .. py:property:: run_summary
      :type: pandas.DataFrame


      Return a summary of runs belonging to the station.

      :returns: Columns include ``id``, ``start``, ``end``, ``components``,
                ``measurement_type``, ``sample_rate``, and ``hdf5_reference``.
      :rtype: pandas.DataFrame

      .. rubric:: Notes

      Channel lists stored as byte arrays or JSON strings are normalized
      before summarization.

      .. rubric:: Examples

      >>> station.run_summary.head()  # doctest: +SKIP


   .. py:method:: make_run_name(alphabet: bool = False) -> str | None

      Generate the next run name using an alphabetic or numeric suffix.

      :param alphabet: If ``True`` use letters (``a``, ``b``, ...); otherwise use
                       numeric suffixes (``001``).
      :type alphabet: bool, default False

      :returns: Proposed run name or ``None`` if generation fails.
      :rtype: str or None

      .. rubric:: Examples

      >>> station.metadata.id = "MT001"
      >>> station.make_run_name()
      'MT001a'



   .. py:method:: locate_run(sample_rate: float, start: str | mt_metadata.common.mttime.MTime) -> pandas.DataFrame | None

      Locate runs matching a sample rate and start time.

      :param sample_rate: Sample rate in samples per second.
      :type sample_rate: float
      :param start: Start time string or ``MTime`` instance.
      :type start: str or MTime

      :returns: Matching rows from ``run_summary`` or ``None`` when no match exists.
      :rtype: pandas.DataFrame or None

      .. rubric:: Examples

      >>> station.locate_run(256.0, "2020-01-01T00:00:00")  # doctest: +SKIP



   .. py:method:: add_run(run_name: str, run_metadata: mt_metadata.timeseries.Run | None = None) -> mth5.groups.RunGroup

      Add a run under this station.

      :param run_name: Run identifier (for example ``id`` + suffix).
      :type run_name: str
      :param run_metadata: Metadata container to seed the run attributes.
      :type run_metadata: mt_metadata.timeseries.Run, optional

      :returns: Wrapper for the created or existing run.
      :rtype: RunGroup

      .. rubric:: Examples

      >>> run = station.add_run("MT001a")
      >>> run.metadata.id
      'MT001a'



   .. py:method:: get_run(run_name: str) -> mth5.groups.RunGroup

      Return a run by name.

      :param run_name: Existing run name.
      :type run_name: str

      :returns: Wrapper for the requested run.
      :rtype: RunGroup

      :raises MTH5Error: If the run does not exist.

      .. rubric:: Examples

      >>> existing_run = station.get_run("MT001a")
      >>> existing_run.name
      'MT001a'



   .. py:method:: remove_run(run_name: str) -> None

      Remove a run from this station.

      :param run_name: Existing run name.
      :type run_name: str

      .. rubric:: Notes

      Deleting removes the reference only; storage is not reclaimed.

      .. rubric:: Examples

      >>> station.remove_run("MT001a")



   .. py:method:: update_station_metadata() -> None

      Deprecated alias for :py:meth:`update_metadata`.

      :raises DeprecationWarning: Always raised to direct callers to ``update_metadata``.

      .. rubric:: Examples

      >>> station.update_station_metadata()  # doctest: +ELLIPSIS
      Traceback (most recent call last):
      ...
      DeprecationWarning: 'update_station_metadata' has been deprecated use 'update_metadata()'



   .. py:method:: update_metadata() -> None

      Synchronize station metadata from contained runs.

      .. rubric:: Notes

      The station ``time_period`` is set to the min/max of all runs, and
      ``channels_recorded`` combines all recorded components.

      .. rubric:: Examples

      >>> _ = station.update_metadata()
      >>> station.metadata.time_period.start  # doctest: +SKIP
      '2020-01-01T00:00:00'



