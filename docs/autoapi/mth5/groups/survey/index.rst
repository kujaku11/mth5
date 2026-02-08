mth5.groups.survey
==================

.. py:module:: mth5.groups.survey


Classes
-------

.. autoapisummary::

   mth5.groups.survey.MasterSurveyGroup
   mth5.groups.survey.SurveyGroup


Module Contents
---------------

.. py:class:: MasterSurveyGroup(group: h5py.Group, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Collection helper for surveys under ``Experiment/Surveys``.

   Provides helpers to add, fetch, or remove surveys and to summarize all
   channels in the experiment.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> m5 = mth5.MTH5()
   >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
   >>> surveys = m5.surveys_group
   >>> _ = surveys.add_survey("survey_01")
   >>> surveys.channel_summary.head()  # doctest: +SKIP


   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Return a DataFrame summarizing all channels across surveys.

      :returns: Columns include survey, station, run, location, component,
                start/end, sample info, orientation, units, and HDF5 reference.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      >>> summary = surveys.channel_summary
      >>> set(summary.columns) >= {"survey", "station", "run", "component"}
      True


   .. py:method:: add_survey(survey_name: str, survey_metadata: mt_metadata.timeseries.Survey | None = None) -> SurveyGroup

      Add or fetch a survey at ``/Experiment/Surveys/<name>``.

      :param survey_name: Survey identifier; validated with ``validate_name``.
      :type survey_name: str
      :param survey_metadata: Metadata container used to seed the survey attributes.
      :type survey_metadata: Survey, optional

      :returns: Wrapper for the created or existing survey.
      :rtype: SurveyGroup

      :raises ValueError: If ``survey_name`` is empty.
      :raises MTH5Error: If the provided metadata id conflicts with the group name.

      .. rubric:: Examples

      >>> survey = surveys.add_survey("survey_01")
      >>> survey.metadata.id
      'survey_01'



   .. py:method:: get_survey(survey_name: str) -> SurveyGroup

      Return an existing survey by name.

      :param survey_name: Existing survey name.
      :type survey_name: str

      :returns: Wrapper for the requested survey.
      :rtype: SurveyGroup

      :raises MTH5Error: If the survey does not exist.

      .. rubric:: Examples

      >>> existing = surveys.get_survey("survey_01")
      >>> existing.metadata.id
      'survey_01'



   .. py:method:: remove_survey(survey_name: str) -> None

      Delete a survey reference from the file.

      :param survey_name: Existing survey name.
      :type survey_name: str

      .. rubric:: Notes

      HDF5 deletion removes the reference only; storage is not reclaimed.

      .. rubric:: Examples

      >>> surveys.remove_survey("survey_01")



.. py:class:: SurveyGroup(group: h5py.Group, survey_metadata: mt_metadata.timeseries.Survey | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Wrapper for a single survey at ``Experiment/Surveys/<id>``.

   Handles survey-level metadata, child groups (stations, reports, filters,
   standards), and synchronization utilities.

   .. rubric:: Examples

   >>> survey = surveys.add_survey("survey_01")
   >>> survey.metadata.id
   'survey_01'


   .. py:method:: initialize_group(**kwargs: Any) -> None

      Create default subgroups and write survey metadata.

      :param \*\*kwargs: Additional attributes to set on the instance before initialization.

      .. rubric:: Examples

      >>> survey.initialize_group()



   .. py:method:: metadata() -> mt_metadata.timeseries.Survey

      Survey metadata enriched with station and filter information.



   .. py:method:: write_metadata() -> None

      Write HDF5 attributes from the survey metadata object.



   .. py:property:: stations_group
      :type: mth5.groups.MasterStationGroup



   .. py:property:: filters_group
      :type: mth5.groups.FiltersGroup


      Convenience accessor for ``/Survey/Filters`` group.


   .. py:property:: reports_group
      :type: mth5.groups.ReportsGroup


      Convenience accessor for ``/Survey/Reports`` group.


   .. py:property:: standards_group
      :type: mth5.groups.StandardsGroup


      Convenience accessor for ``/Survey/Standards`` group.


   .. py:method:: update_survey_metadata(survey_dict: dict[str, Any] | None = None) -> None

      Deprecated alias for :py:meth:`update_metadata`.

      :raises DeprecationWarning: Always raised to direct callers to ``update_metadata``.

      .. rubric:: Examples

      >>> survey.update_survey_metadata()  # doctest: +ELLIPSIS
      Traceback (most recent call last):
      ...
      DeprecationWarning: 'update_survey_metadata' has been deprecated use 'update_metadata()'



   .. py:method:: update_metadata(survey_dict: dict[str, Any] | None = None) -> None

      Synchronize survey metadata from station summaries.

      :param survey_dict: Additional metadata values to merge before synchronization.
      :type survey_dict: dict, optional

      .. rubric:: Notes

      Updates survey start/end dates and bounding box from station summaries,
      then writes metadata to HDF5.

      .. rubric:: Examples

      >>> _ = survey.update_metadata()
      >>> survey.metadata.time_period.start_date  # doctest: +SKIP
      '2020-01-01'



