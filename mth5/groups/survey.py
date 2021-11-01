# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:59:45 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT

"""

# =============================================================================
# Imports
# =============================================================================
from mth5.groups import (
    BaseGroup,
    MasterStationGroup,
    FiltersGroup,
    ReportsGroup,
    StandardsGroup,
)
from mth5.utils.exceptions import MTH5Error

from mt_metadata.timeseries import Survey

# =============================================================================
# Survey Group
# =============================================================================
class MasterSurveyGroup(BaseGroup):
    """
    Utility class to hold information about the surveys within an experiment and
    accompanying metadata.  This class is next level down from Experiment for
    stations ``Experiment/Surveys``.  This class provides methods to add and
    get surveys.  

    To access MasterSurveyGroup from an open MTH5 file:

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> surveys = mth5_obj.surveys_group

    To check what stations exist

    >>> surveys.groups_list
    ['survey_01', 'survey_02']

    To access the hdf5 group directly use `SurveyGroup.hdf5_group`.

    >>> stations.hdf5_group.ref
    <HDF5 Group Reference>

    .. note:: All attributes should be input into the metadata object, that
             way all input will be validated against the metadata standards.
             If you change attributes in metadata object, you should run the
             `SurveyGroup.write_metadata()` method.  This is a temporary
             solution, working on an automatic updater if metadata is changed.

    If you want to add a new attribute this should be done using the
    `metadata.add_base_attribute` method.

    >>> stations.metadata.add_base_attribute('new_attribute',
    >>> ...                                'new_attribute_value',
    >>> ...                                {'type':str,
    >>> ...                                 'required':True,
    >>> ...                                 'style':'free form',
    >>> ...                                 'description': 'new attribute desc.',
    >>> ...                                 'units':None,
    >>> ...                                 'options':[],
    >>> ...                                 'alias':[],
    >>> ...                                 'example':'new attribute

    To add a survey:

        >>> new_survey = surveys.add_survey('new_survey')
        >>> surveys
        Experiment/Surveys:
        ====================
            |- Group: new_survey
            ---------------------
                |- Group: Filters
                ------------------
                |- Group: Reports
                -----------------
                |- Group: Standards
                -------------------
                |- Group: Stations
                ------------------


    Add a survey with metadata:

        >>> from mth5.metadata import Survey
        >>> survey_metadata = Survey()
        >>> survey_metadata.id = 'MT004'
        >>> survey_metadata.time_period.start = '2020-01-01T12:30:00'
        >>> new_survey = surveys.add_survey('Test_01', survey_metadata)
        >>> # to look at the metadata
        >>> new_survey.metadata
        {
            "survey": {
                "acquired_by.author": null,
                "acquired_by.comments": null,
                "id": "MT004",
                ...
                }
        }


    .. seealso:: `mth5.metadata` for details on how to add metadata from
                 various files and python objects.

    To remove a survey:

    >>> surveys.remove_survey('new_survey')
    >>> surveys
    /Survey/Stations:
    ====================

    .. note:: Deleting a survey is not as simple as del(survey).  In HDF5
              this does not free up memory, it simply removes the reference
              to that survey.  The common way to get around this is to
              copy what you want into a new file, or overwrite the survey.

    To get a survey:

    >>> existing_survey = surveys.get_survey('existing_survey_name')

    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

    @property
    def channel_summary(self):
        """
        Summary of all channels in the file.
        """

        for ii, survey in enumerate(self.groups_list):
            survey_group = self.get_survey(survey)
            if ii == 0:
                channel_summary = survey_group.channel_summary

            else:
                channel_summary = channel_summary.append(survey_group.channel_summary)

        return channel_summary

    def add_survey(self, survey_name, survey_metadata=None):
        """
        Add a survey with metadata if given with the path:
            ``/Survey/surveys/survey_name``

        If the survey already exists, will return that survey and nothing
        is added.

        :param survey_name: Name of the survey, should be the same as
                             metadata.id
        :type survey_name: string
        :param survey_metadata: Station metadata container, defaults to None
        :type survey_metadata: :class:`mth5.metadata.Station`, optional
        :return: A convenience class for the added survey
        :rtype: :class:`mth5_groups.StationGroup`

        :To add a survey:

        >>> new_survey = surveys.add_survey('new_survey')
        >>> surveys
        Experiment/Surveys:
        ====================
            |- Group: new_survey
            ---------------------
                |- Group: Filters
                ------------------
                |- Group: Reports
                -----------------
                |- Group: Standards
                -------------------
                |- Group: Stations
                ------------------


        :Add a survey with metadata:

        >>> from mth5.metadata import Survey
        >>> survey_metadata = Survey()
        >>> survey_metadata.id = 'MT004'
        >>> survey_metadata.time_period.start = '2020-01-01T12:30:00'
        >>> new_survey = surveys.add_survey('Test_01', survey_metadata)
        >>> # to look at the metadata
        >>> new_survey.metadata
        {
            "survey": {
                "acquired_by.author": null,
                "acquired_by.comments": null,
                "id": "MT004",
                ...
                }
        }

        .. seealso:: `mth5.metadata` for details on how to add metadata from
                     various files and python objects.
        """
        if survey_name is None:
            raise Exception("survey name is None, do not know what to name it")
        try:
            survey_group = self.hdf5_group.create_group(survey_name)
            self.logger.debug("Created group %s", survey_group.name)

            if survey_metadata is None:
                survey_metadata = Survey(id=survey_name)

            else:
                if survey_metadata.id != survey_name:
                    msg = (
                        f"survey group name {survey_name} must be same as "
                        + f"survey id {survey_metadata.id}"
                    )
                    self.logger.error(msg)
                    raise MTH5Error(msg)

            survey_obj = SurveyGroup(
                survey_group,
                survey_metadata=survey_metadata,
                **self.dataset_options,
            )
            survey_obj.initialize_group()

        except ValueError:
            msg = "survey %s already exists, returning existing group."
            self.logger.info(msg, survey_name)
            survey_obj = self.get_survey(survey_name)

        return survey_obj

    def get_survey(self, survey_name):
        """
        Get a survey with the same name as survey_name

        :param survey_name: existing survey name
        :type survey_name: string
        :return: convenience survey class
        :rtype: :class:`mth5.mth5_groups.surveyGroup`
        :raises MTH5Error:  if the survey name is not found.

        :Example:

        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
        >>> # one option
        >>> existing_survey = mth5_obj.get_survey('MT001')
        >>> # another option
        >>> existing_survey = mth5_obj.experiment_group.surveys_group.get_survey('MT001')
        MTH5Error: MT001 does not exist, check survey_list for existing names

        """

        try:
            return SurveyGroup(self.hdf5_group[survey_name], **self.dataset_options)
        except KeyError:
            msg = (
                f"{survey_name} does not exist, "
                + "check survey_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def remove_survey(self, survey_name):
        """
        Remove a survey from the file.

        .. note:: Deleting a survey is not as simple as del(survey).  In HDF5
              this does not free up memory, it simply removes the reference
              to that survey.  The common way to get around this is to
              copy what you want into a new file, or overwrite the survey.

        :param survey_name: existing survey name
        :type survey_name: string

        :Example: ::

            >>> from mth5 import mth5
            >>> mth5_obj = mth5.MTH5()
            >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
            >>> # one option
            >>> mth5_obj.remove_survey('MT001')
            >>> # another option
            >>> mth5_obj.surveys_group.remove_survey('MT001')

        """

        try:
            del self.hdf5_group[survey_name]
            self.logger.info(
                "Deleting a survey does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{survey_name} does not exist, "
                + "check survey_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)


class SurveyGroup(BaseGroup):
    """
    Utility class to holds general information about the survey and
    accompanying metadata for an MT survey.

    To access the hdf5 group directly use `SurveyGroup.hdf5_group`.

    >>> survey = SurveyGroup(hdf5_group)
    >>> survey.hdf5_group.ref
    <HDF5 Group Reference>

    .. note:: All attributes should be input into the metadata object, that
             way all input will be validated against the metadata standards.
             If you change attributes in metadata object, you should run the
             `SurveyGroup.write_metadata()` method.  This is a temporary
             solution, working on an automatic updater if metadata is changed.

    >>> survey.metadata.existing_attribute = 'update_existing_attribute'
    >>> survey.write_metadata()

    If you want to add a new attribute this should be done using the
    `metadata.add_base_attribute` method.

    >>> survey.metadata.add_base_attribute('new_attribute',
    >>> ...                                'new_attribute_value',
    >>> ...                                {'type':str,
    >>> ...                                 'required':True,
    >>> ...                                 'style':'free form',
    >>> ...                                 'description': 'new attribute desc.',
    >>> ...                                 'units':None,
    >>> ...                                 'options':[],
    >>> ...                                 'alias':[],
    >>> ...                                 'example':'new attribute

    .. tip:: If you want ot add surveys, reports, etc to the survey this
              should be done from the MTH5 object.  This is to avoid
              duplication, at least for now.

    To look at what the structure of ``/Survey`` looks like:

        >>> survey
        /Survey:
        ====================
            |- Group: Filters
            -----------------
                --> Dataset: summary
            -----------------
            |- Group: Reports
            -----------------
                --> Dataset: summary
                -----------------
            |- Group: Standards
            -------------------
                --> Dataset: summary
                -----------------
            |- Group: Stations
            ------------------
                --> Dataset: summary
                -----------------

    """

    def __init__(self, group, survey_metadata=None, **kwargs):

        super().__init__(group, group_metadata=survey_metadata, **kwargs)

        self._default_subgroup_names = [
            "Stations",
            "Reports",
            "Filters",
            "Standards",
        ]

    def initialize_group(self, **kwargs):
        """
        Initialize group by making a summary table and writing metadata

        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.write_metadata()

        for group_name in self._default_subgroup_names:
            self.hdf5_group.create_group(f"{group_name}")
            m5_grp = getattr(self, f"{group_name.lower()}_group")
            m5_grp.initialize_group()

    @property
    def stations_group(self):
        return MasterStationGroup(self.hdf5_group["Stations"])

    @property
    def filters_group(self):
        """Convenience property for /Survey/Filters group"""
        return FiltersGroup(self.hdf5_group["Filters"], **self.dataset_options)

    @property
    def reports_group(self):
        """Convenience property for /Survey/Reports group"""
        return ReportsGroup(self.hdf5_group["Reports"], **self.dataset_options)

    @property
    def standards_group(self):
        """Convenience property for /Survey/Standards group"""
        return StandardsGroup(self.hdf5_group["Standards"], **self.dataset_options)

    def update_survey_metadata(self, survey_dict=None):
        """
        update start end dates and location corners from stations_group.summary_table

        """

        station_summary = self.stations_group.station_summary.copy()
        self.logger.debug("Updating survey metadata from stations summary table")

        if survey_dict:
            self.metadata.from_dict(survey_dict, skip_none=True)

        self.metadata.time_period.start_date = (
            station_summary.start.min().isoformat().split("T")[0]
        )
        self.metadata.time_period.end_date = (
            station_summary.end.max().isoformat().split("T")[0]
        )
        self.metadata.northwest_corner.latitude = station_summary.latitude.max()
        self.metadata.northwest_corner.longitude = station_summary.longitude.min()
        self.metadata.southeast_corner.latitude = station_summary.latitude.min()
        self.metadata.southeast_corner.longitude = station_summary.longitude.max()

        self.write_metadata()
