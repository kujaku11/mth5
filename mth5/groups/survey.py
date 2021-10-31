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
    BaseGroup, MasterStationGroup, FiltersGroup, ReportsGroup, StandardsGroup)
from mth5.utils.exceptions import MTH5Error

from mt_metadata.timeseries import Survey

# =============================================================================
# Survey Group
# =============================================================================
class MasterSurveyGroup(BaseGroup):
    """
    Utility class to holds information about the stations within a survey and
    accompanying metadata.  This class is next level down from Survey for
    stations ``/Survey/Stations``.  This class provides methods to add and
    get stations.  A summary table of all existing stations is also provided
    as a convenience look up table to make searching easier.

    To access MasterStationGroup from an open MTH5 file:

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> stations = mth5_obj.stations_group

    To check what stations exist

    >>> stations.groups_list
    ['summary', 'MT001', 'MT002', 'MT003']

    To access the hdf5 group directly use `SurveyGroup.hdf5_group`.

    >>> stations.hdf5_group.ref
    <HDF5 Group Reference>

    .. note:: All attributes should be input into the metadata object, that
             way all input will be validated against the metadata standards.
             If you change attributes in metadata object, you should run the
             `SurveyGroup.write_metadata()` method.  This is a temporary
             solution, working on an automatic updater if metadata is changed.

    >>> stations.metadata.existing_attribute = 'update_existing_attribute'
    >>> stations.write_metadata()

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

    To add a station:

        >>> new_station = stations.add_station('new_station')
        >>> stations
        /Survey/Stations:
        ====================
            --> Dataset: summary
            ......................
            |- Group: new_station
            ---------------------
                --> Dataset: summary
                ......................

    Add a station with metadata:

        >>> from mth5.metadata import Station
        >>> station_metadata = Station()
        >>> station_metadata.id = 'MT004'
        >>> station_metadata.time_period.start = '2020-01-01T12:30:00'
        >>> station_metadata.location.latitude = 40.000
        >>> station_metadata.location.longitude = -120.000
        >>> new_station = stations.add_station('Test_01', station_metadata)
        >>> # to look at the metadata
        >>> new_station.metadata
        {
            "station": {
                "acquired_by.author": null,
                "acquired_by.comments": null,
                "id": "MT004",
                ...
                }
        }


    .. seealso:: `mth5.metadata` for details on how to add metadata from
                 various files and python objects.

    To remove a station:

    >>> stations.remove_station('new_station')
    >>> stations
    /Survey/Stations:
    ====================
        --> Dataset: summary
        ......................

    .. note:: Deleting a station is not as simple as del(station).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the station.

    To get a station:

    >>> existing_station = stations.get_station('existing_station_name')
    >>> existing_station
    /Survey/Stations/existing_station_name:
    =======================================
        --> Dataset: summary
        ......................
        |- Group: run_01
        ----------------
            --> Dataset: summary
            ......................
            --> Dataset: Ex
            ......................
            --> Dataset: Ey
            ......................
            --> Dataset: Hx
            ......................
            --> Dataset: Hy
            ......................
            --> Dataset: Hz
            ......................

    A summary table is provided to make searching easier.  The table
    summarized all stations within a survey. To see what names are in the
    summary table:

    >>> stations.summary_table.dtype.descr
    [('id', ('|S5', {'h5py_encoding': 'ascii'})),
     ('start', ('|S32', {'h5py_encoding': 'ascii'})),
     ('end', ('|S32', {'h5py_encoding': 'ascii'})),
     ('components', ('|S100', {'h5py_encoding': 'ascii'})),
     ('measurement_type', ('|S12', {'h5py_encoding': 'ascii'})),
     ('sample_rate', '<f8')]


    .. note:: When a station is added an entry is added to the summary table,
              where the information is pulled from the metadata.

    >>> stations.summary_table
    index |   id    |            start             |             end
     | components | measurement_type | sample_rate
     -------------------------------------------------------------------------
     --------------------------------------------------
     0   |  Test_01   |  1980-01-01T00:00:00+00:00 |  1980-01-01T00:00:00+00:00
     |  Ex,Ey,Hx,Hy,Hz   |  BBMT   | 100

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

        If the survey already exists, will return that station and nothing
        is added.

        :param station_name: Name of the station, should be the same as
                             metadata.id
        :type station_name: string
        :param station_metadata: Station metadata container, defaults to None
        :type station_metadata: :class:`mth5.metadata.Station`, optional
        :return: A convenience class for the added station
        :rtype: :class:`mth5_groups.StationGroup`

        :Example: ::

            >>> from mth5 import mth5
            >>> mth5_obj = mth5.MTH5()
            >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
            >>> # one option
            >>> stations = mth5_obj.stations_group
            >>> new_station = stations.add_station('MT001')
            >>> # another option
            >>> new_staiton = mth5_obj.stations_group.add_station('MT001')

        .. todo:: allow dictionaries, json string, xml elements as metadata
                  input.

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
        >>> surveys = mth5_obj.surveys_group
        >>> existing_survey = surveys.get_survey('MT001')
        >>> # another option
        >>> existing_staiton = mth5_obj.surveys_group.get_survey('MT001')
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
            >>> surveys = mth5_obj.surveys_group
            >>> surveys.remove_survey('MT001')
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
                + "check station_list for existing names"
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

    .. tip:: If you want ot add stations, reports, etc to the survey this
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

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

    @BaseGroup.metadata.getter
    def metadata(self):
        """Overwrite get metadata to include station information"""

        # need the try statement for when the file is initiated there is no
        # /Station group yet
        try:
            self._metadata.stations = []
            for key in self.stations_group.groups_list:
                key_group = self.stations_group.get_station(key)
                self._metadata.stations.append(key_group.metadata)

            # need to add filters
            flt_group = FiltersGroup(self.hdf5_group["Filters"])
            for key in flt_group.filter_dict.keys():
                self._metadata.filters[key] = flt_group.to_filter_object(key)

        except KeyError:
            pass

        return self._metadata

    @property
    def stations_group(self):
        return MasterStationGroup(self.hdf5_group["Stations"])
    
    @property
    def filters_group(self):
        """Convenience property for /Survey/Filters group"""
        return FiltersGroup(
                self.hdf5_group["Filters"], **self.dataset_options
            )
    
    @property
    def reports_group(self):
        """Convenience property for /Survey/Reports group"""
        return ReportsGroup(
                self.hdf5_group["Reports"], **self.dataset_options
            )
    
    @property
    def standards_group(self):
        """Convenience property for /Survey/Standards group"""
        return StandardsGroup(
                self.hdf5_group["Standards"], **self.dataset_options
            )


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
