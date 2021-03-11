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
from mth5.groups import BaseGroup, MasterStationGroup

# =============================================================================
# Survey Group
# =============================================================================


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
        """ Overwrite get metadata to include station information """
        
        # need the try statement for when the file is initiated there is no
        # /Station group yet
        try:
            self._metadata.stations = []
            for key in self.stations_group.groups_list:
                key_group = self.stations_group.get_station(key)
                self._metadata.stations.append(key_group.metadata)
        except KeyError:
            pass
            
        return self._metadata

    @property
    def stations_group(self):
        return MasterStationGroup(self.hdf5_group["Stations"])

    def update_survey_metadata(self):
        """
        update start end dates and location corners from stations_group.summary_table

        """

        self.logger.debug("Updating survey metadata from stations summary table")
        self.metadata.time_period.start_date = (
            self.stations_group.station_summary.start.min().isoformat().split("T")[0]
        )
        self.metadata.time_period.end_date = (
            self.stations_group.station_summary.end.max().isoformat().split("T")[0]
        )
        self.metadata.northwest_corner.latitude = (
            self.stations_group.station_summary.latitude.max()
        )
        self.metadata.northwest_corner.longitude = (
            self.stations_group.station_summary.longitude.min()
        )
        self.metadata.southeast_corner.latitude = (
            self.stations_group.station_summary.latitude.min()
        )
        self.metadata.southeast_corner.longitude = (
            self.stations_group.station_summary.longitude.max()
        )

        self.write_metadata()
