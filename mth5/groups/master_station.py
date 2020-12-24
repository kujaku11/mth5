# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:14:40 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import h5py

from mth5.groups import BaseGroup, StationGroup
from mth5.utils.exceptions import MTH5Error
from mth5 import metadata
# =============================================================================
# Standards Group
# =============================================================================
class MasterStationGroup(BaseGroup):
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

    >>> stations.group_list
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

        # # summary of stations
        # self._defaults_summary_attrs = {
        #     "name": "summary",
        #     "max_shape": (1000,),
        #     "dtype": np.dtype(
        #         [
        #             ("id", "S5"),
        #             ("start", "S32"),
        #             ("end", "S32"),
        #             ("components", "S100"),
        #             ("measurement_type", "S12"),
        #             ("location.latitude", np.float),
        #             ("location.longitude", np.float),
        #             ("location.elevation", np.float),
        #             ("hdf5_reference", h5py.ref_dtype),
        #         ]
        #     ),
        # }
        # summary of stations
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (10000,),
            "dtype": np.dtype(
                [
                    ("station", "S10"),
                    ("run", "S11"),
                    ("latitude", np.float),
                    ("longitude", np.float),
                    ("elevation", np.float),
                    ("component", "S20"),
                    ("start", "S32"),
                    ("end", "S32"),
                    ("n_samples", np.int),
                    ("measurement_type", "S12"),
                    ("azimuth", np.float),
                    ("tilt", np.float),
                    ("units", "S25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        }

    def add_station(self, station_name, station_metadata=None):
        """
        Add a station with metadata if given with the path:
            ``/Survey/Stations/station_name``

        If the station already exists, will return that station and nothing
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
        if station_name is None:
            print("Hey! Name your station!")
            raise Exception
        try:
            station_group = self.hdf5_group.create_group(station_name)
            self.logger.debug("Created group {0}".format(station_group.name))

            if station_metadata is None:
                station_metadata = metadata.Station(id=station_name)

            else:
                if station_metadata.id != station_name:
                    msg = (
                        f"Station group name {station_name} must be same as "
                        + f"station id {station_metadata.id}"
                    )
                    self.logger.error(msg)
                    raise MTH5Error(msg)
            station_obj = StationGroup(
                station_group, station_metadata=station_metadata, **self.dataset_options
            )
            station_obj.initialize_group()

            # be sure to add a table entry
            # self.summary_table.add_row(station_obj.table_entry)

        except ValueError:
            msg = (
                f"Station {station_name} already exists, " + "returning existing group."
            )
            self.logger.info(msg)
            station_obj = StationGroup(self.hdf5_group[station_name])
            station_obj.read_metadata()

        return station_obj

    def get_station(self, station_name):
        """
        Get a station with the same name as station_name

        :param station_name: existing station name
        :type station_name: string
        :return: convenience station class
        :rtype: :class:`mth5.mth5_groups.StationGroup`
        :raises MTH5Error:  if the station name is not found.

        :Example:

        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
        >>> # one option
        >>> stations = mth5_obj.stations_group
        >>> existing_station = stations.get_station('MT001')
        >>> # another option
        >>> existing_staiton = mth5_obj.stations_group.get_station('MT001')
        MTH5Error: MT001 does not exist, check station_list for existing names

        """

        try:
            return StationGroup(self.hdf5_group[station_name], **self.dataset_options)
        except KeyError:
            msg = (
                f"{station_name} does not exist, "
                + "check station_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def remove_station(self, station_name):
        """
        Remove a station from the file.

        .. note:: Deleting a station is not as simple as del(station).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the station.

        :param station_name: existing station name
        :type station_name: string

        :Example: ::

            >>> from mth5 import mth5
            >>> mth5_obj = mth5.MTH5()
            >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
            >>> # one option
            >>> stations = mth5_obj.stations_group
            >>> stations.remove_station('MT001')
            >>> # another option
            >>> mth5_obj.stations_group.remove_station('MT001')

        """

        try:
            del self.hdf5_group[station_name]
            self.logger.info(
                "Deleting a station does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{station_name} does not exist, "
                + "check station_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)
            