# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:18:29 2020

.. note:: Need to keep these groups together, if you split them into files you
get a circular import.

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import inspect
import weakref
import logging

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime
from mt_metadata.base import Base

from mth5.groups.base import BaseGroup
from mth5.utils.exceptions import MTH5Error
from mth5.helpers import to_numpy_type, inherit_doc_string
from mth5.timeseries import ChannelTS, RunTS

meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
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
                    ("sample_rate", np.float),
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
                f"Station {station_name} already exists, " +
                "returning existing group."
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


# =============================================================================
# Station Group
# =============================================================================
class StationGroup(BaseGroup):
    """
    StationGroup is a utility class to hold information about a single station
    and accompanying metadata.  This class is the next level down from
    Stations --> ``/Survey/Stations/station_name``.

    This class provides methods to add and get runs.  A summary table of all
    existing runs in the station is also provided as a convenience look up
    table to make searching easier.

    :param group: HDF5 group for a station, should have a path
                  ``/Survey/Stations/station_name``
    :type group: :class:`h5py.Group`
    :param station_metadata: metadata container, defaults to None
    :type station_metadata: :class:`mth5.metadata.Station`, optional

    :Usage:

    :Access StationGroup from an open MTH5 file:

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> station = mth5_obj.stations_group.get_station('MT001')

    :Check what runs exist:

    >>> station.group_list
    ['MT001a', 'MT001b', 'MT001c', 'MT001d']

    To access the hdf5 group directly use `StationGroup.hdf5_group`.

    >>> station.hdf5_group.ref
    <HDF5 Group Reference>

    .. note:: All attributes should be input into the metadata object, that
             way all input will be validated against the metadata standards.
             If you change attributes in metadata object, you should run the
             `SurveyGroup.write_metadata()` method.  This is a temporary
             solution, working on an automatic updater if metadata is changed.

    >>> station.metadata.existing_attribute = 'update_existing_attribute'
    >>> station.write_metadata()

    If you want to add a new attribute this should be done using the
    `metadata.add_base_attribute` method.

    >>> station.metadata.add_base_attribute('new_attribute',
    >>> ...                                 'new_attribute_value',
    >>> ...                                 {'type':str,
    >>> ...                                  'required':True,
    >>> ...                                  'style':'free form',
    >>> ...                                  'description': 'new attribute desc.',
    >>> ...                                  'units':None,
    >>> ...                                  'options':[],
    >>> ...                                  'alias':[],
    >>> ...                                  'example':'new attribute

    :To add a run:

    >>> new_run = stations.add_run('MT001e')
    >>> new_run
    /Survey/Stations/Test_01:
    =========================
        |- Group: MT001e
        -----------------
            --> Dataset: summary
            ......................
        --> Dataset: summary
        ......................

    :Add a run with metadata:

    >>> from mth5.metadata import Run
    >>> run_metadata = Run()
    >>> run_metadata.time_period.start = '2020-01-01T12:30:00'
    >>> run_metadata.time_period.end = '2020-01-03T16:30:00'
    >>> run_metadata.location.latitude = 40.000
    >>> run_metadata.location.longitude = -120.000
    >>> new_run = runs.add_run('Test_01', run_metadata)
    >>> # to look at the metadata
    >>> new_run.metadata
    {
        "run": {
            "acquired_by.author": "new_user",
            "acquired_by.comments": "First time",
            "channels_recorded_auxiliary": ['T'],
            ...
            }
    }


    .. seealso:: `mth5.metadata` for details on how to add metadata from
                 various files and python objects.

    :Remove a run:

    >>> station.remove_run('new_run')
    >>> station
    /Survey/Stations/Test_01:
    =========================
        --> Dataset: summary
        ......................

    .. note:: Deleting a station is not as simple as del(station).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the station.

    :Get a run:

    >>> existing_run = stations.get_station('existing_run')
    >>> existing_run
    /Survey/Stations/MT001/MT001a:
    =======================================
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

    :summary Table:

    A summary table is provided to make searching easier.  The table
    summarized all stations within a survey. To see what names are in the
    summary table:

    >>> new_run.summary_table.dtype.descr
    [('id', ('|S20', {'h5py_encoding': 'ascii'})),
     ('start', ('|S32', {'h5py_encoding': 'ascii'})),
     ('end', ('|S32', {'h5py_encoding': 'ascii'})),
     ('components', ('|S100', {'h5py_encoding': 'ascii'})),
     ('measurement_type', ('|S12', {'h5py_encoding': 'ascii'})),
     ('sample_rate', '<f8'),
     ('hdf5_reference', ('|O', {'ref': h5py.h5r.Reference}))]

    .. note:: When a run is added an entry is added to the summary table,
              where the information is pulled from the metadata.

    >>> station.summary_table
    index | id | start | end | components | measurement_type | sample_rate |
    hdf5_reference
    --------------------------------------------------------------------------
    -------------
    """

    def __init__(self, group, station_metadata=None, **kwargs):
        super().__init__(group, group_metadata=station_metadata, **kwargs)

        # summary of runs
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (1000,),
            "dtype": np.dtype(
                [
                    ("id", "S20"),
                    ("start", "S32"),
                    ("end", "S32"),
                    ("components", "S100"),
                    ("measurement_type", "S12"),
                    ("sample_rate", np.float),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        }

    @property
    def master_station_group(self):
        """ shortcut to master station group """
        return MasterStationGroup(self.hdf5_group.parent)

    @property
    def name(self):
        return self.metadata.id

    @name.setter
    def name(self, name):
        self.metadata.id = name

    @property
    def table_entry(self):
        """ make table entry """

        return np.array(
            [
                (
                    self.metadata.id,
                    self.metadata.time_period.start,
                    self.metadata.time_period.end,
                    ",".join(self.metadata.channels_recorded),
                    self.metadata.data_type,
                    self.metadata.location.latitude,
                    self.metadata.location.longitude,
                    self.metadata.location.elevation,
                    self.hdf5_group.ref,
                )
            ],
            dtype=np.dtype(
                [
                    ("id", "S5"),
                    ("start", "S32"),
                    ("end", "S32"),
                    ("components", "S100"),
                    ("measurement_type", "S12"),
                    ("location.latitude", np.float),
                    ("location.longitude", np.float),
                    ("location.elevation", np.float),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

    def make_run_name(self):
        """
        Make a run name that will be the next alphabet letter extracted from
        the run list.  Expects that all runs are labled as id{a-z}.

        :return: metadata.id + next letter
        :rtype: string

        >>> station.metadata.id = 'MT001'
        >>> station.make_run_name()
        'MT001a'

        """
        if self.name is None:
            msg = "id is not set, cannot make a run name"
            self.logger.error(msg)
            raise MTH5Error(msg)

        run_list = sorted(
            [group[-1:] for group in self.groups_list if self.name in group]
        )

        if len(run_list) == 0:
            next_letter = "a"
        else:
            next_letter = chr(ord(run_list[-1]) + 1)

        return "{0}{1}".format(self.name, next_letter)

    def locate_run(self, sample_rate, start):
        """
        Locate a run based on sample rate and start time from the summary table

        :param sample_rate: sample rate in samples/seconds
        :type sample_rate: float
        :param start: start time
        :type start: string or :class:`mth5.utils.mttime.MTime`
        :return: appropriate run name, None if not found
        :rtype: string or None

        """

        if not isinstance(start, MTime):
            start = MTime(start)

        if self.summary_table.nrows < 1:
            self.logger.debug("No rows in summary table")
            return None

        sr_find = list(self.summary_table.locate("sample_rate", sample_rate))
        if sr_find == []:
            self.logger.debug(
                f"no summary entries with sample rate {sample_rate}")
            return None

        for ff in sr_find:
            row = self.summary_table.array[ff]
            if MTime(row["start"].decode()) == start:
                return row["id"]

        self.logger.debug(f"no summary entries with start time {start}")
        return None

    def add_run(self, run_name, run_metadata=None):
        """
        Add a run to a station.

        :param run_name: run name, should be id{a-z}
        :type run_name: string
        :param metadata: metadata container, defaults to None
        :type metadata: :class:`mth5.metadata.Station`, optional

        need to be able to fill an entry in the summary table.

        .. todo:: auto fill run name if none is given.

        .. todo:: add ability to add a run with data.

        """

        try:
            run_group = self.hdf5_group.create_group(run_name)
            self.logger.debug("Created group {0}".format(run_group.name))
            if run_metadata is None:
                run_metadata = metadata.Run(id=run_name)
            elif run_metadata.id != run_name:
                msg = (
                    f"Run name {run_name} must be the same as "
                    + f"run_metadata.id {run_metadata.id}"
                )
                self.logger.error(msg)
                raise MTH5Error(msg)

            run_obj = RunGroup(
                run_group, run_metadata=run_metadata, **self.dataset_options
            )
            run_obj.initialize_group()

            self.summary_table.add_row(run_obj.table_entry)

        except ValueError:
            msg = f"run {run_name} already exists, " + \
                "returning existing group."
            self.logger.info(msg)
            run_obj = RunGroup(self.hdf5_group[run_name])
            run_obj.read_metadata()

        return run_obj

    def get_run(self, run_name):
        """
        get a run from run name

        :param run_name: existing run name
        :type run_name: string
        :return: Run object
        :rtype: :class:`mth5.mth5_groups.RunGroup`

        >>> existing_run = station.get_run('MT001')

        """
        try:
            return RunGroup(self.hdf5_group[run_name], **self.dataset_options)
        except KeyError:
            msg = (
                f"{run_name} does not exist, " +
                "check groups_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def remove_run(self, run_name):
        """
        Remove a run from the station.

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
            del self.hdf5_group[run_name]
            self.logger.info(
                "Deleting a run does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{run_name} does not exist, " +
                "check station_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def validate_station_metadata(self):
        """
        Check metadata from the runs and make sure it matches the station metadata

        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.logger.debug("Updating station metadata from summary table")
        self.metadata.time_period.start = min(
            self.summary_table.array["start"]
        ).decode()
        self.metadata.time_period.end = max(
            self.summary_table.array["end"]).decode()
        self.metadata.channels_recorded = ",".join(
            list(
                set(
                    ",".join(
                        list(self.summary_table.array["components"].astype(
                            np.unicode_))
                    ).split(",")
                )
            )
        )

        self.write_metadata()


# =============================================================================
# Run Group
# =============================================================================
class RunGroup(BaseGroup):
    """
    RunGroup is a utility class to hold information about a single run
    and accompanying metadata.  This class is the next level down from
    Stations --> ``/Survey/Stations/station/station{a-z}``.

    This class provides methods to add and get channels.  A summary table of
    all existing channels in the run is also provided as a convenience look up
    table to make searching easier.

    :param group: HDF5 group for a station, should have a path
                  ``/Survey/Stations/station_name/run_name``
    :type group: :class:`h5py.Group`
    :param station_metadata: metadata container, defaults to None
    :type station_metadata: :class:`mth5.metadata.Station`, optional

    :Access RunGroup from an open MTH5 file:

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')

    :Check what channels exist:

    >>> station.group_list
    ['Ex', 'Ey', 'Hx', 'Hy']

    To access the hdf5 group directly use `RunGroup.hdf5_group`

    >>> station.hdf5_group.ref
    <HDF5 Group Reference>

    .. note:: All attributes should be input into the metadata object, that
             way all input will be validated against the metadata standards.
             If you change attributes in metadata object, you should run the
             `SurveyGroup.write_metadata()` method.  This is a temporary
             solution, working on an automatic updater if metadata is changed.

    >>> run.metadata.existing_attribute = 'update_existing_attribute'
    >>> run.write_metadata()

    If you want to add a new attribute this should be done using the
    `metadata.add_base_attribute` method.

    >>> station.metadata.add_base_attribute('new_attribute',
    >>> ...                                 'new_attribute_value',
    >>> ...                                 {'type':str,
    >>> ...                                  'required':True,
    >>> ...                                  'style':'free form',
    >>> ...                                  'description': 'new attribute desc.',
    >>> ...                                  'units':None,
    >>> ...                                  'options':[],
    >>> ...                                  'alias':[],
    >>> ...                                  'example':'new attribute

    :Add a channel:

    >>> new_channel = run.add_channel('Ex', 'electric',
    >>> ...                            data=numpy.random.rand(4096))
    >>> new_run
    /Survey/Stations/MT001/MT001a:
    =======================================
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


    :Add a channel with metadata:

    >>> from mth5.metadata import Electric
    >>> ex_metadata = Electric()
    >>> ex_metadata.time_period.start = '2020-01-01T12:30:00'
    >>> ex_metadata.time_period.end = '2020-01-03T16:30:00'
    >>> new_ex = run.add_channel('Ex', 'electric',
    >>> ...                       channel_metadata=ex_metadata)
    >>> # to look at the metadata
    >>> new_ex.metadata
    {
         "electric": {
            "ac.end": 1.2,
            "ac.start": 2.3,
            ...
            }
    }


    .. seealso:: `mth5.metadata` for details on how to add metadata from
                 various files and python objects.

    :Remove a channel:

    >>> run.remove_channel('Ex')
    >>> station
    /Survey/Stations/MT001/MT001a:
    =======================================
        --> Dataset: summary
        ......................
        --> Dataset: Ey
        ......................
        --> Dataset: Hx
        ......................
        --> Dataset: Hy
        ......................

    .. note:: Deleting a station is not as simple as del(station).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the station.

    :Get a channel:

    >>> existing_ex = stations.get_channel('Ex')
    >>> existing_ex
    Channel Electric:
    -------------------
        data type:        Ex
        data type:        electric
        data format:      float32
        data shape:       (4096,)
        start:            1980-01-01T00:00:00+00:00
        end:              1980-01-01T00:32:+08:00
        sample rate:      8


    :summary Table:

    A summary table is provided to make searching easier.  The table
    summarized all stations within a survey. To see what names are in the
    summary table:

    >>> run.summary_table.dtype.descr
    [('component', ('|S5', {'h5py_encoding': 'ascii'})),
     ('start', ('|S32', {'h5py_encoding': 'ascii'})),
     ('end', ('|S32', {'h5py_encoding': 'ascii'})),
     ('n_samples', '<i4'),
     ('measurement_type', ('|S12', {'h5py_encoding': 'ascii'})),
     ('units', ('|S25', {'h5py_encoding': 'ascii'})),
     ('hdf5_reference', ('|O', {'ref': h5py.h5r.Reference}))]


    .. note:: When a run is added an entry is added to the summary table,
              where the information is pulled from the metadata.

    >>> new_run.summary_table
    index | component | start | end | n_samples | measurement_type | units |
    hdf5_reference
    --------------------------------------------------------------------------
    -------------
    """

    def __init__(self, group, run_metadata=None, **kwargs):

        super().__init__(group, group_metadata=run_metadata, **kwargs)

        # summary of channels in run
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (20,),
            "dtype": np.dtype(
                [
                    ("component", "S20"),
                    ("start", "S32"),
                    ("end", "S32"),
                    ("n_samples", np.int),
                    ("measurement_type", "S12"),
                    ("units", "S25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        }

    @property
    def station_group(self):
        """ shortcut to station group """
        return StationGroup(self.hdf5_group.parent)

    @property
    def master_station_group(self):
        """ shortcut to master station group """
        return MasterStationGroup(self.hdf5_group.parent.parent)

    @property
    def table_entry(self):
        """
        Get a run table entry

        :return: a properly formatted run table entry
        :rtype: :class:`numpy.ndarray` with dtype:

        >>> dtype([('id', 'S20'),
                 ('start', 'S32'),
                 ('end', 'S32'),
                 ('components', 'S100'),
                 ('measurement_type', 'S12'),
                 ('sample_rate', np.float),
                 ('hdf5_reference', h5py.ref_dtype)])

        """
        return np.array(
            [
                (
                    self.metadata.id,
                    self.metadata.time_period.start,
                    self.metadata.time_period.end,
                    ",".join(self.metadata.channels_recorded_all),
                    self.metadata.data_type,
                    self.metadata.sample_rate,
                    self.hdf5_group.ref,
                )
            ],
            dtype=np.dtype(
                [
                    ("id", "S20"),
                    ("start", "S32"),
                    ("end", "S32"),
                    ("components", "S100"),
                    ("measurement_type", "S12"),
                    ("sample_rate", np.float),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

    def add_channel(
        self,
        channel_name,
        channel_type,
        data,
        channel_dtype="f",
        max_shape=(None,),
        chunks=True,
        channel_metadata=None,
        **kwargs,
    ):
        """
        add a channel to the run

        :param channel_name: name of the channel
        :type channel_name: string
        :param channel_type: [ electric | magnetic | auxiliary ]
        :type channel_type: string
        :raises MTH5Error: If channel type is not correct

        :param channel_metadata: metadata container, defaults to None
        :type channel_metadata: [ :class:`mth5.metadata.Electric` |
                                 :class:`mth5.metadata.Magnetic` |
                                 :class:`mth5.metadata.Auxiliary` ], optional
        :return: Channel container
        :rtype: [ :class:`mth5.mth5_groups.ElectricDatset` |
                 :class:`mth5.mth5_groups.MagneticDatset` |
                 :class:`mth5.mth5_groups.AuxiliaryDatset` ]

        >>> new_channel = run.add_channel('Ex', 'electric', None)
        >>> new_channel
        Channel Electric:
        -------------------
                        component:        None
                data type:        electric
                data format:      float32
                data shape:       (1,)
                start:            1980-01-01T00:00:00+00:00
                end:              1980-01-01T00:00:00+00:00
                sample rate:      None


        """
        channel_name = channel_name.lower()
        for key, value in kwargs.items():
            setattr(self, key, value)

        if data is not None:
            if data.size < 1024:
                chunks = None

        try:
            if data is not None:
                channel_group = self.hdf5_group.create_dataset(
                    channel_name,
                    data=data,
                    maxshape=max_shape,
                    dtype=data.dtype,
                    chunks=chunks,
                    **self.dataset_options,
                )
            # initialize an resizable data array
            else:
                channel_group = self.hdf5_group.create_dataset(
                    channel_name,
                    shape=(1,),
                    maxshape=max_shape,
                    dtype=channel_dtype,
                    chunks=chunks,
                    **self.dataset_options,
                )

            self.logger.debug("Created group {0}".format(channel_group.name))
            if channel_type.lower() in ["magnetic"]:
                channel_obj = MagneticDataset(
                    channel_group, dataset_metadata=channel_metadata
                )
            elif channel_type.lower() in ["electric"]:
                channel_obj = ElectricDataset(
                    channel_group, dataset_metadata=channel_metadata
                )
            elif channel_type.lower() in ["auxiliary"]:
                channel_obj = AuxiliaryDataset(
                    channel_group, dataset_metadata=channel_metadata
                )
            else:
                msg = (
                    "`channel_type` must be in [ electric | magnetic | "
                    + "auxiliary ]. Input was {0}".format(channel_type)
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
            if channel_obj.metadata.component is None:
                channel_obj.metadata.component = channel_name
            channel_obj.write_metadata()
            self.summary_table.add_row(channel_obj.table_entry)
            self.master_station_group.summary_table.add_row(
                channel_obj.channel_entry)

        except (OSError, RuntimeError):
            msg = (
                f"channel {channel_name} already exists, " +
                "returning existing group."
            )
            self.logger.info(msg)
            if channel_type in ["magnetic"]:
                channel_obj = MagneticDataset(self.hdf5_group[channel_name])
            elif channel_type in ["electric"]:
                channel_obj = ElectricDataset(self.hdf5_group[channel_name])
            elif channel_type in ["auxiliary"]:
                channel_obj = AuxiliaryDataset(self.hdf5_group[channel_name])
            channel_obj.read_metadata()

        return channel_obj

    def get_channel(self, channel_name):
        """

        Get a channel from an existing name.  Returns the appropriate
        container.

        :param channel_name: name of the channel
        :type channel_name: string
        :return: Channel container
        :rtype: [ :class:`mth5.mth5_groups.ElectricDatset` |
                  :class:`mth5.mth5_groups.MagneticDatset` |
                  :class:`mth5.mth5_groups.AuxiliaryDatset` ]
        :raises MTH5Error:  If no channel is found

        :Example:

        >>> existing_channel = run.get_channel('Ex')
        MTH5Error: Ex does not exist, check groups_list for existing names'

        >>> run.group_list
        ['Ey', 'Hx', 'Hz']

        >>> existing_channel = run.get_channel('Ey')
        >>> existing_channel
        Channel Electric:
        -------------------
                        component:        Ey
                data type:        electric
                data format:      float32
                data shape:       (4096,)
                start:            1980-01-01T00:00:00+00:00
                end:              1980-01-01T00:00:01+00:00
                sample rate:      4096


        """

        try:
            ch_dataset = self.hdf5_group[channel_name]
            if ch_dataset.attrs["mth5_type"].lower() in ["electric"]:
                ch_metadata = meta_classes["Electric"]()
                ch_metadata.from_dict({"Electric": ch_dataset.attrs})
                channel = ElectricDataset(
                    ch_dataset, dataset_metadata=ch_metadata, write_metadata=False
                )
            elif ch_dataset.attrs["mth5_type"].lower() in ["magnetic"]:
                ch_metadata = meta_classes["Magnetic"]()
                ch_metadata.from_dict({"Magnetic": ch_dataset.attrs})
                channel = MagneticDataset(
                    ch_dataset, dataset_metadata=ch_metadata, write_metadata=False
                )
            elif ch_dataset.attrs["mth5_type"].lower() in ["auxiliary"]:
                ch_metadata = meta_classes["Auxiliary"]()
                ch_metadata.from_dict({"Auxiliary": ch_dataset.attrs})
                channel = AuxiliaryDataset(
                    ch_dataset, dataset_metadata=ch_metadata, write_metadata=False
                )
            else:
                channel = ChannelDataset(ch_dataset)

            return channel

        except KeyError:
            msg = (
                f"{channel_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def remove_channel(self, channel_name):
        """
        Remove a run from the station.

        .. note:: Deleting a channel is not as simple as del(channel).  In HDF5
              this does not free up memory, it simply removes the reference
              to that channel.  The common way to get around this is to
              copy what you want into a new file, or overwrite the channel.

        :param station_name: existing station name
        :type station_name: string

        :Example: 

        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
        >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')
        >>> run.remove_channel('Ex')

        .. todo:: Need to remove summary table entry as well.

        """

        channel_name = channel_name.lower()

        try:
            component = self.hdf5_group[channel_name].attrs["component"]
            del self.hdf5_group[channel_name]
            self.summary_table.remove_row(
                self.summary_table.locate("component", component)
            )
            self.logger.info(
                "Deleting a channel does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{channel_name} does not exist, "
                + "check group_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def to_runts(self):
        """
        create a :class:`mth5.timeseries.RunTS` object from channels of the 
        run

        :return: DESCRIPTION
        :rtype: TYPE

        """
        ch_list = []
        for channel in self.groups_list:
            if channel in ["summary"]:
                continue
            ch_obj = self.get_channel(channel)
            ts_obj = ch_obj.to_channel_ts()
            ch_list.append(ts_obj)
        return RunTS(ch_list, run_metadata=self.metadata)

    def from_runts(self, run_ts_obj):
        """
        create channel datasets from a :class:`mth5.timeseries.RunTS` object 
        and update metadata.  

        :parameter :class:`mth5.timeseries.RunTS` run_ts_obj: Run object with all
        the appropriate channels and metadata.

        Will create a run group and appropriate channel datasets.
        """

        if not isinstance(run_ts_obj, RunTS):
            msg = f"Input must be a mth5.timeseries.RunTS object not {type(run_ts_obj)}"
            self.logger.error(msg)
            raise MTH5Error(msg)

        self.metadata.from_dict(run_ts_obj.metadata.to_dict())

        channels = []

        for comp in run_ts_obj.channels:

            if comp[0] in ["e"]:
                channel_type = "electric"
                ch_metadata = metadata.Electric()

            elif comp[0] in ["h", "b"]:
                channel_type = "magnetic"
                ch_metadata = metadata.Magnetic()
            else:
                channel_type = "auxiliary"
                ch_metadata = metadata.Auxiliary()

            ch_metadata.from_dict(
                {channel_type: run_ts_obj.dataset[comp].attrs})
            ch_metadata.hdf5_type = channel_type

            channels.append(
                self.add_channel(
                    comp,
                    channel_type,
                    run_ts_obj.dataset[comp].values,
                    channel_metadata=ch_metadata,
                )
            )
        return channels

    def from_channel_ts(self, channel_ts_obj):
        """
        create a channel data set from a :class:`mth5.timeseries.ChannelTS` object and 
        update metadata.

        :param channel_ts_obj: a single time series object
        :type channel_ts_obj: :class:`mth5.timeseries.ChannelTS`
        :return: new channel dataset
        :rtype: :class:`mth5.groups.ChannelDataset

        """

        if not isinstance(channel_ts_obj, ChannelTS):
            msg = f"Input must be a mth5.timeseries.ChannelTS object not {type(channel_ts_obj)}"
            self.logger.error(msg)
            raise MTH5Error(msg)

        ch_obj = self.add_channel(
            channel_ts_obj.component,
            channel_ts_obj.metadata.type,
            channel_ts_obj.ts.values,
            channel_metadata=channel_ts_obj.metadata,
        )

        # need to update the channels recorded
        if channel_ts_obj.metadata.type == "electric":
            if self.metadata.channels_recorded_electric is None:
                self.metadata.channels_recorded_electric = [
                    channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_electric
            ):
                self.metadata.channels_recorded_electric.append(
                    channel_ts_obj.component
                )

        elif channel_ts_obj.metadata.type == "magnetic":
            if self.metadata.channels_recorded_magnetic is None:
                self.metadata.channels_recorded_magnetic = [
                    channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_magnetic
            ):
                self.metadata.channels_recorded_magnetic.append(
                    channel_ts_obj.component
                )

        elif channel_ts_obj.metadata.type == "auxiliary":
            if self.metadata.channels_recorded_auxiliary is None:
                self.metadata.channels_recorded_auxiliary = [
                    channel_ts_obj.component]
            elif (
                channel_ts_obj.component
                not in self.metadata.channels_recorded_auxiliary
            ):
                self.metadata.channels_recorded_auxiliary.append(
                    channel_ts_obj.component
                )

        return ch_obj

    def validate_run_metadata(self):
        """
        Update metadata and table entries to ensure consistency

        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.logger.debug("Updating run metadata from summary table.")
        channels_recorded = list(
            self.summary_table.array["component"].astype(np.unicode_)
        )
        self.metadata.channels_recorded_electric = [
            cc for cc in channels_recorded if cc[0] in ["e"]
        ]
        self.metadata.channels_recorded_magnetic = [
            cc for cc in channels_recorded if cc[0] in ["h", "b"]
        ]
        self.metadata.channels_recorded_auxiliary = [
            cc for cc in channels_recorded if cc[0] not in ["e", "h", "b"]
        ]

        self.metadata.time_period.start = min(
            self.summary_table.array["start"]
        ).decode()

        self.metadata.time_period.end = max(
            self.summary_table.array["end"]).decode()


class ChannelDataset:
    """
    Holds a channel dataset.  This is a simple container for the data to make
    sure that the user has the flexibility to turn the channel into an object
    they want to deal with.

    For now all the numpy type slicing can be used on `hdf5_dataset`

    :param dataset: dataset object for the channel
    :type dataset: :class:`h5py.Dataset`
    :param dataset_metadata: metadata container, defaults to None
    :type dataset_metadata: [ :class:`mth5.metadata.Electric` |
                              :class:`mth5.metadata.Magnetic` |
                              :class:`mth5.metadata.Auxiliary` ], optional
    :raises MTH5Error: If the dataset is not of the correct type

    Utilities will be written to create some common objects like:

        * xarray.DataArray
        * pandas.DataFrame
        * zarr
        * dask.Array

    The benefit of these other objects is that they can be indexed by time,
    and they have much more buit-in funcionality.

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')
    >>> channel = run.get_channel('Ex')
    >>> channel
    Channel Electric:
    -------------------
                component:        Ey
        data type:        electric
        data format:      float32
        data shape:       (4096,)
        start:            1980-01-01T00:00:00+00:00
        end:              1980-01-01T00:00:01+00:00
        sample rate:      4096

    """

    def __init__(self, dataset, dataset_metadata=None, **kwargs):

        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()

        self.logger = logging.getLogger(
            "{0}.{1}".format(__name__, self._class_name))

        # set metadata to the appropriate class.  Standards is not a
        # Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        self.metadata = Base()
        try:
            self.metadata = meta_classes[self._class_name]()
        except KeyError:
            self.metadata = Base()

        if not hasattr(self.metadata, "mth5_type"):
            self._add_base_attributes()

        # set summary attributes
        self.logger.debug(
            "Metadata class for {0} is {1}".format(
                self._class_name, type(self.metadata)
            )
        )

        # if the input data set already has filled attributes, namely if the
        # channel data already exists then read them in with our writing back
        if "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.metadata.from_dict(
                {self.hdf5_dataset.attrs["mth5_type"]: self.hdf5_dataset.attrs}
            )

        # if metadata is input, make sure that its the same class type amd write
        # to the hdf5 dataset
        if dataset_metadata is not None:
            if not isinstance(dataset_metadata, type(self.metadata)):
                msg = "metadata must be type metadata.{0} not {1}".format(
                    self._class_name, type(dataset_metadata)
                )
                self.logger.error(msg)
                raise MTH5Error(msg)

            # load from dict because of the extra attributes for MTH5
            self.metadata.from_dict(dataset_metadata.to_dict())
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name

            # write out metadata to make sure that its in the file.
            self.write_metadata()

        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.write_metadata()

        # if any other keywords
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _add_base_attributes(self):
        # add 2 attributes that will help with querying
        # 1) the metadata class name
        self.metadata.add_base_attribute(
            "mth5_type",
            self._class_name,
            {
                "type": str,
                "required": True,
                "style": "free form",
                "description": "type of group",
                "units": None,
                "options": [],
                "alias": [],
                "example": "group_name",
            },
        )

        # 2) the HDF5 reference that can be used instead of paths
        self.metadata.add_base_attribute(
            "hdf5_reference",
            self.hdf5_dataset.ref,
            {
                "type": "h5py_reference",
                "required": True,
                "style": "free form",
                "description": "hdf5 internal reference",
                "units": None,
                "options": [],
                "alias": [],
                "example": "<HDF5 Group Reference>",
            },
        )

    def __str__(self):
        try:
            lines = ["Channel {0}:".format(self._class_name)]
            lines.append("-" * (len(lines[0]) + 2))
            info_str = "\t{0:<18}{1}"
            lines.append(info_str.format(
                "component:", self.metadata.component))
            lines.append(info_str.format("data type:", self.metadata.type))
            lines.append(info_str.format(
                "data format:", self.hdf5_dataset.dtype))
            lines.append(info_str.format(
                "data shape:", self.hdf5_dataset.shape))
            lines.append(info_str.format(
                "start:", self.metadata.time_period.start))
            lines.append(info_str.format(
                "end:", self.metadata.time_period.end))
            lines.append(info_str.format(
                "sample rate:", self.metadata.sample_rate))
            return "\n".join(lines)
        except ValueError:
            return "MTH5 file is closed and cannot be accessed."

    def __repr__(self):
        return self.__str__()

    @property
    def _class_name(self):
        return self.__class__.__name__.split("Dataset")[0]

    @property
    def run_group(self):
        """ shortcut to run group """
        return RunGroup(self.hdf5_dataset.parent)

    @property
    def station_group(self):
        """ shortcut to station group """
        return StationGroup(self.hdf5_dataset.parent.parent)

    @property
    def master_station_group(self):
        """ shortcut to master station group """
        return MasterStationGroup(self.hdf5_dataset.parent.parent.parent)

    @property
    def start(self):
        return self.metadata.time_period._start_dt

    @start.setter
    def start(self, value):
        """ set start time and validate through metadata validator """
        if isinstance(value, MTime):
            self.metadata.time_period.start = value.iso_str
        else:
            self.metadata.time_period.start = value

    @property
    def end(self):
        """ return end time based on the data"""
        return self.start + (self.n_samples / self.sample_rate)

    @property
    def sample_rate(self):
        return self.metadata.sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        """ set sample rate through metadata validator """
        self.metadata.sample_rate = value

    @property
    def n_samples(self):
        return self.hdf5_dataset.size

    def read_metadata(self):
        """
        Read metadata from the HDF5 file into the metadata container, that
        way it can be validated.

        """

        self.metadata.from_dict({self._class_name: self.hdf5_dataset.attrs})

    def write_metadata(self):
        """
        Write metadata from the metadata container to the HDF5 attrs
        dictionary.

        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            value = to_numpy_type(value)
            self.logger.debug("wrote metadata {0} = {1}".format(key, value))
            self.hdf5_dataset.attrs.create(key, value)

    def replace_dataset(self, new_data_array):
        """
        replace the entire dataset with a new one, nothing left behind

        :param new_data_array: new data array shape (npts, )
        :type new_data_array: :class:`numpy.ndarray`

        """
        if not isinstance(new_data_array, np.ndarray):
            try:
                new_data_array = np.array(new_data_array)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_data_array)}"
                self.logger.exception(msg)
                raise TypeError(msg)

        if new_data_array.shape != self.hdf5_dataset.shape:
            msg = (
                f"resizing dataset {self.hdf5_dataset.name} "
                + f"from {self.hdf5_dataset.shape} to "
                + f"{new_data_array.shape}"
            )
            self.logger.debug(msg)
            self.hdf5_dataset.resize(new_data_array.shape)
        self.hdf5_dataset[...] = new_data_array
        self.logger.debug(f"replacing {self.hdf5_dataset.name}")

    def extend_dataset(
        self,
        new_data_array,
        start_time,
        sample_rate,
        fill=None,
        max_gap_seconds=1,
        fill_window=10,
    ):
        """
        Append data according to how the start time aligns with existing
        data.  If the start time is before existing start time the data is
        prepended, similarly if the start time is near the end data will be
        appended.  

        If the start time is within the existing time range, existing data 
        will be replace with the new data.

        If there is a gap between start or end time of the new data with
        the existing data you can either fill the data with a constant value
        or an error will be raise depending on the value of fill.

        :param new_data_array: new data array with shape (npts, )
        :type new_data_array: :class:`numpy.ndarray`
        :param start_time: start time of the new data array in UTC
        :type start_time: string or :class:`mth5.utils.mttime.MTime`
        :param sample_rate: Sample rate of the new data array, must match 
                            existing sample rate
        :type sample_rate: float
        :param fill: If there is a data gap how do you want to fill the gap
            * None: will raise an  :class:`mth5.utils.exceptions.MTH5Error`
            * 'mean': will fill with the mean of each data set within
            the fill window
            * 'median': will fill with the median of each data set 
            within the fill window
            * value: can be an integer or float to fill the gap
            * 'nan': will fill the gap with NaN
        :type fill: string, None, float, integer
        :param max_gap_seconds: sets a maximum number of seconds the gap can
                                be.  Anything over this number will raise 
                                a :class:`mth5.utils.exceptions.MTH5Error`.
        :type max_gap_seconds: float or integer
        :param fill_window: number of points from the end of each data set
                            to estimate fill value from.
        :type fill_window: integer

        :raises: :class:`mth5.utils.excptions.MTH5Error` if sample rate is 
                 not the same, or fill value is not understood, 

        Append Example
        ---------------

        :rubric:

        >>> ex = mth5_obj.get_channel('MT001', 'MT001a', 'Ex')
        >>> ex.n_samples
        4096
        >>> ex.end
        2015-01-08T19:32:09.500000+00:00
        >>> t = timeseries.ChannelTS('electric',
        ...                     data=2*np.cos(4 * np.pi * .05 * \
        ...                                   np.linspace(0,4096l num=4096) *
        ...                                   .01),
        ...                     channel_metadata={'electric':{
        ...                        'component': 'ex',
        ...                        'sample_rate': 8, 
        ...                        'time_period.start':(ex.end+(1)).iso_str}}) 
        >>> ex.extend_dataset(t.ts, t.start, t.sample_rate, fill='median',
        ...                   max_gap_seconds=2)
        2020-07-02T18:02:47 - mth5.groups.Electric.extend_dataset - INFO -
        filling data gap with 1.0385180759767025
        >>> ex.n_samples
        8200
        >>> ex.end
        2015-01-08T19:40:42.500000+00:00

        """
        fw = fill_window
        # check input parameters
        if sample_rate != self.sample_rate:
            msg = (
                "new data must have the same sampling rate as existing data.\n"
                + f"\tnew sample rate =      {sample_rate}\n"
                + f"\texisting sample rate = {self.sample_rate}"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)

        if not isinstance(new_data_array, np.ndarray):
            try:
                new_data_array = np.array(new_data_array)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_data_array)}"
                self.logger.exception(msg)
                raise TypeError(msg)

        if not isinstance(start_time, MTime):
            start_time = MTime(start_time)

        # get end time will need later
        end_time = start_time + (new_data_array.size / sample_rate)

        # check start time
        start_t_diff = self._get_diff_new_array_start(start_time)
        end_t_diff = self._get_diff_new_array_end(end_time)

        self.logger.debug("Extending data.")
        self.logger.debug(f"Existing start time {self.start}")
        self.logger.debug(f"New start time      {start_time}")
        self.logger.debug(f"Existing end time   {self.end}")
        self.logger.debug(f"New end time        {end_time}")

        # prepend data
        if start_t_diff < 0:
            self.logger.info("Prepending: ")
            self.logger.info(
                f"new start time {start_time} is before existing {self.start}"
            )
            if end_time.iso_no_tz not in self.time_index:
                gap = abs(end_time - self.start)
                if gap > 0:
                    if gap > max_gap_seconds:
                        msg = (
                            f"Time gap of {gap} seconds "
                            + f"is more than max_gap_seconds = {max_gap_seconds}."
                            + " Consider making a new run."
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)

                    if fill is None:
                        msg = (
                            f"A time gap of {gap} seconds is found "
                            + "between new and existing data sets. \n"
                            + f"\tnew end time:        {end_time}\n"
                            + f"\texisting start time: {self.start}"
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)

                    # set new start time
                    old_slice = self.time_slice(self.start, end_time=self.end)
                    old_start = self.start.copy()
                    self.start = start_time

                    # resize the existing data to make room for new data
                    self.hdf5_dataset.resize(
                        (
                            int(
                                new_data_array.size
                                + self.hdf5_dataset.size
                                + gap * sample_rate
                            ),
                        )
                    )

                    # fill based on time, refill existing data first
                    self.hdf5_dataset[
                        self.get_index_from_time(old_start):
                    ] = old_slice.ts.values
                    self.hdf5_dataset[
                        0: self.get_index_from_time(end_time)
                    ] = new_data_array

                    if fill == "mean":
                        fill_value = np.mean(
                            np.array(
                                [
                                    new_data_array[-fw:].mean(),
                                    float(old_slice.ts[0:fw].mean()),
                                ]
                            )
                        )
                    elif fill == "median":
                        fill_value = np.median(
                            np.array(
                                [
                                    np.median(new_data_array[-fw:]),
                                    np.median(old_slice.ts[0:fw]),
                                ]
                            )
                        )
                    elif fill == "nan":
                        fill_value = np.nan

                    elif isinstance(fill, (int, float)):
                        fill_value = fill

                    else:
                        msg = f"fill value {fill} is not understood"
                        self.logger.error(msg)
                        raise MTH5Error(msg)

                    self.logger.info(f"filling data gap with {fill_value}")
                    self.hdf5_dataset[
                        self.get_index_from_time(end_time): self.get_index_from_time(
                            old_start
                        )
                    ] = fill_value
            else:
                self.logger.debug("Prepending data with overlaps.")
                new_size = (self.n_samples +
                            int(abs(start_t_diff) * sample_rate),)
                overlap = abs(end_time - self.start)
                self.logger.warning(
                    f"New data is overlapping by {overlap} s."
                    + " Any overlap will be overwritten."
                )
                # set new start time
                old_slice = self.time_slice(self.start, end_time=self.end)
                old_start = self.start.copy()
                self.start = start_time
                self.logger.debug(
                    f"resizing data set from {self.n_samples} to {new_size}"
                )
                self.hdf5_dataset.resize(new_size)

                # put back the existing data, which any overlapping times
                # will be overwritten
                self.hdf5_dataset[
                    self.get_index_from_time(old_start):
                ] = old_slice.ts.values
                self.hdf5_dataset[
                    0: self.get_index_from_time(end_time)
                ] = new_data_array

        # append data
        elif start_t_diff > 0:
            old_end = self.end.copy()
            if start_time.iso_no_tz not in self.time_index:
                gap = abs(self.end - start_time)
                if gap > 0:
                    if gap > max_gap_seconds:
                        msg = (
                            f"Time gap of {gap} seconds "
                            + f"is more than max_gap_seconds = {max_gap_seconds}."
                            + " Consider making a new run."
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)

                    if fill is None:
                        msg = (
                            f"A time gap of {gap} seconds is found "
                            + "between new and existing data sets. \n"
                            + f"\tnew start time:        {start_time}\n"
                            + f"\texisting end time:     {self.end}"
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)

                    # resize the existing data to make room for new data
                    self.hdf5_dataset.resize(
                        (
                            int(
                                new_data_array.size
                                + self.hdf5_dataset.size
                                + gap * sample_rate
                            ),
                        )
                    )

                    self.hdf5_dataset[
                        self.get_index_from_time(start_time):
                    ] = new_data_array
                    old_index = self.get_index_from_time(old_end)
                    if fill == "mean":
                        fill_value = np.mean(
                            np.array(
                                [
                                    new_data_array[0:fw].mean(),
                                    np.mean(
                                        self.hdf5_dataset[old_index - fw:]),
                                ]
                            )
                        )
                    elif fill == "median":
                        fill_value = np.median(
                            np.array(
                                [
                                    np.median(new_data_array[0:fw]),
                                    np.median(
                                        self.hdf5_dataset[old_index - fw:]),
                                ]
                            )
                        )
                    elif fill == "nan":
                        fill_value = np.nan

                    elif isinstance(fill, (int, float)):
                        fill_value = fill

                    else:
                        msg = f"fill value {fill} is not understood"
                        self.logger.error(msg)
                        raise MTH5Error(msg)

                    self.logger.info(f"filling data gap with {fill_value}")
                    self.hdf5_dataset[
                        self.get_index_from_time(old_end): self.get_index_from_time(
                            start_time
                        )
                    ] = fill_value

            else:
                # if the new data fits within the extisting time span
                if end_t_diff < 0:
                    self.logger.debug(
                        "New data fits within existing time span"
                        + " all data in the window : "
                        f"{start_time} -- {end_time} " + "will be overwritten."
                    )
                    self.hdf5_dataset[
                        self.get_index_from_time(start_time): self.get_index_from_time(
                            end_time
                        )
                    ] = new_data_array
                else:
                    self.logger.debug("Appending data with overlaps.")
                    new_size = (self.n_samples +
                                int(abs(start_t_diff) * sample_rate),)
                    overlap = abs(self.end - start_time)
                    self.logger.warning(
                        f"New data is overlapping by {overlap} s."
                        + " Any overlap will be overwritten."
                    )

                    self.logger.debug(
                        f"resizing data set from {self.n_samples} to {new_size}"
                    )
                    self.hdf5_dataset.resize(new_size)

                    # put back the existing data, which any overlapping times
                    # will be overwritten
                    self.hdf5_dataset[
                        self.get_index_from_time(start_time):
                    ] = new_data_array

    def to_channel_ts(self):
        """
        :return: a Timeseries with the appropriate time index and metadata
        :rtype: :class:`mth5.timeseries.ChannelTS`

        loads from memory (nearly half the size of xarray alone, not sure why)

        """

        return ChannelTS(
            self.metadata.type,
            data=self.hdf5_dataset[()],
            channel_metadata=self.metadata,
            run_metadata=self.run_group.metadata,
            station_metadata=self.station_group.metadata,
        )

    def to_xarray(self):
        """
        :return: an xarray DataArray with appropriate metadata and the 
                 appropriate time index.
        :rtype: :class:`xarray.DataArray`

        .. note:: that metadta will not be validated if changed in an xarray.

        loads from memory
        """

        return xr.DataArray(
            self.hdf5_dataset[()],
            coords=[("time", self.time_index)],
            attrs=self.metadata.to_dict(single=True),
        )

    def to_dataframe(self):
        """

        :return: a dataframe where data is stored in the 'data' column and
                 attributes are stored in the experimental attrs attribute 
        :rtype: :class:`pandas.DataFrame`

        .. note:: that metadta will not be validated if changed in an xarray.

        loads into RAM
        """

        df = pd.DataFrame(
            {"data": self.hdf5_dataset[()]}, index=self.time_index)
        df.attrs.update(self.metadata.to_dict(single=True))

        return df

    def to_numpy(self):
        """
        :return: a numpy structured array with 2 columns (time, channel_data)
        :rtype: :class:`numpy.core.records`
        .. data:: data is a builtin to numpy and cannot be used as a name

        loads into RAM

        """

        return np.core.records.fromarrays(
            [self.time_index.to_numpy(), self.hdf5_dataset[()]],
            names="time,channel_data",
        )

    def from_channel_ts(
        self,
        channel_ts_obj,
        how="replace",
        fill=None,
        max_gap_seconds=1,
        fill_window=10,
    ):
        """
        fill data set from a :class:`mth5.timeseries.ChannelTS` object.

        Will check for time alignement, and metadata.

        :param channel_ts_obj: time series object
        :type channel_ts_obj: :class:`mth5.timeseries.ChannelTS`
        :param how: how the new array will be input to the existing dataset:

            - 'replace' -> replace the entire dataset nothing is left over.
            - 'extend' -> add onto the existing dataset, any  overlapping
              values will be rewritten, if there are gaps between data sets 
              those will be handled depending on the value of fill.

         :param fill: If there is a data gap how do you want to fill the gap:

            - None -> will raise an :class:`mth5.utils.exceptions.MTH5Error`
            - 'mean'-> will fill with the mean of each data set within
              the fill window
            - 'median' -> will fill with the median of each data set 
              within the fill window
            - value -> can be an integer or float to fill the gap
            - 'nan' -> will fill the gap with NaN

        :type fill: string, None, float, integer
        :param max_gap_seconds: sets a maximum number of seconds the gap can
                                be.  Anything over this number will raise 
                                a :class:`mth5.utils.exceptions.MTH5Error`.

        :type max_gap_seconds: float or integer
        :param fill_window: number of points from the end of each data set
                            to estimate fill value from.

        :type fill_window: integer

        """

        if not isinstance(channel_ts_obj, ChannelTS):
            msg = f"Input must be a ChannelTS object not {type(channel_ts_obj)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if how == "replace":
            self.metadata = channel_ts_obj.metadata
            self.replace_dataset(channel_ts_obj.ts.values)
            self.write_metadata()

        elif how == "extend":
            self.extend_dataset(
                channel_ts_obj.ts.values,
                channel_ts_obj.start,
                channel_ts_obj.sample_rate,
                fill=fill,
            )

            # TODO need to check on metadata.

    def from_xarray(
        self, data_array, how="replace", fill=None, max_gap_seconds=1, fill_window=10
    ):
        """
        fill data set from a :class:`xarray.DataArray` object.

        Will check for time alignement, and metadata.

        :param data_array_obj: Xarray data array
        :type channel_ts_obj: :class:`xarray.DataArray`
        :param how: how the new array will be input to the existing dataset:

            - 'replace' -> replace the entire dataset nothing is left over.
            - 'extend' -> add onto the existing dataset, any  overlapping
            values will be rewritten, if there are gaps between data sets 
            those will be handled depending on the value of fill.

         :param fill: If there is a data gap how do you want to fill the gap:

            - None -> will raise an :class:`mth5.utils.exceptions.MTH5Error`
            - 'mean'-> will fill with the mean of each data set within
               the fill window
            - 'median' -> will fill with the median of each data set 
               within the fill window
            - value -> can be an integer or float to fill the gap
            - 'nan' -> will fill the gap with NaN

        :type fill: string, None, float, integer
        :param max_gap_seconds: sets a maximum number of seconds the gap can
                                be.  Anything over this number will raise 
                                a :class:`mth5.utils.exceptions.MTH5Error`.

        :type max_gap_seconds: float or integer
        :param fill_window: number of points from the end of each data set
                            to estimate fill value from.

        :type fill_window: integer

        """

        if not isinstance(data_array, xr.DataArray):
            msg = f"Input must be a xarray.DataArray object not {type(data_array)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if how == "replace":
            self.metadata.from_dict(
                {self.metadata._class_name: data_array.attrs})
            self.replace_dataset(data_array.values)
            self.write_metadata()

        elif how == "extend":
            self.extend_dataset(
                data_array.values,
                data_array.coords.indexes["time"][0].isoformat(),
                1e9 / data_array.coords.indexes["time"][0].freq.nanos,
                fill=fill,
            )

        # TODO need to check on metadata.

    def _get_diff_new_array_start(self, start_time):
        """
        Make sure the new array has the same start time if not return the 
        time difference

        :param start_time: start time of the new array
        :type start_time: string, int or :class:`mth5.utils.MTime`
        :return: time difference in seconds as new start time minus old.
            *  A positive number means new start time is later than old
               start time.
            * A negative number means the new start time is earlier than
              the old start time.
        :rtype: float

        """
        if not isinstance(start_time, MTime):
            start_time = MTime(start_time)

        t_diff = 0
        if start_time != self.start:
            t_diff = start_time - self.start

        return t_diff

    def _get_diff_new_array_end(self, end_time):
        """
        Make sure the new array has the same end time if not return the 
        time difference

        :param end_time: end time of the new array
        :type end_time: string, int or :class:`mth5.utils.MTime`
        :return: time difference in seconds as new end time minus old.
            *  A positive number means new end time is later than old
               end time.
            * A negative number means the new end time is earlier than
              the old end time.
        :rtype: float

        """
        if not isinstance(end_time, MTime):
            end_time = MTime(end_time)

        t_diff = 0
        if end_time != self.end:
            t_diff = end_time - self.end

        return t_diff

    @property
    def time_index(self):
        """

        :return: time index given parameters in metadata
        :rtype: :class:`pandas.DatetimeIndex`

        """
        dt_freq = "{0:.0f}N".format(1.0e9 / (self.sample_rate))
        return pd.date_range(
            start=self.start.iso_no_tz, periods=self.n_samples, freq=dt_freq
        )

    @property
    def table_entry(self):
        """
        Creat a table entry to put into the run summary table.
        """

        return np.array(
            [
                (
                    self.metadata.component,
                    self.metadata.time_period.start,
                    self.metadata.time_period.end,
                    self.hdf5_dataset.size,
                    self.metadata.type,
                    self.metadata.units,
                    self.hdf5_dataset.ref,
                )
            ],
            dtype=np.dtype(
                [
                    ("component", "S20"),
                    ("start", "S32"),
                    ("end", "S32"),
                    ("n_samples", np.int),
                    ("measurement_type", "S12"),
                    ("units", "S25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

    @property
    def channel_entry(self):
        """
        channel entry that will go into a full channel summary of the entire survey

        """
        return np.array(
            [
                (
                    self.station_group.metadata.id,
                    self.run_group.metadata.id,
                    self.station_group.metadata.location.latitude,
                    self.station_group.metadata.location.longitude,
                    self.station_group.metadata.location.elevation,
                    self.metadata.component,
                    self.metadata.time_period.start,
                    self.metadata.time_period.end,
                    self.hdf5_dataset.size,
                    self.metadata.sample_rate,
                    self.metadata.type,
                    self.metadata.measurement_azimuth,
                    self.metadata.measurement_tilt,
                    self.metadata.units,
                    self.hdf5_dataset.ref,
                )
            ],
            dtype=np.dtype(
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
                    ("sample_rate", np.float),
                    ("measurement_type", "S12"),
                    ("azimuth", np.float),
                    ("tilt", np.float),
                    ("units", "S25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

    def time_slice(
        self, start_time, end_time=None, n_samples=None, return_type="channel_ts"
    ):
        """
        Get a time slice from the channel and return the appropriate type

            * numpy array with metadata
            * pandas.Dataframe with metadata
            * xarray.DataFrame with metadata
            * :class:`mth5.timeseries.ChannelTS` 'default'
            * dask.DataFrame with metadata 'not yet'

        :param start_time: start time of the slice
        :type start_time: string or :class:`mth5.utils.mttime.MTime`
        :param end_time: end time of the slice
        :type end_time: string or :class:`mth5.utils.mttime.MTime`, optional
        :param n_samples: number of samples to read in
        :type n_samples: integer, optional
        :return: the correct container for the time series.
        :rtype: [ :class:`xarray.DataArray` | :class:`pandas.DataFrame` |
                 :class:`mth5.timeseries.ChannelTS` | :class:`numpy.ndarray` ]
        :raises: ValueError if both end_time and n_samples are None or given.

        :Example with number of samples:

        .. code-block:: 

            >>> ex = mth5_obj.get_channel('FL001', 'FL001a', 'Ex')
            >>> ex_slice = ex.time_slice("2015-01-08T19:49:15", n_samples=4096)
            >>> ex_slice
            <xarray.DataArray (time: 4096)>
            array([0.93115046, 0.14233688, 0.87917119, ..., 0.26073634, 0.7137319 ,
                   0.88154395])
            Coordinates:
              * time     (time) datetime64[ns] 2015-01-08T19:49:15 ... 2015-01-08T19:57:46.875000
            Attributes:
                ac.end:                      None
                ac.start:                    None
                ...

            >>> type(ex_slice)
            mth5.timeseries.ChannelTS

            # plot the time series
            >>> ex_slice.ts.plot()

        :Example with start and end time:

        >>> ex_slice = ex.time_slice("2015-01-08T19:49:15",
        ...                          end_time="2015-01-09T19:49:15")

        :Raises Example:

        >>> ex_slice = ex.time_slice("2015-01-08T19:49:15",
        ...                          end_time="2015-01-09T19:49:15",
        ...                          n_samples=4096)
        ValueError: Must input either end_time or n_samples, not both.

        """

        if not isinstance(start_time, MTime):
            start_time = MTime(start_time)

        if end_time is not None:
            if not isinstance(end_time, MTime):
                end_time = MTime(end_time)

        if n_samples is not None:
            n_samples = int(n_samples)

        if n_samples is None and end_time is None:
            msg = "Must input either end_time or n_samples."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_samples is not None and end_time is not None:
            msg = "Must input either end_time or n_samples, not both."
            self.logger.error(msg)
            raise ValueError(msg)

        # if end time is given
        if end_time is not None and n_samples is None:
            start_index = self.get_index_from_time(start_time)
            end_index = self.get_index_from_time(end_time)
            npts = int(end_index - start_index)

        # if n_samples are given
        elif end_time is None and n_samples is not None:
            start_index = self.get_index_from_time(start_time)
            end_index = start_index + n_samples
            npts = n_samples

        if npts > self.hdf5_dataset.size:
            msg = (
                "Requested slice is larger than data.  "
                + f"Slice length = {npts}, data length = {self.hdf5_dataset.shape}"
                + " Check start and end times."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # create a regional reference that can be used, need +1 to be inclusive
        regional_ref = self.hdf5_dataset.regionref[start_index: end_index + 1]

        dt_freq = "{0:.0f}N".format(1.0e9 / (self.metadata.sample_rate))

        dt_index = pd.date_range(
            start=start_time.iso_str.split("+", 1)[0],
            periods=npts,
            freq=dt_freq,
            closed=None,
        )

        self.logger.debug(
            f"Slicing from {dt_index[0]} (index={start_index}) to"
            + f"{dt_index[-1]} (index={end_index})"
        )

        self.logger.debug(
            f"Slice start={dt_index[0]}, stop={dt_index[-1]}, "
            + f"step={dt_freq}, n_samples={npts}"
        )

        meta_dict = self.metadata.to_dict()[self.metadata._class_name]
        meta_dict["time_period.start"] = dt_index[0].isoformat()
        meta_dict["time_period.end"] = dt_index[-1].isoformat()

        data = None
        if return_type == "xarray":
            # need the +1 to be inclusive of the last point
            data = xr.DataArray(
                self.hdf5_dataset[regional_ref], coords=[("time", dt_index)]
            )
            data.attrs.update(meta_dict)

        elif return_type == "pandas":
            data = pd.DataFrame(
                {"data": self.hdf5_dataset[regional_ref]}, index=dt_index
            )
            data.attrs.update(meta_dict)

        elif return_type == "numpy":
            data = self.hdf5_dataset[regional_ref]

        elif return_type == "channel_ts":
            data = ChannelTS(
                self.metadata.type,
                data=self.hdf5_dataset[regional_ref],
                channel_metadata={self.metadata.type: meta_dict},
            )
        else:
            msg = "return_type not understood, must be [ pandas | numpy | channel_ts ]"
            self.logger.error(msg)
            raise ValueError(msg)

        return data

    def get_index_from_time(self, given_time):
        """
        get the appropriate index for a given time.

        :param given_time: DESCRIPTION
        :type given_time: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(given_time, MTime):
            given_time = MTime(given_time)

        index = (
            given_time - self.metadata.time_period.start
        ) * self.metadata.sample_rate

        return int(round(index))


@inherit_doc_string
class ElectricDataset(ChannelDataset):
    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)


@inherit_doc_string
class MagneticDataset(ChannelDataset):
    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)


@inherit_doc_string
class AuxiliaryDataset(ChannelDataset):
    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)
