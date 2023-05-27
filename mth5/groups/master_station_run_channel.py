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

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime
from mt_metadata.base import Base
from mt_metadata.timeseries.filters import ChannelResponseFilter

from mth5 import CHUNK_SIZE, CHANNEL_DTYPE
from mth5.groups.base import BaseGroup, RunGroup, TransferFunctionGroup
from mth5.groups import FiltersGroup, TransferFunctionGroup
from mth5.groups.fourier_coefficients import MasterFCGroup, FCGroup, FCChannel
from mth5.utils.exceptions import MTH5Error
from mth5.helpers import (
    to_numpy_type,
    from_numpy_type,
    inherit_doc_string,
    validate_name,
)

from mth5.timeseries import ChannelTS, RunTS
from mth5.timeseries.channel_ts import make_dt_coordinates
from mth5.utils.mth5_logger import setup_logger

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
        ch_list = []
        for station in self.groups_list:
            s_group = StationGroup(self.hdf5_group[station])
            for run in s_group.groups_list:
                r_group = RunGroup(s_group.hdf5_group[run])
                for ch in r_group.groups_list:
                    ds_type = r_group.hdf5_group[ch].attrs["mth5_type"]
                    if ds_type.lower() in ["electric"]:
                        ch_dataset = ElectricDataset(r_group.hdf5_group[ch])
                    elif ds_type.lower() in ["magnetic"]:
                        ch_dataset = MagneticDataset(r_group.hdf5_group[ch])
                    elif ds_type.lower() in ["auxiliary"]:
                        ch_dataset = AuxiliaryDataset(r_group.hdf5_group[ch])
                    ch_list.append(ch_dataset.channel_entry)
        ch_list = np.array(ch_list)
        return pd.DataFrame(ch_list.flatten())

    @property
    def station_summary(self):
        """
        Summary of stations in the file

        :return: DESCRIPTION
        :rtype: TYPE

        """
        st_list = []
        for key, group in self.hdf5_group.items():
            entry = {
                "station": key,
                "start": group.attrs["time_period.start"],
                "end": group.attrs["time_period.end"],
                "latitude": group.attrs["location.latitude"],
                "longitude": group.attrs["location.longitude"],
            }
            st_list.append(entry)

        df = pd.DataFrame(st_list)
        df.start = pd.to_datetime(df.start)
        df.end = pd.to_datetime(df.end)

        return df

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
            raise Exception("station name is None, do not know what to name it")
        station_name = validate_name(station_name)
        try:
            station_group = self.hdf5_group.create_group(station_name)
            self.logger.debug("Created group %s", station_group.name)

            if station_metadata is None:
                station_metadata = metadata.Station(id=station_name)
            else:
                if validate_name(station_metadata.id) != station_name:
                    msg = (
                        f"Station group name {station_name} must be same as "
                        + f"station id {station_metadata.id}"
                    )
                    self.logger.error(msg)
                    raise MTH5Error(msg)
            station_obj = StationGroup(
                station_group,
                station_metadata=station_metadata,
                **self.dataset_options,
            )
            station_obj.initialize_group()

            # be sure to add a table entry
            # self.summary_table.add_row(station_obj.table_entry)
        except ValueError:
            msg = "Station %s already exists, returning existing group."
            self.logger.info(msg, station_name)
            station_obj = self.get_station(station_name)
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
        station_name = validate_name(station_name)
        try:
            return StationGroup(
                self.hdf5_group[station_name], **self.dataset_options
            )
        except KeyError:
            msg = (
                f"{station_name} does not exist, "
                + "check station_list for existing names"
            )
            self.logger.debug("Error" + msg)
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

        station_name = validate_name(station_name)
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
            self.logger.debug("Error" + msg)
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

    >>> station.groups_list
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

        self._defaults_summary_keys = [
            "id",
            "start",
            "end",
            "components",
            "measurement_type",
            "sample_rate",
            "hdf5_reference",
            "mth5_type",
        ]

        self._default_subgroup_names = [
            "Transfer_Functions",
            "Fourier_Coefficients",
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
    def master_station_group(self):
        """shortcut to master station group"""
        return MasterStationGroup(self.hdf5_group.parent)

    @property
    def transfer_functions_group(self):
        """Convinience method for /Station/Transfer_Functions"""
        return TransferFunctionsGroup(
            self.hdf5_group["Transfer_Functions"], **self.dataset_options
        )

    @property
    def fourier_coefficients_group(self):
        """Convinience method for /Station/Fourier_Coefficients"""
        return MasterFCGroup(
            self.hdf5_group["Fourier_Coefficients"], **self.dataset_options
        )

    @BaseGroup.metadata.getter
    def metadata(self):
        """Overwrite get metadata to include run information in the station"""

        self._metadata.runs = []
        for key in self.groups_list:
            if key.lower() == "transfer_functions":
                continue
            try:
                key_group = self.get_run(key)
                self._metadata.runs.append(key_group.metadata)
            except MTH5Error:
                self.logger.warning(f"Could not find run {key}")
        return self._metadata

    @property
    def name(self):
        return self.metadata.id

    @name.setter
    def name(self, name):
        self.metadata.id = name

    @property
    def run_summary(self):
        """
        Summary of runs in the station

        :return: DESCRIPTION
        :rtype: TYPE

        """

        run_list = []
        for key, group in self.hdf5_group.items():
            if group.attrs["mth5_type"].lower() in ["run"]:
                comps = ",".join(
                    [
                        ii.decode()
                        for ii in group.attrs[
                            "channels_recorded_auxiliary"
                        ].tolist()
                        + group.attrs["channels_recorded_electric"].tolist()
                        + group.attrs["channels_recorded_magnetic"].tolist()
                    ]
                )
                run_list.append(
                    (
                        group.attrs["id"],
                        group.attrs["time_period.start"].split("+")[0],
                        group.attrs["time_period.end"].split("+")[0],
                        comps,
                        group.attrs["data_type"],
                        group.attrs["sample_rate"],
                        self.hdf5_group.ref,
                    )
                )
        run_summary = np.array(
            run_list,
            dtype=np.dtype(
                [
                    ("id", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("components", "U100"),
                    ("measurement_type", "U12"),
                    ("sample_rate", float),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(run_summary)

    def make_run_name(self, alphabet=False):
        """
        Make a run name that will be the next alphabet letter extracted from
        the run list.  Expects that all runs are labled as id{a-z}.

        :return: metadata.id + next letter
        :rtype: string

        >>> station.metadata.id = 'MT001'
        >>> station.make_run_name()
        'MT001a'

        """

        run_list = sorted(
            [group[-1:] for group in self.groups_list if self.name in group]
        )

        next_letter = None
        if len(run_list) == 0:
            if alphabet:
                next_letter = "a"
            else:
                next_letter = "001"
        else:
            try:
                next_letter = chr(ord(run_list[-1]) + 1)
            except TypeError:
                try:
                    next_letter = f"{int(run_list[-1]) + 1}"
                except ValueError:
                    self.logger.info("Could not create a new run name")
        return next_letter

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
        if self.run_summary.size < 1:
            return None
        sr_find = self.run_summary[
            (self.run_summary.sample_rate == sample_rate)
            & (self.run_summary.start == start)
        ]
        if sr_find.size < 1:
            return None
        return sr_find

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

        run_name = validate_name(run_name)
        try:
            run_group = self.hdf5_group.create_group(run_name)
            if run_metadata is None:
                run_metadata = metadata.Run(id=run_name)
            elif validate_name(run_metadata.id) != run_name:
                msg = "Run name %s must be the same as run_metadata.id %s"
                self.logger.error(msg, run_name, run_metadata.id)
                raise MTH5Error(msg % (run_name, run_metadata.id))
            run_obj = RunGroup(
                run_group, run_metadata=run_metadata, **self.dataset_options
            )
            run_obj.initialize_group()
        except ValueError:
            msg = "run %s already exists, returning existing group."
            self.logger.info(msg, run_name)
            run_obj = self.get_run(run_name)
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

        run_name = validate_name(run_name)
        try:
            return RunGroup(self.hdf5_group[run_name], **self.dataset_options)
        except KeyError:
            msg = (
                f"{run_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.debug("Error" + msg)
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

        run_name = validate_name(run_name)
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
                f"{run_name} does not exist, "
                + "check station_list for existing names"
            )
            self.logger.debug("Error" + msg)
            raise MTH5Error(msg)

    def update_station_metadata(self):
        """
        Check metadata from the runs and make sure it matches the station metadata

        :return: DESCRIPTION
        :rtype: TYPE

        """

        run_summary = self.run_summary.copy()
        self._metadata.time_period.start = run_summary.start.min().isoformat()
        self._metadata.time_period.end = run_summary.end.max().isoformat()
        self._metadata.channels_recorded = list(
            set(",".join(run_summary.components.to_list()).split(","))
        )

        self.write_metadata()
