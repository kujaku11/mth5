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

import h5py
import numpy as np
import pandas as pd

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime

from mth5.groups import (
    BaseGroup,
    RunGroup,
    TransferFunctionsGroup,
    MasterFCGroup,
)
from mth5.helpers import from_numpy_type
from mth5.utils.exceptions import MTH5Error

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

    >>> stations.station_summary

    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

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

        return self._add_group(
            station_name, StationGroup, station_metadata, match="id"
        )

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
        return self._get_group(station_name, StationGroup)

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

        self._remove_group(station_name)


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
        self._default_subgroup_names = [
            "Transfer_Functions",
            "Fourier_Coefficients",
        ]
        super().__init__(group, group_metadata=station_metadata, **kwargs)

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

    @property
    def survey_metadata(self):
        """survey metadata"""

        meta_dict = dict(self.hdf5_group.parent.parent.attrs)
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        survey_metadata = metadata.Survey()
        survey_metadata.from_dict({"survey": meta_dict})
        survey_metadata.add_station(self.metadata)
        return survey_metadata

    @BaseGroup.metadata.getter
    def metadata(self):
        """Overwrite get metadata to include run information in the station"""

        self._metadata.runs = []
        for key in self.groups_list:
            if key.lower() in [
                name.lower() for name in self._default_subgroup_names
            ]:
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
                        group.ref,
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

        run_summary = self.run_summary.copy()
        if run_summary.size < 1:
            return None
        sr_find = run_summary[
            (run_summary.sample_rate == sample_rate)
            & (run_summary.start == start)
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

        return self._add_group(
            run_name, RunGroup, group_metadata=run_metadata, match="id"
        )

    def get_run(self, run_name):
        """
        get a run from run name

        :param run_name: existing run name
        :type run_name: string
        :return: Run object
        :rtype: :class:`mth5.mth5_groups.RunGroup`

        >>> existing_run = station.get_run('MT001')

        """

        return self._get_group(run_name, RunGroup)

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

        self._remove_group(run_name)

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
