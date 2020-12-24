# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:18:29 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import h5py

from mth5.groups import BaseGroup, RunGroup
from mth5.utils.exceptions import MTH5Error
from mth5.utils.mttime import MTime
from mth5 import metadata
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
            self.logger.debug(f"no summary entries with sample rate {sample_rate}")
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
            msg = f"run {run_name} already exists, " + "returning existing group."
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
                f"{run_name} does not exist, " + "check groups_list for existing names"
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
                f"{run_name} does not exist, " + "check station_list for existing names"
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
        self.metadata.time_period.end = max(self.summary_table.array["end"]).decode()
        self.metadata.channels_recorded = ",".join(
            list(
                set(
                    ",".join(
                        list(self.summary_table.array["components"].astype(np.unicode_))
                    ).split(",")
                )
            )
        )

        self.write_metadata()

