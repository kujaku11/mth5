# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:59:03 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import inspect

import h5py
import numpy as np
import pandas as pd
from mt_metadata import timeseries as metadata

from mth5 import CHUNK_SIZE
from mth5.groups import (
    AuxiliaryDataset,
    BaseGroup,
    ChannelDataset,
    ElectricDataset,
    MagneticDataset,
)
from mth5.helpers import from_numpy_type, to_numpy_type, validate_name
from mth5.timeseries import ChannelTS, RunTS
from mth5.utils.exceptions import MTH5Error


meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
# =============================================================================


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

    >>> station.groups_list
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
        # Channel metadata cache to share objects between add_channel and metadata property
        self._channel_metadata_cache = {}

    @property
    def station_metadata(self):
        """station metadata"""

        meta_dict = dict(self.hdf5_group.parent.attrs)
        meta_dict["run_list"] = [self.metadata.id]
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        station_metadata = metadata.Station()
        station_metadata.from_dict({"station": meta_dict})
        station_metadata.add_run(self.metadata)

        return station_metadata

    @property
    def survey_metadata(self):
        """survey metadata"""

        meta_dict = dict(self.hdf5_group.parent.parent.parent.attrs)
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        survey_metadata = metadata.Survey()
        survey_metadata.from_dict({"survey": meta_dict})
        survey_metadata.add_station(self.station_metadata)
        return survey_metadata

    @BaseGroup.metadata.getter
    def metadata(self):
        """Overwrite get metadata to include channel information in the runs"""

        if not self._has_read_metadata:
            self.read_metadata()
            self._has_read_metadata = True

        # Only rebuild channels if they haven't been built yet or if the group list has changed
        if not self._metadata.channels or len(self._metadata.channels) != len(
            self.groups_list
        ):
            # Get current channel names from the groups and existing channels
            current_group_names = set(self.groups_list)
            existing_channel_names = set(ch.component for ch in self._metadata.channels)

            # Only rebuild if there's actually a difference in the channel sets
            if current_group_names != existing_channel_names:
                # Clear and rebuild the channels list
                self._metadata.channels = []
                for ch in self.groups_list:
                    # Check if we have cached metadata for this channel
                    if ch in self._channel_metadata_cache:
                        # Reuse cached metadata to prevent duplicate processing
                        cached_metadata = self._channel_metadata_cache[ch]
                        self._metadata.add_channel(cached_metadata)
                    else:
                        # Create new metadata if not cached
                        meta_dict = dict(self.hdf5_group[ch].attrs)
                        for key, value in meta_dict.items():
                            meta_dict[key] = from_numpy_type(value)
                        ch_metadata = meta_classes[meta_dict["type"].capitalize()]()
                        ch_metadata.from_dict(meta_dict)
                        # Cache the metadata for future use
                        self._channel_metadata_cache[ch] = ch_metadata
                        self._metadata.add_channel(ch_metadata)
            # If channel sets are identical, skip rebuilding to prevent duplicates
        self._metadata.hdf5_reference = self.hdf5_group.ref
        return self._metadata

    @property
    def channel_summary(self):
        """

         summary of channels in run
        :return: DESCRIPTION
        :rtype: TYPE

        """

        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["type"]
                if ch_type in ["electric", "magnetic", "auxiliary"]:
                    ch_list.append(
                        (
                            group.attrs["component"],
                            group.attrs["time_period.start"].split("+")[0],
                            group.attrs["time_period.end"].split("+")[0],
                            group.size,
                            group.attrs["sample_rate"],
                            group.attrs["type"],
                            group.attrs["units"],
                            group.ref,
                        )
                    )
            except KeyError:
                pass
        ch_summary = np.array(
            ch_list,
            dtype=np.dtype(
                [
                    ("component", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("n_samples", int),
                    ("sample_rate", float),
                    ("measurement_type", "U12"),
                    ("units", "U25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(ch_summary)

    def write_metadata(self):
        """
        Overwrite Base.write_metadata to include updating table entry
        Write HDF5 metadata from metadata object.

        """

        for key, value in self.metadata.to_dict(single=True).items():
            value = to_numpy_type(value)
            self.hdf5_group.attrs.create(key, value)

    def add_channel(
        self,
        channel_name,
        channel_type,
        data,
        channel_dtype="int32",
        shape=None,
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
        :param shape: Set the shape of the array, uses the data if input. Can
         be useful if you are setting up the file. This will set the size of
         the dataset, whereas `max_shape` sets the max shape which ends up in
         different memory size. If you are not sure about the size of the array
         suggest using `max_shape`, but if you already know and want to start
         with an array of 0's use `shape`, defaults to None
        :type shape: tuple, optional
        :param max_shape: Absolute max shape of the data to be stored, this
         means the data can be extended up to the given shape. If None is
         given then the data can be extended infinitely (or until memory runs
         out), defaults to (None,)
        :type max_shape: tuple, optional
        :param chunks: Use chunked storage, defaults to True
        :type chunks: bool, optional
        :param channel_metadata: metadata container, defaults to None
        :type channel_metadata: [ :class:`mth5.metadata.Electric` |
                                 :class:`mth5.metadata.Magnetic` |
                                 :class:`mth5.metadata.Auxiliary` ], optional
        :param **kwargs: Key word arguments
        :type **kwargs: dictionary
        :raises MTH5Error: If channel type is not correct
        :return: DESCRIPTION
        :rtype: TYPE


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
        channel_name = validate_name(channel_name.lower())
        estimate_size = (1,)
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
                    dtype=data.dtype,
                    chunks=chunks,
                    maxshape=max_shape,
                    **self.dataset_options,
                )
            # initialize a resizable data array
            # need to set the chunk size to something useful, if the chunk
            # size is 1 this causes performance issues and bloating of the
            # hdf5 file.  Set to 8196 for now.
            else:
                if shape is None:
                    if channel_metadata is not None:
                        # can estimate a size, this will help with allocating
                        # and set the chunk size to a realistic value
                        if (
                            channel_metadata.time_period.start
                            != channel_metadata.time_period.end
                        ):
                            if channel_metadata.sample_rate > 0:
                                estimate_size = (
                                    int(
                                        (
                                            channel_metadata.time_period.end
                                            - channel_metadata.time_period.start
                                        )
                                        * channel_metadata.sample_rate
                                    ),
                                )
                        else:
                            estimate_size = (1,)
                            chunks = CHUNK_SIZE
                    else:
                        estimate_size = (1,)
                        chunks = CHUNK_SIZE
                    if estimate_size[0] > 2**31:
                        estimate_size = (1,)
                        self.logger.warning(
                            "Estimated size is too large. Check start and end "
                            "times, initializing with size (1,)"
                        )
                else:
                    estimate_size = shape
                ## Create the dataset
                channel_group = self.hdf5_group.create_dataset(
                    channel_name,
                    shape=estimate_size,
                    maxshape=max_shape,
                    dtype=channel_dtype,
                    chunks=chunks,
                    **self.dataset_options,
                )
            if channel_metadata:
                if channel_metadata.component != channel_name:
                    self.logger.warning(
                        f"Channel name {channel_name} != "
                        f"channel_metadata.component "
                        f"{channel_metadata.component}, setting to {channel_name}"
                    )
                    channel_metadata.component = channel_name
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
                    f"auxiliary ]. Input was {channel_type}"
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
        except (OSError, RuntimeError, ValueError):
            msg = f"channel {channel_name} already exists, returning existing group."
            self.logger.debug(msg)
            channel_obj = self.get_channel(channel_name)

            if data is not None:
                self.logger.debug(f"Replacing data with new shape {data.shape}")
                channel_obj.replace_dataset(data)

                self.logger.debug("Updating metadata")
                channel_obj.metadata.update(channel_metadata)
                channel_obj.write_metadata()
                self.logger.debug(f"Done with {channel_name}")
        # need to make sure the channel name is passed.
        if channel_obj.metadata.component != channel_name:
            channel_obj.metadata.component = channel_name
            channel_obj.write_metadata()

        # Cache the processed channel metadata to prevent duplicate processing in metadata property
        # Use the channel object's metadata which has already been processed through from_dict
        self._channel_metadata_cache[channel_name] = channel_obj.metadata

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

        >>> run.groups_list
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

        channel_name = validate_name(channel_name.lower())
        try:
            ch_dataset = self.hdf5_group[channel_name]
        except KeyError:
            msg = (
                f"{channel_name} does not exist, check groups_list "
                "for existing names"
            )
            self.logger.debug(msg)
            raise MTH5Error(msg)
        if ch_dataset.attrs["mth5_type"].lower() in ["electric"]:
            ch_metadata = meta_classes["Electric"]()
            ch_metadata.from_dict({"Electric": dict(ch_dataset.attrs)})
            channel = ElectricDataset(
                ch_dataset,
                dataset_metadata=ch_metadata,
                write_metadata=False,
            )
        elif ch_dataset.attrs["mth5_type"].lower() in ["magnetic"]:
            ch_metadata = meta_classes["Magnetic"]()
            ch_metadata.from_dict({"Magnetic": dict(ch_dataset.attrs)})
            channel = MagneticDataset(
                ch_dataset,
                dataset_metadata=ch_metadata,
                write_metadata=False,
            )
        elif ch_dataset.attrs["mth5_type"].lower() in ["auxiliary"]:
            ch_metadata = meta_classes["Auxiliary"]()
            ch_metadata.from_dict({"Auxiliary": dict(ch_dataset.attrs)})
            channel = AuxiliaryDataset(
                ch_dataset,
                dataset_metadata=ch_metadata,
                write_metadata=False,
            )
        else:
            channel = ChannelDataset(ch_dataset)
        channel.read_metadata()

        return channel

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

        channel_name = validate_name(channel_name.lower())

        try:
            del self.hdf5_group[channel_name]
            # Remove from metadata cache if present
            if channel_name in self._channel_metadata_cache:
                del self._channel_metadata_cache[channel_name]
            self.logger.info(
                "Deleting a channel does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{channel_name} does not exist, "
                "check groups_list for existing names"
            )
            self.logger.debug("Error: " + msg)
            raise MTH5Error(msg)

    def has_data(self):
        """
        make sure there is data in the run, values are non-zero or not empty

        :return: True if has data, False if empty or all zeros
        :rtype: bool

        """
        has_data_list = []
        has_data = True
        for channel in self.groups_list:
            if channel in ["summary"]:
                continue
            ch_obj = self.get_channel(channel)
            has_data_list.append(f"{ch_obj.metadata.component}: {ch_obj.has_data()}")
            if not ch_obj.has_data():
                has_data = False

        if not has_data:
            self.logger.info(", ".join(has_data_list))
        return has_data

    def to_runts(self, start=None, end=None, n_samples=None):
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

            if start is not None:
                ts_obj = ch_obj.time_slice(start, end=end, n_samples=n_samples)
            else:
                ts_obj = ch_obj.to_channel_ts()
            ch_list.append(ts_obj)
        return RunTS(
            ch_list,
            run_metadata=self.metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )

    def from_runts(self, run_ts_obj, **kwargs):
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
        self._metadata.update(run_ts_obj.run_metadata)

        channels = []

        for comp in run_ts_obj.channels:
            ch = getattr(run_ts_obj, comp)

            if ch.station_metadata.id is not None:
                if ch.station_metadata.id != self.station_metadata.id:
                    if ch.station_metadata.id not in ["0", None]:
                        self.logger.warning(
                            f"Channel station.id {ch.station_metadata.id} != "
                            f" group station.id {self.station_metadata.id}. "
                            f"Setting to ch.station_metadata.id to {self.station_metadata.id}"
                        )
                        ch.station_metadata.id = self.station_metadata.id
            if ch.run_metadata.id is not None:
                if ch.run_metadata.id != self.metadata.id:
                    if ch.run_metadata.id not in ["0", None]:
                        self.logger.warning(
                            f"Channel run.id {ch.run_metadata.id} != "
                            f" group run.id {self.metadata.id}. "
                            f"Setting to ch.run_metadata.id to {self.metadata.id}"
                        )
                        ch.run_metadata.id = self.metadata.id

            channels.append(self.from_channel_ts(ch))
        self.update_metadata()
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
        ## Need to add in the filters
        if channel_ts_obj.channel_response.filters_list != []:
            from mth5.groups import FiltersGroup

            fg = FiltersGroup(self.hdf5_group.parent.parent.parent["Filters"])
            for ff in channel_ts_obj.channel_response.filters_list:
                fg.add_filter(ff)
        ch_obj = self.add_channel(
            channel_ts_obj.component,
            channel_ts_obj.channel_metadata.type,
            channel_ts_obj.ts,
            channel_metadata=channel_ts_obj.channel_metadata,
        )

        # need to update the channels recorded
        if channel_ts_obj.channel_metadata.type == "electric":
            if self.metadata.channels_recorded_electric is None:
                self.metadata.channels_recorded_electric = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_electric
            ):
                self.metadata.channels_recorded_electric.append(
                    channel_ts_obj.component
                )
        elif channel_ts_obj.channel_metadata.type == "magnetic":
            if self.metadata.channels_recorded_magnetic is None:
                self.metadata.channels_recorded_magnetic = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_magnetic
            ):
                self.metadata.channels_recorded_magnetic.append(
                    channel_ts_obj.component
                )
        elif channel_ts_obj.channel_metadata.type == "auxiliary":
            if self.metadata.channels_recorded_auxiliary is None:
                self.metadata.channels_recorded_auxiliary = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component
                not in self.metadata.channels_recorded_auxiliary
            ):
                self.metadata.channels_recorded_auxiliary.append(
                    channel_ts_obj.component
                )
        return ch_obj

    def update_run_metadata(self):
        """
        Update metadata and table entries to ensure consistency

        :return: DESCRIPTION
        :rtype: TYPE

        """

        raise DeprecationWarning(
            "'update_run_metadata' has been deprecated use 'update_metadata()'"
        )

    def update_metadata(self):
        """
        Update metadata and table entries to ensure consistency

        :return: DESCRIPTION
        :rtype: TYPE

        """
        channel_summary = self.channel_summary.copy()

        self._metadata.time_period.start = channel_summary.start.min().isoformat()
        self._metadata.time_period.end = channel_summary.end.max().isoformat()
        try:
            self._metadata.sample_rate = channel_summary.sample_rate.unique()[0]
        except IndexError:
            msg = "There maybe no channels associated with this run -- setting sample_rate to 0"
            self.logger.critical(msg)
            self._metadata.sample_rate = 0
        self.write_metadata()

    def plot(self, start=None, end=None, n_samples=None):
        """
        Produce a simple matplotlib plot using runts
        """

        runts = self.to_runts(start=start, end=end, n_samples=n_samples)

        return runts.plot()
