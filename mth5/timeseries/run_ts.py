# -*- coding: utf-8 -*-
"""
.. module:: timeseries
   :synopsis: Deal with MT time series

.. todo:: Check the conversion to netcdf.  There are some weird serializations of
lists and arrays that goes on, seems easiest to convert all lists to strings and then
convert them back if read in.


:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT
"""

# ==============================================================================
# Imports
# ==============================================================================
import inspect

import xarray as xr
from matplotlib import pyplot as plt

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime

from mth5.utils.exceptions import MTTSError
from .channel_ts import ChannelTS
from mth5.utils.mth5_logger import setup_logger

from obspy.core import Stream

# =============================================================================
# make a dictionary of available metadata classes
# =============================================================================
meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))


# =============================================================================
# run container
# =============================================================================
class RunTS:
    """
    holds all run ts in one aligned array

    components --> {'ex': ex_xarray, 'ey': ey_xarray}

    """

    def __init__(self, array_list=None, run_metadata=None, station_metadata=None):

        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.run_metadata = metadata.Run()
        self.station_metadata = metadata.Station()
        self._dataset = xr.Dataset()

        # load the arrays first this will write run and station metadata
        if array_list is not None:
            self.dataset = array_list

        # if the use inputs metadata, overwrite all values in the metadata element
        if run_metadata is not None:
            if isinstance(run_metadata, dict):
                # make sure the input dictionary has the correct form
                if "Run" not in list(run_metadata.keys()):
                    run_metadata = {"Run": run_metadata}
                self.run_metadata.from_dict(run_metadata)

            elif isinstance(run_metadata, metadata.Run):
                self.run_metadata.from_dict(run_metadata.to_dict())
            else:
                msg = (
                    "Input metadata must be a dictionary or Run object, "
                    f"not {type(run_metadata)}"
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        # add station metadata, this will be important when propogating a run
        if station_metadata is not None:
            if isinstance(station_metadata, metadata.Station):
                self.station_metadata.from_dict(station_metadata.to_dict())

            elif isinstance(station_metadata, dict):
                if "Station" not in list(station_metadata.keys()):
                    station_metadata = {"Station": station_metadata}
                self.station_metadata.from_dict(station_metadata)

            else:
                msg = "input metadata must be type %s or dict, not %s"
                self.logger.error(
                    msg, type(self.station_metadata), type(station_metadata)
                )
                raise MTTSError(
                    msg % (type(self.station_metadata), type(station_metadata))
                )

    def __str__(self):
        s_list = [
            f"Station:     {self.station_metadata.id}",
            f"Run:         {self.run_metadata.id}",
            f"Start:       {self.start}",
            f"End:         {self.end}",
            f"Sample Rate: {self.sample_rate}",
            f"Components:  {self.channels}",
        ]
        return "\n\t".join(["RunTS Summary:"] + s_list)

    def __repr__(self):
        return self.__str__()

    def _validate_array_list(self, array_list):
        """check to make sure all entries are a :class:`ChannelTS` object"""

        if not isinstance(array_list, (tuple, list)):
            msg = f"array_list must be a list or tuple, not {type(array_list)}"
            self.logger.error(msg)
            raise TypeError(msg)

        valid_list = []
        for index, item in enumerate(array_list):
            if not isinstance(item, (ChannelTS, xr.DataArray)):
                msg = f"array entry {index} must be ChannelTS object not {type(item)}"
                self.logger.error(msg)
                raise TypeError(msg)
            if isinstance(item, ChannelTS):
                valid_list.append(item.to_xarray())

                # if a channelTS is input then it comes with run and station metadata
                # use those first, then the user can update later.
                self.run_metadata.channels.append(item.channel_metadata)
                if index == 0:
                    self.station_metadata.from_dict(item.station_metadata.to_dict())
                    self.run_metadata.from_dict(item.run_metadata.to_dict())
                else:
                    self.station_metadata.update(item.station_metadata, match=["id"])
                    self.run_metadata.update(item.run_metadata, match=["id"])
            else:
                valid_list.append(item)

        # probably should test for sampling rate.
        sr_test = dict([(item.component, (item.sample_rate)) for item in valid_list])

        if len(set([v for k, v in sr_test.items()])) != 1:
            msg = f"sample rates are not all the same {sr_test}"
            self.logger.error(msg)
            raise MTTSError(msg)

        return valid_list

    def __getattr__(self, name):
        # change to look for keys directly and use type to set channel type
        if name in self.dataset.keys():
            return ChannelTS(self.dataset[name].attrs["type"], self.dataset[name])
        else:
            # this is a hack for now until figure out who is calling shape, size
            if name[0] == "_":
                return None
            if name not in ["shape", "size"]:
                try:
                    return super().__getattribute__(name)
                except AttributeError:
                    # elif name not in self.__dict__.keys() and name not in [
                    #     "shape",
                    #     "size",
                    #     "sample_rate",
                    #     "start",
                    #     "end",
                    # ]:
                    msg = f"RunTS has no attribute {name}"
                    self.logger.error(msg)
                    raise NameError(msg)

    @property
    def has_data(self):
        """check to see if there is data"""
        if len(self.channels) > 0:
            return True
        return False

    @property
    def summarize_metadata(self):
        """

        Get a summary of all the metadata

        :return: DESCRIPTION
        :rtype: TYPE

        """
        meta_dict = {}
        for comp in self.dataset.data_vars:
            for mkey, mvalue in self.dataset[comp].attrs.items():
                meta_dict[f"{comp}.{mkey}"] = mvalue

        return meta_dict

    def validate_metadata(self):
        """
        Check to make sure that the metadata matches what is in the data set.

        updates metadata from the data.

        Check the start and end times, channels recorded
        :return: DESCRIPTION
        :rtype: TYPE

        """

        # check sampling rate
        if self.has_data:
            # check start time
            if self.start != self.run_metadata.time_period.start:
                if self.run_metadata.time_period.start != "1980-01-01T00:00:00+00:00":
                    msg = (
                        f"start time of dataset {self.start} does not "
                        f"match metadata start {self.run_metadata.time_period.start} "
                        f"updating metatdata value to {self.start}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.time_period.start = self.start.iso_str

            # check end time
            if self.end != self.run_metadata.time_period.end:
                if self.run_metadata.time_period.end != "1980-01-01T00:00:00+00:00":
                    msg = (
                        f"end time of dataset {self.end} does not "
                        f"match metadata end {self.run_metadata.time_period.end} "
                        f"updating metatdata value to {self.end}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.time_period.end = self.end.iso_str
            if self.sample_rate != self.run_metadata.sample_rate:
                if self.run_metadata.sample_rate is not None:
                    msg = (
                        f"sample rate of dataset {self.sample_rate} does not "
                        f"match metadata sample rate {self.run_metadata.sample_rate} "
                        f"updating metatdata value to {self.sample_rate}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.sample_rate = self.sample_rate

            # update channels recorded
            self.run_metadata.channels_recorded_auxiliary = []
            self.run_metadata.channels_recorded_electric = []
            self.run_metadata.channels_recorded_magnetic = []
            for ch in self.channels:
                if ch[0] in ["e"]:
                    self.run_metadata.channels_recorded_electric.append(ch)
                elif ch[0] in ["h", "b"]:
                    self.run_metadata.channels_recorded_magnetic.append(ch)
                else:
                    self.run_metadata.channels_recorded_auxiliary.append(ch)

            self.station_metadata.runs.append(self.run_metadata)

    def set_dataset(self, array_list, align_type="outer"):
        """

        :param array_list: list of xarrays
        :type array_list: list of :class:`mth5.timeseries.ChannelTS` objects
        :param align_type: how the different times will be aligned
            * ’outer’: use the union of object indexes
            * ’inner’: use the intersection of object indexes
            * ’left’: use indexes from the first object with each dimension
            * ’right’: use indexes from the last object with each dimension
            * ’exact’: instead of aligning, raise ValueError when indexes to
            be aligned are not equal
            * ’override’: if indexes are of same size, rewrite indexes to
            be those of the first object with that dimension. Indexes for
            the same dimension must have the same size in all objects.
        :type align_type: string

        """
        x_array_list = self._validate_array_list(array_list)

        # first need to align the time series.
        x_array_list = xr.align(*x_array_list, join=align_type)

        # input as a dictionary
        xdict = dict([(x.component.lower(), x) for x in x_array_list])
        self._dataset = xr.Dataset(xdict)
        self.validate_metadata()
        self._dataset.attrs.update(self.run_metadata.to_dict(single=True))

    def add_channel(self, channel):
        """
        Add a channel to the dataset, can be an :class:`xarray.DataArray` or
        :class:`mth5.timeseries.ChannelTS` object.

        Need to be sure that the coordinates and dimensions are the same as the
        existing dataset, namely coordinates are time, and dimensions are the same,
        if the dimesions are larger than the existing dataset then the added channel
        will be clipped to the dimensions of the existing dataset.

        If the start time is not the same nan's will be placed at locations where the
        timing does not match the current start time.  This is a feature of xarray.


        :param channel: a channel xarray or ChannelTS to add to the run
        :type channel: :class:`xarray.DataArray` or :class:`mth5.timeseries.ChannelTS`


        """

        if isinstance(channel, xr.DataArray):
            c = ChannelTS()
            c.ts = channel
        elif isinstance(channel, ChannelTS):
            c = channel
            self.run_metadata.channels.append(c.channel_metadata)
        else:
            raise ValueError("Input Channel must be type xarray.DataArray or ChannelTS")

        ### need to validate the channel to make sure sample rate is the same
        if c.sample_rate != self.sample_rate:
            msg = (
                f"Channel sample rate is not correct, current {self.sample_rate} "
                + f"input {c.sample_rate}"
            )
            self.logger.error(msg)
            raise MTTSError(msg)

        ### should probably check for other metadata like station and run?

        self._dataset[c.component] = c.ts

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, array_list):
        msg = (
            "Data will be aligned using the min and max time. "
            "If that is not correct use set_dataset and change the alignment type."
        )
        self.logger.debug(msg)
        self.set_dataset(array_list)

    @property
    def start(self):
        if self.has_data:
            return MTime(self.dataset.coords["time"].to_index()[0].isoformat())
        return self.run_metadata.time_period.start

    @property
    def end(self):
        if self.has_data:
            return MTime(self.dataset.coords["time"].to_index()[-1].isoformat())
        return self.run_metadata.time_period.end

    @property
    def sample_rate(self):
        if self.has_data:
            try:
                return 1e9 / self.dataset.coords["time"].to_index().freq.n
            except AttributeError:
                self.logger.warning("Something weird happend with xarray time indexing")

                raise ValueError("Something weird happend with xarray time indexing")
        return self.run_metadata.sample_rate

    @property
    def channels(self):
        return [cc for cc in list(self.dataset.data_vars)]

    def to_obspy_stream(self):
        """
        convert time series to an :class:`obspy.core.Stream` which is like a
        list of :class:`obspy.core.Trace` objects.

        :return: An Obspy Stream object from the time series data
        :rtype: :class:`obspy.core.Stream`

        """

        trace_list = []
        for channel in self.channels:
            if channel[0] in ["e"]:
                ch_type = "electric"
            elif channel[0] in ["h", "b"]:
                ch_type = "magnetic"
            else:
                ch_type = "auxiliary"
            ts_obj = ChannelTS(ch_type, self.dataset[channel])
            trace_list.append(ts_obj.to_obspy_trace())

        return Stream(traces=trace_list)

    def from_obspy_stream(self, obspy_stream, run_metadata=None):
        """
        Get a run from an :class:`obspy.core.stream` which is a list of
        :class:`obspy.core.Trace` objects.

        :param obspy_stream: Obspy Stream object
        :type obspy_stream: :class:`obspy.core.Stream`


        """

        if not isinstance(obspy_stream, Stream):
            msg = f"Input must be obspy.core.Stream not {type(obspy_stream)}"
            self.logger.error(msg)
            raise MTTSError(msg)

        array_list = []
        station_list = []
        for obs_trace in obspy_stream:
            channel_ts = ChannelTS()
            channel_ts.from_obspy_trace(obs_trace)
            if channel_ts.channel_metadata.component == "e1":
                channel_ts.channel_metadata.component = "ex"
            if channel_ts.channel_metadata.component == "e2":
                channel_ts.channel_metadata.component = "ey"
            if channel_ts.channel_metadata.component == "h1":
                channel_ts.channel_metadata.component = "hx"
            if channel_ts.channel_metadata.component == "h2":
                channel_ts.channel_metadata.component = "hy"
            if channel_ts.channel_metadata.component == "h3":
                channel_ts.channel_metadata.component = "hz"
            if run_metadata:
                try:
                    ch = [
                        ch
                        for ch in run_metadata.channels
                        if ch.component == channel_ts.component
                    ][0]
                    channel_ts.channel_metadata.update(ch)
                except IndexError:
                    self.logger.warning("could not find %s" % channel_ts.component)
            station_list.append(channel_ts.station_metadata.fdsn.id)

            array_list.append(channel_ts)

        ### need to merge metadata into something useful, station name is the only
        ### name that is preserved
        try:
            station = list(set([ss for ss in station_list if ss is not None]))[0]
        except IndexError:
            station = None
            msg = "Could not find station name"
            self.logger.warn(msg)

        self.station_metadata.fdsn.id = station

        self.set_dataset(array_list)

        self.validate_metadata()

    def plot(self):
        """

        plot the time series probably slow for large data sets

        :return: DESCRIPTION
        :rtype: TYPE

        """

        n_channels = len(self.channels)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0)
        ax1 = fig.add_subplot(n_channels, 1, 1)
        self.dataset[self.channels[0]].plot()
        ax_list = [ax1]
        for ii, comp in enumerate(self.channels[1:], 2):
            ax = plt.subplot(n_channels, 1, ii, sharex=ax1)
            self.dataset[comp].plot()
            ax_list.append(ax)

        for ax in ax_list:
            ax.grid(which="major", color=(0.65, 0.65, 0.65), ls="--", lw=0.75)
            ax.grid(which="minor", color=(0.85, 0.85, 0.85), ls="--", lw=0.5)
            ax.set_axisbelow(True)
