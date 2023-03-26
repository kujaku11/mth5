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
import numpy as np

from matplotlib import pyplot as plt

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime
from mt_metadata.utils.list_dict import ListDict
from mt_metadata.timeseries.filters import ChannelResponseFilter

from .channel_ts import ChannelTS, make_dt_coordinates
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

    ToDo, have a single Survey object under the hood and properties to other
    metadata objects for get/set.

    """

    def __init__(
        self,
        array_list=None,
        run_metadata=None,
        station_metadata=None,
        survey_metadata=None,
    ):

        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self._survey_metadata = self._initialize_metadata()
        self._dataset = xr.Dataset()
        self._filters = {}

        self.survey_metadata = survey_metadata
        self.station_metadata = station_metadata
        self.run_metadata = run_metadata

        # load the arrays first this will write run and station metadata
        if array_list is not None:
            self.dataset = array_list

    def __str__(self):
        s_list = [
            f"Survey:      {self.survey_metadata.id}",
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

    def __add__(self, other):
        """
        Add two runs together in the following steps

        1. xr.combine_by_coords([original, other])
        2. compute monotonic time index
        3. reindex(new_time_index, method='nearest')

        If you want a different method or more control use merge

        :param other: Another run
        :type other: :class:`mth5.timeseries.RunTS`
        :raises TypeError: If input is not a RunTS
        :raises ValueError: if the components are different
        :return: Combined channel with monotonic time index and same metadata
        :rtype: :class:`mth5.timeseries.RunTS`

        """
        if not isinstance(other, RunTS):
            raise TypeError(f"Cannot combine {type(other)} with RunTS.")

        # combine into a data set use override to keep attrs from original
        combined_ds = xr.combine_by_coords(
            [self.dataset, other.dataset], combine_attrs="override"
        )

        n_samples = (
            self.sample_rate
            * float(
                combined_ds.time.max().values - combined_ds.time.min().values
            )
            / 1e9
        ) + 1

        new_dt_index = make_dt_coordinates(
            combined_ds.time.min().values,
            self.sample_rate,
            n_samples,
            self.logger,
        )

        new_run = RunTS(
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )

        new_run.dataset = combined_ds.reindex(
            {"time": new_dt_index}, method=None
        ).interpolate_na(dim="time", method="slinear")

        new_run.run_metadata.update_time_period()
        new_run.station_metadata.update_time_period()
        new_run.survey_metadata.update_time_period()
        new_run.filters = self.filters
        new_run.filters.update(other.filters)

        return new_run

    def _initialize_metadata(self):
        """
        Create a single `Survey` object to store all metadata

        :param channel_type: DESCRIPTION
        :type channel_type: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        survey_metadata = metadata.Survey(id="0")
        survey_metadata.stations.append(metadata.Station(id="0"))
        survey_metadata.stations[0].runs.append(metadata.Run(id="0"))

        return survey_metadata

    def _validate_run_metadata(self, run_metadata):
        """
        validate run metadata

        """

        if not isinstance(run_metadata, metadata.Run):
            if isinstance(run_metadata, dict):
                if "run" not in [cc.lower() for cc in run_metadata.keys()]:
                    run_metadata = {"Run": run_metadata}
                r_metadata = metadata.Run()
                r_metadata.from_dict(run_metadata)
                self.logger.debug("Loading from metadata dict")
                return r_metadata
            else:
                msg = "input metadata must be type %s or dict, not %s"
                self.logger.error(
                    msg, type(self.run_metadata), type(run_metadata)
                )
                raise TypeError(
                    msg % (type(self.run_metadata), type(run_metadata))
                )
        return run_metadata.copy()

    def _validate_station_metadata(self, station_metadata):
        """
        validate station metadata
        """

        if not isinstance(station_metadata, metadata.Station):
            if isinstance(station_metadata, dict):
                if "station" not in [
                    cc.lower() for cc in station_metadata.keys()
                ]:
                    station_metadata = {"Station": station_metadata}

                st_metadata = metadata.Station()
                st_metadata.from_dict(station_metadata)
                self.logger.debug("Loading from metadata dict")
                return st_metadata
            else:
                msg = "input metadata must be type {0} or dict, not {1}".format(
                    type(self.station_metadata), type(station_metadata)
                )
                self.logger.error(msg)
                raise TypeError(msg)

        return station_metadata.copy()

    def _validate_survey_metadata(self, survey_metadata):
        """
        validate station metadata
        """

        if not isinstance(survey_metadata, metadata.Survey):
            if isinstance(survey_metadata, dict):
                if "station" not in [
                    cc.lower() for cc in survey_metadata.keys()
                ]:
                    survey_metadata = {"Survey": survey_metadata}

                sv_metadata = metadata.Station()
                sv_metadata.from_dict(survey_metadata)
                self.logger.debug("Loading from metadata dict")
                return sv_metadata
            else:
                msg = "input metadata must be type {0} or dict, not {1}".format(
                    type(self.survey_metadata), type(survey_metadata)
                )
                self.logger.error(msg)
                raise TypeError(msg)

        return survey_metadata.copy()

    def _validate_array_list(self, array_list):
        """check to make sure all entries are a :class:`ChannelTS` object"""

        if not isinstance(array_list, (tuple, list)):
            msg = f"array_list must be a list or tuple, not {type(array_list)}"
            self.logger.error(msg)
            raise TypeError(msg)

        valid_list = []
        station_metadata = metadata.Station()
        run_metadata = metadata.Run()
        channels = ListDict()

        for index, item in enumerate(array_list):
            if not isinstance(item, (ChannelTS, xr.DataArray)):
                msg = f"array entry {index} must be ChannelTS object not {type(item)}"
                self.logger.error(msg)
                raise TypeError(msg)
            if isinstance(item, ChannelTS):
                valid_list.append(item.to_xarray())

                # if a channelTS is input then it comes with run and station metadata
                # use those first, then the user can update later.

                if item.station_metadata.id not in ["0", None]:
                    if station_metadata.id not in ["0", None]:
                        station_metadata.update(
                            item.station_metadata, match=["id"]
                        )
                    else:
                        station_metadata.update(item.station_metadata)

                if item.run_metadata.id not in ["0", None]:
                    if run_metadata.id not in ["0", None]:
                        run_metadata.update(item.run_metadata, match=["id"])
                    else:
                        run_metadata.update(item.run_metadata)

                channels.append(item.channel_metadata)

                # get the filters from the channel
                if item.channel_response_filter.filters_list != []:
                    for ff in item.channel_response_filter.filters_list:
                        self._filters[ff.name] = ff

            else:
                valid_list.append(item)

        # need to make sure that the station metadata was actually updated,
        # should have an ID.
        run_metadata.channels = channels
        if station_metadata.id not in ["0", None]:
            station_metadata.runs = ListDict()
            station_metadata.runs.append(run_metadata)
            # need to add the other runs that are in the metadata for
            # completeness.
            for run in self.station_metadata.runs:
                if run.id not in [run_metadata.id, "0", None]:
                    station_metadata.runs.append(run)
            self.station_metadata = station_metadata
        # if the run metadata was updated
        elif run_metadata.id not in ["0", None]:
            self.run_metadata = run_metadata
        # if the run metadata or station metadata was not updated from channel
        # metadata, then update just the channels.
        else:
            self.run_metadata.channels = channels

        # probably should test for sampling rate.
        sr_test = dict(
            [(item.component, (item.sample_rate)) for item in valid_list]
        )

        if len(set([v for k, v in sr_test.items()])) != 1:
            msg = f"sample rates are not all the same {sr_test}"
            self.logger.error(msg)
            raise ValueError(msg)

        return valid_list

    def _get_channel_response_filter(self, ch_name):
        """
        Get the channel response filter from the filter dictionary

        :param ch_name: DESCRIPTION
        :type ch_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        filter_list = []
        if ch_name in self.dataset.keys():

            for filter_name in self.dataset[ch_name].attrs["filter.name"]:
                try:
                    filter_list.append(self.filters[filter_name])
                except KeyError:
                    self.logger.debug(
                        f"Could not find {filter_name} in filters"
                    )

        return ChannelResponseFilter(filters_list=filter_list)

    def __getattr__(self, name):
        # change to look for keys directly and use type to set channel type
        if name in self.dataset.keys():

            ch_response_filter = self._get_channel_response_filter(name)
            # if cannot get filters, but the filters name indicates that
            # filters should be there don't input the channel response filter
            # cause then an empty filters_list will set filter.name to []
            if ch_response_filter.filters_list == []:
                ch_response_filter = None

            return ChannelTS(
                self.dataset[name].attrs["type"],
                self.dataset[name],
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                channel_response_filter=ch_response_filter,
            )
        else:
            # this is a hack for now until figure out who is calling shape, size
            if name[0] == "_":
                return None
            if name not in ["shape", "size"]:
                try:
                    return super().__getattribute__(name)
                except AttributeError:
                    msg = f"RunTS has no attribute {name}"
                    self.logger.error(msg)
                    raise NameError(msg)

    ### Properties ------------------------------------------------------------
    @property
    def survey_metadata(self):
        """
        survey metadata
        """
        return self._survey_metadata

    @survey_metadata.setter
    def survey_metadata(self, survey_metadata):
        """

        :param survey_metadata: survey metadata object or dictionary
        :type survey_metadata: :class:`mt_metadata.timeseries.Survey` or dict

        """

        if survey_metadata is not None:
            survey_metadata = self._validate_survey_metadata(survey_metadata)
            self._survey_metadata.update(
                self._validate_survey_metadata(survey_metadata)
            )

    @property
    def station_metadata(self):
        """
        station metadata
        """

        return self.survey_metadata.stations[0]

    @station_metadata.setter
    def station_metadata(self, station_metadata):
        """
        set station metadata from a valid input
        """

        if station_metadata is not None:
            station_metadata = self._validate_station_metadata(station_metadata)

            runs = ListDict()
            if self.run_metadata.id not in ["0", 0]:
                runs.append(self.run_metadata.copy())
            runs.extend(station_metadata.runs)
            if len(runs) == 0:
                runs[0] = metadata.Run(id="0")

            # be sure there is a level below
            if len(runs[0].channels) == 0:
                ch_metadata = metadata.Auxiliary()
                ch_metadata.type = "auxiliary"
                runs[0].channels.append(ch_metadata)

            stations = ListDict()
            stations.append(station_metadata)
            stations[0].runs = runs

            self.survey_metadata.stations = stations

    @property
    def run_metadata(self):
        """
        station metadata
        """
        run_metadata = self.survey_metadata.stations[0].runs[0]

        return run_metadata

    @run_metadata.setter
    def run_metadata(self, run_metadata):
        """
        set run metadata from a valid input
        """

        if run_metadata is not None:
            run_metadata = self._validate_run_metadata(run_metadata)
            runs = ListDict()
            runs.append(run_metadata)
            runs.extend(
                self.station_metadata.runs, skip_keys=[run_metadata.id, "0"]
            )
            self._survey_metadata.stations[0].runs = runs

    def has_data(self):
        """check to see if there is data"""
        if len(self.channels) > 0:
            return True
        return False

    @property
    def summarize_metadata(self):
        """

        Get a summary of all the metadata

        :return: A summary of all channel metadata in one place
        :rtype: dictionary

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

        """

        # check sampling rate
        if self.has_data():
            # check start time
            if self.start != self.run_metadata.time_period.start:
                if (
                    self.run_metadata.time_period.start
                    != "1980-01-01T00:00:00+00:00"
                ):
                    msg = (
                        f"start time of dataset {self.start} does not "
                        f"match metadata start {self.run_metadata.time_period.start} "
                        f"updating metatdata value to {self.start}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.time_period.start = self.start.iso_str

            # check end time
            if self.end != self.run_metadata.time_period.end:
                if (
                    self.run_metadata.time_period.end
                    != "1980-01-01T00:00:00+00:00"
                ):
                    msg = (
                        f"end time of dataset {self.end} does not "
                        f"match metadata end {self.run_metadata.time_period.end} "
                        f"updating metatdata value to {self.end}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.time_period.end = self.end.iso_str
            if self.sample_rate != self.run_metadata.sample_rate:
                if self.run_metadata.sample_rate == 0.0:
                    pass
                elif self.run_metadata.sample_rate is not None:
                    msg = (
                        f"sample rate of dataset {self.sample_rate} does not "
                        f"match metadata sample rate {self.run_metadata.sample_rate} "
                        f"updating metatdata value to {self.sample_rate}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.sample_rate = self.sample_rate

            if self.run_metadata.id not in self.station_metadata.runs.keys():
                self.station_metadata.runs[0].update(self.run_metadata)

            self.station_metadata.update_time_period()
            self.survey_metadata.update_time_period()

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
        if isinstance(array_list, (list, tuple)):
            x_array_list = self._validate_array_list(array_list)

            # first need to align the time series.
            x_array_list = xr.align(*x_array_list, join=align_type)

            # input as a dictionary
            xdict = dict([(x.component.lower(), x) for x in x_array_list])
            self._dataset = xr.Dataset(xdict)

        elif isinstance(array_list, xr.Dataset):
            self._dataset = array_list

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
            for ff in c.channel_response_filter.filters_list:
                self._filters[ff.name] = ff

        else:
            raise ValueError(
                "Input Channel must be type xarray.DataArray or ChannelTS"
            )

        ### need to validate the channel to make sure sample rate is the same
        if c.sample_rate != self.sample_rate:
            msg = (
                f"Channel sample rate is not correct, current {self.sample_rate} "
                + f"input {c.sample_rate}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        ### should probably check for other metadata like station and run?
        if len(self.dataset.dims) == 0:
            self.dataset = c._ts.to_dataset()
        else:
            self.dataset = xr.merge([self.dataset, c._ts.to_dataset()])

    @property
    def dataset(self):
        """:class:`xarray.Dataset`"""
        return self._dataset

    @dataset.setter
    def dataset(self, array_list):
        """Set the dataset"""
        msg = (
            "Data will be aligned using the min and max time. "
            "If that is not correct use set_dataset and change the alignment type."
        )
        self.logger.debug(msg)
        self.set_dataset(array_list)

    @property
    def start(self):
        """Start time UTC"""
        if self.has_data():
            return MTime(self.dataset.coords["time"].to_index()[0].isoformat())
        return self.run_metadata.time_period.start

    @property
    def end(self):
        """End time UTC"""
        if self.has_data():
            return MTime(self.dataset.coords["time"].to_index()[-1].isoformat())
        return self.run_metadata.time_period.end

    @property
    def sample_rate(self):
        """
        Sample rate, this is estimated by the mdeian difference between
        samples in time, if data is present. Otherwise return the metadata
        sample rate.
        """
        if self.has_data():
            try:
                return round(
                    1.0
                    / np.float64(
                        (
                            np.median(
                                np.diff(self.dataset.coords["time"].to_index())
                                / np.timedelta64(1, "s")
                            )
                        )
                    ),
                    0,
                )
            except AttributeError:
                self.logger.warning(
                    "Something weird happend with xarray time indexing"
                )

                raise ValueError(
                    "Something weird happend with xarray time indexing"
                )
        return self.run_metadata.sample_rate

    @property
    def sample_interval(self):
        """
        Sample interval = 1 / sample_rate
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.sample_rate != 0:
            return 1.0 / self.sample_rate
        return 0.0

    @property
    def channels(self):
        """List of channel names in dataset"""
        return [cc for cc in list(self.dataset.data_vars)]

    @property
    def filters(self):
        """Dictionary of filters used by the channels"""
        return self._filters

    @filters.setter
    def filters(self, value):
        """
        a dictionary of filters found in the channel objects.

        Should use the dictionary methods to update a dictionary.

        :param value: dictionary of :module:`mt_metadata.timeseries.filters`
        objects
        :type value: dictionary
        :raises TypeError: If input is anything other than a dictionary

        """
        if not isinstance(value, dict):
            raise TypeError("input must be a dictionary")

        self._filters = value

    def to_obspy_stream(self):
        """
        convert time series to an :class:`obspy.core.Stream` which is like a
        list of :class:`obspy.core.Trace` objects.

        :return: An Obspy Stream object from the time series data
        :rtype: :class:`obspy.core.Stream`

        """

        trace_list = []
        for channel in self.channels:
            ts_obj = getattr(self, channel)
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
            raise TypeError(msg)

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
                    self.logger.warning(
                        "could not find %s" % channel_ts.component
                    )
            station_list.append(channel_ts.station_metadata.fdsn.id)

            array_list.append(channel_ts)

        ### need to merge metadata into something useful, station name is the only
        ### name that is preserved
        try:
            station = list(set([ss for ss in station_list if ss is not None]))[
                0
            ]
        except IndexError:
            station = None
            msg = "Could not find station name"
            self.logger.warn(msg)

        self.station_metadata.fdsn.id = station

        self.set_dataset(array_list)

        # need to be sure update any input metadata.
        if run_metadata is not None:
            self.station_metadata.runs = ListDict()
            self.station_metadata.add_run(run_metadata)

        self.validate_metadata()

    def get_slice(self, start, end=None, n_samples=None):
        """

        :param start: DESCRIPTION
        :type start: TYPE
        :param end: DESCRIPTION, defaults to None
        :type end: TYPE, optional
        :param n_samples: DESCRIPTION, defaults to None
        :type n_samples: TYPE, optional
        :raises ValueError: DESCRIPTION
        :return: DESCRIPTION
        :rtype: TYPE

        """
        """
        Get just a chunk of data from the run, this will attempt to find the
        closest points to the given parameters.  
        
        .. note:: We use pandas `slice_indexer` because xarray slice does not
        seem to work as well, even though they should be based on the same 
        code.

        :param start: start time of the slice
        :type start: string or :class:`mt_metadata.utils.mttime.MTime`
        :param end: end time of the slice, defaults to None
        :type end: string or :class:`mt_metadata.utils.mttime.MTime`, optional
        :param n_samples: number of samples to get, defaults to None
        :type n_samples: int, optional
        :raises ValueError: If end and n_samples are not input
        :return: slice of data requested
        :rtype: :class:`mth5.timeseries.RunTS`

        """
        if not isinstance(start, MTime):
            start = MTime(start)

        if n_samples is not None:
            seconds = (n_samples - 1) / self.sample_rate
            end = start + seconds

        elif end is not None:
            if not isinstance(end, MTime):
                end = MTime(end)
        else:
            raise ValueError("Must input n_samples or end")

        chunk = self.dataset.indexes["time"].slice_indexer(
            start=np.datetime64(start.iso_no_tz),
            end=np.datetime64(end.iso_no_tz),
        )

        new_runts = RunTS()
        new_runts.station_metadata = self.station_metadata
        new_runts.run_metadata = self.run_metadata
        new_runts.filters = self.filters
        new_runts.dataset = self._dataset.isel(indexers={"time": chunk})

        return new_runts

    def calibrate(self, **kwargs):
        """
        Calibrate the data according to the filters in each channel.

        :return: calibrated run
        :rtype: :class:`mth5.timeseries.RunTS`

        """

        new_run = RunTS()
        new_run.run_metadata = self.run_metadata
        new_run.station_metadata = self.station_metadata

        for channel in self.channels:
            ch_ts = getattr(self, channel)
            calibrated_ch_ts = ch_ts.remove_instrument_response(**kwargs)
            new_run.add_channel(calibrated_ch_ts)

        return new_run

    def resample(self, new_sample_rate, inplace=False):
        """
        Resample data to new sample rate.

        :param new_sample_rate: DESCRIPTION
        :type new_sample_rate: TYPE
        :param inplace: DESCRIPTION, defaults to False
        :type inplace: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        new_dt_freq = "{0:.0f}N".format(1e9 / (new_sample_rate))

        new_ds = self.dataset.resample(time=new_dt_freq).nearest(
            tolerance=new_dt_freq
        )
        new_ds.attrs["sample_rate"] = new_sample_rate
        self.run_metadata.sample_rate = new_ds.attrs["sample_rate"]

        if inplace:
            self.dataset = new_ds
        else:
            # return new_ts
            return RunTS(
                new_ds,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                survey_metadata=self.survey_metadata,
            )

    def merge(self, other, gap_method="slinear", new_sample_rate=None):
        """
        merg two runs or list of runs together in the following steps

        1. xr.combine_by_coords([original, other])
        2. compute monotonic time index
        3. reindex(new_time_index, method=gap_method)

        If you want a different method or more control use merge

        :param other: Another run
        :type other: :class:`mth5.timeseries.RunTS`
        :raises TypeError: If input is not a RunTS
        :raises ValueError: if the components are different
        :return: Combined run with monotonic time index and same metadata
        :rtype: :class:`mth5.timeseries.RunTS`

        """
        if new_sample_rate is not None:
            merge_sample_rate = new_sample_rate
        else:
            merge_sample_rate = self.sample_rate

        combine_list = [self.dataset]
        ts_filters = self.filters
        if isinstance(other, (list, tuple)):
            for run in other:
                if not isinstance(run, RunTS):
                    raise TypeError(f"Cannot combine {type(run)} with RunTS.")

                if new_sample_rate is not None:
                    run = run.resample(new_sample_rate)
                combine_list.append(run.dataset)
                ts_filters.update(other.filters)
        else:
            if not isinstance(other, RunTS):
                raise TypeError(f"Cannot combine {type(other)} with RunTS.")

            if new_sample_rate is not None:
                other = other.resample(new_sample_rate)
            combine_list.append(other.dataset)
            ts_filters.update(other.filters)

        # combine into a data set use override to keep attrs from original

        combined_ds = xr.combine_by_coords(
            combine_list, combine_attrs="override"
        )

        n_samples = (
            merge_sample_rate
            * float(
                combined_ds.time.max().values - combined_ds.time.min().values
            )
            / 1e9
        ) + 1

        new_dt_index = make_dt_coordinates(
            combined_ds.time.min().values,
            merge_sample_rate,
            n_samples,
            self.logger,
        )

        run_metadata = self.run_metadata.copy()
        run_metadata.sample_rate = merge_sample_rate

        new_run = RunTS(
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )

        new_run.dataset = combined_ds.reindex(
            {"time": new_dt_index},
            method=None,
        ).interpolate_na(dim="time", method=gap_method)

        new_run.run_metadata.update_time_period()
        new_run.station_metadata.update_time_period()
        new_run.survey_metadata.update_time_period()
        new_run.filters = ts_filters

        return new_run

    def plot(
        self,
        color_map={
            "ex": (1, 0.2, 0.2),
            "ey": (1, 0.5, 0),
            "hx": (0, 0.5, 1),
            "hy": (0.5, 0.2, 1),
            "hz": (0.2, 1, 1),
        },
        channel_order=None,
    ):
        """

        plot the time series probably slow for large data sets

        """

        if channel_order is not None:
            ch_list = channel_order()
        else:
            ch_list = self.channels

        n_channels = len(self.channels)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0)
        ax_list = []
        for ii, comp in enumerate(ch_list, 1):
            try:
                color = color_map[comp]
            except KeyError:
                color = (0, 0.4, 0.8)
            if ii == 1:
                ax = plt.subplot(n_channels, 1, ii)
            else:
                ax = plt.subplot(n_channels, 1, ii, sharex=ax_list[0])
            self.dataset[comp].plot.line(ax=ax, color=color)
            ax.grid(which="major", color=(0.65, 0.65, 0.65), ls="--", lw=0.75)
            ax.grid(which="minor", color=(0.85, 0.85, 0.85), ls="--", lw=0.5)
            ax.set_axisbelow(True)
            if ii != len(ch_list):
                plt.setp(ax.get_xticklabels(), visible=False)

            ax_list.append(ax)
