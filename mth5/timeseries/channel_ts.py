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

import numpy as np
import pandas as pd
import xarray as xr

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime
from mt_metadata.timeseries.filters import ChannelResponseFilter

from mth5.utils.exceptions import MTTSError
from mth5.utils.mth5_logger import setup_logger
from mth5.utils import fdsn_tools

from obspy.core import Trace

# =============================================================================
# make a dictionary of available metadata classes
# =============================================================================
meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))


def make_dt_coordinates(start_time, sample_rate, n_samples, logger):
    """
    get the date time index from the data

    :param string start_time: start time in time format
    :param float sample_rate: sample rate in samples per seconds
    :param int n_samples: number of samples in time series
    :param :class:`logging.logger` logger: logger class object

    :return: date-time index

    """

    if sample_rate in [0, None]:
        msg = (
            f"Need to input a valid sample rate. Not {sample_rate}, "
            + "returning a time index assuming a sample rate of 1"
        )
        logger.warning(msg)
        sample_rate = 1

    if start_time is None:
        msg = (
            f"Need to input a start time. Not {start_time}, "
            + "returning a time index with start time of "
            + "1980-01-01T00:00:00"
        )
        logger.warning(msg)
        start_time = "1980-01-01T00:00:00"

    if n_samples < 1:
        msg = f"Need to input a valid n_samples. Not {n_samples}"
        logger.error(msg)
        raise ValueError(msg)

    if not isinstance(start_time, MTime):
        start_time = MTime(start_time)

    dt_freq = "{0:.0f}N".format(1.0e9 / (sample_rate))

    dt_index = pd.date_range(
        start=start_time.iso_str.split("+", 1)[0],
        periods=n_samples,
        freq=dt_freq,
        closed=None,
    )

    return dt_index


# ==============================================================================
# Channel Time Series Object
# ==============================================================================
class ChannelTS:
    """

    .. note:: Assumes equally spaced samples from the start time.

    The time series is stored in an :class:`xarray.Dataset` that has
    coordinates of time and is a 1-D array labeled 'data'.  The :class:`xarray.Dataset`
    can be accessed and set from the :attribute:`ts`.  The data is stored in
    :attribute:'ts.data' and the time index is a coordinate of :attribute:`ts`.

    The time coordinate is made from the start time, sample rate and
    number of samples.  Currently, End time is a derived property and
    cannot be set.

    Channel time series object is based on xarray and :class:`mth5.metadata` therefore
    any type of interpolation, resampling, groupby, etc can be done using xarray
    methods.

    There are 3 metadata classes that hold important metadata

        * :class:`mth5.metadata.Station` holds information about the station
        * :class:`mth5.metadata.Run` holds information about the run the channel
        belongs to.
        * :class`mth5.metadata.Channel` holds information specific to the channel.

    This way a single channel will hold all information needed to represent the
    channel.

    :rubric:

    Example
    ---------

        >>> from mth5.timeseries import ChannelTS
        >>> ts_obj = ChannelTS('auxiliary')
        >>> ts_obj.sample_rate = 8
        >>> ts_obj.start = '2020-01-01T12:00:00+00:00'
        >>> ts_obj.ts = range(4096)
        >>> ts_obj.station_metadata.id = 'MT001'
        >>> ts_obj.run_metadata.id = 'MT001a'
        >>> ts_obj.component = 'temperature'
        >>> print(ts_obj)
                Station      = MT001
                Run          = MT001a
                Channel Type = auxiliary
            Component    = temperature
                Sample Rate  = 8.0
                Start        = 2020-01-01T12:00:00+00:00
                End          = 2020-01-01T12:08:31.875000+00:00
                N Samples    = 4096

    Plot time series with xarray
    ------------------------------

        >>> p = ts_obj.ts.plot()



    """

    def __init__(
        self,
        channel_type="auxiliary",
        data=None,
        channel_metadata=None,
        station_metadata=None,
        run_metadata=None,
        **kwargs,
    ):

        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.station_metadata = metadata.Station()
        self.run_metadata = metadata.Run()
        self._ts = xr.DataArray([1], coords=[("time", [1])], name="ts")
        self._channel_response = ChannelResponseFilter()

        # get correct metadata class
        try:
            self.channel_metadata = meta_classes[channel_type.capitalize()]()
            self.channel_metadata.type = channel_type.lower()
        except KeyError:
            msg = (
                "Channel type is undefined, must be [ electric | "
                + "magnetic | auxiliary ]"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if channel_metadata is not None:
            if isinstance(channel_metadata, type(self.channel_metadata)):
                self.channel_metadata.update(channel_metadata)
                self.logger.debug(
                    "Loading from metadata class {0}".format(
                        type(self.channel_metadata)
                    )
                )
            elif isinstance(channel_metadata, dict):
                if not channel_type in [cc.lower() for cc in channel_metadata.keys()]:
                    channel_metadata = {channel_type: channel_metadata}
                self.channel_metadata.from_dict(channel_metadata)
                self.logger.debug("Loading from metadata dict")

            else:
                msg = "input metadata must be type %s or dict, not %s"
                self.logger.error(
                    msg, type(self.channel_metadata), type(channel_metadata)
                )
                raise MTTSError(
                    msg % (type(self.channel_metadata), type(channel_metadata))
                )

        # add station metadata, this will be important when propogating a single
        # channel such that it can stand alone.
        if station_metadata is not None:
            if isinstance(station_metadata, metadata.Station):
                self.station_metadata.update(station_metadata)

            elif isinstance(station_metadata, dict):
                if not "station" in [cc.lower() for cc in station_metadata.keys()]:
                    station_metadata = {"Station": station_metadata}
                self.station_metadata.from_dict(station_metadata)
                self.logger.debug("Loading from metadata dict")

            else:
                msg = "input metadata must be type {0} or dict, not {1}".format(
                    type(self.station_metadata), type(station_metadata)
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        # add run metadata, this will be important when propogating a single
        # channel such that it can stand alone.
        if run_metadata is not None:
            if isinstance(run_metadata, metadata.Run):
                self.run_metadata.update(run_metadata)

            elif isinstance(run_metadata, dict):
                if not "run" in [cc.lower() for cc in run_metadata.keys()]:
                    run_metadata = {"Run": run_metadata}
                self.run_metadata.from_dict(run_metadata)
                self.logger.debug("Loading from metadata dict")

            else:
                msg = "input metadata must be type %s or dict, not %s"
                self.logger.error(msg, type(self.run_metadata), type(run_metadata))
                raise MTTSError(msg % (type(self.run_metadata), type(run_metadata)))

        # input data
        if data is not None:
            self.ts = data

        self._update_xarray_metadata()

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

    def __str__(self):
        lines = [
            f"Station:      {self.station_metadata.id}",
            f"Run:          {self.run_metadata.id}",
            f"Channel Type: {self.channel_type}",
            f"Component:    {self.component}",
            f"Sample Rate:  {self.sample_rate}",
            f"Start:        {self.start}",
            f"End:          {self.end}",
            f"N Samples:    {self.n_samples}",
        ]

        return "\n\t".join(["Channel Summary:"] + lines)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):

        if not isinstance(other, ChannelTS):
            raise ValueError(f"Cannot compare ChannelTS with {type(other)}")

        if not other.metadata == self.channel_metadata:
            return False

        if self.ts.equals(other.ts) is False:
            msg = "timeseries are not equal"
            self.logger.info(msg)
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, ChannelTS):
            raise ValueError(f"Cannot compare ChannelTS with {type(other)}")

        self.logger.info("Only testing start time")
        if other.start < self.start and other.sample_rate == self.sample_rate:
            return True
        return False

    def __gt__(self, other):
        return not self.__lt__(other)

    ### Properties ------------------------------------------------------------
    @property
    def ts(self):
        return self._ts.data

    @ts.setter
    def ts(self, ts_arr):
        """
        if setting ts with a pandas data frame, make sure the data is in a
        column name 'data'
        """

        if isinstance(ts_arr, (np.ndarray, list, tuple)):
            if not isinstance(ts_arr, np.ndarray):
                ts_arr = np.array(ts_arr)

            # Validate an input array to make sure its 1D
            if len(ts_arr.shape) == 2:
                if 1 in ts_arr.shape:
                    ts_arr = ts_arr.reshape(ts_arr.size)
                else:
                    msg = f"Input array must be 1-D array not {ts_arr.shape}"
                    self.logger.error(msg)
                    raise ValueError(msg)

            dt = make_dt_coordinates(
                self.start, self.sample_rate, ts_arr.size, self.logger
            )
            self._ts = xr.DataArray(ts_arr, coords=[("time", dt)], name="ts")
            self._update_xarray_metadata()

        elif isinstance(ts_arr, pd.core.frame.DataFrame):
            if isinstance(ts_arr.index[0], pd._libs.tslibs.timestamps.Timestamp):
                dt = ts_arr.index
            else:
                dt = make_dt_coordinates(
                    self.start,
                    self.sample_rate,
                    ts_arr["data"].size,
                    self.logger,
                )
            try:
                self._ts = xr.DataArray(
                    ts_arr["data"], coords=[("time", dt)], name="ts"
                )
                self._update_xarray_metadata()

            except AttributeError:
                msg = (
                    "Data frame needs to have a column named `data` "
                    + "where the time series data is stored"
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        elif isinstance(ts_arr, pd.core.series.Series):
            if isinstance(ts_arr.index[0], pd._libs.tslibs.timestamps.Timestamp):
                dt = ts_arr.index
            else:
                dt = make_dt_coordinates(
                    self.start,
                    self.sample_rate,
                    ts_arr["data"].size,
                    self.logger,
                )

            self._ts = xr.DataArray(ts_arr.values, coords=[("time", dt)], name="ts")
            self._update_xarray_metadata()

        elif isinstance(ts_arr, xr.DataArray):
            # TODO: need to validate the input xarray
            self._ts = ts_arr
            # need to pull out the metadata as a separate dictionary
            meta_dict = dict([(k, v) for k, v in ts_arr.attrs.items()])

            # need to get station and run metadata out
            station_keys = [k for k in meta_dict.keys() if "station." in k]
            run_keys = [k for k in meta_dict.keys() if "run." in k]
            station_dict = {}
            run_dict = {}
            for key in station_keys:
                station_dict[key.split("station.")[-1]] = meta_dict.pop(key)
            for key in run_keys:
                run_dict[key.split("run.")[-1]] = meta_dict.pop(key)

            self.channel_metadata.from_dict({meta_dict["type"]: meta_dict})
            self.station_metadata.from_dict({"station": station_dict})
            self.run_metadata.from_dict({"run": run_dict})
            # need to run this incase things are different.
            self._update_xarray_metadata()

        else:
            msg = (
                "Data type {0} not supported".format(type(ts_arr))
                + ", ts needs to be a numpy.ndarray, pandas DataFrame, "
                + "or xarray.DataArray."
            )
            raise MTTSError(msg)

    @property
    def channel_type(self):
        """Channel Type"""
        return self.channel_metadata._class_name

    @channel_type.setter
    def channel_type(self, value):
        """change channel type means changing the metadata type"""

        if value.lower() != self.channel_metadata._class_name.lower():
            m_dict = self.channel_metadata.to_dict()[self.channel_metadata._class_name]
            try:
                self.channel_metadata = meta_classes[value.capitalize()]()
                msg = (
                    f"Changing metadata to {value.capitalize()}"
                    + "will translate any similar attributes."
                )
                self.logger.info(msg)
            except KeyError:
                msg = (
                    f"Channel type {value} not understood, must be "
                    + "[ Electrict | Magnetic | Auxiliary ]"
                )
                self.logger.error(msg)

            for key in self.channel_metadata.to_dict()[
                self.channel_metadata._class_name
            ].keys():
                try:
                    self.channel_metadata.set_attr_from_name(key, m_dict[key])
                except KeyError:
                    pass
        return

    def _update_xarray_metadata(self):
        """
        Update xarray attrs dictionary with metadata.  Here we are assuming that
        self.channel_metadata is the parent and attrs in xarray are children because all
        metadata will be validated by :class:`mth5.metadata` class objects.

        Eventually there should be a way that this is automatic, but I'm not that
        clever yet.

        This should be mainly used internally but gives the user a way to update
        metadata.

        """
        self.logger.debug("Updating xarray attributes")

        self.channel_metadata.time_period.start = self.start.iso_no_tz
        self.channel_metadata.time_period.end = self.end.iso_no_tz
        self.channel_metadata.sample_rate = self.sample_rate

        self._ts.attrs.update(
            self.channel_metadata.to_dict()[self.channel_metadata._class_name]
        )
        # add station and run id's here, for now this is all we need but may need
        # more metadata down the road.
        self._ts.attrs["station.id"] = self.station_metadata.id
        self._ts.attrs["run.id"] = self.run_metadata.id

    @property
    def component(self):
        """component"""
        return self.channel_metadata.component

    @component.setter
    def component(self, comp):
        """set component in metadata and carry through"""
        if self.channel_metadata.type == "electric":
            if comp[0].lower() != "e":
                msg = (
                    "The current timeseries is an electric channel. "
                    "Cannot change channel type, create a new ChannelTS object."
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        elif self.channel_metadata.type == "magnetic":
            if comp[0].lower() not in ["h", "b"]:
                msg = (
                    "The current timeseries is a magnetic channel. "
                    "Cannot change channel type, create a new ChannelTS object."
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        if self.channel_metadata.type == "auxiliary":
            if comp[0].lower() in ["e", "h", "b"]:
                msg = (
                    "The current timeseries is an auxiliary channel. "
                    "Cannot change channel type, create a new ChannelTS object."
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        self.channel_metadata.component = comp
        self._update_xarray_metadata()

    # --> number of samples just to make sure there is consistency
    @property
    def n_samples(self):
        """number of samples"""
        return int(self.ts.size)

    @n_samples.setter
    def n_samples(self, n_samples):
        """number of samples (int)"""
        self.logger.warning(
            "Cannot set the number of samples. Use `ChannelTS.resample` or `get_slice`"
        )

    @property
    def has_data(self):
        """
        check to see if there is an index in the time series
        """
        if len(self._ts) > 1:
            if isinstance(
                self._ts.indexes["time"][0],
                pd._libs.tslibs.timestamps.Timestamp,
            ):
                return True
            return False
        else:
            return False

    # --> sample rate
    @property
    def sample_rate(self):
        """sample rate in samples/second"""
        if self.has_data:
            sr = 1./ np.float64((np.median(np.diff(self._ts.coords.indexes["time"])) / np.timedelta64(1, 's')))
            
        else:
            self.logger.debug("Data has not been set yet, sample rate is from metadata")
            sr = self.channel_metadata.sample_rate
            if sr is None:
                sr = 0.0
        return np.round(sr, 0)

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        """
        sample rate in samples/second

        type float
        """
        if self.has_data:
            self.logger.warning(
                "Resetting sample_rate assumes same start time and "
                + "same number of samples, resulting in new end time. "
                + "If you want to downsample existing time series "
                + "use the method channelTS.resample()"
            )
            self.logger.debug(
                f"Resetting sample rate from {self.sample_rate} to {sample_rate}"
            )
            new_dt = make_dt_coordinates(
                self.start, sample_rate, self.n_samples, self.logger
            )
            self._ts.coords["time"] = new_dt
        else:
            if self.channel_metadata.sample_rate not in [0.0, None]:
                self.logger.warning(
                    f"Resetting ChannelTS.channel_metadata.sample_rate to {sample_rate}. "
                )
            self.channel_metadata.sample_rate = sample_rate
        self._update_xarray_metadata()
        
    @property
    def sample_interval(self):
        """
        Sample interval = 1 / sample_rate
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        if self.sample_rate != 0:
            return 1./self.sample_rate
        return 0.

    ## set time and set index
    @property
    def start(self):
        """MTime object"""
        if self.has_data:
            return MTime(self._ts.coords.indexes["time"][0].isoformat())
        else:
            self.logger.debug(
                "Data not set yet, pulling start time from "
                + "metadata.time_period.start"
            )
            return MTime(self.channel_metadata.time_period.start)

    @start.setter
    def start(self, start_time):
        """
        start time of time series in UTC given in some format or a datetime
        object.

        Resets epoch seconds if the new value is not equivalent to previous
        value.

        Resets how the ts data frame is indexed, setting the starting time to
        the new start time.

        :param start_time: start time of time series, can be string or epoch seconds

        """

        if not isinstance(start_time, MTime):
            start_time = MTime(start_time)

        self.channel_metadata.time_period.start = start_time.iso_str
        if self.has_data:
            if start_time == MTime(self._ts.coords.indexes["time"][0].isoformat()):
                return
            else:
                new_dt = make_dt_coordinates(
                    start_time, self.sample_rate, self.n_samples, self.logger
                )
                self._ts.coords["time"] = new_dt

        # make a time series that the data can be indexed by
        else:
            self.logger.debug("No data, just updating metadata start")

        self._update_xarray_metadata()

    @property
    def end(self):
        """MTime object"""
        if self.has_data:
            return MTime(self._ts.coords.indexes["time"][-1].isoformat())
        else:
            self.logger.debug(
                "Data not set yet, pulling end time from " + "metadata.time_period.end"
            )
            return MTime(self.channel_metadata.time_period.end)

    @end.setter
    def end(self, end_time):
        """
        end time of time series in UTC given in some format or a datetime
        object.

        Resets epoch seconds if the new value is not equivalent to previous
        value.

        Resets how the ts data frame is indexed, setting the starting time to
        the new start time.
        """
        self.logger.warning(
            "Cannot set `end`. If you want a slice, then " + "use get_slice method"
        )

    @property
    def channel_response_filter(self):
        """
        Full channel response filter

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._channel_response

    @channel_response_filter.setter
    def channel_response_filter(self, value):
        """

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(value, ChannelResponseFilter):
            msg = (
                "channel response must be a "
                "mt_metadata.timeseries.filters.ChannelResponseFilter object "
                f"not {type(value)}."
            )
            self.logger.error(msg)
            raise TypeError(msg)

        self._channel_response = value

    def get_slice(self, start, end=None, n_samples=None):
        """
        Get a slice from the time series given a start and end time.

        Looks for >= start & <= end

        Uses loc to be exact with milliseconds

        :param start: DESCRIPTION
        :type start: TYPE
        :param end: DESCRIPTION
        :type end: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if n_samples is None and end is None:
            msg = "Must input either end_time or n_samples."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_samples is not None and end is not None:
            msg = "Must input either end_time or n_samples, not both."
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(start, MTime):
            start = MTime(start)
            
        if n_samples is not None:
            n_samples = int(n_samples)
            end = start + n_samples / self.sample_rate
            
        if end is not None:
            if not isinstance(end, MTime):
                end = MTime(end)

        new_ts = self._ts.loc[
            (self._ts.indexes["time"] >= start.iso_no_tz)
            & (self._ts.indexes["time"] <= end.iso_no_tz)
        ]
        
        new_ch_ts = ChannelTS(channel_type=self.channel_type,
                              data=new_ts,
                              channel_metadata=self.channel_metadata,
                              run_metadata=self.run_metadata,
                              station_metadata=self.station_metadata)
        
        return new_ch_ts

    # decimate data
    def resample(self, dec_factor=1, inplace=False):
        """
        decimate the data by using scipy.signal.decimate

        :param dec_factor: decimation factor
        :type dec_factor: int

        * refills ts.data with decimated data and replaces sample_rate

        """

        new_dt_freq = "{0:.0f}N".format(1e9 / (self.sample_rate / dec_factor))

        new_ts = self._ts.resample(time=new_dt_freq).nearest(tolerance=new_dt_freq)
        new_ts.attrs["sample_rate"] = self.sample_rate / dec_factor
        self.channel_metadata.sample_rate = new_ts.attrs["sample_rate"]

        if inplace:
            self.ts = new_ts

        else:
            new_ts.attrs.update(
                self.channel_metadata.to_dict()[self.channel_metadata._class_name]
            )
            # return new_ts
            return ChannelTS(
                self.channel_metadata.type, data=new_ts, metadata=self.channel_metadata
            )

    def to_xarray(self):
        """
        Returns a :class:`xarray.DataArray` object of the channel timeseries
        this way metadata from the metadata class is updated upon return.

        :return: Returns a :class:`xarray.DataArray` object of the channel timeseries
        this way metadata from the metadata class is updated upon return.
        :rtype: :class:`xarray.DataArray`


        >>> import numpy as np
        >>> from mth5.timeseries import ChannelTS
        >>> ex = ChannelTS("electric")
        >>> ex.start = "2020-01-01T12:00:00"
        >>> ex.sample_rate = 16
        >>> ex.ts = np.random.rand(4096)


        """
        self._update_xarray_metadata()
        return self._ts

    def to_obspy_trace(self):
        """
        Convert the time series to an :class:`obspy.core.trace.Trace` object.  This
        will be helpful for converting between data pulled from IRIS and data going
        into IRIS.

        :return: DESCRIPTION
        :rtype: TYPE

        """

        obspy_trace = Trace(self.ts.data)
        obspy_trace.stats.channel = self.component
        obspy_trace.stats.starttime = self.start.iso_str
        obspy_trace.stats.sampling_rate = self.sample_rate
        obspy_trace.stats.station = self.station_metadata.fdsn.id

        return obspy_trace

    def from_obspy_trace(self, obspy_trace):
        """
        Fill data from an :class:`obspy.core.Trace`

        :param obspy.core.trace obspy_trace: Obspy trace object

        """

        if not isinstance(obspy_trace, Trace):
            msg = f"Input must be obspy.core.Trace, not {type(obspy_trace)}"
            self.logger.error(msg)
            raise MTTSError(msg)

        if obspy_trace.stats.channel[1].lower() in ["e", "q"]:
            self.channel_metadata = metadata.Electric()
        elif obspy_trace.stats.channel[1].lower() in ["h", "b", "f"]:
            self.channel_metadata = metadata.Magnetic()
        else:
            self.channel_metadata = metadata.Auxiliary()

        mt_code = fdsn_tools.make_mt_channel(
            fdsn_tools.read_channel_code(obspy_trace.stats.channel)
        )
        self.channel_metadata.component = mt_code
        self.start = obspy_trace.stats.starttime.isoformat()
        self.sample_rate = obspy_trace.stats.sampling_rate
        self.station_metadata.fdsn.id = obspy_trace.stats.station
        self.station_metadata.fdsn.network = obspy_trace.stats.network
        self.station_metadata.id = obspy_trace.stats.station
        self.channel_metadata.units = "counts"
        self.ts = obspy_trace.data
