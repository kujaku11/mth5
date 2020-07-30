# -*- coding: utf-8 -*-
"""
.. module:: timeseries
   :synopsis: Deal with MT time series

:copyright:
    Jared Peacock (jpeacock@usgs.gov)
    
:license: 
    MIT
"""

# ==============================================================================
# Imports
# ==============================================================================
import logging
import inspect

import numpy as np
import pandas as pd
import xarray as xr

from mth5 import metadata
from mth5.utils.mttime import MTime
from mth5.utils.exceptions import MTTSError

# make a dictionary of available metadata classes
meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
# ==============================================================================

# ==============================================================================
class MTTS():
    """
    
    .. note:: Assumes equally spaced samples from the start time.
    
    The time series is stored in an :class:`xarray.Dataset` that is has 
    coordinates of time and is a 1-D array labeled 'data'
    
    The time coordinate is made from the start time, sample rate and 
    number of samples.  Currently, End time is a derived property and 
    cannot be set. 
    
    
    MT time series object is based on xarray and :class:`mth5.metadata`

    """

    def __init__(self, channel_type, data=None, channel_metadata=None, **kwargs):
        self.logger = logging.getLogger("{0}.{1}".format(__name__, self._class_name))

        # get correct metadata class
        try:
            self.metadata = meta_classes[channel_type.capitalize()]()
        except KeyError:
            msg = (
                "Channel type is undefined, must be [ electric | "
                + "magnetic | auxiliary ]"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if channel_metadata is not None:
            if isinstance(channel_metadata, type(self.metadata)):
                self.metadata.from_dict(channel_metadata.to_dict())
                self.logger.debug(
                    "Loading from metadata class {0}".format(type(self.metadata))
                )
            elif isinstance(channel_metadata, dict):
                self.metadata.from_dict(channel_metadata)
                self.logger.debug("Loading from metadata dict")

            else:
                msg = "input metadata must be type {0} or dict, not {1}".format(
                    type(self.metadata), type(channel_metadata)
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        self._ts = xr.DataArray([1], coords=[("time", [1])])
        self.update_xarray_metadata()
        
        if data is not None:
            self.ts = data

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

    def __str__(self):
        return self.ts.__str__()

    def __repr__(self):
        return self.ts.__repr__()

    ###-------------------------------------------------------------
    ## make sure some attributes have the correct data type
    # make sure that the time series is a pandas data frame
    @property
    def _class_name(self):
        return self.__class__.__name__

    @property
    def ts(self):
        return self._ts

    @ts.setter
    def ts(self, ts_arr):
        """
        if setting ts with a pandas data frame, make sure the data is in a
        column name 'data'
        """
        if isinstance(ts_arr, np.ndarray):
            dt = self._make_dt_coordinates(self.start, self.sample_rate, ts_arr.size)
            self._ts = xr.DataArray(ts_arr, coords=[("time", dt)])
            self.update_xarray_metadata()

        elif isinstance(ts_arr, pd.core.frame.DataFrame):
            try:
                dt = self._make_dt_index(
                    self.start_time_utc, self.sample_rate, ts_arr["data"].size
                )
                self._ts = xr.DataArray(ts_arr["data"], coords=[("time", dt)])

            except AttributeError:
                msg = (
                    "Data frame needs to have a column named `data` "
                    + "where the time series data is stored"
                )
                self.logger.error(msg)
                raise MTTSError(msg)

        elif isinstance(ts_arr, xr.DataArray):
            # TODO: need to validate the input xarray
            self._ts = ts_arr
            meta_dict = dict([(k, v) for k, v in ts_arr.attrs.items()])
            self.metadata.from_dict({self.metadata.type: meta_dict})

        else:
            msg = (
                "Data type {0} not supported".format(type(ts_arr))
                + ", ts needs to be a numpy.ndarray, pandas DataFrame, "
                + "or xarray.DataArray."
            )
            raise MTTSError(msg)

    def update_xarray_metadata(self):
        """
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.metadata.time_period.start = self.start_time_utc
        self.metadata.time_period.end = self.end_time_utc
        self.metadata.sample_rate = self.sample_rate

        self._ts.attrs.update(self.metadata.to_dict()[self.metadata._class_name])

    # --> number of samples just to make sure there is consistency
    @property
    def n_samples(self):
        """number of samples"""
        return int(self.ts.size)

    @n_samples.setter
    def n_samples(self, n_samples):
        """number of samples (int)"""
        self.logger.warning(
            "Cannot set the number of samples, " + "Use `MTTS.resample`"
        )

    def _check_for_index(self):
        """
        check to see if there is an index in the time series
        """
        if len(self._ts) > 1:
            return True
        else:
            return False

    # --> sample rate
    @property
    def sample_rate(self):
        """sample rate in samples/second"""
        if self._check_for_index():
            sr = 1e9 / self._ts.coords.indexes["time"][0].freq.nanos
        else:
            self.logger.debug(
                "Data has not been set yet, " + " sample rate is from metadata"
            )
            sr = self.metadata.sample_rate
            if sr is None:
                sr = 0.0
        return np.round(sr, 0)

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        """
        sample rate in samples/second

        type float
        """
        self.metadata.sample_rate = sample_rate
        self.logger.warning(
            "Setting MTTS.metadata.sample_rate. "
            + "If you want to change the time series sample"
            + " rate use method `resample`."
        )

    ## set time and set index
    @property
    def start(self):
        """MTime object"""
        if self._check_for_index():
            return MTime(self._ts.coords.indexes["time"][0].isoformat())
        else:
            self.logger.debug(
                "Data not set yet, pulling start time from "
                + "metadata.time_period.start"
            )
            return MTime(self.metadata.time_period.start)

    @start.setter
    def start(self, start_time):
        """
        start time of time series in UTC given in some format or a datetime
        object.

        Resets epoch seconds if the new value is not equivalent to previous
        value.

        Resets how the ts data frame is indexed, setting the starting time to
        the new start time.
        """

        if not isinstance(start_time, MTime):
            start_time = MTime(start_time)

        self.metadata.time_period.start = start_time.iso_str
        if self._check_for_index():
            if start_time == MTime(self.ts.coords.indexes["time"][0].isoformat()):
                return
            else:
                new_dt = self._make_dt_coordinates(
                    start_time, self.sample_rate, self.n_samples
                )
                self.ts.coords["time"] = new_dt

        # make a time series that the data can be indexed by
        else:
            self.logger.warning("No data, just updating metadata start")

    @property
    def end(self):
        """MTime object"""
        if self._check_for_index():
            return MTime(self._ts.coords.indexes["time"][-1].isoformat())
        else:
            self.logger.info(
                "Data not set yet, pulling end time from " + "metadata.time_period.end"
            )
            return MTime(self.metadata.time_period.end)

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
            "Cannot set `end`. If you want a slice, then "
            + "use MTTS.ts.sel['time'=slice(start, end)]"
        )

        # if not isinstance(end_time, MTime):
        #     end_time = MTime(end_time)

        # self.metadata.time_period.end = end_time.iso_str
        # if self._check_for_index():
        #     if start_time == MTime(self.ts.coords.indexes['time'][0].isoformat()):
        #         return
        #     else:
        #         new_dt = self._make_dt_coordinates(start_time,
        #                                            self.sample_rate,
        #                                            self.n_samples)
        #         self.ts.coords['time'] = new_dt

        # # make a time series that the data can be indexed by
        # else:
        #     self.logger.warning("No data, just updating metadata start")


    def _make_dt_coordinates(self, start_time, sample_rate, n_samples):
        """
        get the date time index from the data

        :param start_time: start time in time format
        :type start_time: string
        """
        if len(self.ts) == 0:
            return

        if sample_rate in [0, None]:
            msg = (
                f"Need to input a valid sample rate. Not {sample_rate}, "
                + "returning a time index assuming a sample rate of 1"
            )
            self.logger.warning(msg)
            sample_rate = 1

        if start_time is None:
            msg = (
                f"Need to input a valid sample rate. Not {start_time}, "
                + "returning a time index with start time of "
                + "1980-01-01T00:00:00"
            )
            self.logger.warning(msg)
            start_time = "1980-01-01T00:00:00"

        if n_samples < 1:
            msg = f"Need to input a valid n_samples. Not {n_samples}"
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(start_time, MTime):
            start_time = MTime(start_time)

        dt_freq = "{0:.0f}N".format(1.0e9 / (sample_rate))

        dt_index = pd.date_range(
            start=start_time.iso_str.split("+", 1)[0], periods=n_samples, freq=dt_freq
        )

        return dt_index

    def get_slice(self, start, end):
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

        if not isinstance(start, MTime):
            start = MTime(start)
        if not isinstance(end, MTime):
            end = MTime(end)

        new_ts = self.ts.loc[
            (self.ts.indexes["time"] >= start.iso_no_tz)
            & (self.ts.indexes["time"] <= end.iso_no_tz)
        ]
        new_ts.attrs["time_period.start"] = new_ts.coords.indexes["time"][0].isoformat()
        new_ts.attrs["time_period.end"] = new_ts.coords.indexes["time"][-1].isoformat()

        return new_ts

    # decimate data
    def resample(self, dec_factor=1, inplace=False):
        """
        decimate the data by using scipy.signal.decimate

        :param dec_factor: decimation factor
        :type dec_factor: int

        * refills ts.data with decimated data and replaces sample_rate

        """

        new_dt_freq = "{0:.0f}N".format(1e9 / (self.sample_rate / dec_factor))

        new_ts = self.ts.resample(time=new_dt_freq).nearest(tolerance=new_dt_freq)
        new_ts.attrs["sample_rate"] = self.sample_rate / dec_factor
        self.metadata.sample_rate = new_ts.attrs["sample_rate"]

        if inplace:
            self.ts = new_ts

        else:
            new_ts.attrs.update(self.metadata.to_dict()[self.metadata._class_name])
            # return new_ts
            return MTTS(self.metadata.type,
                             data=new_ts,
                             metadata=self.metadata)

# =============================================================================
# run container
# =============================================================================
class RunTS():
    """
    holds all run ts in one aligned array
    
    components --> {'ex': ex_xarray, 'ey': ey_xarray}
    
    """
    
    def __init__(self, array_list=None):
        self.logger = logging.getLogger(f"{__name__}.{self._class_name}")
        self.metadata = metadata.Run()
        self._dataset = xr.Dataset()
        
        if array_list is not None:
            self.build_dataset(array_list)
            
    @property
    def _class_name(self):
        return self.__class__.__name__
    
    def _validate_array_list(self, array_list):
        """ check to make sure all entries are a :class:`MTTS` object"""
        
        if not isinstance(array_list, (tuple, list)):
            msg = f"array_list must be a list or tuple, not {type(array_list)}"
            self.logger.error(msg)
            raise TypeError(msg)
            
        for index, item in enumerate(array_list):
            if not isinstance(item, MTTS):
                msg = f"array entry {index} must be MTTS object not {type(item)}"
                self.logger.error(msg)
                raise TypeError(msg)
                
        x_array_list = [x.ts for x in array_list]
        meta_list = dict([(x.metadata.component, 
                           x.metadata.to_dict()[list(x.metadata.to_dict().keys())[0]]) 
                          for x in array_list])   
            
        return x_array_list, meta_list
    
    def build_dataset(self, array_list, align_type='outer'):
        """
        
        :param array_list: list of xarrays
        :type array_list: list of :class:`mth5.timeseries.MTTS` objects
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
        x_array_list, meta_dict = self._validate_array_list(array_list)
        
        # first need to align the time series.
        x_array_list = xr.align(*x_array_list, join=align_type)
        
        # input as a dictionary
        xdict = dict([(x.component, x) for x in x_array_list])
        self._dataset = xr.Dataset(xdict)
        
        self._dataset.attrs.update(meta_dict)
        
    @property
    def dataset(self):
        return self._dataset
            
    @dataset.setter
    def dataset(self):
        msg = "Cannot set dataset, use build_dataset instead."
        self.logger.error(msg)
        raise AttributeError(msg)
        
    @property
    def start(self):
        return self.dataset.coords['time'].to_index()[0].isoformat()
    
    @property
    def end(self):
        return self.dataset.coords['time'].to_index()[-1].isoformat()
    
    @property
    def sample_rate(self):
        return 1E9/self.dataset.coords['time'].to_index().freq.n
        
        
        
    
    
        
    
    



