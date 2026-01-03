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
from __future__ import annotations

import inspect

import numpy as np
import scipy
import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mt_metadata import timeseries as metadata
from mt_metadata.common.list_dict import ListDict
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries.filters import ChannelResponse
from obspy.core import Stream

from .channel_ts import ChannelTS
from .ts_helpers import get_decimation_sample_rates, make_dt_coordinates


# =============================================================================
# make a dictionary of available metadata classes
# =============================================================================
meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))


# =============================================================================
# run container
# =============================================================================
class RunTS:
    """
    Container for MT time series data from a single run.

    Holds all run time series in one aligned xarray Dataset with channels as
    data variables and time as the coordinate. Manages metadata for survey,
    station, and run levels.

    Parameters
    ----------
    array_list : list[ChannelTS] | list[xr.DataArray] | xr.Dataset | None, optional
        List of ChannelTS objects, xarray DataArrays, or an xarray Dataset
        containing the time series data. All channels will be aligned to a
        common time index.
    run_metadata : metadata.Run | dict | None, optional
        Metadata for the run. Can be a Run object or dictionary.
    station_metadata : metadata.Station | dict | None, optional
        Metadata for the station. Can be a Station object or dictionary.
    survey_metadata : metadata.Survey | dict | None, optional
        Metadata for the survey. Can be a Survey object or dictionary.

    Attributes
    ----------
    dataset : xr.Dataset
        xarray Dataset containing all channel data with time coordinate
    survey_metadata : metadata.Survey
        Survey-level metadata
    station_metadata : metadata.Station
        Station-level metadata
    run_metadata : metadata.Run
        Run-level metadata
    filters : dict[str, Filter]
        Dictionary of channel response filters keyed by filter name
    sample_rate : float
        Sample rate in samples per second
    channels : list[str]
        List of channel names in the dataset

    Examples
    --------
    Create an empty RunTS:

    >>> from mth5.timeseries import RunTS
    >>> run = RunTS()

    Create RunTS from ChannelTS objects:

    >>> from mth5.timeseries import ChannelTS, RunTS
    >>> ex = ChannelTS('electric', data=ex_data,
    ...                channel_metadata={'component': 'ex'})
    >>> ey = ChannelTS('electric', data=ey_data,
    ...                channel_metadata={'component': 'ey'})
    >>> run = RunTS(array_list=[ex, ey])
    >>> print(run.channels)
    ['ex', 'ey']

    Access individual channels:

    >>> ex_channel = run.ex  # Returns ChannelTS object
    >>> print(ex_channel.sample_rate)
    256.0

    See Also
    --------
    ChannelTS : Individual channel time series container

    Notes
    -----
    When multiple channels are provided with different start/end times,
    they will be automatically aligned using the earliest start and latest
    end times, with NaN values filling gaps.

    """

    def __init__(
        self,
        array_list: list[ChannelTS] | list[xr.DataArray] | xr.Dataset | None = None,
        run_metadata: metadata.Run | dict | None = None,
        station_metadata: metadata.Station | dict | None = None,
        survey_metadata: metadata.Survey | dict | None = None,
    ) -> None:
        self.logger = logger
        self._survey_metadata = self._initialize_metadata()
        self._dataset = xr.Dataset()
        self._filters: dict[str, ChannelResponse] = {}

        self.survey_metadata = survey_metadata
        self.station_metadata = station_metadata
        self.run_metadata = run_metadata

        self._sample_rate: float | None = self._check_sample_rate_at_init()

        # load the arrays first this will write run and station metadata
        if array_list is not None:
            self.dataset = array_list

    def _check_sample_rate_at_init(self) -> float | None:
        """
        Check if sample_rate is specified in run_metadata at initialization.

        Returns
        -------
        float | None
            Sample rate from run_metadata if available, otherwise None.

        Notes
        -----
        If data is subsequently loaded, a check will be done to ensure
        sample rates match. If they don't, the data sample_rate is used.

        """
        sr = None
        if self.run_metadata is not None:
            sr = self.run_metadata.sample_rate

        return sr

    def __str__(self) -> str:
        """String representation of RunTS."""
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

    def __repr__(self) -> str:
        """String representation of RunTS."""
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """
        Test equality between two RunTS objects.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            True if objects are equal, False otherwise.

        Raises
        ------
        TypeError
            If other is not a RunTS object.
        """
        if not isinstance(other, RunTS):
            raise TypeError(f"Cannot compare RunTS with {type(other)}.")
        if not other.survey_metadata == self.survey_metadata:
            return False
        if not other.station_metadata == self.station_metadata:
            return False
        if not other.run_metadata == self.run_metadata:
            return False
        if self.dataset.equals(other.dataset) is False:
            return False
        return True

    def __neq__(self, other: object) -> bool:
        """Test inequality between two RunTS objects."""
        return not self.__eq__(other)

    def __add__(self, other: RunTS) -> RunTS:
        """
        Add two runs together to create a combined run.

        Combines runs using the following steps:

        1. xr.combine_by_coords([original, other])
        2. Compute monotonic time index spanning full time range
        3. Interpolate to new time index using slinear method

        Parameters
        ----------
        other : RunTS
            Another RunTS object to combine with this one.

        Returns
        -------
        RunTS
            Combined run with monotonic time index and metadata from the
            first run.

        Raises
        ------
        TypeError
            If input is not a RunTS object.

        Examples
        --------
        >>> run1 = RunTS(array_list=[ex1, ey1])
        >>> run2 = RunTS(array_list=[ex2, ey2])
        >>> combined = run1 + run2
        >>> print(combined.start, combined.end)

        Notes
        -----
        For more control over the merging process (gap filling method,
        resampling, etc.), use the `merge()` method instead.

        See Also
        --------
        merge : More flexible merging with customization options

        """
        if not isinstance(other, RunTS):
            raise TypeError(f"Cannot combine {type(other)} with RunTS.")
        # combine into a data set use override to keep attrs from original
        combined_ds = xr.combine_by_coords(
            [self.dataset, other.dataset], combine_attrs="override"
        )

        n_samples = (
            self.sample_rate
            * float(combined_ds.time.max().values - combined_ds.time.min().values)
            / 1e9
        ) + 1

        new_dt_index = make_dt_coordinates(
            combined_ds.time.min().values, self.sample_rate, n_samples
        )

        new_run = RunTS(
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )

        new_run.dataset = combined_ds.interp(time=new_dt_index, method="slinear")

        new_run.run_metadata.update_time_period()
        new_run.station_metadata.update_time_period()
        new_run.survey_metadata.update_time_period()
        new_run.filters = self.filters
        new_run.filters.update(other.filters)

        return new_run

    def _initialize_metadata(self) -> metadata.Survey:
        """
        Create a hierarchical metadata structure with default values.

        Creates a Survey object containing a Station which contains a Run,
        all with default IDs of "0". This provides the base structure for
        storing metadata at all levels.

        Returns
        -------
        metadata.Survey
            Survey metadata object with nested station and run.

        """

        survey_metadata = metadata.Survey(id="0")
        survey_metadata.stations.append(metadata.Station(id="0"))
        survey_metadata.stations[0].runs.append(metadata.Run(id="0"))

        return survey_metadata

    def _validate_run_metadata(self, run_metadata: metadata.Run | dict) -> metadata.Run:
        """
        Validate and convert run metadata to proper format.

        Parameters
        ----------
        run_metadata : metadata.Run | dict
            Run metadata as a Run object or dictionary.

        Returns
        -------
        metadata.Run
            Validated Run metadata object (copy of input).

        Raises
        ------
        TypeError
            If input is neither a Run object nor a dictionary.

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
                msg = (
                    "input metadata must be type {type(self.run_metadata)} "
                    "or dict, not {type(run_metadata)}"
                )
                self.logger.error(msg)
                raise TypeError(msg)
        return run_metadata.copy()

    def _validate_station_metadata(
        self, station_metadata: metadata.Station | dict
    ) -> metadata.Station:
        """
        Validate and convert station metadata to proper format.

        Parameters
        ----------
        station_metadata : metadata.Station | dict
            Station metadata as a Station object or dictionary.

        Returns
        -------
        metadata.Station
            Validated Station metadata object (copy of input).

        Raises
        ------
        TypeError
            If input is neither a Station object nor a dictionary.
        """

        if not isinstance(station_metadata, metadata.Station):
            if isinstance(station_metadata, dict):
                if "station" not in [cc.lower() for cc in station_metadata.keys()]:
                    station_metadata = {"Station": station_metadata}
                st_metadata = metadata.Station()
                st_metadata.from_dict(station_metadata)
                self.logger.debug("Loading from metadata dict")
                return st_metadata
            else:
                msg = (
                    f"input metadata must be type {type(self.station_metadata)} "
                    "or dict, not {type(station_metadata)}"
                )
                self.logger.error(msg)
                raise TypeError(msg)
        return station_metadata.copy()

    def _validate_survey_metadata(
        self, survey_metadata: metadata.Survey | dict
    ) -> metadata.Survey:
        """
        Validate and convert survey metadata to proper format.

        Parameters
        ----------
        survey_metadata : metadata.Survey | dict
            Survey metadata as a Survey object or dictionary.

        Returns
        -------
        metadata.Survey
            Validated Survey metadata object (copy of input).

        Raises
        ------
        TypeError
            If input is neither a Survey object nor a dictionary.
        """

        if not isinstance(survey_metadata, metadata.Survey):
            if isinstance(survey_metadata, dict):
                if "survey" not in [cc.lower() for cc in survey_metadata.keys()]:
                    survey_metadata = {"Survey": survey_metadata}
                sv_metadata = metadata.Survey()
                sv_metadata.from_dict(survey_metadata)
                self.logger.debug("Loading from metadata dict")
                return sv_metadata
            else:
                msg = (
                    f"input metadata must be type {type(self.survey_metadata)} "
                    "or dict, not {type(survey_metadata)}"
                )
                self.logger.error(msg)
                raise TypeError(msg)
        return survey_metadata.copy()

    def _validate_array_list(
        self, array_list: list[ChannelTS] | list[xr.DataArray] | tuple
    ) -> list[xr.DataArray]:
        """
        Validate and convert array list to proper format.

        Checks that all entries are ChannelTS or xarray.DataArray objects,
        converts to DataArray format, extracts metadata and filters, and
        aligns all channels to a common time index.

        Parameters
        ----------
        array_list : list[ChannelTS] | list[xr.DataArray] | tuple
            List or tuple of ChannelTS objects or xarray DataArrays.

        Returns
        -------
        list[xr.DataArray]
            List of validated and aligned xarray DataArrays.

        Raises
        ------
        TypeError
            If array_list is not a list or tuple, or if any element is not
            a ChannelTS or DataArray object.

        Notes
        -----
        This method also updates the station and run metadata from the
        ChannelTS objects if present, and extracts channel response filters.

        """
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

                if item.station_metadata.id not in ["0", None, ""]:
                    if station_metadata.id not in ["0", None, ""]:
                        station_metadata.update(item.station_metadata, match=["id"])
                    else:
                        station_metadata.update(item.station_metadata)
                if item.run_metadata.id not in ["0", None, ""]:
                    if run_metadata.id not in ["0", None, ""]:
                        run_metadata.update(item.run_metadata, match=["id"])
                    else:
                        run_metadata.update(item.run_metadata)
                channels.append(item.channel_metadata)

                # get the filters from the channel
                if item.channel_response.filters_list != []:
                    for ff in item.channel_response.filters_list:
                        self._filters[ff.name] = ff
            else:
                valid_list.append(item)
        # need to make sure that the station metadata was actually updated,
        # should have an ID.
        run_metadata.channels = channels
        if station_metadata.id not in ["0", None, ""]:
            station_metadata.runs = ListDict()
            station_metadata.runs.append(run_metadata)
            # need to add the other runs that are in the metadata for
            # completeness.
            for run in self.station_metadata.runs:
                if run.id not in [run_metadata.id, "0", None, ""]:
                    station_metadata.add_run(run)

            if self.station_metadata.id != station_metadata.id:
                logger.warning(
                    f"Station ID {station_metadata.id} from ChannelTS does "
                    "not match original station ID {self.station_metadata.id}. "
                    "Updating ID to match."
                )
            self.station_metadata = station_metadata
        # if the run metadata was updated
        elif run_metadata.id not in ["0", None, ""]:
            if self.run_metadata.id != run_metadata.id:
                logger.warning(
                    f"Run ID {run_metadata.id} from ChannelTS does "
                    "not match original run ID {self.run_metadata.id}. "
                    "Updating ID to match."
                )
            self.run_metadata = run_metadata
        # if the run metadata or station metadata was not updated from channel
        # metadata, then update just the channels.
        else:
            self.run_metadata.channels = channels
        # need to align the time series.
        valid_list = self._align_channels(valid_list)

        return valid_list

    def _align_channels(self, valid_list: list[xr.DataArray]) -> list[xr.DataArray]:
        """
        Align channels to a common time index.

        Checks for common start and end times across all channels. If not
        common, reindexes each channel to a new time index spanning from
        the earliest start to latest end at the common sample rate.

        Parameters
        ----------
        valid_list : list[xr.DataArray]
            List of channel DataArrays to align.

        Returns
        -------
        list[xr.DataArray]
            List of aligned DataArrays with common time index.

        Notes
        -----
        Uses 'nearest' method for reindexing with tolerance of one sample.
        Missing data is filled with NaN values.

        """

        earliest_start = self._get_earliest_start(valid_list)
        latest_end = self._get_latest_end(valid_list)
        reindex = False
        if not self._check_common_start(valid_list):
            self.logger.info(
                f"Channels do not have a common start, using earliest: {earliest_start}"
            )
            reindex = True
        if not self._check_common_end(valid_list):
            self.logger.info(
                f"Channels do not have a common end, using latest: {latest_end}"
            )
            reindex = True
        if reindex:
            sample_rate = self._check_sample_rate(valid_list)

            new_time_index = self._get_common_time_index(
                earliest_start, latest_end, sample_rate
            )
            tolerance = f"{(1e9 / sample_rate):.0f}N"
            aligned_list = []
            for ch in valid_list:
                aligned_list.append(
                    ch.reindex(
                        time=new_time_index,
                        method="nearest",
                        tolerance=tolerance,
                    )
                )
        else:
            aligned_list = valid_list
        return aligned_list

    def _check_sample_rate(self, valid_list: list[xr.DataArray]) -> float:
        """
        Check that all channels have the same sample rate.

        Parameters
        ----------
        valid_list : list[xr.DataArray]
            List of channel DataArrays.

        Returns
        -------
        float
            The common sample rate.

        Raises
        ------
        ValueError
            If channels have different sample rates.
        """
        sr_test = list(
            set(
                [(item.sample_rate) for item in valid_list]
                + [np.round(item.sps_filters.fs, 3) for item in valid_list]
            )
        )

        if len(sr_test) != 1:
            msg = f"sample rates are not all the same {sr_test}"
            self.logger.error(msg)
            raise ValueError(msg)
        return sr_test[0]

    def _check_common_start(self, valid_list: list[xr.DataArray]) -> bool:
        """
        Check if all channels have the same start time.

        Parameters
        ----------
        valid_list : list[xr.DataArray]
            List of channel DataArrays.

        Returns
        -------
        bool
            True if all channels start at the same time, False otherwise.

        """
        start_list = list(set([item.coords["time"].values[0] for item in valid_list]))
        if len(start_list) != 1:
            return False
        return True

    def _check_common_end(self, valid_list: list[xr.DataArray]) -> bool:
        """
        Check if all channels have the same end time.

        Parameters
        ----------
        valid_list : list[xr.DataArray]
            List of channel DataArrays.

        Returns
        -------
        bool
            True if all channels end at the same time, False otherwise.

        """
        end_list = list(set([item.coords["time"].values[-1] for item in valid_list]))
        if len(end_list) != 1:
            return False
        return True

    def _get_earliest_start(self, valid_list: list[xr.DataArray]) -> np.datetime64:
        """
        Get the earliest start time from all channels.

        Parameters
        ----------
        valid_list : list[xr.DataArray]
            List of channel DataArrays.

        Returns
        -------
        np.datetime64
            Earliest start time.
        """

        return min([item.coords["time"].values[0] for item in valid_list])

    def _get_latest_end(self, valid_list: list[xr.DataArray]) -> np.datetime64:
        """
        Get the latest end time from all channels.

        Parameters
        ----------
        valid_list : list[xr.DataArray]
            List of channel DataArrays.

        Returns
        -------
        np.datetime64
            Latest end time.
        """

        return max([item.coords["time"].values[-1] for item in valid_list])

    def _get_common_time_index(
        self, start: np.datetime64, end: np.datetime64, sample_rate: float
    ) -> np.ndarray:
        """
        Generate a common time index for channel alignment.

        Parameters
        ----------
        start : np.datetime64
            Start time.
        end : np.datetime64
            End time.
        sample_rate : float
            Sample rate in samples per second.

        Returns
        -------
        np.ndarray
            Array of datetime64 timestamps.
        """

        n_samples = int(sample_rate * float(end - start) / 1e9) + 1

        return make_dt_coordinates(start, sample_rate, n_samples)

    def _get_channel_response(self, ch_name: str) -> ChannelResponse:
        """
        Get the channel response filter from the filter dictionary.

        Parameters
        ----------
        ch_name : str
            Name of the channel.

        Returns
        -------
        ChannelResponse
            ChannelResponse object containing the filter list for the channel.

        Notes
        -----
        Looks for filters in the dataset attributes under 'filter.name' or
        'filters' keys and retrieves them from the internal filters dictionary.

        """

        filter_list = []
        if ch_name in self.dataset.keys():
            if "filter.name" in self.dataset[ch_name].attrs.keys():
                for filter_name in self.dataset[ch_name].attrs["filter.name"]:
                    try:
                        filter_list.append(self.filters[filter_name])
                    except KeyError:
                        self.logger.debug(f"Could not find {filter_name} in filters")
            elif "filters" in self.dataset[ch_name].attrs.keys():
                for ch_filter in self.dataset[ch_name].attrs["filters"]:
                    try:
                        filter_list.append(
                            self.filters[ch_filter["applied_filter"]["name"]]
                        )
                    except KeyError:
                        self.logger.debug(
                            f"Could not find {ch_filter['applied_filter']['name']} in filters"
                        )
        return ChannelResponse(filters_list=filter_list)

    def __getattr__(self, name: str) -> ChannelTS | None:
        """Enable accessing channels as attributes (e.g., run.ex)."""
        # change to look for keys directly and use type to set channel type
        if name in self.dataset.keys():
            ch_response_filter = self._get_channel_response(name)
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
                channel_response=ch_response_filter,
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

    def copy(self, data: bool = True) -> RunTS:
        """
        Create a copy of the RunTS object.

        Parameters
        ----------
        data : bool, optional
            If True, copy the data along with metadata. If False, only
            copy the metadata (default is True).

        Returns
        -------
        RunTS
            A copy of the RunTS object.

        Examples
        --------
        Create a copy with data:

        >>> run_copy = run.copy()

        Create a metadata-only copy:

        >>> run_meta = run.copy(data=False)
        >>> print(run_meta.has_data())
        False

        """

        if not data:
            return RunTS(
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )
        else:
            return RunTS(
                array_list=self.dataset,
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )

    ### Properties ------------------------------------------------------------
    @property
    def survey_metadata(self) -> metadata.Survey:
        """
        Survey metadata.

        Returns
        -------
        metadata.Survey
            Survey-level metadata object.
        """
        return self._survey_metadata

    @survey_metadata.setter
    def survey_metadata(self, survey_metadata: metadata.Survey | dict | None) -> None:
        """
        Set survey metadata.

        Parameters
        ----------
        survey_metadata : metadata.Survey | dict | None
            Survey metadata object or dictionary. If None, no action is taken.

        """

        if survey_metadata is not None:
            survey_metadata = self._validate_survey_metadata(survey_metadata)
            self._survey_metadata.update(survey_metadata)
            for station in survey_metadata.stations:
                if station.id not in self._survey_metadata.stations.keys():
                    self._survey_metadata.add_station(
                        self._validate_station_metadata(station), update=False
                    )

    @property
    def station_metadata(self) -> metadata.Station:
        """
        Station metadata.

        Returns
        -------
        metadata.Station
            Station-level metadata object (first station in survey).
        """

        return self.survey_metadata.stations[0]

    @station_metadata.setter
    def station_metadata(
        self, station_metadata: metadata.Station | dict | None
    ) -> None:
        """
        Set station metadata.

        Parameters
        ----------
        station_metadata : metadata.Station | dict | None
            Station metadata object or dictionary. If None, no action is taken.

        Notes
        -----
        Preserves existing run metadata and merges with new station metadata.

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
    def run_metadata(self) -> metadata.Run:
        """
        Run metadata.

        Returns
        -------
        metadata.Run
            Run-level metadata object (first run in first station).
        """
        run_metadata = self.survey_metadata.stations[0].runs[0]

        return run_metadata

    @run_metadata.setter
    def run_metadata(self, run_metadata: metadata.Run | dict | None) -> None:
        """
        Set run metadata.

        Parameters
        ----------
        run_metadata : metadata.Run | dict | None
            Run metadata object or dictionary. If None, no action is taken.

        Notes
        -----
        Updates the runs list while preserving other existing runs.

        """

        if run_metadata is not None:
            run_metadata = self._validate_run_metadata(run_metadata)
            runs = ListDict()
            runs.append(run_metadata)
            runs.extend(self.station_metadata.runs, skip_keys=[run_metadata.id, "0"])
            self._survey_metadata.stations[0].runs = runs

    def has_data(self) -> bool:
        """
        Check if the RunTS contains any data.

        Returns
        -------
        bool
            True if channels with data exist, False otherwise.

        Examples
        --------
        >>> run = RunTS()
        >>> print(run.has_data())
        False
        >>> run.add_channel(ex_channel)
        >>> print(run.has_data())
        True
        """
        if len(self.channels) > 0:
            return True
        return False

    @property
    def summarize_metadata(self) -> dict[str, any]:
        """
        Get a summary of all channel metadata.

        Flattens the metadata from all channels into a single dictionary
        with keys in the format 'channel.attribute'.

        Returns
        -------
        dict[str, any]
            Dictionary with flattened metadata from all channels.

        Examples
        --------
        >>> meta_summary = run.summarize_metadata
        >>> print(meta_summary.keys())
        dict_keys(['ex.time_period.start', 'ex.sample_rate', ...])

        """
        meta_dict = {}
        for comp in self.dataset.data_vars:
            for mkey, mvalue in self.dataset[comp].attrs.items():
                meta_dict[f"{comp}.{mkey}"] = mvalue
        return meta_dict

    def validate_metadata(self) -> None:
        """
        Validate and synchronize metadata with dataset contents.

        Checks that metadata (start time, end time, sample rate, channels)
        matches the actual data in the dataset. Updates metadata from data
        if discrepancies are found.

        Notes
        -----
        This method is automatically called when setting the dataset. It:

        - Validates start and end times
        - Validates sample rate
        - Updates station and survey time periods
        - Logs warnings for any discrepancies found

        Examples
        --------
        >>> run.validate_metadata()
        >>> print(run.run_metadata.time_period.start)
        2020-01-01T00:00:00+00:00

        """

        # check sampling rate
        if self.has_data():
            # check start time
            if self.start != self.run_metadata.time_period.start:
                if self.run_metadata.time_period.start != "1980-01-01T00:00:00+00:00":
                    msg = (
                        f"start time of dataset {self.start} does not "
                        f"match metadata start {self.run_metadata.time_period.start} "
                        f"updating metatdata value to {self.start}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.time_period.start = self.start.isoformat()
            # check end time
            if self.end != self.run_metadata.time_period.end:
                if self.run_metadata.time_period.end != "1980-01-01T00:00:00+00:00":
                    msg = (
                        f"end time of dataset {self.end} does not "
                        f"match metadata end {self.run_metadata.time_period.end} "
                        f"updating metatdata value to {self.end}"
                    )
                    self.logger.warning(msg)
                self.run_metadata.time_period.end = self.end.isoformat()
            # check sample rate
            data_sr = self._compute_sample_rate()
            if self.sample_rate != data_sr:
                if self.run_metadata.sample_rate not in [0.0, None]:
                    msg = (
                        f"sample rate of dataset {data_sr} does not "
                        f"match metadata sample rate {self.sample_rate} "
                        f"updating metatdata value to {data_sr}"
                    )
                    self.logger.critical(msg)
                self._sample_rate = data_sr
                self.run_metadata.sample_rate = data_sr

            if self.sample_rate != self.run_metadata.sample_rate:
                msg = (
                    f"sample rate of dataset {data_sr} does not "
                    f"match metadata sample rate {self.sample_rate} "
                    f"updating metatdata value to {data_sr}"
                )
                self.logger.critical(msg)
                self.run_metadata.sample_rate = self._sample_rate
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

            # input as a dictionary
            xdict = dict([(x.component.lower(), x) for x in x_array_list])
            self._dataset = xr.Dataset(xdict)
        elif isinstance(array_list, xr.Dataset):
            self._dataset = array_list
        self.validate_metadata()
        self._dataset.attrs.update(self.run_metadata.to_dict(single=True))

    def add_channel(self, channel: xr.DataArray | ChannelTS) -> None:
        """
        Add a channel to the dataset.

        The channel must have the same sample rate and time coordinates that
        are compatible with the existing dataset. If start times don't match,
        NaN values will be placed where timing doesn't align.

        Parameters
        ----------
        channel : xr.DataArray | ChannelTS
            A channel as an xarray DataArray or ChannelTS object to add.

        Raises
        ------
        ValueError
            If the channel has a different sample rate than the run, or if
            the input is not a DataArray or ChannelTS.

        Examples
        --------
        Add a ChannelTS:

        >>> hz = ChannelTS('magnetic', data=hz_data,
        ...                channel_metadata={'component': 'hz'})
        >>> run.add_channel(hz)
        >>> print(run.channels)
        ['ex', 'ey', 'hx', 'hy', 'hz']

        Add an xarray DataArray:

        >>> import xarray as xr
        >>> data_array = xr.DataArray(data, coords={'time': times})
        >>> run.add_channel(data_array)

        """

        if isinstance(channel, xr.DataArray):
            c = ChannelTS()
            c.ts = channel
        elif isinstance(channel, ChannelTS):
            c = channel
            self.run_metadata.channels.append(c.channel_metadata)
            for ff in c.channel_response.filters_list:
                self._filters[ff.name] = ff
        else:
            raise ValueError("Input Channel must be type xarray.DataArray or ChannelTS")
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
            self.dataset = c.data_array.to_dataset()
        else:
            self.dataset = xr.merge([self.dataset, c.data_array.to_dataset()])

    @property
    def dataset(self) -> xr.Dataset:
        """
        The xarray Dataset containing all channel data.

        Returns
        -------
        xr.Dataset
            Dataset with channels as data variables and time as coordinate.

        Examples
        --------
        >>> print(run.dataset)
        <xarray.Dataset>
        Dimensions:  (time: 4096)
        Coordinates:
          * time     (time) datetime64[ns] ...
        Data variables:
            ex       (time) float64 ...
            ey       (time) float64 ...
        """
        return self._dataset

    @dataset.setter
    def dataset(
        self, array_list: list[ChannelTS] | list[xr.DataArray] | xr.Dataset
    ) -> None:
        """
        Set the dataset.

        Parameters
        ----------
        array_list : list[ChannelTS] | list[xr.DataArray] | xr.Dataset
            Data to set as the dataset.

        Notes
        -----
        Data will be aligned using min and max times. For different alignment,
        use set_dataset() with the align_type parameter.
        """
        msg = (
            "Data will be aligned using the min and max time. "
            "If that is not correct use set_dataset and change the alignment type."
        )
        self.logger.debug(msg)
        self.set_dataset(array_list)

    @property
    def start(self) -> MTime:
        """
        Start time of the run in UTC.

        Returns
        -------
        MTime
            Start time from the dataset if data exists, otherwise from
            run_metadata.

        Examples
        --------
        >>> print(run.start)
        2020-01-01T00:00:00+00:00
        """
        if self.has_data():
            return MTime(
                time_stamp=self.dataset.coords["time"].to_index()[0].isoformat()
            )
        return self.run_metadata.time_period.start

    @property
    def end(self) -> MTime:
        """
        End time of the run in UTC.

        Returns
        -------
        MTime
            End time from the dataset if data exists, otherwise from
            run_metadata.

        Examples
        --------
        >>> print(run.end)
        2020-01-01T01:00:00+00:00
        """
        if self.has_data():
            return MTime(
                time_stamp=self.dataset.coords["time"].to_index()[-1].isoformat()
            )
        return self.run_metadata.time_period.end

    def _compute_sample_rate(self) -> float:
        """
        Compute sample rate from the time coordinate.

        Returns
        -------
        float
            Sample rate in samples per second, rounded to nearest integer.

        Raises
        ------
        ValueError
            If time indexing fails.

        Notes
        -----
        Uses scipy.stats.mode to find the most common time difference.

        """

        try:
            dt_array = np.diff(self.dataset.coords["time"].to_index()) / np.timedelta64(
                1, "s"
            )
            best_dt, counts = scipy.stats.mode(dt_array)
            return round(
                1.0 / np.float64(best_dt),
                0,
            )
        except AttributeError:
            self.logger.warning("Something weird happend with xarray time indexing")

            raise ValueError("Something weird happend with xarray time indexing")

    @property
    def sample_rate(self) -> float:
        """
        Sample rate in samples per second.

        Returns
        -------
        float
            Sample rate estimated from time differences if data exists,
            otherwise from metadata.

        Examples
        --------
        >>> print(run.sample_rate)
        256.0
        """
        if self.has_data():
            if self._sample_rate is None:
                self._sample_rate = self._compute_sample_rate()

        return self._sample_rate

    @property
    def sample_interval(self) -> float:
        """
        Sample interval in seconds (inverse of sample_rate).

        Returns
        -------
        float
            Sample interval = 1 / sample_rate, or 0.0 if sample_rate is 0.

        Examples
        --------
        >>> print(run.sample_interval)
        0.00390625  # for 256 Hz

        """

        if self.sample_rate != 0:
            return 1.0 / self.sample_rate
        return 0.0

    @property
    def channels(self) -> list[str]:
        """
        List of channel names in the dataset.

        Returns
        -------
        list[str]
            List of channel component names (e.g., ['ex', 'ey', 'hx']).

        Examples
        --------
        >>> print(run.channels)
        ['ex', 'ey', 'hx', 'hy', 'hz']
        """
        return [cc for cc in list(self.dataset.data_vars)]

    @property
    def filters(self) -> dict[str, ChannelResponse]:
        """
        Dictionary of channel response filters.

        Returns
        -------
        dict[str, ChannelResponse]
            Dictionary keyed by filter name containing ChannelResponse objects.

        Examples
        --------
        >>> print(run.filters.keys())
        dict_keys(['v_to_counts', 'dipole_100m'])
        """
        return self._filters

    @filters.setter
    def filters(self, value: dict[str, ChannelResponse]) -> None:
        """
        Set the filters dictionary.

        Parameters
        ----------
        value : dict[str, ChannelResponse]
            Dictionary of filter name to ChannelResponse object mappings.

        Raises
        ------
        TypeError
            If value is not a dictionary.

        Notes
        -----
        Use dictionary methods (update, etc.) to modify the filters dict.

        """
        if not isinstance(value, dict):
            raise TypeError("input must be a dictionary")
        self._filters = value

    def to_obspy_stream(
        self, network_code: str | None = None, encoding: str | None = None
    ) -> Stream:
        """
        Convert time series to an ObsPy Stream object.

        Creates an ObsPy Stream containing a Trace for each channel in the run.

        Parameters
        ----------
        network_code : str | None, optional
            Two-letter network code provided by FDSN DMC. If None, uses
            station metadata.
        encoding : str | None, optional
            Data encoding format (e.g., 'STEIM2', 'FLOAT64'). If None, uses
            default encoding.

        Returns
        -------
        obspy.core.Stream
            Stream object containing Trace objects for all channels.

        Examples
        --------
        >>> stream = run.to_obspy_stream(network_code='MT')
        >>> print(stream)
        3 Trace(s) in Stream:
        MT.MT001..EX | 2020-01-01T00:00:00 - ... | 256.0 Hz, 4096 samples

        See Also
        --------
        from_obspy_stream : Create RunTS from ObsPy Stream
        ChannelTS.to_obspy_trace : Convert single channel

        """

        trace_list = []
        for channel in self.channels:
            ts_obj = getattr(self, channel)
            trace_list.append(
                ts_obj.to_obspy_trace(network_code=network_code, encoding=encoding)
            )
        return Stream(traces=trace_list)

    def wrangle_leap_seconds_from_obspy(
        self, array_list: list[ChannelTS]
    ) -> list[ChannelTS]:
        """
        Handle potential leap second issues from ObsPy streams.

        Removes runs with only one sample that are numerically identical to
        adjacent samples, which may be artifacts of leap second handling.

        Parameters
        ----------
        array_list : list[ChannelTS]
            List of ChannelTS objects from ObsPy conversion.

        Returns
        -------
        list[ChannelTS]
            Filtered list with single-sample runs removed.

        Notes
        -----
        This is experimental handling for issue #169. The exact behavior of
        ObsPy's leap second handling is not fully documented.

        """
        msg = f"Possible Leap Second Bug -- see issue #169"
        self.logger.warning(msg)
        return [x for x in array_list if x.n_samples != 1]

    def from_obspy_stream(
        self, obspy_stream: Stream, run_metadata: metadata.Run | None = None
    ) -> None:
        """
        Populate the run from an ObsPy Stream object.

        Converts each Trace in the Stream to a ChannelTS and adds it to the
        run. Updates metadata from the ObsPy trace headers.

        Parameters
        ----------
        obspy_stream : obspy.core.Stream
            ObsPy Stream object containing one or more Trace objects.
        run_metadata : metadata.Run | None, optional
            Additional run metadata to apply. If provided, will be merged
            with metadata from the Stream.

        Raises
        ------
        TypeError
            If obspy_stream is not an ObsPy Stream object.

        Examples
        --------
        >>> from obspy import read
        >>> stream = read('data.mseed')
        >>> run = RunTS()
        >>> run.from_obspy_stream(stream)
        >>> print(run.channels)
        ['ex', 'ey', 'hx', 'hy', 'hz']

        Notes
        -----
        Component names are automatically mapped:

        - e1 -> ex, e2 -> ey
        - h1 -> hx, h2 -> hy, h3 -> hz

        See Also
        --------
        to_obspy_stream : Convert to ObsPy Stream
        ChannelTS.from_obspy_trace : Convert single trace

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
                    self.logger.warning(f"could not find {channel_ts.component}")

            # else:
            #     run_metadata = metadata.Run(id="001")
            station_list.append(channel_ts.station_metadata.fdsn.id)

            array_list.append(channel_ts)
        ### need to merge metadata into something useful, station name is the only
        ### name that is preserved

        ### TODO need to updata run metadata and station metadata to reflect the
        ### renaming of the channels.
        try:
            station = list(set([ss for ss in station_list if ss is not None]))[0]
        except IndexError:
            station = None
            msg = "Could not find station name"
            self.logger.warn(msg)
        self.station_metadata.fdsn.id = station

        if len(run_metadata.channels) != len(array_list):
            array_list = self.wrangle_leap_seconds_from_obspy(array_list)
        self.set_dataset(array_list)

        # need to be sure update any input metadata.
        if run_metadata is not None:
            self.station_metadata.runs = ListDict()
            self.station_metadata.add_run(run_metadata)
        self.validate_metadata()

    def get_slice(
        self,
        start: str | MTime,
        end: str | MTime | None = None,
        n_samples: int | None = None,
    ) -> RunTS:
        """
        Extract a time slice from the run.

        Gets a chunk of data from the run, finding the closest points to the
        given parameters. Uses pandas slice_indexer for robust slicing.

        Parameters
        ----------
        start : str | MTime
            Start time of the slice (ISO format string or MTime object).
        end : str | MTime | None, optional
            End time of the slice. Required if n_samples not provided.
        n_samples : int | None, optional
            Number of samples to get. Required if end not provided.

        Returns
        -------
        RunTS
            New RunTS object containing the requested slice with copies of
            metadata and filters.

        Raises
        ------
        ValueError
            If neither end nor n_samples is provided.

        Examples
        --------
        Get slice by start and end times:

        >>> slice1 = run.get_slice('2020-01-01T00:00:00',
        ...                         '2020-01-01T00:01:00')
        >>> print(slice1.start, slice1.end)

        Get slice by start time and number of samples:

        >>> slice2 = run.get_slice('2020-01-01T00:00:00', n_samples=1024)
        >>> print(len(slice2.dataset.time))
        1024

        Notes
        -----
        Uses pandas slice_indexer which handles near-matches better than
        xarray's native slicing. The actual slice may be slightly adjusted
        to match available data points.

        """
        if not isinstance(start, MTime):
            start = MTime(time_stamp=start)
        if n_samples is not None:
            seconds = (n_samples - 1) / self.sample_rate
            end = start + seconds
        elif end is not None:
            if not isinstance(end, MTime):
                end = MTime(time_stamp=end)
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

    def calibrate(self, **kwargs) -> RunTS:
        """
        Remove instrument response from all channels.

        Applies the channel response filters to calibrate each channel,
        creating a new run with calibrated data.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to each channel's
            remove_instrument_response method.

        Returns
        -------
        RunTS
            New RunTS object with calibrated channels.

        Examples
        --------
        >>> calibrated_run = run.calibrate()
        >>> # Calibration typically converts from counts to physical units

        See Also
        --------
        ChannelTS.remove_instrument_response : Calibrate single channel

        """

        new_run = RunTS(run_metadata=self.run_metadata)
        new_run.station_metadata = self.station_metadata

        for channel in self.channels:
            ch_ts = getattr(self, channel)
            calibrated_ch_ts = ch_ts.remove_instrument_response(**kwargs)
            new_run.add_channel(calibrated_ch_ts)
        return new_run

    def decimate(
        self, new_sample_rate: float, inplace: bool = False, max_decimation: int = 8
    ) -> RunTS | None:
        """
        Decimate data to a new sample rate using multi-stage decimation.

        Applies FIR filtering and downsampling in multiple stages to achieve
        the target sample rate while preventing aliasing.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in samples per second.
        inplace : bool, optional
            If True, modify the current run. If False, return a new run
            (default is False).
        max_decimation : int, optional
            Maximum decimation factor for each stage (default is 8).

        Returns
        -------
        RunTS | None
            If inplace is False, returns new decimated RunTS. Otherwise None.

        Examples
        --------
        Decimate from 256 Hz to 1 Hz:

        >>> decimated_run = run.decimate(1.0)
        >>> print(decimated_run.sample_rate)
        1.0

        Decimate in place:

        >>> run.decimate(16.0, inplace=True)
        >>> print(run.sample_rate)
        16.0

        Notes
        -----
        NaN values are filled with 0 before decimation to prevent NaN
        propagation. Multi-stage decimation is used to maintain signal
        quality and prevent aliasing.

        See Also
        --------
        resample_poly : Alternative resampling using polyphase filtering
        resample : Simple resampling without anti-aliasing

        """

        sr_list = get_decimation_sample_rates(
            self.sample_rate, new_sample_rate, max_decimation
        )
        # need to fill nans with 0 otherwise they wipeout the decimation values
        # and all becomes nan.
        new_ds = self.dataset.fillna(0)
        for step_sr in sr_list:
            new_ds = new_ds.sps_filters.decimate(step_sr)
        new_ds.attrs["sample_rate"] = new_sample_rate
        self.run_metadata.sample_rate = new_ds.attrs["sample_rate"]

        if inplace:
            self.dataset = new_ds
        else:
            # return new_ds
            return RunTS(
                new_ds,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                survey_metadata=self.survey_metadata,
            )

    def resample_poly(
        self, new_sample_rate: float, pad_type: str = "mean", inplace: bool = False
    ) -> RunTS | None:
        """
        Resample data using polyphase filtering.

        Uses scipy.signal.resample_poly to resample while applying an FIR
        filter to remove aliasing. Generally more accurate than simple
        resampling but slower than decimation.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in samples per second.
        pad_type : str, optional
            Padding method for edge effects: 'mean', 'median', 'zero'
            (default is 'mean').
        inplace : bool, optional
            If True, modify current run. If False, return new run
            (default is False).

        Returns
        -------
        RunTS | None
            If inplace is False, returns new resampled RunTS. Otherwise None.

        Examples
        --------
        Resample from 256 Hz to 100 Hz:

        >>> resampled_run = run.resample_poly(100.0)
        >>> print(resampled_run.sample_rate)
        100.0

        Notes
        -----
        NaN values are filled with 0 before resampling. The polyphase method
        is particularly good for arbitrary sample rate ratios.

        See Also
        --------
        decimate : Multi-stage decimation for downsampling
        resample : Simple nearest-neighbor resampling

        """

        # need to fill nans with 0 otherwise they wipeout the decimation values
        # and all becomes nan.
        new_ds = self.dataset.fillna(0)
        new_ds = new_ds.sps_filters.resample_poly(new_sample_rate, pad_type=pad_type)

        new_ds.attrs["sample_rate"] = new_sample_rate
        self.run_metadata.sample_rate = new_ds.attrs["sample_rate"]

        if inplace:
            self.dataset = new_ds
        else:
            return RunTS(
                new_ds,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                survey_metadata=self.survey_metadata,
            )

    def resample(self, new_sample_rate: float, inplace: bool = False) -> RunTS | None:
        """
        Resample data to a new sample rate using nearest-neighbor method.

        Simple resampling without anti-aliasing filtering. Use decimate or
        resample_poly for better quality when downsampling.

        Parameters
        ----------
        new_sample_rate : float
            Target sample rate in samples per second.
        inplace : bool, optional
            If True, modify current run. If False, return new run
            (default is False).

        Returns
        -------
        RunTS | None
            If inplace is False, returns new resampled RunTS. Otherwise None.

        Examples
        --------
        >>> resampled_run = run.resample(128.0)
        >>> print(resampled_run.sample_rate)
        128.0

        Warnings
        --------
        This method does not apply anti-aliasing filtering. For downsampling,
        consider using decimate() or resample_poly() instead.

        See Also
        --------
        decimate : Proper downsampling with anti-aliasing
        resample_poly : High-quality resampling with polyphase filtering

        """

        new_dt_freq = "{0:.0f}N".format(1e9 / (new_sample_rate))

        new_ds = self.dataset.resample(time=new_dt_freq).nearest(tolerance=new_dt_freq)
        new_ds.attrs["sample_rate"] = new_sample_rate
        self.run_metadata.sample_rate = new_ds.attrs["sample_rate"]

        if inplace:
            self.dataset = new_ds
        else:
            # return new_ds
            return RunTS(
                new_ds,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                survey_metadata=self.survey_metadata,
            )

    def merge(
        self,
        other: RunTS | list[RunTS],
        gap_method: str = "slinear",
        new_sample_rate: float | None = None,
        resample_method: str = "poly",
    ) -> RunTS:
        """
        Merge multiple runs into a single run.

        Combines this run with one or more other runs, optionally resampling
        to a common sample rate and filling gaps with interpolation.

        Parameters
        ----------
        other : RunTS | list[RunTS]
            Another RunTS object or list of RunTS objects to merge.
        gap_method : str, optional
            Interpolation method for filling gaps: 'linear', 'nearest',
            'zero', 'slinear', 'quadratic', 'cubic' (default is 'slinear').
        new_sample_rate : float | None, optional
            If provided, all runs will be resampled to this rate before
            merging. If None, uses the sample rate of the first run.
        resample_method : str, optional
            Resampling method if new_sample_rate is provided: 'decimate'
            or 'poly' (default is 'poly').

        Returns
        -------
        RunTS
            New merged RunTS object with monotonic time index.

        Raises
        ------
        TypeError
            If other is not a RunTS or list of RunTS objects.

        Examples
        --------
        Merge two runs:

        >>> run1 = RunTS(array_list=[ex1, ey1])
        >>> run2 = RunTS(array_list=[ex2, ey2])
        >>> merged = run1.merge(run2)

        Merge multiple runs with resampling:

        >>> runs = [run1, run2, run3]
        >>> merged = run1.merge(runs, new_sample_rate=1.0,
        ...                     resample_method='poly')

        Notes
        -----
        The merge process:

        1. Optionally resample all runs to common sample rate
        2. Combine datasets using xr.combine_by_coords
        3. Create monotonic time index spanning full range
        4. Interpolate to new index filling gaps
        5. Merge all filter dictionaries

        Metadata is taken from the first run (self).

        See Also
        --------
        __add__ : Simple merging with + operator

        """
        if new_sample_rate is not None:
            merge_sample_rate = new_sample_rate
            if resample_method == "decimate":
                combine_list = [self.decimate(new_sample_rate).dataset]
            elif resample_method == "poly":
                combine_list = [self.resample_poly(new_sample_rate).dataset]
        else:
            merge_sample_rate = self.sample_rate
            combine_list = [self.dataset]
        ts_filters = self.filters
        if isinstance(other, (list, tuple)):
            for run in other:
                if not isinstance(run, RunTS):
                    raise TypeError(f"Cannot combine {type(run)} with RunTS.")
                if new_sample_rate is not None:
                    if resample_method == "decimate":
                        run = run.decimate(new_sample_rate)
                    elif resample_method == "poly":
                        run = run.resample_poly(new_sample_rate)
                combine_list.append(run.dataset)
                ts_filters.update(run.filters)
        else:
            if not isinstance(other, RunTS):
                raise TypeError(f"Cannot combine {type(other)} with RunTS.")
            if new_sample_rate is not None:
                if resample_method == "decimate":
                    other = other.decimate(new_sample_rate)
                elif resample_method == "poly":
                    other = other.resample_poly(new_sample_rate)
            combine_list.append(other.dataset)
            ts_filters.update(other.filters)
        # combine into a data set use override to keep attrs from original

        combined_ds = xr.combine_by_coords(combine_list, combine_attrs="override")

        n_samples = (
            merge_sample_rate
            * float(combined_ds.time.max().values - combined_ds.time.min().values)
            / 1e9
        ) + 1

        new_dt_index = make_dt_coordinates(
            combined_ds.time.min().values, merge_sample_rate, n_samples
        )

        run_metadata = self.run_metadata.copy()
        run_metadata.sample_rate = merge_sample_rate

        new_run = RunTS(
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )

        ## tried reindex then interpolate_na, but that has issues if the
        ## intial time index does not exactly match up with the new time index
        ## and then get a bunch of Nan, unless use nearest or pad, but then
        ## gaps are not filled correctly, just do a interp seems easier.
        new_run.dataset = combined_ds.interp(time=new_dt_index, method=gap_method)

        # update channel attributes
        for ch in new_run.channels:
            new_run.dataset[ch].attrs["time_period.start"] = new_run.start
            new_run.dataset[ch].attrs["time_period.end"] = new_run.end
        new_run.run_metadata.update_time_period()
        new_run.station_metadata.update_time_period()
        new_run.survey_metadata.update_time_period()
        new_run.filters = ts_filters

        return new_run

    def plot(
        self,
        color_map: dict[str, tuple[float, float, float]] | None = None,
        channel_order: list[str] | None = None,
    ) -> Figure:
        """
        Plot all channels as time series.

        Creates a multi-panel figure with each channel in its own subplot,
        sharing a common time axis.

        Parameters
        ----------
        color_map : dict[str, tuple[float, float, float]] | None, optional
            Dictionary mapping channel names to RGB color tuples (values 0-1).
            Default colors:

            - ex: (1, 0.2, 0.2) - red
            - ey: (1, 0.5, 0) - orange
            - hx: (0, 0.5, 1) - blue
            - hy: (0.5, 0.2, 1) - purple
            - hz: (0.2, 1, 1) - cyan

        channel_order : list[str] | None, optional
            Order of channels from top to bottom. If None, uses order from
            self.channels.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plot.

        Examples
        --------
        Plot with default settings:

        >>> fig = run.plot()
        >>> fig.savefig('timeseries.png')

        Plot with custom colors and order:

        >>> colors = {'ex': (1, 0, 0), 'ey': (0, 1, 0)}
        >>> fig = run.plot(color_map=colors, channel_order=['ey', 'ex'])

        Warnings
        --------
        May be slow for large datasets (millions of points). Consider
        using get_slice() first to plot a subset.

        """

        if color_map is None:
            color_map = {
                "ex": (1, 0.2, 0.2),
                "ey": (1, 0.5, 0),
                "hx": (0, 0.5, 1),
                "hy": (0.5, 0.2, 1),
                "hz": (0.2, 1, 1),
            }

        if channel_order is not None:
            ch_list = channel_order
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
        return fig

    def plot_spectra(
        self,
        spectra_type: str = "welch",
        color_map: dict[str, tuple[float, float, float]] | None = None,
        **kwargs,
    ) -> Figure:
        """
        Plot power spectral density for all channels.

        Computes and plots the power spectrum of each channel on a single
        log-log plot with period on x-axis.

        Parameters
        ----------
        spectra_type : str, optional
            Type of spectral estimate to compute. Currently only 'welch'
            is supported (default is 'welch').
        color_map : dict[str, tuple[float, float, float]] | None, optional
            Dictionary mapping channel names to RGB color tuples (values 0-1).
            Uses same defaults as plot().
        **kwargs
            Additional keyword arguments passed to the spectra computation
            method (e.g., nperseg, window for Welch's method).

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the spectra plot.

        Examples
        --------
        Plot spectra with default settings:

        >>> fig = run.plot_spectra()

        Plot with custom Welch parameters:

        >>> fig = run.plot_spectra(nperseg=1024, window='hann')

        Notes
        -----
        The plot shows:

        - Period (seconds) on bottom x-axis
        - Frequency (Hz) on top x-axis
        - Power (dB) on y-axis

        See Also
        --------
        ChannelTS.welch_spectra : Compute Welch power spectrum

        """

        if color_map is None:
            color_map = {
                "ex": (1, 0.2, 0.2),
                "ey": (1, 0.5, 0),
                "hx": (0, 0.5, 1),
                "hy": (0.5, 0.2, 1),
                "hz": (0.2, 1, 1),
            }

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        line_list = []
        label_list = []
        for comp in self.channels:
            ch = getattr(self, comp)
            plot_freq, power = ch.welch_spectra(**kwargs)
            (l1,) = ax.loglog(1.0 / plot_freq, power, lw=1.5, color=color_map[comp])
            line_list.append(l1)
            label_list.append(comp)
        ax.set_xlabel("Period (s)", fontdict={"size": 10, "weight": "bold"})
        ax.set_ylabel("Power (dB)", fontdict={"size": 10, "weight": "bold"})
        ax.axis("tight")
        ax.grid(which="both")

        ax2 = ax.twiny()
        ax2.loglog(plot_freq, power, lw=0)
        ax2.set_xlabel("Frequency (Hz)", fontdict={"size": 10, "weight": "bold"})
        ax2.set_xlim([1 / cc for cc in ax.get_xlim()])

        ax.legend(line_list, label_list)

        plt.show()

        return fig
