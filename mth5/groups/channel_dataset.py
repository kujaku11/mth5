# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:03:23 2023

@author: jpeacock
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
from mth5.groups.base import BaseGroup
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

    .. code-block:: python

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

    def __init__(
        self, dataset, dataset_metadata=None, write_metadata=True, **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()
        self.logger = setup_logger(f"{__name__}.{self._class_name}")

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
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name
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
                msg = "metadata must be type metadata.%s not %s"
                self.logger.error(msg, self._class_name, type(dataset_metadata))
                raise MTH5Error(
                    msg % (self._class_name, type(dataset_metadata))
                )
            # load from dict because of the extra attributes for MTH5
            self.metadata.from_dict(dataset_metadata.to_dict())
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name

            # write out metadata to make sure that its in the file.
            if write_metadata:
                self.write_metadata()
        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.write_metadata()

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
                "default": None,
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
                "default": None,
            },
        )

    def __str__(self):
        try:
            lines = ["Channel {0}:".format(self._class_name)]
            lines.append("-" * (len(lines[0]) + 2))
            info_str = "\t{0:<18}{1}"
            lines.append(info_str.format("component:", self.metadata.component))
            lines.append(info_str.format("data type:", self.metadata.type))
            lines.append(
                info_str.format("data format:", self.hdf5_dataset.dtype)
            )
            lines.append(
                info_str.format("data shape:", self.hdf5_dataset.shape)
            )
            lines.append(
                info_str.format("start:", self.metadata.time_period.start)
            )
            lines.append(info_str.format("end:", self.metadata.time_period.end))
            lines.append(
                info_str.format("sample rate:", self.metadata.sample_rate)
            )
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
        """shortcut to run group"""
        return RunGroup(self.hdf5_dataset.parent)

    @property
    def run_metadata(self):
        """run metadata"""

        meta_dict = dict(self.hdf5_dataset.parent.attrs)
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        run_metadata = metadata.Run()
        run_metadata.from_dict({"run": meta_dict})
        return run_metadata

    @property
    def station_group(self):
        """shortcut to station group"""

        return StationGroup(self.hdf5_dataset.parent.parent)

    @property
    def station_metadata(self):
        """station metadata"""

        meta_dict = dict(self.hdf5_dataset.parent.parent.attrs)
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        station_metadata = metadata.Station()
        station_metadata.from_dict({"station": meta_dict})
        return station_metadata

    @property
    def master_station_group(self):
        """shortcut to master station group"""

        return MasterStationGroup(self.hdf5_dataset.parent.parent.parent)

    @property
    def survey_metadata(self):
        """survey metadata"""

        meta_dict = dict(self.hdf5_dataset.parent.parent.parent.parent.attrs)
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        survey_metadata = metadata.Survey()
        survey_metadata.from_dict({"survey": meta_dict})
        return survey_metadata

    @property
    def survey_id(self):
        """shortcut to survey group"""

        return self.hdf5_dataset.parent.parent.parent.parent.attrs["id"]

    @property
    def channel_response_filter(self):
        # get the filters to make a channel response
        filters_group = FiltersGroup(
            self.hdf5_dataset.parent.parent.parent.parent["Filters"]
        )
        f_list = []
        for name in self.metadata.filter.name:
            name = name.replace("/", " per ").lower()
            try:
                f_list.append(filters_group.to_filter_object(name))
            except KeyError:
                self.logger.warning("Could not locate filter %s", name)
                continue
        return ChannelResponseFilter(filters_list=f_list)

    @property
    def start(self):
        return self.metadata.time_period._start_dt

    @start.setter
    def start(self, value):
        """set start time and validate through metadata validator"""
        if isinstance(value, MTime):
            self.metadata.time_period.start = value.iso_str
        else:
            self.metadata.time_period.start = value

    @property
    def end(self):
        """return end time based on the data"""
        return self.start + (self.n_samples / self.sample_rate)

    @property
    def sample_rate(self):
        return self.metadata.sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        """set sample rate through metadata validator"""
        self.metadata.sample_rate = value

    @property
    def n_samples(self):
        return self.hdf5_dataset.size

    @property
    def time_index(self):
        """
        Create a time index based on the metadata.  This can help when asking
        for time windows from the data

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return make_dt_coordinates(
            self.start, self.sample_rate, self.n_samples, self._logger
        )

    def read_metadata(self):
        """
        Read metadata from the HDF5 file into the metadata container, that
        way it can be validated.

        """
        meta_dict = dict(self.hdf5_dataset.attrs)
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        self.metadata.from_dict({self._class_name: meta_dict})

    def write_metadata(self):
        """
        Write metadata from the metadata container to the HDF5 attrs
        dictionary.

        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            value = to_numpy_type(value)
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
            self.hdf5_dataset.resize(new_data_array.shape)
        self.hdf5_dataset[...] = new_data_array

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

        :Append Example:

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

        self.logger.info("Extending data.")
        self.logger.info(f"Existing start time {self.start}")
        self.logger.info(f"New start time      {start_time}")
        self.logger.info(f"Existing end time   {self.end}")
        self.logger.info(f"New end time        {end_time}")

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
                        self.get_index_from_time(old_start) :
                    ] = old_slice.ts.values
                    self.hdf5_dataset[
                        0 : self.get_index_from_time(end_time)
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
                        self.get_index_from_time(
                            end_time
                        ) : self.get_index_from_time(old_start)
                    ] = fill_value
            else:
                new_size = (
                    self.n_samples + int(abs(start_t_diff) * sample_rate),
                )
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
                    self.get_index_from_time(old_start) :
                ] = old_slice.ts.values
                self.hdf5_dataset[
                    0 : self.get_index_from_time(end_time)
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
                        self.get_index_from_time(start_time) :
                    ] = new_data_array
                    old_index = self.get_index_from_time(old_end)
                    if fill == "mean":
                        fill_value = np.mean(
                            np.array(
                                [
                                    new_data_array[0:fw].mean(),
                                    np.mean(
                                        self.hdf5_dataset[old_index - fw :]
                                    ),
                                ]
                            )
                        )
                    elif fill == "median":
                        fill_value = np.median(
                            np.array(
                                [
                                    np.median(new_data_array[0:fw]),
                                    np.median(
                                        self.hdf5_dataset[old_index - fw :]
                                    ),
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
                        self.get_index_from_time(
                            old_end
                        ) : self.get_index_from_time(start_time)
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
                        self.get_index_from_time(
                            start_time
                        ) : self.get_index_from_time(end_time)
                    ] = new_data_array
                else:
                    new_size = (
                        self.n_samples + int(abs(start_t_diff) * sample_rate),
                    )
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
                        self.get_index_from_time(start_time) :
                    ] = new_data_array

    def to_channel_ts(self):
        """
        :return: a Timeseries with the appropriate time index and metadata
        :rtype: :class:`mth5.timeseries.ChannelTS`

        loads from memory (nearly half the size of xarray alone, not sure why)

        """
        return ChannelTS(
            channel_type=self.metadata.type,
            data=self.hdf5_dataset[()],
            channel_metadata=self.metadata,
            run_metadata=self.run_metadata.copy(),
            station_metadata=self.station_metadata.copy(),
            survey_metadata=self.survey_metadata.copy(),
            channel_response_filter=self.channel_response_filter,
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
            {"data": self.hdf5_dataset[()]}, index=self.time_index
        )
        df.attrs.update(self.metadata.to_dict(single=True))

        return df

    def to_numpy(self):
        """
        :return: a numpy structured array with 2 columns (time, channel_data)
        :rtype: :class:`numpy.core.records`

        .. note:: data is a builtin to numpy and cannot be used as a name

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
            self.metadata.from_dict(channel_ts_obj.channel_metadata.to_dict())
            self.replace_dataset(channel_ts_obj.ts)
            # apparently need to reset these otherwise they get overwritten with None
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name
            self.write_metadata()
        elif how == "extend":
            self.extend_dataset(
                channel_ts_obj.ts,
                channel_ts_obj.start,
                channel_ts_obj.sample_rate,
                fill=fill,
            )
        #
        # TODO need to check on metadata.

    def from_xarray(
        self,
        data_array,
        how="replace",
        fill=None,
        max_gap_seconds=1,
        fill_window=10,
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
         be.  Anything over this number will raise a
         :class:`mth5.utils.exceptions.MTH5Error`.

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
                {self.metadata._class_name: data_array.attrs}
            )
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

            * A positive number means new end time is later than old
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
    def table_entry(self):
        """
        Creat a table entry to put into the run summary table.

        """

        return np.array(
            [
                (
                    self.metadata.component,
                    self.metadata.time_period._start_dt.iso_no_tz,
                    self.metadata.time_period._end_dt.iso_no_tz,
                    self.hdf5_dataset.size,
                    self.metadata.type,
                    self.metadata.units,
                    self.hdf5_dataset.ref,
                )
            ],
            dtype=np.dtype(
                [
                    ("component", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("n_samples", int),
                    ("measurement_type", "U12"),
                    ("units", "U25"),
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
                    self.survey_id,
                    self.hdf5_dataset.parent.parent.attrs["id"],
                    self.hdf5_dataset.parent.attrs["id"],
                    self.hdf5_dataset.parent.parent.attrs["location.latitude"],
                    self.hdf5_dataset.parent.parent.attrs["location.longitude"],
                    self.hdf5_dataset.parent.parent.attrs["location.elevation"],
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
                    self.hdf5_dataset.parent.ref,
                    self.hdf5_dataset.parent.parent.ref,
                )
            ],
            dtype=CHANNEL_DTYPE,
        )

    def time_slice(
        self,
        start,
        end=None,
        n_samples=None,
        return_type="channel_ts",
    ):
        """
        Get a time slice from the channel and return the appropriate type

            * numpy array with metadata
            * pandas.Dataframe with metadata
            * xarray.DataFrame with metadata
            * :class:`mth5.timeseries.ChannelTS` 'default'
            * dask.DataFrame with metadata 'not yet'

        :param start: start time of the slice
        :type start: string or :class:`mth5.utils.mttime.MTime`
        :param end: end time of the slice
        :type end: string or :class:`mth5.utils.mttime.MTime`, optional
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

        if not isinstance(start, MTime):
            start = MTime(start)
        if end is not None:
            if not isinstance(end, MTime):
                end = MTime(end)
        if n_samples is not None:
            n_samples = int(n_samples)
        if n_samples is None and end is None:
            msg = "Must input either end_time or n_samples."
            self.logger.error(msg)
            raise ValueError(msg)
        if n_samples is not None and end is not None:
            msg = "Must input either end_time or n_samples, not both."
            self.logger.error(msg)
            raise ValueError(msg)
        # if end time is given
        if end is not None and n_samples is None:
            start_index = self.get_index_from_time(start)
            end_index = self.get_index_from_time(end)
            npts = int(end_index - start_index)
        # if n_samples are given
        elif end is None and n_samples is not None:
            start_index = self.get_index_from_time(start)
            end_index = start_index + (n_samples - 1)
            npts = n_samples
        if npts > self.hdf5_dataset.size or end_index > self.hdf5_dataset.size:
            msg = (
                "Requested slice is larger than data.  "
                f"Slice length = {npts}, data length = {self.hdf5_dataset.shape}. "
                f"Setting end_index to {self.hdf5_dataset.shape}"
            )
            end_index = self.hdf5_dataset.size - 1
            self.logger.warning(msg)
        # create a regional reference that can be used, need +1 to be inclusive
        try:
            regional_ref = self.hdf5_dataset.regionref[
                start_index : end_index + 1
            ]
        except (OSError, RuntimeError):
            self.logger.debug(
                "file is in read mode cannot set an internal reference, using index values"
            )
            regional_ref = slice(start_index, end_index)
        dt_index = make_dt_coordinates(
            start, self.sample_rate, npts, self.logger
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
                channel_response_filter=self.channel_response_filter,
            )
        else:
            msg = "return_type not understood, must be [ pandas | numpy | channel_ts ]"
            self.logger.error(msg)
            raise ValueError(msg)
        return data

    def get_index_from_time(self, given_time):
        """
        get the appropriate index for a given time.

        :param given_time: time string
        :type given_time: string or MTime
        :return: index value
        :rtype: int

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
