# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:22:10 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import inspect
import numpy as np
import h5py

from mth5.groups import BaseGroup, MasterStationGroup, StationGroup
from mth5.datasets import ElectricDataset, MagneticDataset, AuxiliaryDataset, ChannelDataset
from mth5.utils.exceptions import MTH5Error
from mth5 import metadata
from mth5.timeseries import ChannelTS, RunTS

meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
# =============================================================================
# Station Group
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

    >>> station.group_list
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

        # summary of channels in run
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (20,),
            "dtype": np.dtype(
                [
                    ("component", "S20"),
                    ("start", "S32"),
                    ("end", "S32"),
                    ("n_samples", np.int),
                    ("measurement_type", "S12"),
                    ("units", "S25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        }

    @property
    def station_group(self):
        """ shortcut to station group """
        return StationGroup(self.hdf5_group.parent)

    @property
    def master_station_group(self):
        """ shortcut to master station group """
        return MasterStationGroup(self.hdf5_group.parent.parent)

    @property
    def table_entry(self):
        """
        Get a run table entry

        :return: a properly formatted run table entry
        :rtype: :class:`numpy.ndarray` with dtype:
            
        >>> dtype([('id', 'S20'),
                 ('start', 'S32'),
                 ('end', 'S32'),
                 ('components', 'S100'),
                 ('measurement_type', 'S12'),
                 ('sample_rate', np.float),
                 ('hdf5_reference', h5py.ref_dtype)])

        """
        return np.array(
            [
                (
                    self.metadata.id,
                    self.metadata.time_period.start,
                    self.metadata.time_period.end,
                    ",".join(self.metadata.channels_recorded_all),
                    self.metadata.data_type,
                    self.metadata.sample_rate,
                    self.hdf5_group.ref,
                )
            ],
            dtype=np.dtype(
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
        )

    def add_channel(
        self,
        channel_name,
        channel_type,
        data,
        channel_dtype="f",
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
        :raises MTH5Error: If channel type is not correct

        :param channel_metadata: metadata container, defaults to None
        :type channel_metadata: [ :class:`mth5.metadata.Electric` |
                                 :class:`mth5.metadata.Magnetic` |
                                 :class:`mth5.metadata.Auxiliary` ], optional
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
        channel_name = channel_name.lower()
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
                    maxshape=max_shape,
                    dtype=data.dtype,
                    chunks=chunks,
                    **self.dataset_options,
                )
            # initialize an resizable data array
            else:
                channel_group = self.hdf5_group.create_dataset(
                    channel_name,
                    shape=(1,),
                    maxshape=max_shape,
                    dtype=channel_dtype,
                    chunks=chunks,
                    **self.dataset_options,
                )

            self.logger.debug("Created group {0}".format(channel_group.name))
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
                    + "auxiliary ]. Input was {0}".format(channel_type)
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
            if channel_obj.metadata.component is None:
                channel_obj.metadata.component = channel_name
            channel_obj.write_metadata()
            self.summary_table.add_row(channel_obj.table_entry)
            self.master_station_group.summary_table.add_row(channel_obj.channel_entry)

        except (OSError, RuntimeError):
            msg = (
                f"channel {channel_name} already exists, " + "returning existing group."
            )
            self.logger.info(msg)
            if channel_type in ["magnetic"]:
                channel_obj = MagneticDataset(self.hdf5_group[channel_name])
            elif channel_type in ["electric"]:
                channel_obj = ElectricDataset(self.hdf5_group[channel_name])
            elif channel_type in ["auxiliary"]:
                channel_obj = AuxiliaryDataset(self.hdf5_group[channel_name])
            channel_obj.read_metadata()

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

        >>> run.group_list
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

        try:
            ch_dataset = self.hdf5_group[channel_name]
            if ch_dataset.attrs["mth5_type"].lower() in ["electric"]:
                ch_metadata = meta_classes["Electric"]()
                ch_metadata.from_dict({"Electric": ch_dataset.attrs})
                channel = ElectricDataset(
                    ch_dataset, dataset_metadata=ch_metadata, write_metadata=False
                )
            elif ch_dataset.attrs["mth5_type"].lower() in ["magnetic"]:
                ch_metadata = meta_classes["Magnetic"]()
                ch_metadata.from_dict({"Magnetic": ch_dataset.attrs})
                channel = MagneticDataset(
                    ch_dataset, dataset_metadata=ch_metadata, write_metadata=False
                )
            elif ch_dataset.attrs["mth5_type"].lower() in ["auxiliary"]:
                ch_metadata = meta_classes["Auxiliary"]()
                ch_metadata.from_dict({"Auxiliary": ch_dataset.attrs})
                channel = AuxiliaryDataset(
                    ch_dataset, dataset_metadata=ch_metadata, write_metadata=False
                )
            else:
                channel = ChannelDataset(ch_dataset)

            return channel

        except KeyError:
            msg = (
                f"{channel_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

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

        channel_name = channel_name.lower()

        try:
            component = self.hdf5_group[channel_name].attrs["component"]
            del self.hdf5_group[channel_name]
            self.summary_table.remove_row(
                self.summary_table.locate("component", component)
            )
            self.logger.info(
                "Deleting a channel does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{channel_name} does not exist, "
                + "check group_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def to_runts(self):
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
            ts_obj = ch_obj.to_channel_ts()
            ch_list.append(ts_obj)
        return RunTS(ch_list, run_metadata=self.metadata)

    def from_runts(self, run_ts_obj):
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

        self.metadata.from_dict(run_ts_obj.metadata.to_dict())

        channels = []

        for comp in run_ts_obj.channels:

            if comp[0] in ["e"]:
                channel_type = "electric"
                ch_metadata = metadata.Electric()

            elif comp[0] in ["h", "b"]:
                channel_type = "magnetic"
                ch_metadata = metadata.Magnetic()
            else:
                channel_type = "auxiliary"
                ch_metadata = metadata.Auxiliary()

            ch_metadata.from_dict({channel_type: run_ts_obj.dataset[comp].attrs})
            ch_metadata.hdf5_type = channel_type

            channels.append(
                self.add_channel(
                    comp,
                    channel_type,
                    run_ts_obj.dataset[comp].values,
                    channel_metadata=ch_metadata,
                )
            )
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

        ch_obj = self.add_channel(
            channel_ts_obj.component,
            channel_ts_obj.metadata.type,
            channel_ts_obj.ts.values,
            channel_metadata=channel_ts_obj.metadata,
        )

        # need to update the channels recorded
        if channel_ts_obj.metadata.type == "electric":
            if self.metadata.channels_recorded_electric is None:
                self.metadata.channels_recorded_electric = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_electric
            ):
                self.metadata.channels_recorded_electric.append(
                    channel_ts_obj.component
                )

        elif channel_ts_obj.metadata.type == "magnetic":
            if self.metadata.channels_recorded_magnetic is None:
                self.metadata.channels_recorded_magnetic = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_magnetic
            ):
                self.metadata.channels_recorded_magnetic.append(
                    channel_ts_obj.component
                )

        elif channel_ts_obj.metadata.type == "auxiliary":
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

    def validate_run_metadata(self):
        """
        Update metadata and table entries to ensure consistency
        
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.logger.debug("Updating run metadata from summary table.")
        channels_recorded = list(
            self.summary_table.array["component"].astype(np.unicode_)
        )
        self.metadata.channels_recorded_electric = [
            cc for cc in channels_recorded if cc[0] in ["e"]
        ]
        self.metadata.channels_recorded_magnetic = [
            cc for cc in channels_recorded if cc[0] in ["h", "b"]
        ]
        self.metadata.channels_recorded_auxiliary = [
            cc for cc in channels_recorded if cc[0] not in ["e", "h", "b"]
        ]

        self.metadata.time_period.start = min(
            self.summary_table.array["start"]
        ).decode()

        self.metadata.time_period.end = max(self.summary_table.array["end"]).decode()

