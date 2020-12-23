# -*- coding: utf-8 -*-
"""
==================
MTH5
==================

MTH5 deals with reading and writing an MTH5 file, which are HDF5 files
developed for magnetotelluric (MT) data.  The code is based on h5py and
therefor numpy.  This is the simplest and we are not really dealing with
large tables of data to warrant using pytables.

Created on Sun Dec  9 20:50:41 2018

:copyright:
    Jared Peacock (jpeacock@usgs.gov)
    
:license: 
    MIT

"""

# =============================================================================
# Imports
# =============================================================================
import logging

from pathlib import Path
from platform import platform

import h5py

from mth5.utils.exceptions import MTH5Error
from mth5 import __version__ as mth5_version
from mth5.utils.mttime import get_now_utc
from mth5 import groups as groups
from mth5 import helpers

# =============================================================================
# Acceptable parameters
# =============================================================================
acceptable_file_types = ["mth5", "MTH5"]
acceptable_file_versions = ["0.1.0"]
acceptable_data_levels = [0, 1, 2, 3]

# =============================================================================
# MT HDF5 file
# =============================================================================
class MTH5:
    """
    MTH5 is the main container for the HDF5 file format developed for MT data
    
    It uses the metadata standards developled by the
    `IRIS PASSCAL software group 
    <https://www.iris.edu/hq/about_iris/governance/mt_soft>`_
    and defined in the
    `metadata documentation 
    <https://github.com/kujaku11/MTarchive/blob/tables/docs/mt_metadata_guide.pdf>`_.
    
    MTH5 is built with h5py and therefore numpy.  The structure follows the
    different levels of MT data collection:
    Survey
       |_Reports
       |_Standards
       |_Filters
       |_Station
           |_Run
               |_Channel
            
    All timeseries data are stored as individual channels with the appropriate
    metadata defined for the given channel, i.e. electric, magnetic, auxiliary.
    
    Each level is represented as a mth5 group class object which has methods
    to add, remove, and get a group from the level below.  Each group has a 
    metadata attribute that is the approprate metadata class object.  For 
    instance the SurveyGroup has an attribute metadata that is a 
    :class:`mth5.metadata.Survey` object.  Metadata is stored in the HDF5 group 
    attributes as (key, value) pairs.
    
    All groups are represented by their structure tree and can be shown
    at any time from the command line.
    
    Each level has a summary array of the contents of the levels below to 
    hopefully make searching easier. 
    
    :param filename: name of the to be or existing file
    :type filename: string or :class:`pathlib.Path`
    :param compression: compression type.  Supported lossless compressions are
        * 'lzf' - Available with every installation of h5py 
                 (C source code also available). Low to 
                 moderate compression, very fast. No options.
        * 'gzip' - Available with every installation of HDF5,
                  so it’s best where portability is required. 
                  Good compression, moderate speed. 
                  compression_opts sets the compression level 
                  and may be an integer from 0 to 9, 
                  default is 3.
        * 'szip' - Patent-encumbered filter used in the NASA
                   community. Not available with all 
                   installations of HDF5 due to legal reasons.
                   Consult the HDF5 docs for filter options.
    :param compression_opts: compression options, see above
    :type compression_opts: string or int depending on compression type
    :param shuffle: Block-oriented compressors like GZIP or LZF work better
                    when presented with runs of similar values. Enabling the 
                    shuffle filter rearranges the bytes in the chunk and may
                    improve compression ratio. No significant speed penalty,
                    lossless.
    :type shuffle: boolean
    :param fletcher32: Adds a checksum to each chunk to detect data corruption.
                       Attempts to read corrupted chunks will fail with an 
                       error. No significant speed penalty. Obviously 
                       shouldn’t be used with lossy compression filters.
    :type fletcher32: boolean
    :param data_level: level the data are stored following levels defined by
       `NASA ESDS <https://earthdata.nasa.gov/collaborate/open-data-services-and-software/data-information-policy/data-levels>`_
         * 0 - Raw data 
         * 1 - Raw data with response information and full metadata
         * 2 - Derived product, raw data has been manipulated 
    :type data_level: integer, defaults to 1                

    
    :Usage:

    * Open a new file and show initialized file
    
    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> # Have a look at the dataset options
    >>> mth5.dataset_options
    {'compression': 'gzip',
     'compression_opts': 3,
     'shuffle': True,
     'fletcher32': True}
    >>> mth5_obj.open_mth5(r"/home/mtdata/mt01.mth5", 'w')
    >>> mth5_obj
    /:
    ====================
        |- Group: Survey
        ----------------
            |- Group: Filters
            -----------------
                --> Dataset: summary
                ......................
            |- Group: Reports
            -----------------
                --> Dataset: summary
                ......................
            |- Group: Standards
            -------------------
                --> Dataset: summary
                ......................
            |- Group: Stations
            ------------------
                --> Dataset: summary
                ......................

    
    * Add metadata for survey from a dictionary
    
    >>> survey_dict = {'survey':{'acquired_by': 'me', 'archive_id': 'MTCND'}}
    >>> survey = mth5_obj.survey_group
    >>> survey.metadata.from_dict(survey_dict)
    >>> survey.metadata
    {
    "survey": {
        "acquired_by.author": "me",
        "acquired_by.comments": null,
        "archive_id": "MTCND"
        ...}
    }
    
    * Add a station from the convenience function
    
    >>> station = mth5_obj.add_station('MT001')
    >>> mth5_obj
    /:
    ====================
        |- Group: Survey
        ----------------
            |- Group: Filters
            -----------------
                --> Dataset: summary
                ......................
            |- Group: Reports
            -----------------
                --> Dataset: summary
                ......................
            |- Group: Standards
            -------------------
                --> Dataset: summary
                ......................
            |- Group: Stations
            ------------------
                |- Group: MT001
                ---------------
                    --> Dataset: summary
                    ......................
                --> Dataset: summary
                ......................
    >>> station
    /Survey/Stations/MT001:
    ====================
        --> Dataset: summary
        ......................

    >>> data.schedule_01.ex[0:10] = np.nan
    >>> data.calibration_hx[...] = np.logspace(-4, 4, 20)

    .. note:: if replacing an entire array with a new one you need to use [...]
              otherwise the data will not be updated.

    .. warning:: You can only replace entire arrays with arrays of the same
                 size.  Otherwise you need to delete the existing data and
                 make a new dataset.

    .. seealso:: https://www.hdfgroup.org/ and https://www.h5py.org/
    
    """

    def __init__(
        self,
        filename=None,
        compression="gzip",
        compression_opts=3,
        shuffle=True,
        fletcher32=True,
        data_level=1,
    ):

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # make these private so the user cant accidentally change anything.
        self.__hdf5_obj = None
        self.__compression, self.__compression_opts = helpers.validate_compression(
            compression, compression_opts
        )
        self.__shuffle = True
        self.__fletcher32 = True
        self.__data_level = 1
        self.__filename = None
        self.filename = filename

        self._default_root_name = "Survey"
        self._default_subgroup_names = ["Stations", "Reports", "Filters", "Standards"]

        self._file_attrs = {
            "file.type": "MTH5",
            "file.version": acceptable_file_versions[-1],
            "file.access.platform": platform(),
            "file.access.time": get_now_utc(),
            "mth5.software.version": mth5_version,
            "mth5.software.name": "mth5",
            "data_level": self.__data_level,
        }

    def __str__(self):
        if self.h5_is_write():
            return helpers.get_tree(self.__hdf5_obj)

        return "HDF5 file is closed and cannot be accessed."

    def __repr__(self):
        return self.__str__()

    @property
    def dataset_options(self):
        """ summary of dataset options"""
        return {
            "compression": self.__compression,
            "compression_opts": self.__compression_opts,
            "shuffle": self.__shuffle,
            "fletcher32": self.__fletcher32,
        }

    @property
    def filename(self):
        """ file name of the hdf5 file """
        if self.h5_is_write():
            return Path(self.__hdf5_obj.filename)
        msg = (
            "MTH5 file is not open or has not been created yet. "
            + "Returning default name"
        )
        self.logger.warning(msg)
        return self.__filename

    @filename.setter
    def filename(self, value):
        """ make sure file has the proper extension """
        if value is None:
            return None

        if not isinstance(value, Path):
            value = Path(value)

        if value.suffix not in [".h5"]:
            msg = (
                f"file extension {value.suffix} is not correct. "
                "Changing to default .h5"
            )
            self.logger.info(msg)
            self.__filename = value.with_suffix(".h5")

        else:
            self.__filename = value

    @property
    def file_type(self):
        """ File Type should be MTH5 """

        if self.h5_is_write():
            return self.__hdf5_obj.attrs["file.type"]
        return None

    @file_type.setter
    def file_type(self, value):
        """ set file type while validating input """
        if not isinstance(value, str):
            msg = f"Input file type must be a string not {type(value)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if self.h5_is_write():
            if value in acceptable_file_types:
                self.__hdf5_obj.attrs["file.type"] = value
            msg = f"Input file.type is not valid, must be {acceptable_file_types}"
            self.logger.error(msg)
            raise ValueError(msg)

    @property
    def file_version(self):
        """ mth5 file version """
        if self.h5_is_write():
            return self.__hdf5_obj.attrs["file.version"]
        return None

    @file_version.setter
    def file_version(self, value):
        """ set file version while validating input """
        if not isinstance(value, str):
            msg = f"Input file type must be a string not {type(value)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if self.h5_is_write():
            if value in acceptable_file_versions:
                self.__hdf5_obj.attrs["file.version"] = value
            msg = f"Input file.version is not valid, must be {acceptable_file_versions}"
            self.logger.error(msg)
            raise ValueError(msg)

    @property
    def software_name(self):
        """ software name that wrote the file """
        if self.h5_is_write():
            return self.__hdf5_obj.attrs["mth5.software.name"]
        return None

    @property
    def data_level(self):
        """ data level """
        if self.h5_is_write():
            return self.__hdf5_obj.attrs["data_level"]
        return None

    @data_level.setter
    def data_level(self, value):
        """ set data level while validating input """
        if not isinstance(value, int):
            msg = f"Input file type must be an integer not {type(value)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if self.h5_is_write():
            if value in acceptable_file_versions:
                self.__hdf5_obj.attrs["data_level"] = value
            msg = f"Input data_level is not valid, must be {acceptable_data_levels}"
            self.logger.error(msg)
            raise ValueError(msg)

    @property
    def survey_group(self):
        """ Convenience property for /Survey group """
        if self.h5_is_write():
            return groups.SurveyGroup(
                self.__hdf5_obj["/Survey"], **self.dataset_options
            )
        self.logger.info("File is closed cannot access /Survey")
        return None

    @property
    def reports_group(self):
        """ Convenience property for /Survey/Reports group """
        if self.h5_is_write():
            return groups.ReportsGroup(
                self.__hdf5_obj["/Survey/Reports"], **self.dataset_options
            )
        self.logger.info("File is closed cannot access /Reports")
        return None

    @property
    def filters_group(self):
        """ Convenience property for /Survey/Filters group """
        if self.h5_is_write():
            return groups.FiltersGroup(
                self.__hdf5_obj["/Survey/Filters"], **self.dataset_options
            )
        self.logger.info("File is closed cannot access /Filters")
        return None

    @property
    def standards_group(self):
        """ Convenience property for /Survey/Standards group"""
        if self.h5_is_write():
            return groups.StandardsGroup(
                self.__hdf5_obj["/Survey/Standards"], **self.dataset_options
            )
        self.logger.info("File is closed cannot access /Standards")
        return None

    @property
    def stations_group(self):
        """ Convenience property for /Survey/Stations group"""
        if self.h5_is_write():
            return groups.MasterStationGroup(
                self.__hdf5_obj["/Survey/Stations"], **self.dataset_options
            )
        self.logger.info("File is closed cannot access /Stations")
        return None

    @property
    def station_list(self):
        """list of existing stations names"""
        return self.stations_group.groups_list

    def open_mth5(self, filename=None, mode="a"):
        """
        open an mth5 file

        :return: Survey Group
        :type: groups.SurveyGroup

        :Example: 

        >>> from mth5 import mth5
        >>> mth5_object = mth5.MTH5()
        >>> survey_object = mth5_object.open_mth5('Test.mth5', 'w')


        """
        if filename is not None:
            self.__filename = filename
        if not isinstance(self.__filename, Path):
            self.__filename = Path(filename)

        if self.__filename.exists():
            if mode in ["w"]:
                self.logger.warning(
                    f"{self.__filename.name} will be overwritten in 'w' mode"
                )
                try:
                    self._initialize_file()
                except OSError as error:
                    msg = (
                        f"{error}. Need to close any references to {self._filename} first. "
                        + "Then reopen the file in the preferred mode"
                    )
                    self.logger.exception(msg)
            elif mode in ["a", "r", "r+", "w-", "x"]:
                self.__hdf5_obj = h5py.File(self.__filename, mode=mode)
                if not self.validate_file():
                    msg = "Input file is not a valid MTH5 file"
                    self.logger.error(msg)
                    raise MTH5Error(msg)

            else:
                msg = "mode {0} is not understood".format(mode)
                self.logger.error(msg)
                raise MTH5Error(msg)
        else:
            if mode in ["a", "w", "w-", "x"]:
                self._initialize_file()
            else:
                msg = "Cannot open new file in mode {0} ".format(mode)
                self.logger.error(msg)
                raise MTH5Error(msg)

        # TODO need to add a validation step to check for version and legit file

    def _initialize_file(self):
        """
        Initialize the default groups for the file

        :return: Survey Group
        :rtype: groups.SurveyGroup

        """

        self.__hdf5_obj = h5py.File(self.__filename, "w")

        # write general metadata
        self.__hdf5_obj.attrs.update(self._file_attrs)

        survey_group = self.__hdf5_obj.create_group(self._default_root_name)
        survey_obj = groups.SurveyGroup(survey_group)
        survey_obj.write_metadata()

        for group_name in self._default_subgroup_names:
            self.__hdf5_obj.create_group(f"{self._default_root_name}/{group_name}")
            m5_grp = getattr(self, f"{group_name.lower()}_group")
            m5_grp.initialize_group()

        self.logger.info(f"Initialized MTH5 file {self.filename} in mode 'w'")

        return survey_obj

    def validate_file(self):
        """
        Validate an open mth5 file
        
        will test the attribute values and group names
        
        :return: Boolean [ True = valid, False = not valid]
        :rtype: Boolean

        """

        if self.h5_is_write():
            if self.file_type not in acceptable_file_types:
                msg = f"Unaccetable file type {self.file_type}"
                self.logger.error(msg)
                return False
            if self.file_version not in acceptable_file_versions:
                msg = f"Unaccetable file version {self.file_version}"
                self.logger.error(msg)
                return False
            if self.data_level not in acceptable_data_levels:
                msg = f"Unaccetable data_level {self.data_level}"
                self.logger.error(msg)
                return False
            for gr in self.survey_group.groups_list:
                if gr not in self._default_subgroup_names:
                    msg = f"Unaccetable group {gr}"
                    self.logger.error(msg)
                    return False
            return True
        return False

    def close_mth5(self):
        """
        close mth5 file to make sure everything is flushed to the file
        """

        try:
            self.logger.info(f"Flushing and closing {str(self.filename)}")
            self.__hdf5_obj.flush()
            self.__hdf5_obj.close()
            
        except AttributeError:
            helpers.close_open_files()

    def h5_is_write(self):
        """
        check to see if the hdf5 file is open and writeable
        """
        if isinstance(self.__hdf5_obj, h5py.File):
            try:
                if "w" in self.__hdf5_obj.mode or "+" in self.__hdf5_obj.mode:
                    return True
                return False
            except ValueError:
                return False
        return False

    def from_reference(self, h5_reference):
        """
        Get an HDF5 group, dataset, etc from a reference

        :param h5_reference: DESCRIPTION
        :type h5_reference: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        # in the future should allow this to return the proper container.
        referenced = self.__hdf5_obj[h5_reference]
        mth5_type = referenced.attrs["mth5_type"]
        if mth5_type.lower() in ["station"]:
            return groups.StationGroup(referenced)
        elif mth5_type.lower() in ["run"]:
            return groups.RunGroup(referenced)
        elif mth5_type.lower() in ["electric"]:
            return groups.ElectricDataset(referenced)
        elif mth5_type.lower() in ["magnetic"]:
            return groups.MagneticDataset(referenced)
        elif mth5_type.lower() in ["auxiliary"]:
            return groups.AuxiliaryDataset(referenced)
        else:
            self.logger.info(
                f"Could not identify the MTH5 type {mth5_type}, "
                + "returning h5 group."
            )
            return referenced

    def add_station(self, name, station_metadata=None):
        """
        Convenience function to add a station using
        ``mth5.stations_group.add_station``


        Add a station with metadata if given with the path:
            ``/Survey/Stations/station_name``

        If the station already exists, will return that station and nothing
        is added.

        :param station_name: Name of the station, should be the same as
                             metadata.archive_id
        :type station_name: string
        :param station_metadata: Station metadata container, defaults to None
        :type station_metadata: :class:`mth5.metadata.Station`, optional
        :return: A convenience class for the added station
        :rtype: :class:`mth5_groups.StationGroup`

        :Example:

        >>> new_staiton = mth5_obj.add_station('MT001')

        """

        return self.stations_group.add_station(name, station_metadata=station_metadata)

    def get_station(self, station_name):
        """
        Convenience function to get a station using
        ``mth5.stations_group.get_station``

        Get a station with the same name as station_name

        :param station_name: existing station name
        :type station_name: string
        :return: convenience station class
        :rtype: :class:`mth5.mth5_groups.StationGroup`
        :raises MTH5Error:  if the station name is not found.

        :Example:

        >>> existing_staiton = mth5_obj.get_station('MT001')
        MTH5Error: MT001 does not exist, check station_list for existing names

        """

        return self.stations_group.get_station(station_name)

    def remove_station(self, station_name):
        """
        Convenience function to remove a station using
        ``mth5.stations_group.remove_station``

        Remove a station from the file.

        .. note:: Deleting a station is not as simple as del(station).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the station.

        :param station_name: existing station name
        :type station_name: string

        :Example: 

        >>> mth5_obj.remove_station('MT001')

        """

        self.stations_group.remove_station(station_name)

    def add_run(self, station_name, run_name, run_metadata=None):
        """
        Convenience function to add a run using
        ``mth5.stations_group.get_station(station_name).add_run()``

        Add a run to a given station.

        :param run_name: run name, should be archive_id{a-z}
        :type run_name: string
        :param metadata: metadata container, defaults to None
        :type metadata: :class:`mth5.metadata.Station`, optional

        :example:

        >>> new_run = mth5_obj.add_run('MT001', 'MT001a')


        .. todo:: auto fill run name if none is given.

        .. todo:: add ability to add a run with data.


        """

        return self.stations_group.get_station(station_name).add_run(
            run_name, run_metadata=run_metadata
        )

    def get_run(self, station_name, run_name):
        """
        Convenience function to get a run using
        ``mth5.stations_group.get_station(station_name).get_run()``

        get a run from run name for a given station

        :param station_name: existing station name
        :type station_name: string
        :param run_name: existing run name
        :type run_name: string
        :return: Run object
        :rtype: :class:`mth5.mth5_groups.RunGroup`

        :Example:

        >>> existing_run = mth5_obj.get_run('MT001', 'MT001a')

        """

        return self.stations_group.get_station(station_name).get_run(run_name)

    def remove_run(self, station_name, run_name):
        """
        Convenience function to add a run using
        ``mth5.stations_group.get_station(station_name).remove_run()``

        Remove a run from the station.

        .. note:: Deleting a run is not as simple as del(run).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the run.

        :param station_name: existing station name
        :type station_name: string
        :param run_name: existing run name
        :type run_name: string

        :Example:

        >>> mth5_obj.remove_station('MT001', 'MT001a')

        """

        return self.stations_group.get_station(station_name).remove_run(run_name)

    def add_channel(
        self,
        station_name,
        run_name,
        channel_name,
        channel_type,
        data,
        channel_metadata=None,
    ):
        """
        Convenience function to add a channel using
        ``mth5.stations_group.get_station().get_run().add_channel()``

        add a channel to a given run for a given station

        :param station_name: existing station name
        :type station_name: string
        :param run_name: existing run name
        :type run_name: string
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
        
        :Example:
        
        >>> new_channel = mth5_obj.add_channel('MT001', 'MT001a''Ex',
        >>> ...                                'electric', None)
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

        return (
            self.stations_group.get_station(station_name)
            .get_run(run_name)
            .add_channel(
                channel_name,
                channel_type,
                data,
                channel_metadata=channel_metadata,
                **self.dataset_options,
            )
        )

    def get_channel(self, station_name, run_name, channel_name):
        """
        Convenience function to get a channel using
        ``mth5.stations_group.get_station().get_run().get_channel()``

        Get a channel from an existing name.  Returns the appropriate
        container.

        :param station_name: existing station name
        :type station_name: string
        :param run_name: existing run name
        :type run_name: string
        :param channel_name: name of the channel
        :type channel_name: string
        :return: Channel container
        :rtype: [ :class:`mth5.mth5_groups.ElectricDatset` |
                  :class:`mth5.mth5_groups.MagneticDatset` |
                  :class:`mth5.mth5_groups.AuxiliaryDatset` ]
        :raises MTH5Error:  If no channel is found

        :Example:

        >>> existing_channel = mth5_obj.get_channel(station_name,
        >>> ...                                     run_name,
        >>> ...                                     channel_name)
        >>> existing_channel
        Channel Electric:
        -------------------
        		component:        Ex
            	data type:        electric
            	data format:      float32
            	data shape:       (4096,)
            	start:            1980-01-01T00:00:00+00:00
            	end:              1980-01-01T00:00:01+00:00
            	sample rate:      4096

        """

        return (
            self.stations_group.get_station(station_name)
            .get_run(run_name)
            .get_channel(channel_name)
        )

    def remove_channel(self, station_name, run_name, channel_name):
        """
        Convenience function to remove a channel using
        ``mth5.stations_group.get_station().get_run().remove_channel()``

        Remove a channel from a given run and station.

        .. note:: Deleting a channel is not as simple as del(channel).  In HDF5
              this does not free up memory, it simply removes the reference
              to that channel.  The common way to get around this is to
              copy what you want into a new file, or overwrite the channel.

        :param station_name: existing station name
        :type station_name: string
        :param run_name: existing run name
        :type run_name: string
        :param channel_name: existing station name
        :type channel_name: string

        :Example:

        >>> mth5_obj.remove_channel('MT001', 'MT001a', 'Ex')

        """

        return (
            self.stations_group.get_station(station_name)
            .get_run(run_name)
            .remove_channel(channel_name)
        )
