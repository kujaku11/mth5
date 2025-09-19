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

:copyright: Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from platform import platform

import h5py
from loguru import logger

from mth5.utils.exceptions import MTH5Error
from mth5 import __version__ as mth5_version
from mth5 import groups
from mth5.tables import ChannelSummaryTable, FCSummaryTable, TFSummaryTable

from mth5 import helpers
from mth5 import (
    CHANNEL_DTYPE,
    FC_DTYPE,
    TF_DTYPE,
    ACCEPTABLE_FILE_SUFFIXES,
    ACCEPTABLE_FILE_TYPES,
    ACCEPTABLE_FILE_VERSIONS,
    ACCEPTABLE_DATA_LEVELS,
)

from mt_metadata.common.mttime import get_now_utc
from mt_metadata.timeseries import Experiment
from mt_metadata.transfer_functions.core import TF


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

    For version 0.1.0:

        - Survey

           - Reports
           - Standards
           - Filters
           - Stations

               - Run

                   - Channel

    For version 0.2.0:

        - Experiment

            - Reports
            - Standards
            - Surveys

               - Reports
               - Standards
               - Filters
               - Stations

                   - Run

                       -Channel


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
    :param file_version: Version of the file [ '0.1.0' | '0.2.0' ], defaults to "0.2.0"
    :type file_version: string, optional

    :Usage:

    * Open a new file and show initialized file

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5(file_version='0.1.0')
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
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
        data_level=1,
        file_version="0.2.0",
    ):
        self.logger = logger

        # make these private so the user cant accidentally change anything.
        self.__hdf5_obj = None
        (
            self.__compression,
            self.__compression_opts,
        ) = helpers.validate_compression(compression, compression_opts)
        self.__shuffle = shuffle
        self.__fletcher32 = fletcher32

        self.data_level = data_level
        self.filename = filename
        self.file_version = file_version
        self.file_type = "mth5"

        self._set_default_groups()

    def __str__(self):
        if self.h5_is_read():
            return helpers.get_tree(self.__hdf5_obj)
        return "HDF5 file is closed and cannot be accessed."

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_mth5()
        return False

    @property
    def dataset_options(self):
        """summary of dataset options"""
        return {
            "compression": self.__compression,
            "compression_opts": self.__compression_opts,
            "shuffle": self.__shuffle,
            "fletcher32": self.__fletcher32,
        }

    @property
    def file_attributes(self):
        return {
            "file.type": "MTH5",
            "file.version": self.file_version,
            "file.access.platform": platform(),
            "file.access.time": get_now_utc(),
            "mth5.software.version": mth5_version,
            "mth5.software.name": "mth5",
            "data_level": self.data_level,
        }

    @property
    def filename(self):
        """file name of the hdf5 file"""
        if self.h5_is_read():
            return Path(self.__hdf5_obj.filename)
        msg = (
            "MTH5 file is not open or has not been created yet. "
            "Returning default name"
        )
        self.logger.warning(msg)
        return self.__filename

    @filename.setter
    def filename(self, value):
        """make sure file has the proper extension"""
        self.__filename = None
        if value is not None:
            if not isinstance(value, Path):
                value = Path(value)
            if value.suffix not in ACCEPTABLE_FILE_SUFFIXES:
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
        """File Type should be MTH5"""

        if self.h5_is_read():
            # need the try statement for when a file is initialize it does
            # not have the attributes yet.
            try:
                return self.__hdf5_obj.attrs["file.type"]
            except KeyError:
                return self.__file_type
        return self.__file_type

    @file_type.setter
    def file_type(self, value):
        """set file type while validating input"""
        if not isinstance(value, str):
            msg = f"Input file type must be a string not {type(value)}"
            self.logger.error(msg)
            raise ValueError(msg)
        if value not in ACCEPTABLE_FILE_TYPES:
            msg = f"Input file.type is not valid, must be {ACCEPTABLE_FILE_TYPES}"
            self.logger.error(msg)
            raise ValueError(msg)
        self.__file_type = value

        if self.h5_is_read():
            self.__hdf5_obj.attrs["file.type"] = value

    @property
    def file_version(self):
        """mth5 file version"""
        if self.h5_is_read():
            # need the try statement for when a file is initialize it does
            # not have the attributes yet.
            try:
                return self.__hdf5_obj.attrs["file.version"]
            except KeyError:
                return self.__file_version
        return self.__file_version

    @file_version.setter
    def file_version(self, value):
        """set file version while validating input"""
        if not isinstance(value, str):
            msg = f"Input file version must be a string not {type(value)}"
            self.logger.error(msg)
            raise ValueError(msg)
        if value not in ACCEPTABLE_FILE_VERSIONS:
            msg = f"Input file.version is not valid, must be {ACCEPTABLE_FILE_VERSIONS}"
            self.logger.error(msg)
            raise ValueError(msg)
        self.__file_version = value
        self._set_default_groups()

        if self.h5_is_read():
            self.__hdf5_obj.attrs["file.version"] = value

    @property
    def software_name(self):
        """software name that wrote the file"""
        if self.h5_is_read():
            return self.__hdf5_obj.attrs["mth5.software.name"]
        return "mth5"

    @property
    def data_level(self):
        """data level"""
        if self.h5_is_read():
            try:
                return self.__hdf5_obj.attrs["data_level"]
            except KeyError:
                return self.__data_level
        else:
            return self.__data_level

    @data_level.setter
    def data_level(self, value):
        """set data level while validating input"""
        if not isinstance(value, int):
            msg = f"Input file type must be an integer not {type(value)}"
            self.logger.error(msg)
            raise ValueError(msg)
        if value not in ACCEPTABLE_DATA_LEVELS:
            msg = f"Input data_level is not valid, must be {ACCEPTABLE_DATA_LEVELS}"
            self.logger.error(msg)
            raise ValueError(msg)
        self.__data_level = value

        if self.h5_is_read():
            self.__hdf5_obj.attrs["data_level"] = value

    def _set_default_groups(self):
        """get the default groups based on file version"""

        if self.file_version in ["0.1.0"]:
            self._default_root_name = "Survey"
            self._default_subgroup_names = [
                "Stations",
                "Reports",
                "Filters",
                "Standards",
            ]

            self._root_path = "/Survey"
        elif self.file_version in ["0.2.0"]:
            self._default_root_name = "Experiment"
            self._default_subgroup_names = [
                "Surveys",
                "Reports",
                "Standards",
            ]

            self._root_path = "/Experiment"

    @property
    def experiment_group(self):
        """Convenience property for /Experiment group"""
        if self.h5_is_read():
            if self.file_version in ["0.2.0"]:
                return groups.ExperimentGroup(
                    self.__hdf5_obj[f"{self._root_path}"],
                    **self.dataset_options,
                )
            else:
                self.logger.info(
                    f"File version {self.file_version} does not have an Experiment Group"
                )
                return None
        self.logger.info("File is closed cannot access /Experiment")
        return None

    @property
    def survey_group(self):
        """Convenience property for /Survey group"""
        if self.file_version in ["0.1.0"]:
            if self.h5_is_read():
                return groups.SurveyGroup(
                    self.__hdf5_obj[f"{self._root_path}"],
                    **self.dataset_options,
                )
            self.logger.info("File is closed cannot access /Survey")
            return None
        elif self.file_version in ["0.2.0"]:
            self.logger.info(
                f"File version {self.file_version} does not have a survey_group, try surveys_group"
            )

    @property
    def surveys_group(self):
        """Convenience property for /Surveys group"""
        if self.file_version in ["0.1.0"]:
            self.logger.info(
                f"File version {self.file_version} does not have a surveys_group, try survey_group"
            )
        elif self.file_version in ["0.2.0"]:
            if self.h5_is_read():
                return groups.MasterSurveyGroup(
                    self.__hdf5_obj[f"{self._root_path}/Surveys"],
                    **self.dataset_options,
                )
            self.logger.info("File is closed cannot access /Surveys")
            return None

    @property
    def reports_group(self):
        """Convenience property for /Survey/Reports group"""
        if self.h5_is_read():
            return groups.ReportsGroup(
                self.__hdf5_obj[f"{self._root_path}/Reports"],
                **self.dataset_options,
            )
        self.logger.info("File is closed cannot access /Reports")
        return None

    @property
    def filters_group(self):
        """Convenience property for /Survey/Filters group"""
        if self.h5_is_read():
            if self.file_version in ["0.1.0"]:
                return groups.FiltersGroup(
                    self.__hdf5_obj[f"{self._root_path}/Filters"],
                    **self.dataset_options,
                )
            else:
                self.logger.info(
                    "File version 0.2.0 does not have a FiltersGroup at the experiment level"
                )
                return None
        self.logger.info("File is closed cannot access /Filters")
        return None

    @property
    def standards_group(self):
        """Convenience property for /Standards group"""
        if self.h5_is_read():
            return groups.StandardsGroup(
                self.__hdf5_obj[f"{self._root_path}/Standards"],
                **self.dataset_options,
            )
        self.logger.info("File is closed cannot access /Standards")
        return None

    @property
    def stations_group(self):
        """Convenience property for /Survey/Stations group"""
        if self.h5_is_read():
            if self.file_version not in ["0.1.0"]:
                self.logger.info(
                    f"File version {self.file_version} does not have a Stations. "
                    "try surveys_group."
                )
                return None
            return groups.MasterStationGroup(
                self.__hdf5_obj[f"{self._root_path}/Stations"],
                **self.dataset_options,
            )
        self.logger.info("File is closed cannot access /Stations")
        return None

    @property
    def station_list(self):
        """list of existing stations names"""
        if not self.h5_is_read():
            return []
        if self.file_version in ["0.1.0"]:
            return self.stations_group.groups_list
        elif self.file_version in ["0.2.0"]:
            station_list = []
            for survey in self.surveys_group.groups_list:
                sg = self.surveys_group.get_survey(survey)
                station_list += sg.stations_group.groups_list
            return station_list

    def open_mth5(self, filename=None, mode="a", **kwargs):
        """
        open an mth5 file

        :return: Survey Group
        :type: groups.SurveyGroup

        :Example:

        >>> from mth5 import mth5
        >>> mth5_object = mth5.MTH5(file_version='0.1.0')
        >>> survey_object = mth5_object.open_mth5('Test.mth5', 'w')

        >>> from mth5 import mth5
        >>> mth5_object = mth5.MTH5()
        >>> survey_object = mth5_object.open_mth5('Test.mth5', 'w')
        >>> mth5_object.file_version
        '0.2.0'


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
                    self._initialize_file(mode)
                except OSError as error:
                    msg = (
                        f"{error}. Need to close any references to {self.__filename} first. "
                        "Then reopen the file in the preferred mode"
                    )
                    self.logger.exception(msg)
            elif mode in ["a", "w-", "x", "r+"]:
                self.__hdf5_obj = h5py.File(self.__filename, mode=mode, **kwargs)
                self._set_default_groups()
                if not self.validate_file():
                    msg = "Input file is not a valid MTH5 file"
                    self.logger.error(msg)
                    raise MTH5Error(msg)
            elif mode in ["r"]:
                self.__hdf5_obj = h5py.File(self.__filename, mode=mode, **kwargs)
                self._set_default_groups()
                self.validate_file()
            else:
                msg = f"mode {mode} is not understood"
                self.logger.error(msg)
                raise MTH5Error(msg)
        else:
            if mode in ["a", "w", "w-", "x"]:
                self._initialize_file(mode=mode, **kwargs)
            else:
                msg = f"Cannot open new file in mode {mode} "
                self.logger.error(msg)
                raise MTH5Error(msg)
        # TODO need to add a validation step to check for version and legit file
        if not "channel_summary" in self.__hdf5_obj[self._root_path].keys():
            self._initialize_summary()
        return self

    def _initialize_file(self, mode="w", **kwargs):
        """
        Initialize the default groups for the file

        :return: Survey Group
        :rtype: groups.SurveyGroup

        """
        # open an hdf5 file
        self.__hdf5_obj = h5py.File(self.__filename, mode, **kwargs)

        # write general metadata
        self.__hdf5_obj.attrs.update(self.file_attributes)

        # create the default group
        root = self.__hdf5_obj.create_group(self._default_root_name)
        if self._default_root_name == "Survey":
            root_group = groups.SurveyGroup(root)
            root_group.write_metadata()
        for group_name in self._default_subgroup_names:
            try:
                self.__hdf5_obj.create_group(f"{self._default_root_name}/{group_name}")
            except ValueError:
                pass
            m5_grp = getattr(self, f"{group_name.lower()}_group")
            m5_grp.initialize_group()
        self._initialize_summary()

        self.logger.info(
            f"Initialized MTH5 {self.file_version} file {self.filename} in mode {mode}"
        )

    def _initialize_summary(self):
        try:
            # initiate channel and tf summary datasets
            self.__hdf5_obj[self._default_root_name].create_dataset(
                "channel_summary",
                shape=(1,),
                maxshape=(None,),
                dtype=CHANNEL_DTYPE,
                **self.dataset_options,
            )
        except ValueError:
            pass
        try:
            self.__hdf5_obj[self._default_root_name].create_dataset(
                "tf_summary",
                shape=(1,),
                maxshape=(None,),
                dtype=TF_DTYPE,
                **self.dataset_options,
            )
        except ValueError:
            pass
        try:
            self.__hdf5_obj[self._default_root_name].create_dataset(
                "fc_summary",
                shape=(1,),
                maxshape=(None,),
                dtype=FC_DTYPE,
                **self.dataset_options,
            )
        except ValueError:
            pass

    def validate_file(self):
        """
        Validate an open mth5 file

        will test the attribute values and group names

        :return: Boolean [ True = valid, False = not valid]
        :rtype: Boolean

        """

        if self.h5_is_read():
            if self.file_type not in ACCEPTABLE_FILE_TYPES:
                msg = f"Unacceptable file type {self.file_type}"
                self.logger.error(msg)
                return False
            if self.file_version not in ACCEPTABLE_FILE_VERSIONS:
                msg = f"Unacceptable file version {self.file_version}"
                self.logger.error(msg)
                return False
            if self.data_level not in ACCEPTABLE_DATA_LEVELS:
                msg = f"Unacceptable data_level {self.data_level}"
                self.logger.error(msg)
                return False
            if self.file_version in ["0.1.0"]:
                for gr in self.survey_group.groups_list:
                    if "summary" in gr:
                        continue
                    if gr not in self._default_subgroup_names:
                        msg = f"Unacceptable group {gr}"
                        self.logger.error(msg)
                        return False
            elif self.file_version in ["0.2.0"]:
                for gr in self.experiment_group.groups_list:
                    if "summary" in gr:
                        continue
                    if gr not in self._default_subgroup_names:
                        msg = f"Unacceptable group {gr}"
                        self.logger.error(msg)
                        return False
            return True
        self.logger.warning("HDF5 file is not open")
        return False

    def close_mth5(self):
        """
        close mth5 file to make sure everything is flushed to the file
        """

        try:
            # update summay tables
            if self.h5_is_write():
                self.channel_summary.summarize()
                self.tf_summary.summarize()
                self.__hdf5_obj.flush()
            self.logger.info(f"Flushing and closing {str(self.filename)}")
            self.__hdf5_obj.close()
        except (AttributeError, ValueError):
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

    def h5_is_read(self):
        """
        check to see if the hdf5 file is open and readable

        :return: True if readable, False if not
        :rtype: Boolean

        """

        if isinstance(self.__hdf5_obj, h5py.File):
            try:
                if self.__hdf5_obj.mode in ["r", "r+", "a", "w", "w-", "x"]:
                    return True
                return False
            except ValueError:
                return False
        return False

    def has_group(self, group_name):
        """
        Check to see if the group name exists
        """
        if self.h5_is_read():

            def has_name(name):
                if group_name == name:
                    return True

            if self.__hdf5_obj.visit(has_name):
                return True
            return False

    def _make_h5_path(
        self, survey=None, station=None, run=None, channel=None, tf_id=None
    ):
        """
        create an h5 path from inputs
        """

        if self.file_version == "0.1.0":
            h5_path = self._root_path
        elif self.file_version == "0.2.0":
            if survey is None:
                raise ValueError("Survey must be input for file type 0.2.0")
            else:
                survey = helpers.validate_name(survey)
            h5_path = f"{self._root_path}/Surveys/{survey}"
        if station is not None:
            station = helpers.validate_name(station)
            h5_path += f"/Stations/{station}"

            if tf_id is not None:
                tf_id = helpers.validate_name(tf_id)
                h5_path += f"/Transfer_Functions/{tf_id}"
            elif run is not None:
                run = helpers.validate_name(run)
                h5_path += f"/{run}"
                if channel is not None:
                    channel = helpers.validate_name(channel)
                    h5_path += f"/{channel}"
        return h5_path

    def from_reference(self, h5_reference):
        """
        Get an HDF5 group, dataset, etc from a reference

        :param h5_reference: DESCRIPTION
        :type h5_reference: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        ref_dict = {
            "survey": groups.SurveyGroup,
            "station": groups.StationGroup,
            "run": groups.RunGroup,
            "electric": groups.ElectricDataset,
            "magnetic": groups.MagneticDataset,
            "auxiliary": groups.AuxiliaryDataset,
            "transferfunction": groups.TransferFunctionGroup,
        }

        # in the future should allow this to return the proper container.
        referenced = self.__hdf5_obj[h5_reference]
        mth5_type = referenced.attrs["mth5_type"].lower()

        try:
            group = ref_dict[mth5_type](referenced)
            if mth5_type == "transferfunction":
                return group.to_tf_object()
            return group
        except KeyError:
            self.logger.info(
                f"Could not identify the MTH5 type {mth5_type}, " "returning h5 group."
            )
            return referenced

    def to_experiment(self, has_data=True):
        """
        Create an :class:`mt_metadata.timeseries.Experiment` object from the
        metadata contained in the MTH5 file.

        :returns: :class:`mt_metadata.timeseries.Experiment`

        """

        if self.h5_is_read():
            if self.file_version in ["0.1.0"]:
                experiment = Experiment()
                experiment.surveys.append(self.survey_group.metadata)
            elif self.file_version in ["0.2.0"]:
                experiment = self.experiment_group.metadata

            # remove runs that have no data.
            if has_data:
                no_data_runs = self.run_summary[self.run_summary.has_data == False]
                if not no_data_runs.empty:
                    for row in no_data_runs.itertuples():
                        experiment.surveys[row.survey].stations[row.station].remove_run(
                            row.run
                        )

            return experiment

    def from_experiment(self, experiment, survey_index=0, update=False):
        """
        Fill out an MTH5 from a :class:`mt_metadata.timeseries.Experiment` object
        given a survey_id

        :param experiment: Experiment metadata
        :type experiment: :class:`mt_metadata.timeseries.Experiment`
        :param survey_index: Index of the survey to write
        :type survey_index: int, defaults to 0

        """
        if self.h5_is_write():
            if self.file_version in ["0.1.0"]:
                sg = self.survey_group
                sg.metadata.from_dict(experiment.surveys[survey_index].to_dict())
                sg.write_metadata()
                for station in experiment.surveys[0].stations:
                    mt_station = self.add_station(station.id, station_metadata=station)
                    if update:
                        mt_station.metadata.update(station)
                        mt_station.write_metadata()
                    for run in station.runs:
                        mt_run = mt_station.add_run(run.id, run_metadata=run)
                        if update:
                            mt_run.metadata.update(run)
                            mt_run.write_metadata()
                        for channel in run.channels:
                            mt_ch = mt_run.add_channel(
                                channel.component,
                                channel.type,
                                None,
                                channel_metadata=channel,
                            )
                            if update:
                                mt_ch.metadata.update(channel)
                                mt_ch.write_metadata()
                        ### need to update from input metadata for time period
                        ### and channels, runs, stations.
                        mt_run.update_metadata()
                    mt_station.update_metadata()
                sg.update_metadata()
                for k, v in experiment.surveys[0].filters.items():
                    self.filters_group.add_filter(v)
            elif self.file_version in ["0.2.0"]:
                for survey in experiment.surveys:
                    sg = self.add_survey(survey.id, survey_metadata=survey)

                    for station in survey.stations:
                        mt_station = self.add_station(
                            station.id,
                            station_metadata=station,
                            survey=sg.metadata.id,
                        )
                        if update:
                            mt_station.metadata.update(station)
                            mt_station.write_metadata()
                        for run in station.runs:
                            mt_run = mt_station.add_run(run.id, run_metadata=run)
                            if update:
                                mt_run.metadata.update(run)
                                mt_run.write_metadata()
                            for channel in run.channels:
                                mt_ch = mt_run.add_channel(
                                    channel.component,
                                    channel.type,
                                    None,
                                    channel_metadata=channel,
                                )
                                if update:
                                    mt_ch.metadata.update(channel)
                                    mt_ch.write_metadata()
                            mt_run.update_metadata()
                        mt_station.update_metadata()
                    sg.update_metadata()
                    for k, v in survey.filters.items():
                        sg.filters_group.add_filter(v)

    @property
    def channel_summary(self):
        """return a dataframe of channels"""

        return ChannelSummaryTable(
            self.__hdf5_obj[f"{self._root_path}/channel_summary"]
        )

    @property
    def fc_summary(self):
        """return a dataframe of fcs"""

        return FCSummaryTable(self.__hdf5_obj[f"{self._root_path}/fc_summary"])

    @property
    def run_summary(self):
        """return a dataframe of channels"""
        # need to add mth5 file path for future processing classes.
        run_summary_df = self.channel_summary.to_run_summary()
        run_summary_df["mth5_path"] = self.filename.as_posix()

        return run_summary_df

    @property
    def tf_summary(self):
        """return a dataframe of tfs"""

        return TFSummaryTable(self.__hdf5_obj[f"{self._root_path}/tf_summary"])

    def add_survey(self, survey_name, survey_metadata=None):
        """
        Add a survey with metadata if given with the path:
            ``/Experiment/Surveys/survey_name``

        If the survey already exists, will return that survey and nothing
        is added.

        :param survey_name: Name of the survey, should be the same as
                             metadata.id
        :type survey_name: string
        :param survey_metadata: survey metadata container, defaults to None
        :type survey_metadata: :class:`mth5.metadata.survey`, optional
        :return: A convenience class for the added survey
        :rtype: :class:`mth5_groups.SurveyGroup`

        :Example: ::

            >>> from mth5 import mth5
            >>> mth5_obj = mth5.MTH5()
            >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
            >>> # one option
            >>> new_survey = mth5_obj.add_survey('MT001')
            >>> # another option
            >>> new_station = mth5_obj.experiment_group.surveys_group.add_survey('MT001')

        """
        return self.surveys_group.add_survey(
            survey_name, survey_metadata=survey_metadata
        )

    def get_survey(self, survey_name):
        """
        Get a survey with the same name as survey_name

        :param survey_name: existing survey name
        :type survey_name: string
        :return: convenience survey class
        :rtype: :class:`mth5.mth5_groups.surveyGroup`
        :raises MTH5Error:  if the survey name is not found.

        :Example:

        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
        >>> # one option
        >>> existing_survey = mth5_obj.get_survey('MT001')
        >>> # another option
        >>> existing_staiton = mth5_obj.experiment_group.surveys_group.get_survey('MT001')
        MTH5Error: MT001 does not exist, check groups_list for existing names

        """

        survey_path = self._make_h5_path(survey=survey_name)
        try:
            group = groups.SurveyGroup(
                self.__hdf5_obj[survey_path],
                **self.dataset_options,
            )
            group.read_metadata()
            return group
        except KeyError:
            msg = (
                f"{survey_path} does not exist, check survey_list for "
                "existing names."
            )
            self.logger.warning(msg)
            raise MTH5Error(msg)

    def remove_survey(self, survey_name):
        """
        Remove a survey from the file.

        .. note:: Deleting a survey is not as simple as del(survey).  In HDF5
              this does not free up memory, it simply removes the reference
              to that survey.  The common way to get around this is to
              copy what you want into a new file, or overwrite the survey.

        :param survey_name: existing survey name
        :type survey_name: string

        :Example: ::

            >>> from mth5 import mth5
            >>> mth5_obj = mth5.MTH5()
            >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
            >>> # one option
            >>> mth5_obj.remove_survey('MT001')
            >>> # another option
            >>> mth5_obj.experiment_group.surveys_group.remove_survey('MT001')

        """

        survey_path = self._make_h5_path(survey=survey_name)
        try:
            del self.__hdf5_obj[f"{survey_path}"]
            self.logger.info(
                "Deleting a survey does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{survey_path} does not exist, "
                "check station_list for existing names"
            )
            self.logger.warning(msg)
            raise MTH5Error(msg)

    def add_station(self, station_name, station_metadata=None, survey=None):
        """
        Convenience function to add a station using
        ``mth5.stations_group.add_station``


        Add a station with metadata if given with the path [v0.1.0]:
            ``/Survey/Stations/station_name``

        Add a station with metadata if given with the path [v0.2.0]:
            ``Experiment/Surveys/survey/Stations/station_name``

        If the station already exists, will return that station and nothing
        is added.

        :param station_name: Name of the station, should be the same as
                             metadata.archive_id
        :type station_name: string
        :param station_metadata: Station metadata container, defaults to None
        :type station_metadata: :class:`mth5.metadata.Station`, optional
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string
        :return: A convenience class for the added station
        :rtype: :class:`mth5_groups.StationGroup`

        :Example:

        >>> new_staiton = mth5_obj.add_station('MT001')

        """

        if self.file_version in ["0.1.0"]:
            return self.stations_group.add_station(
                station_name, station_metadata=station_metadata
            )
        elif self.file_version in ["0.2.0"]:
            if survey is None:
                msg = f"Need to input 'survey' for file version {self.file_version}"
                self.logger.error(msg)
                raise ValueError(msg)
            sg = self.get_survey(survey)
            return sg.stations_group.add_station(
                station_name, station_metadata=station_metadata
            )

    def get_station(self, station_name, survey=None):
        """
        Convenience function to get a station using

        Get a station with the same name as station_name

        :param station_name: existing station name
        :type station_name: string
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string
        :return: convenience station class
        :rtype: :class:`mth5.mth5_groups.StationGroup`
        :raises MTH5Error:  if the station name is not found.

        :Example:

        >>> existing_staiton = mth5_obj.get_station('MT001')
        MTH5Error: MT001 does not exist, check station_list for existing names

        """

        station_path = self._make_h5_path(survey=survey, station=station_name)
        try:
            group = groups.StationGroup(
                self.__hdf5_obj[station_path], **self.dataset_options
            )
            group.read_metadata()
            return group
        except KeyError:
            raise MTH5Error(f"Could not find station {station_name}")

    def remove_station(self, station_name, survey=None):
        """
        Convenience function to remove a station using

        Remove a station from the file.

        .. note:: Deleting a station is not as simple as del(station).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the station.

        :param station_name: existing station name
        :type station_name: string
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string

        :Example:

        >>> mth5_obj.remove_station('MT001')

        """
        station_name = helpers.validate_name(station_name)
        if self.file_version in ["0.1.0"]:
            return self.stations_group.remove_station(station_name)
        elif self.file_version in ["0.2.0"]:
            if survey is None:
                msg = f"Need to input 'survey' for file version {self.file_version}"
                self.logger.error(msg)
                raise ValueError(msg)
            survey = helpers.validate_name(survey)
            sg = self.get_survey(survey)
            return sg.stations_group.remove_station(station_name)

    def add_run(self, station_name, run_name, run_metadata=None, survey=None):
        """
        Convenience function to add a run using

        Add a run to a given station.

        :param run_name: run name, should be archive_id{a-z}
        :type run_name: string
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string
        :param metadata: metadata container, defaults to None
        :type metadata: :class:`mth5.metadata.Station`, optional

        :example:

        >>> new_run = mth5_obj.add_run('MT001', 'MT001a')


        .. todo:: auto fill run name if none is given.

        .. todo:: add ability to add a run with data.


        """

        return self.get_station(station_name, survey=survey).add_run(
            run_name, run_metadata=run_metadata
        )

    def get_run(self, station_name, run_name, survey=None):
        """
        Convenience function to get a run using
        ``mth5.stations_group.get_station(station_name).get_run()``

        get a run from run name for a given station

        :param station_name: existing station name
        :type station_name: string
        :param run_name: existing run name
        :type run_name: string
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string
        :return: Run object
        :rtype: :class:`mth5.mth5_groups.RunGroup`

        :Example:

        >>> existing_run = mth5_obj.get_run('MT001', 'MT001a')

        """

        run_path = self._make_h5_path(survey=survey, station=station_name, run=run_name)
        try:
            group = groups.RunGroup(self.__hdf5_obj[run_path], **self.dataset_options)
            group.read_metadata()
            return group
        except KeyError:
            raise MTH5Error(f"Could not find {run_path}")

    def remove_run(self, station_name, run_name, survey=None):
        """
        Remove a run from the station.

        .. note:: Deleting a run is not as simple as del(run).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the run.

        :param station_name: existing station name
        :type station_name: string
        :param run_name: existing run name
        :type run_name: string
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string

        :Example:

        >>> mth5_obj.remove_station('MT001', 'MT001a')

        """

        return self.get_station(station_name, survey=survey).remove_run(run_name)

    def add_channel(
        self,
        station_name,
        run_name,
        channel_name,
        channel_type,
        data,
        channel_dtype="int32",
        max_shape=(None,),
        chunks=True,
        channel_metadata=None,
        survey=None,
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
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string
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

        return self.get_run(station_name, run_name, survey=survey).add_channel(
            channel_name,
            channel_type,
            data,
            channel_metadata=channel_metadata,
            channel_dtype=channel_dtype,
            max_shape=max_shape,
            chunks=chunks,
            **self.dataset_options,
        )

    def get_channel(self, station_name, run_name, channel_name, survey=None):
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
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string
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
        run_path = self._make_h5_path(survey=survey, station=station_name, run=run_name)
        rg = groups.RunGroup(self.__hdf5_obj[run_path], **self.dataset_options)
        rg.read_metadata()
        try:
            return rg.get_channel(helpers.validate_name(channel_name))
        except (AttributeError, KeyError):
            raise MTH5Error(f"Could not find channel, {run_path}/{channel_name}")

    def remove_channel(self, station_name, run_name, channel_name, survey=None):
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
        :param survey: existing survey name, needed for file version >= 0.2.0
        :type survey: string

        :Example:

        >>> mth5_obj.remove_channel('MT001', 'MT001a', 'Ex')

        """
        station_name = helpers.validate_name(station_name)
        run_name = helpers.validate_name(run_name)
        channel_name = helpers.validate_name(channel_name)
        return (
            self.get_station(station_name, survey=survey)
            .get_run(run_name)
            .remove_channel(channel_name)
        )

    def add_transfer_function(self, tf_object, update_metadata=True):
        """
        Add a transfer function
        :param tf_object: DESCRIPTION
        :type tf_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(tf_object, TF):
            msg = f"Input must be a TF object not {type(tf_object)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if tf_object.survey_metadata.id == "0":
            tf_object.survey_metadata.id = "unknown_survey"
        tf_object.survey_metadata.id = helpers.validate_name(
            tf_object.survey_metadata.id
        )
        if self.file_version == "0.2.0":
            try:
                # need to check survey metadata to make sure it matches,
                # if it doesn't need to make a new survey group so that
                # when a TF is pulled it gets the proper survey metadata.
                # this should eventually search over each unknonw survey
                # for matching metadata so there aren't 100 groups

                survey_group = self.get_survey(tf_object.survey_metadata.id)
                if tf_object.survey_metadata.id in ["unknown_survey"]:
                    for sg_id in self.surveys_group.groups_list:
                        if "unknown_survey" in sg_id:
                            match = True
                            survey_group = self.get_survey(sg_id)
                            sg_dict = survey_group.metadata.to_dict(
                                single=True, required=False
                            )
                            for (
                                key,
                                value,
                            ) in tf_object.survey_metadata.to_dict(single=True).items():
                                if key in [
                                    "hdf5_reference",
                                    "mth5_type",
                                    "id",
                                ]:
                                    continue
                                if sg_dict[key] != value:
                                    match = False
                                    break
                            if match:
                                break
                    # create a new survey group with a new id, this is likely
                    # not the best way to do this, it should be strongly
                    # encouraged that the user assigne a survey id.
                    if not match:
                        count = 1
                        survey_id = f"unknown_survey_{count:03}"
                        while survey_id in self.surveys_group.groups_list:
                            count += 1
                            survey_id = f"unknown_survey_{count:03}"
                        tf_object.survey_metadata.id = survey_id
                        survey_group = self.add_survey(
                            tf_object.survey_metadata.id,
                            survey_metadata=tf_object.survey_metadata,
                        )
            except MTH5Error:
                survey_group = self.add_survey(
                    tf_object.survey_metadata.id,
                    survey_metadata=tf_object.survey_metadata,
                )
        else:
            survey_group = self.survey_group
            # might need a better test here
            if survey_group.metadata.id is None:
                survey_group.metadata.update(tf_object.survey_metadata)
                survey_group.write_metadata()
        try:
            station_group = survey_group.stations_group.get_station(
                tf_object.station_metadata.id
            )
            # station_group.metadata.update(tf_object.to_ts_station_metadata())
            # station_group.write_metadata()
        except MTH5Error:
            station_group = survey_group.stations_group.add_station(
                tf_object.station_metadata.id,
                station_metadata=tf_object.to_ts_station_metadata(),
            )
        ## need to check for runs and channels
        if tf_object.station_metadata.transfer_function.runs_processed in [
            [],
            [""],
        ]:
            tf_object.station_metadata.transfer_function.runs_processed = (
                tf_object.station_metadata.run_list
            )
        for run_id in tf_object.station_metadata.transfer_function.runs_processed:
            if run_id in ["", None, "None"]:
                continue
            try:
                run_group = station_group.get_run(run_id)
            except MTH5Error:
                run = tf_object.station_metadata.get_run(run_id)
                if run is None:
                    run = tf_object.station_metadata.runs[0].copy()
                    run.id = run_id
                run_group = station_group.add_run(run_id, run_metadata=run)

                if run is not None:
                    for ch in run.channels:
                        try:
                            ch_dataset = run_group.get_channel(ch.component)
                        except MTH5Error:
                            ch_dataset = run_group.add_channel(
                                ch.component,
                                ch.type,
                                None,
                                channel_metadata=ch,
                            )
        try:
            tf_group = station_group.transfer_functions_group.add_transfer_function(
                tf_object.tf_id, tf_object=tf_object
            )
            # need to update time_period from TF here
        except (OSError, RuntimeError, ValueError):
            msg = f"TF {tf_object.tf_id} already exists, returning existing group."
            self.logger.debug(msg)
            tf_group = station_group.transfer_functions_group.get_transfer_function(
                tf_object.tf_id
            )

        if update_metadata:
            survey_group.update_metadata()
        return tf_group

    def get_transfer_function(self, station_id, tf_id, survey=None):
        """
        Get a transfer function

        :param survey_id: DESCRIPTION
        :type survey_id: TYPE
        :param station_id: DESCRIPTION
        :type station_id: TYPE
        :param tf_id: DESCRIPTION
        :type tf_id: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        try:
            tf_df = self.tf_summary.to_dataframe()
            ref = (
                tf_df.loc[
                    (tf_df.station == station_id)
                    & (tf_df.tf_id == tf_id)
                    & (tf_df.survey == "unknown_survey")
                ]
                .iloc[0]
                .hdf5_reference
            )
            return self.from_reference(ref)
        except IndexError:
            tf_path = self._make_h5_path(survey=survey, station=station_id, tf_id=tf_id)
            try:
                tg = groups.TransferFunctionGroup(
                    self.__hdf5_obj[tf_path], **self.dataset_options
                )
                tg.read_metadata()
                return tg.to_tf_object()
            except KeyError:
                raise MTH5Error(f"Could not find {tf_path}")

    def remove_transfer_function(self, station_id, tf_id, survey=None):
        """
        remove a transfer function

        :param survey_id: DESCRIPTION
        :type survey_id: TYPE
        :param station_id: DESCRIPTION
        :type station_id: TYPE
        :param tf_id: DESCRIPTION
        :type tf_id: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        station_group = self.get_station(station_id, survey=survey)

        station_group.transfer_functions_group.remove_transfer_function(tf_id)


def _default_table_names() -> list:
    """
    track a global list of mth5 table names
    """
    return ["channel_summary", "fc_summary", "tf_summary"]
