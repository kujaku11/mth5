# -*- coding: utf-8 -*-
"""
Make MTH5
============

This module provides helper functions to make MTH5 file from various clients

Supported Clients include:

    * FDSN (through Obspy)
    * Science Base (TODO)
    * NCI - Australia (TODO)
    * Phoenix MTU-5C
    * Zen
    * LEMI (424 and 423)
    * Metronix
    * UoA (PR6-24 and Orange Box)


Updated on Wed Aug  25 19:57:00 2021

@author: jpeacock + tronan
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from . import FDSN
from . import USGSGeomag
from . import PhoenixClient
from . import ZenClient
from . import LEMIClient
from . import MetronixClient
from . import UoAClient

# =============================================================================


class MakeMTH5:
    def __init__(self, mth5_version="0.2.0", interact=False, save_path=None, **kwargs):
        """

        :param mth5_version: MTH5 file version, defaults to "0.2.0"
        :type mth5_version: string, optional
        :param interact: keep file open (True) or close it (False), defaults to False
        :type interact: Boolean, optional
        :param save_path: Path to save MTH5 file to, defaults to None
        :type save_path: string or :class:`pathlib.Path`, optional

        """

        self.mth5_version = mth5_version
        self.interact = interact
        self.save_path = save_path
        self.h5_compression = "gzip"
        self.h5_compression_opts = 4
        self.h5_shuffle = True
        self.h5_fletcher32 = True
        self.h5_data_level = 1

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.save_path is None:
            self.save_path = Path().cwd()

    def __str__(self):
        lines = ["MakeMTH5 Attibutes:"]
        for key, value in self.get_h5_kwargs().items():
            lines.append(f"\t{key}: {value}")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def get_h5_kwargs(self):
        h5_params = dict(
            mth5_version=self.mth5_version,
            h5_compression=self.h5_compression,
            h5_compression_opts=self.h5_compression_opts,
            h5_shuffle=self.h5_shuffle,
            h5_fletcher32=self.h5_fletcher32,
            h5_data_level=self.h5_data_level,
        )

        for key, value in self.__dict__.items():
            if key.startswith("h5"):
                h5_params[key] = value

        return h5_params

    @classmethod
    def from_fdsn_client(cls, request_df, client="IRIS", **kwargs):
        """
        Pull data from an FDSN archive like IRIS.  Uses Obspy.Clients.

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        >>> MakeMTH5.from_fdsn_client(
            request_df, **{'h5_compression_opts': 1}
            )

        :param request_df: DataFrame with columns

            - 'network'   --> FDSN Network code
            - 'station'   --> FDSN Station code
            - 'location'  --> FDSN Location code
            - 'channel'   --> FDSN Channel code
            - 'start'     --> Start time YYYY-MM-DDThh:mm:ss
            - 'end'       --> End time YYYY-MM-DDThh:mm:ss

        :type request_df: :class:`pandas.DataFrame`

        :param client: FDSN client name, defaults to "IRIS"
        :type client: string, optional
        :raises AttributeError: If the input DataFrame is not properly
         formatted an Attribute Error will be raised.
        :raises ValueError: If the values of the DataFrame are not correct a
         ValueError will be raised.
        :param interact: Boolean to keep the created MTH5 file open or not
        :type interact: bool
        :return: MTH5 file name
        :rtype: :class:`pathlib.Path`


        .. seealso:: https://docs.obspy.org/packages/obspy.clients.fdsn.html#id1

        .. note:: If any of the column values are blank, then any value will
        searched for.  For example if you leave 'station' blank, any station
        within the given start and end time will be returned.

        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        fdsn_client = FDSN(client=client, **kw_dict)

        mth5_object = fdsn_client.make_mth5_from_fdsn_client(
            request_df, path=maker.save_path, interact=maker.interact
        )

        return mth5_object

    @classmethod
    def from_fdsn_miniseed_and_stationxml(
        cls, station_xml_path, miniseed_files, save_path=None, **kwargs
    ):
        """
        This will create an MTH5 if you already have a StationXML and miniSEED
        files that you created or downloaded from an FDSN client.

        :param station_xml_path: can be either full path to StationXML file
         or an obspy.Inventory object
        :type client: string, pathlib.Path, obspy.Inventory
        :raises TypeError: If the input is not of correct type.
        :param miniseed_files: a list of miniseed file paths or obspy.Stream
         objects.  Can also be a single file path or obspy.Stream object.
        :type miniseed_files: list, pathlib.path, str, obspy.Stream
        :param save_path: directory to save new MTH5 file to. If None
         then the file will be saved to the current working directory.
        :type save_path: str, pathlib.Path, optional
        :return: MTH5 file name, will be
         save_path.joinpath(unique_network_unique_stations.h5)
        :rtype: :class:`pathlib.Path`

        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        fdsn_client = FDSN(**kw_dict)

        return fdsn_client.make_mth5_from_inventory_and_streams(
            station_xml_path, miniseed_files, save_path=save_path
        )

    @classmethod
    def from_usgs_geomag(cls, request_df, **kwargs):
        """
        Download geomagnetic observatory data from USGS webservices into an
        MTH5 using a request dataframe or csv file.

        - **observatory**: Geogmangetic observatory ID
        - **type**: type of data to get 'adjusted'
        - **start**: start date time to request UTC
        - **end**: end date time to request UTC
        - **elements**: components to get
        - **sampling_period**: samples between measurements in seconds

        :param request_df: DataFrame with columns

            - 'observatory'     --> Observatory code
            - 'type'            --> data type [ 'variation' | 'adjusted' | 'quasi-definitive' | 'definitive' ]
            - 'elements'        --> Elements to get [D, DIST, DST, E, E-E, E-N, F, G, H, SQ, SV, UK1, UK2, UK3, UK4, X, Y, Z]
            - 'sampling_period' --> sample period [ 1 | 60 | 3600 ]
            - 'start'           --> Start time YYYY-MM-DDThh:mm:ss
            - 'end'             --> End time YYYY-MM-DDThh:mm:ss

        :type request_df: :class:`pandas.DataFrame`, str or Path if csv file


        :return: if interact is True an MTH5 object is returned otherwise the
         path to the file is returned
        :rtype: Path or :class:`mth5.mth5.MTH5`

        .. seealso:: https://www.usgs.gov/tools/web-service-geomagnetism-data

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        >>> MakeMTH5.from_usgs_geomag(
            request_df, **{'h5_compression_opts': 1}
            )

        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        geomag_client = USGSGeomag(
            save_path=maker.save_path,
            interact=maker.interact,
            **kw_dict,
        )

        return geomag_client.make_mth5_from_geomag(request_df)

    @classmethod
    def from_zen(
        cls,
        data_path,
        sample_rates=[4096, 1024, 256],
        calibration_path=None,
        survey_id=None,
        combine=True,
        **kwargs,
    ):
        """
        Create an MTH5 from zen data.

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        >>> MakeMTH5.from_zen(
            data_path, **{'h5_compression_opts': 1}
            )

        :param data_path: directory to where data are stored
        :type data_path: Path, str
        :param sample_rates: sample rates to include,
         defaults to [4096, 1024, 256]
        :type sample_rates: list, optional
        :param save_path: path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :param calibration_path: path to calibration file amtant.cal,
         defaults to None
        :type calibration_path: str or Path, optional
        :param survey_id: survey ID to apply to all station found under
         `data_path`, defaults to None
        :type survey_id: string
        :param combine: if True combine the runs into a single run sampled at 1s,
         defaults to True
        :type combine: bool
        :return: MTH5 file name
        :rtype: Path

        """

        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        zc = ZenClient(
            data_path,
            sample_rates=sample_rates,
            save_path=maker.save_path,
            calibration_path=calibration_path,
            **kw_dict,
        )

        return zc.make_mth5_from_zen(survey_id=survey_id, combine=combine, **kwargs)

    @classmethod
    def from_phoenix(
        cls,
        data_path,
        mth5_filename=None,
        save_path=None,
        sample_rates=[150, 24000],
        receiver_calibration_dict=None,
        sensor_calibration_dict=None,
        **kwargs,
    ):
        """
        Build an H5 file from Phoenix MTU-5C files.  The key step when working
        with Phoenix data is to export the scal and rxcal files into JSON using
        EMPower.  Place these files in a folder that you can easily find, and
        you can use this folder as inputs for the `receiver_calibration_dict`
        and `sensor_calibration_dict` which will search the folder and read
        in all the rxcal and scal files into appropriate dictionaries such
        that the filters will be linked with the data for appropriate
        calibration.

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        >>> MakeMTH5.from_phoenix(
            data_path, **{'h5_compression_opts': 1}
            )

        :param data_path: Directory where data files are, could be a single
         station or a full directory of stations.
        :type data_path: str or Path
        :param mth5_filename: filename for the H5, defaults to 'from_phoenix.h5'
        :type mth5_filename: str, optional
        :param save_path: path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :param sample_rates: sample rates to include in file, defaults to
         [150, 24000]
        :type sample_rates: list, optional
        :param receiver_calibration_dict: This can either be a directory path to
         where the rxcal.json files are or a dictionary where keys are the
         receiver IDs and the values are the filename of the rxcal.json file,
         defaults to None
        :type receiver_calibration_dict: str, Path or dict, optional
        :param sensor_calibration_dict: This can either be a directory path to
         where the scal.json files are or a dictionary where keys are the
         sensor IDs and the values are the `PhoenixCalibration` objects that
         have read in the scal.json file for that sensor, defaults to None
        :type sensor_calibration_dict: str, Path, dict, optional
        :return: Path to MTH5 file
        :rtype: Path

        """

        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        phx_client = PhoenixClient(
            data_path,
            mth5_filename=mth5_filename,
            sample_rates=sample_rates,
            receiver_calibration_dict=receiver_calibration_dict,
            sensor_calibration_dict=sensor_calibration_dict,
            save_path=save_path,
            **kw_dict,
        )

        return phx_client.make_mth5_from_phoenix()

    @classmethod
    def from_lemi(
        cls,
        data_path,
        survey_id,
        station_id,
        sample_rates=None,
        mth5_filename=None,
        save_path=None,
        **kwargs,
    ):
        """
        Build a MTH5 file from LEMI instrument data.

        Supports both LEMI-424 (long period, .txt) and LEMI-423 (broadband, .B423).
        Works mainly on a station by station basis because there is limited metadata
        in LEMI files.

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        :Example:

            >>> # LEMI-424 data
            >>> MakeMTH5.from_lemi(
            ...     '/data/lemi424', 'Survey01', 'MT001',
            ...     sample_rates=[1],
            ...     **{'h5_compression_opts': 1}
            ... )
            >>>
            >>> # LEMI-423 data at 1000 Hz
            >>> MakeMTH5.from_lemi(
            ...     '/data/lemi423', 'Survey01', 'MT002',
            ...     sample_rates=[1000]
            ... )
            >>>
            >>> # LEMI-423 data, unknown rate (include all possible rates)
            >>> MakeMTH5.from_lemi(
            ...     '/data/lemi423', 'Survey01', 'MT003',
            ...     sample_rates=[4000, 2000, 1000, 500, 250]
            ... )

        :param data_path: Directory where LEMI data files are, could be a single
         station or a full directory of stations.
        :type data_path: str or Path
        :param survey_id: Survey ID for all stations
        :type survey_id: str
        :param station_id: Station ID for station
        :type station_id: str
        :param sample_rates: Sample rates to include. If None, will auto-detect and
         include all sample rates found in files. If specified, only processes files
         matching those rates. Valid rates:
         - LEMI-424: [1] Hz (long period only)
         - LEMI-423: [4000], [2000], [1000], [500], [250] Hz (broadband, auto-detected)
         Examples:
         - None (default): Process all files regardless of sample rate
         - [1]: Only LEMI-424 files
         - [1000]: Only LEMI-423 files recorded at 1000 Hz
         - [4000, 2000, 1000, 500, 250]: All possible LEMI-423 rates
        :type sample_rates: list or None, optional
        :param mth5_filename: Filename for the H5, defaults to 'from_lemi.h5'
        :type mth5_filename: str, optional
        :param save_path: Path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :return: Path to MTH5 file
        :rtype: Path

        .. note::
            - LEMI-424 files are .txt format at 1 Hz sample rate
            - LEMI-423 files are .B423 format with variable sample rates
            - Station and survey IDs must be provided (not in file metadata)
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        # If sample_rates is None, include all possible rates for auto-detection
        # This allows both LEMI-424 (1 Hz) and LEMI-423 (4000/2000/1000/500/250 Hz)
        # to be processed without user having to specify
        if sample_rates is None:
            sample_rates = [1, 4000, 2000, 1000, 500, 250]

        lemi_client = LEMIClient(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates,
            mth5_filename=mth5_filename,
            **kw_dict,
        )

        return lemi_client.make_mth5_from_lemi(survey_id, station_id)

    @classmethod
    def from_lemi424(cls, data_path, survey_id, station_id, sample_rates=None,
                     mth5_filename=None, save_path=None, **kwargs):
        """
        Backward compatibility alias for :meth:`from_lemi`.
        """
        return cls.from_lemi(data_path, survey_id, station_id, sample_rates=sample_rates,
                            mth5_filename=mth5_filename, save_path=save_path, **kwargs)

    @classmethod
    def from_metronix(
        cls,
        data_path,
        sample_rates=[128],
        mth5_filename=None,
        save_path=None,
        run_name_zeros=0,
        **kwargs,
    ):
        """
        Build a MTH5 file from Metronix Geophysics ATSS + JSON files saved in
        their new folder structure.

        :param data_path: Highest level of where you want to archive data from,
         usuall the survey level.  If you want just a single station, then use
         the station folder path.
        :type data_path: str or pathlib.Path
        :param sample_rates: sample rates to archive in samples/sercond,
         defaults to [128]
        :type sample_rates: list of floats, optional
        :param mth5_filename: filename for the H5, defaults to 'from_lemi424.h5'
        :type mth5_filename: str, optional
        :param save_path: path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :param run_name_zeros: number of zeros to include in new run names, will
         be named as 'sr{sample_rate}_{run_id:0{run_name_zeros}}'.  If set to
         0 then will use the original run name, usually 'run_0001'.
        :type run_name_zeros: int, defaults to 0
        :return: Path to MTH5 file
        :rtype: Path

        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        metronix_client = MetronixClient(
            data_path,
            sample_rates=sample_rates,
            save_path=save_path,
            mth5_filename=mth5_filename,
            **kw_dict,
        )

        return metronix_client.make_mth5_from_metronix(run_name_zeros=run_name_zeros)

    @classmethod
    def from_uoa(
        cls,
        data_path,
        survey_id,
        station_id,
        instrument_type,
        run_id="001",
        mth5_filename=None,
        save_path=None,
        **kwargs,
    ):
        """
        Build a MTH5 file from University of Adelaide instrument data.

        Supports PR6-24 (Earth Data Logger) and Orange Box instruments.

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        :Example:

            >>> # PR6-24 long-period data
            >>> MakeMTH5.from_uoa(
            ...     '/data/pr624', 'Survey01', 'MT001',
            ...     instrument_type='pr624',
            ...     sample_rate=10.0,
            ...     sensor_type='bartington',
            ...     dipole_length_ex=50.0,
            ...     dipole_length_ey=50.0,
            ...     **{'h5_compression_opts': 1}
            ... )
            >>>
            >>> # Orange Box data
            >>> MakeMTH5.from_uoa(
            ...     '/data/orange', 'Survey01', 'ST61',
            ...     instrument_type='orange',
            ...     dipole_length_ex=100.0,
            ...     dipole_length_ey=100.0
            ... )

        :param data_path: Directory or file path to UoA data
        :type data_path: str or Path
        :param survey_id: Survey ID
        :type survey_id: str
        :param station_id: Station ID
        :type station_id: str
        :param instrument_type: 'pr624' or 'orange'
        :type instrument_type: str
        :param run_id: Run ID, defaults to "001"
        :type run_id: str, optional
        :param mth5_filename: Filename for the H5, defaults to 'from_uoa.h5'
        :type mth5_filename: str, optional
        :param save_path: Path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :param kwargs: Additional arguments for reader:

            **PR6-24 arguments:**
                - sample_rate (float): REQUIRED - sample rate in Hz
                - sensor_type (str): 'bartington' or 'lemi120'
                - dipole_length_ex (float): Ex dipole length in meters
                - dipole_length_ey (float): Ey dipole length in meters
                - calibration_fn_bx (str): LEMI-120 .rsp file for Bx
                - calibration_fn_by (str): LEMI-120 .rsp file for By
                - calibration_fn_bz (str): LEMI-120 .rsp file for Bz
                - latitude (float): Station latitude
                - longitude (float): Station longitude
                - elevation (float): Station elevation

            **Orange Box arguments:**
                - dipole_length_ex (float): Ex dipole length in meters
                - dipole_length_ey (float): Ey dipole length in meters
                - latitude (float): Station latitude
                - longitude (float): Station longitude
                - elevation (float): Station elevation

        :return: Path to MTH5 file
        :rtype: Path

        .. note::
            - PR6-24 requires sample_rate parameter (not in file metadata)
            - Orange Box auto-detects sample rate from file header
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        uoa_client = UoAClient(
            data_path,
            instrument_type=instrument_type,
            save_path=save_path,
            mth5_filename=mth5_filename,
            **kw_dict,
        )

        return uoa_client.make_mth5_from_uoa(survey_id, station_id, run_id, **kwargs)
