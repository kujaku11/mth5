# -*- coding: utf-8 -*-
"""
Make MTH5
============

This module provides helper functions to make MTH5 file from various clients

Supported Clients include:
    
    * FDSN (through Obspy)
    * Science Base (TODO)
    * NCI - Australia (TODO)

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
from . import LEMI424Client

# =============================================================================


class MakeMTH5:
    def __init__(
        self, mth5_version="0.2.0", interact=False, save_path=None, **kwargs
    ):
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

        return zc.make_mth5_from_zen(
            survey_id=survey_id, combine=combine, **kwargs
        )

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
    def from_lemi424(
        cls,
        data_path,
        survey_id,
        station_id,
        mth5_filename=None,
        save_path=None,
        **kwargs,
    ):
        """
        Build a MTH5 file from LEMI 424 long period data.  Works mainly on a
        station by station basis because there is limited metadata.

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        >>> MakeMTH5.from_lemi424(
            data_path, 'test', 'mt01', **{'h5_compression_opts': 1}
            )


        :param data_path: Directory where data files are, could be a single
         station or a full directory of stations.
        :type data_path: str or Path
        :param survey_id: survey ID for all stations
        :type survey_id: str
        :param station_id: station ID for station
        :type station_id: string
        :param mth5_filename: filename for the H5, defaults to 'from_lemi424.h5'
        :type mth5_filename: str, optional
        :param save_path: path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :return: Path to MTH5 file
        :rtype: Path
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        lemi_client = LEMI424Client(
            data_path,
            save_path=save_path,
            mth5_filename=mth5_filename,
            **kw_dict,
        )

        return lemi_client.make_mth5_from_lemi424(survey_id, station_id)
