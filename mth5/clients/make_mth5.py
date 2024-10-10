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
        self.compression = "gzip"
        self.compression_opts = 4
        self.shuffle = True
        self.fletcher32 = True
        self.data_level = 1

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.save_path is None:
            self.save_path = Path().cwd()

    @classmethod
    def from_fdsn_client(self, request_df, client="IRIS", **kwargs):
        """
        Pull data from an FDSN archive like IRIS.  Uses Obspy.Clients.

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

        fdsn_client = FDSN(
            client=client,
            mth5_version=self.mth5_version,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            data_level=self.data_level,
            **kwargs,
        )

        mth5_object = fdsn_client.make_mth5_from_fdsn_client(
            request_df, path=self.save_path, interact=self.interact
        )

        return mth5_object

    @classmethod
    def from_usgs_geomag(self, request_df, **kwargs):
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

        """

        geomag_client = USGSGeomag(
            mth5_version=self.mth5_version,
            save_path=self.save_path,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            data_level=self.data_level,
            interact=self.interact,
            **kwargs,
        )

        return geomag_client.make_mth5_from_geomag(request_df)

    @classmethod
    def from_zen(
        self,
        data_path,
        sample_rates=[4096, 1024, 256],
        calibration_path=None,
        survey_id=None,
        combine=True,
        **kwargs
    ):
        """
        Create an MTH5 from zen data.

        :param data_path: DESCRIPTION
        :type data_path: TYPE
        :param sample_rates: DESCRIPTION, defaults to [4096, 1024, 256]
        :type sample_rates: TYPE, optional
        :param save_path: DESCRIPTION, defaults to None
        :type save_path: TYPE, optional
        :param calibration_path: DESCRIPTION, defaults to None
        :type calibration_path: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        kwargs["compression"] = self.compression
        kwargs["compression_opts"] = self.compression_opts
        kwargs["shuffle"] = self.shuffle
        kwargs["fletcher32"] = self.fletcher32
        kwargs["data_level"] = self.data_level

        zc = ZenClient(
            data_path,
            sample_rates=sample_rates,
            save_path=self.save_path,
            calibration_path=calibration_path,
            **kwargs,
        )

        return zc.make_mth5_from_zen(
            survey_id=survey_id, combine=combine, **kwargs
        )

    @classmethod
    def from_phoenix(
        self,
        data_path,
        mth5_filename=None,
        sample_rates=[150, 24000],
        receiver_calibration_dict=None,
        sensor_calibration_dict=None,
        save_path=None,
    ):

        phx_client = PhoenixClient(
            data_path,
            mth5_filename=mth5_filename,
            sample_rates=sample_rates,
            receiver_calibration_dict=receiver_calibration_dict,
            sensor_calibration_dict=sensor_calibration_dict,
            save_path=save_path,
        )

        return phx_client.make_mth5_from_phoenix()
