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

from . import FDSN

# =============================================================================


class MakeMTH5:
    def __init__(self, mth5_version="0.2.0", interact=False, save_path=None):
        """

        :param mth5_version: MTH5 file version, defaults to "0.2.0"
        :type mth5_version: string, optional
        :param interact: keep file open (True) or close it (False), defaults to False
        :type interact: Boolean, optional
        :param save_path: Path to save MTH5 file to, defaults to None
        :type save_path: string or :class:`pathlib.Path`, optional

        """

        self.mth5_version = mth5_version
        self.interact = False
        self.save_path = None
        self.compression = "gzip"
        self.compression_opts = 4
        self.shuffle = True
        self.fletcher32 = True
        self.data_level = 1
        self.file_version = "0.2.0"

    def from_fdsn_client(self, request_df, client="IRIS"):
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

        fdsn_client = FDSN(client=client, mth5_version=self.mth5_version)
        mth5_object = fdsn_client.make_mth5_from_fdsnclient(
            request_df, path=self.save_path, interact=self.interact
        )

        return mth5_object
