#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z3DCollection
=================

An object to hold Z3D file information to make processing easier.


Created on Sat Apr  4 12:40:40 2020

@author: peacock
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mth5.io.collection import Collection
from mth5.io.zen import Z3D
from mth5.io.zen.coil_response import CoilResponse

from mt_metadata.timeseries import Station

# =============================================================================
# Collection of Z3D Files
# =============================================================================


class Z3DCollection(Collection):
    """
    An object to deal with a collection of Z3D files. Metadata and information
    are contained with in Pandas DataFrames for easy searching.

    """

    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.station_metadata_dict = {}
        self.file_ext = "z3d"

    def get_calibrations(self, antenna_calibration_file):
        """
        Get coil calibrations from the antenna.cal file

        :param antenna_calibration_file: DESCRIPTION
        :type antenna_calibration_file: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return CoilResponse(antenna_calibration_file)

    def _sort_station_metadata(self, station_list):
        """

        :param station_list: DESCRIPTION
        :type station_list: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        sdf = pd.DataFrame(station_list)
        info = {}
        for station in sdf.id.unique():
            station_df = sdf[sdf.id == station]
            station_metadata = Station()
            station_metadata.id = station
            station_metadata.location.latitude = station_df[
                "location.latitude"
            ].median()
            station_metadata.location.longitude = station_df[
                "location.longitude"
            ].median()
            station_metadata.location.elevation = station_df[
                "location.elevation"
            ].median()

            info[station] = station_metadata

        return info

    def to_dataframe(
        self, sample_rates=[256, 4096], run_name_zeros=4, calibration_path=None
    ):
        """
        Get general z3d information and put information in a dataframe

        :param z3d_fn_list: List of files Paths to z3d files
        :type z3d_fn_list: list

        :return: Dataframe of z3d information
        :rtype: Pandas.DataFrame

        :Example: ::

            >>> zc_obj = zc.Z3DCollection(r"/home/z3d_files")
            >>> z3d_fn_list = zc.get_z3d_fn_list()
            >>> z3d_df = zc.get_z3d_info(z3d_fn_list)
            >>> # write dataframe to a file to use later
            >>> z3d_df.to_csv(r"/home/z3d_files/z3d_info.csv")

        """

        station_metadata = []
        cal_obj = self.get_calibrations(calibration_path)
        entries = []
        for z3d_fn in set(
            self.get_files(
                [self.file_ext, self.file_ext.lower(), self.file_ext.upper()]
            )
        ):
            z3d_obj = Z3D(z3d_fn)
            z3d_obj.read_all_info()
            station_metadata.append(
                z3d_obj.station_metadata.to_dict(single=True)
            )
            if not int(z3d_obj.sample_rate) in sample_rates:
                self.logger.warning(
                    f"{z3d_obj.sample_rate} not in {sample_rates}"
                )
                return

            entry = self.get_empty_entry_dict()
            entry["survey"] = z3d_obj.metadata.job_name
            entry["station"] = z3d_obj.station
            entry["run"] = None
            entry["start"] = z3d_obj.start.isoformat()
            entry["end"] = z3d_obj.end.isoformat()
            entry["channel_id"] = z3d_obj.channel_number
            entry["component"] = z3d_obj.component
            entry["fn"] = z3d_fn
            entry["sample_rate"] = z3d_obj.sample_rate
            entry["file_size"] = z3d_obj.file_size
            entry["n_samples"] = z3d_obj.n_samples
            entry["sequence_number"] = 0
            entry["dipole"] = z3d_obj.dipole_length
            entry["coil_number"] = z3d_obj.coil_number
            entry["latitude"] = z3d_obj.latitude
            entry["longitude"] = z3d_obj.longitude
            entry["elevation"] = z3d_obj.elevation
            entry["instrument_id"] = f"ZEN_{int(z3d_obj.header.box_number):03}"
            entry["coil_number"] = z3d_obj.coil_number
            if cal_obj.has_coil_number(z3d_obj.coil_number):
                entry["calibration_fn"] = cal_obj.calibration_file
            else:
                entry["calibration_fn"] = None

            entries.append(entry)
        # make pandas dataframe and set data types
        df = self._sort_df(
            self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
        )

        self.station_metadata_dict = self._sort_station_metadata(
            station_metadata
        )

        return df

    def assign_run_names(self, df, zeros=3):
        # assign run names
        for station in df.station.unique():
            starts = sorted(df[df.station == station].start.unique())
            for block_num, start in enumerate(starts, 1):
                sample_rate = df[
                    (df.station == station) & (df.start == start)
                ].sample_rate.unique()[0]

                df.loc[
                    (df.station == station) & (df.start == start), "run"
                ] = f"sr{sample_rate:.0f}_{block_num:0{zeros}}"
                df.loc[
                    (df.station == station) & (df.start == start),
                    "sequence_number",
                ] = block_num
        return df
