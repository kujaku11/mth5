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
import numpy as np
import pandas as pd
from pathlib import Path

from mth5.io.collection import Collection
from mth5.io.zen import Z3D

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

    def get_calibrations(self, calibration_path):
        """
        get coil calibrations
        """
        if calibration_path is None:
            self.logger.warning("Calibration path is None")
            return {}
        if not isinstance(calibration_path, Path):
            calibration_path = Path(calibration_path)
        if not calibration_path.exists():
            self.logger.warning(
                "WARNING: could not find calibration path: "
                "{0}".format(calibration_path)
            )
            return {}
        calibration_dict = {}
        for cal_fn in calibration_path.glob("*.csv"):
            cal_num = cal_fn.stem
            calibration_dict[cal_num] = cal_fn
        return calibration_dict

    def to_dataframe(self, calibration_path=None):
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

        cal_dict = self.get_calibrations(calibration_path)
        z3d_info_list = []
        for z3d_fn in self._get_files(".z3d"):
            z3d_obj = Z3D(z3d_fn)
            z3d_obj.read_all_info()

            entry = {}
            entry["survey"] = z3d_obj.metadata.job_name
            entry["station"] = z3d_obj.station
            entry["run"] = None
            entry["start"] = z3d_obj.start
            entry["end"] = z3d_obj.end
            entry["channel_id"] = z3d_obj.channel_number
            entry["component"] = z3d_obj.component
            entry["fn"] = z3d_fn
            entry["sample_rate"] = z3d_obj.sample_rate
            entry["file_size"] = z3d_fn.stat().st_size
            entry["n_samples"] = 0
            entry["sequence_number"] = 0
            entry["instrument_id"] = z3d_obj.header.box_number
            if cal_dict:
                try:
                    entry["calibration_fn"] = cal_dict[z3d_obj.coil_number]
                except KeyError:
                    self.logger.warning(
                        f"Could not find {z3d_obj.coil_number}"
                    )

            z3d_info_list.append(entry)
        # make pandas dataframe and set data types
        z3d_df = pd.DataFrame(z3d_info_list)
        z3d_df.start = pd.to_datetime(z3d_df.start, errors="coerce")
        z3d_df.stop = pd.to_datetime(z3d_df.stop, errors="coerce")

        z3d_df = self.assign_runs(z3d_df)

        return z3d_df

    def assign_runs(self, df, zeros=3):

        # assign block numbers
        for sr in df.sample_rate.unique():
            starts = sorted(df[df.sampling_rate == sr].start.unique())
            for block_num, start in enumerate(starts):
                df.loc[(df.start == start), "block"] = block_num
        return df

    def _validate_block_dict(self, z3d_df, block_dict):
        """ """
        if block_dict is None:
            block_dict = {}
            for sr in z3d_df.sampling_rate.unique():
                block_dict[sr] = list(
                    z3d_df[z3d_df.sampling_rate == sr].block.unique()
                )
        else:
            assert isinstance(block_dict, dict), "Blocks is not a dictionary."
            for key, value in block_dict.items():
                if isinstance(value, str):
                    if value == "all":
                        block_dict[key] = list(
                            z3d_df[z3d_df.sampling_rate == key].block.unique()
                        )
        return block_dict
