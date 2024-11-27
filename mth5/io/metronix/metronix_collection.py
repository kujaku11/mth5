# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:22:44 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5.io.collection import Collection
from mth5.io.metronix import ATSS

# =============================================================================


class MetronixCollection(Collection):
    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext = ["atss"]

    def to_dataframe(
        self, sample_rates=[128], rename_run=False, run_name_zeros=4
    ):
        """
        Create dataframe for metronix timeseries atss + json file sets

        :param sample_rates: DESCRIPTION, defaults to [128]
        :type sample_rates: TYPE, optional
        :param run_name_zeros: DESCRIPTION, defaults to 4
        :type run_name_zeros: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        entries = []
        for atss_fn in set(self.get_files(self.file_ext)):
            atss_obj = ATSS(atss_fn)
            ch_metadata = atss_obj.channel_metadata

            entry = self.get_empty_entry_dict()
            entry["survey"] = atss_obj.survey_id
            entry["station"] = atss_obj.station_id
            entry["run"] = atss_obj.run_id
            entry["start"] = ch_metadata.time_period.start
            entry["end"] = ch_metadata.time_period.end
            entry["channel_id"] = atss_obj.channel_number
            entry["component"] = atss_obj.component
            entry["fn"] = atss_fn
            entry["sample_rate"] = ch_metadata.sample_rate
            entry["file_size"] = atss_obj.file_size
            entry["n_samples"] = atss_obj.n_samples
            entry["sequence_number"] = 0
            entry["dipole"] = 0
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
