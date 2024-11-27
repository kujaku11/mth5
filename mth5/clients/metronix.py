# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:23:50 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5.mth5 import MTH5
from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.io.metronix import ATSS, MetronixCollection

# =============================================================================


class MetronixClient(ClientBase):
    def __init__(
        self,
        data_path,
        sample_rates=[128],
        save_path=None,
        calibration_path=None,
        mth5_filename="from_metronix.h5",
        **kwargs,
    ):

        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates,
            mth5_filename=mth5_filename,
            **kwargs,
        )

        self.collection = MetronixCollection(self.data_path)

    def get_run_dict(self, run_name_zeros=0):
        """
        get run information

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(
            sample_rates=self.sample_rates,
            run_name_zeros=run_name_zeros,
            calibration_path=self.calibration_path,
        )

    def make_mth5_from_metronix(self, run_name_zeros=0, **kwargs):
        """
        Create an MTH5 from new ATSS + JSON style Metronix data.

        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        runs = self.get_run_dict(run_name_zeros=run_name_zeros)

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, "w")
