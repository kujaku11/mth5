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

    def get_run_dict(self):
        """
        get run information

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(
            sample_rates=self.sample_rates,
            calibration_path=self.calibration_path,
        )
