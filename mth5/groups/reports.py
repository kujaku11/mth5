# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:03:53 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import h5py

from mth5.groups.base import BaseGroup
# =============================================================================
# Reports Group
# =============================================================================
class ReportsGroup(BaseGroup):
    """
    Not sure how to handle this yet

    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

        # summary of reports
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (1000,),
            "dtype": np.dtype(
                [
                    ("name", "S5"),
                    ("type", "S32"),
                    ("summary", "S200"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        }

    def add_report(self, report_name, report_metadata=None, report_data=None):
        """

        :param report_name: DESCRIPTION
        :type report_name: TYPE
        :param report_metadata: DESCRIPTION, defaults to None
        :type report_metadata: TYPE, optional
        :param report_data: DESCRIPTION, defaults to None
        :type report_data: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.logger.error("Not Implemented yet")
        