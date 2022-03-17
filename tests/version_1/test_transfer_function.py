# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:24:28 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5 import MTH5

from mt_metadata.transfer_functions.tf import StatisticalEstimate
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML


fn_path = Path(__file__).parent
# =============================================================================


class TestTFGroup(unittest.TestCase):
    def setUp(self):
        
        self.maxDiff = None
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        
        self.st_metadata = StatisticalEstimate()
        self.st_metadata.name = "impedance"
        self.st_metadata.input_channels = ["hx", "hy"]
        self.st_metadata.output_channels = ["ex", "ey"]
        
        self.tf_obj = TF(TF_XML)
        
#     def test_add_estimate(self):
        
#         z = self.tf_group.add_statistical_estimate(
#             self.st_metadata, self.tf_obj.impedance.to_numpy())
        
        
        
        
        

