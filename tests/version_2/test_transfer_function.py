# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:24:28 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

import h5py

from mt_metadata.transfer_functions.tf import StatisticalEstimate
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML

from mth5.groups.transfer_function import TransferFunctionGroup

# =============================================================================


class TestTFGroup(unittest.TestCase):
    def setUp(self):

        self.hdf5 = h5py.File("tf_test.h5", "w")
        tf_group = self.hdf5.create_group("tf")
        self.tf_group = TransferFunctionGroup(tf_group)

        self.st_metadata = StatisticalEstimate()
        self.st_metadata.name = "impedance"
        self.st_metadata.input_channels = ["hx", "hy"]
        self.st_metadata.output_channels = ["ex", "ey"]

        self.tf_obj = TF(TF_XML)


#     def test_add_estimate(self):

#         z = self.tf_group.add_statistical_estimate(
#             self.st_metadata, self.tf_obj.impedance.to_numpy())
