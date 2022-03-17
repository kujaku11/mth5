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
from collections import OrderedDict

from mth5.mth5 import MTH5

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
        
        self.tf_group = self.mth5_obj.add_transfer_function(self.tf_obj)
        self.tf_h5 = self.mth5_obj.get_transfer_function(
            self.tf_obj.station, self.tf_obj.station)
        


    def test_station_metadta(self):
        
        meta_dict = OrderedDict([('acquired_by.author', 'National Geoelectromagnetic Facility'),
                     ('channels_recorded', ['ex', 'ey', 'hx', 'hy', 'hz']),
                     ('data_type', 'mt'),
                     ('fdsn.id', 'USMTArray.NMX20.2020'),
                     ('geographic_name', 'Nations Draw, NM, USA'),
                     ('id', 'NMX20'),
                     ('location.datum', 'WGS84'),
                     ('location.declination.epoch', '2020.0'),
                     ('location.declination.model', 'WMM'),
                     ('location.declination.value', 9.0899999999999999),
                     ('location.elevation', 1940.05),
                     ('location.latitude', 34.470528),
                     ('location.longitude', -108.712288),
                     ('orientation.angle_to_geographic_north', 0.0),
                     ('orientation.method', None),
                     ('orientation.reference_frame', 'geographic'),
                     ('provenance.creation_time', '2021-03-17T14:47:44+00:00'),
                     ('provenance.software.author', 'none'),
                     ('provenance.software.name',
                      'EMTF File Conversion Utilities 4.0'),
                     ('provenance.software.version', None),
                     ('provenance.submitter.author', 'Anna Kelbert'),
                     ('provenance.submitter.email', 'akelbert@usgs.gov'),
                     ('provenance.submitter.organization',
                      'U.S. Geological Survey, Geomagnetism Program'),
                     ('run_list', ['NMX20a', 'NMX20b']),
                     ('time_period.end', '2020-10-07T20:28:00+00:00'),
                     ('time_period.start', '2020-09-20T19:03:06+00:00'),
                     ('transfer_function.coordinate_system', 'geopgraphic'),
                     ('transfer_function.processing_parameters', ['{type: None}']),
                     ('transfer_function.remote_references',
                      ['NMX20b',
                       'NMX20',
                       'NMW20',
                       'COR21',
                       'NMY21-NMX20b',
                       'NMX20',
                       'UTS18']),
                     ('transfer_function.runs_processed', ['NMX20a','NMX20b']),
                     ('transfer_function.sign_convention', 'exp(+ i\\omega t)')])
        
        self.assertDictEqual(meta_dict, self.tf_h5.station_metadata.to_dict(single=True))
        
    def tearDown(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()
        
        
