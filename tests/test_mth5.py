# -*- coding: utf-8 -*-
"""
Tests for MTh5

Created on Thu Jun 18 16:54:19 2020

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

import unittest
from pathlib import Path
import numpy as np

from mth5 import mth5
from mth5.standards import schema
from mth5.utils.exceptions import MTH5Error, MTH5TableError

fn_path = Path(__file__).parent
# =============================================================================
# 
# =============================================================================
mth5.helpers.close_open_files()

class TestMTH5(unittest.TestCase):
    def setUp(self):
        self.fn = fn_path.joinpath('test.mth5')
        self.mth5_obj = mth5.MTH5()
        self.mth5_obj.open_mth5(self.fn, mode='w')
        
    def test_initial_standards_group_size(self):
        stable = self.mth5_obj.standards_group.summary_table
        self.assertGreater(stable.nrows, 1)
        
    def test_initial_standards_keys(self):
        stable = self.mth5_obj.standards_group.summary_table
        standards_obj = schema.Standards()
        standards_dict = standards_obj.summarize_standards()
        standards_keys = sorted(list(standards_dict.keys()))
        
        stable_keys = sorted([ss.decode() for ss in 
                              list(stable.array['attribute'])])
        
        self.assertListEqual(standards_keys, stable_keys)
        
    def test_default_group_names(self):
        groups = sorted(self.mth5_obj.survey_group.groups_list)
        defaults = sorted(self.mth5_obj._default_subgroup_names)
        
        self.assertListEqual(defaults, groups)
        
    def test_summary_tables_exist(self):
        for group_name in self.mth5_obj._default_subgroup_names:
            group_name = group_name.lower()
            group = getattr(self.mth5_obj, f'{group_name}_group')
            self.assertIn('Summary', 
                          group.groups_list)
            
    def test_add_station(self):
        new_station = self.mth5_obj.add_station('MT001')
        self.assertIn('MT001', self.mth5_obj.stations_group.groups_list)
        self.assertIsInstance(new_station, mth5.groups.StationGroup)
        
    def test_remove_station(self):
        self.mth5_obj.add_station('MT001')
        self.mth5_obj.remove_station('MT001')
        self.assertNotIn('MT001', self.mth5_obj.stations_group.groups_list)
        
    def test_get_station_fail(self):
        self.assertRaises(MTH5Error, self.mth5_obj.get_station, 'MT002')
        
    def test_add_run(self):
        new_station = self.mth5_obj.add_station('MT001')
        new_run = new_station.add_run('MT001a')
        self.assertIn('MT001a', new_station.groups_list)
        self.assertIsInstance(new_run, mth5.groups.RunGroup)
        
    def test_remove_run(self):
        new_station = self.mth5_obj.add_station('MT001')
        new_station.add_run('MT001a')
        new_station.remove_run('MT001a')
        self.assertNotIn('MT001a', new_station.groups_list)
        
    def test_get_run_fail(self):
        self.assertRaises(MTH5Error, self.mth5_obj.get_run, 
                          'MT001', 'MT002a')
        
    def test_add_channel(self):
        new_station = self.mth5_obj.add_station('MT001')
        new_run = new_station.add_run('MT001a')
        new_channel = new_run.add_channel('Ex', 'electric', None)
        self.assertIn('Ex', new_run.groups_list)
        self.assertIsInstance(new_channel, mth5.groups.ElectricDataset)
        
        self.assertIn('Ex', 
                      (new_run.summary_table.array['component']
                       .astype(np.unicode_).tolist()))
        
        
    def test_remove_channel(self):
        new_station = self.mth5_obj.add_station('MT001')
        new_run = new_station.add_run('MT001a')
        new_channel = new_run.add_channel('Ex', 'electric', None)
        new_run.remove_channel('Ex')
        self.assertNotIn('Ex', new_run.groups_list)
        
    def test_get_channel_fail(self):
        new_station = self.mth5_obj.add_station('MT001')
        new_station.add_run('MT001a')
        self.assertRaises(MTH5Error, self.mth5_obj.get_channel, 
                          'MT001', 'MT001a', 'Ey')

    def tearDown(self):
        self.mth5_obj.close_mth5()
    

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
    