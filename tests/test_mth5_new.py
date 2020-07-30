# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:49:44 2020

@author: jpeacock
"""

import unittest
from mth5 import mth5
from pathlib import Path

class TestMTH5(unittest.TestCase):
    
    def setUp(self):
        fn = Path(r'c:\Users\jpeacock\Documents\GitHub\MTarchive\tests\test.mth5')
        self.base = mth5.MTH5()
        self.base.open_mth5(fn, mode='w')
        
    def test_initialized_groups(self):
        master_key = self.base._default_master_group
        initial_groups = list(sorted(self.base.mth5_obj.keys()))
        self.assertListEqual(initial_groups, list(master_key))
        
        initial_groups = list(sorted(self.base.mth5_obj[master_key].keys()))
        self.assertListEqual(initial_groups, self.base._default_subgroups)
        
    def tearDown(self):
        self.base.close_mth5()
        
# =============================================================================
# Run 
# =============================================================================
if __name__ == '__main__':
    unittest.main()
    