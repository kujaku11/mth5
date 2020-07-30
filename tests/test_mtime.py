# -*- coding: utf-8 -*-
"""
time test
Created on Thu May 21 14:09:17 2020

@author: jpeacock
"""

import unittest
from mth5.utils.mttime import MTime
from mth5.utils.exceptions import MTTimeError

# =============================================================================
# tests
# =============================================================================
class TestMTime(unittest.TestCase):
    def setUp(self):
        self.date_str_01 = '2020-01-02'
        self.date_str_01 = '01-02-20'
        self.year = 2020
        self.month = 1
        self.day = 2
        self.hour = 12
        self.minute = 15
        self.second = 20
        self.ms = 123400
        self.dt_str_01 = '2020-01-02 12:15:20.1234' 
        self.dt_true = '2020-01-02T12:15:20.123400+00:00'
        
        self.epoch_seconds = 1577967320.1234
        self.input_fail = '01294055'
        self.mtime_obj = MTime()
    
    def test_string_input_date(self):
        self.mtime_obj.from_str(self.date_str_01)
        self.assertEqual(self.year, self.mtime_obj.year)
        self.assertEqual(self.month, self.mtime_obj.month)
        self.assertEqual(self.day, self.mtime_obj.day)
        
    def test_string_input_dt(self):
        self.mtime_obj.from_str(self.dt_str_01)
        self.assertEqual(self.year, self.mtime_obj.year)
        self.assertEqual(self.month, self.mtime_obj.month)
        self.assertEqual(self.day, self.mtime_obj.day)
        self.assertEqual(self.hour, self.mtime_obj.hour)
        self.assertEqual(self.minute, self.mtime_obj.minutes)
        self.assertEqual(self.second, self.mtime_obj.seconds)
        self.assertEqual(self.ms, self.mtime_obj.microseconds)
        self.assertEqual(self.dt_true, self.mtime_obj.iso_str)
        self.assertAlmostEqual(self.epoch_seconds, 
                               self.mtime_obj.epoch_seconds,
                               places=4)
        
    def test_input_fail(self):
        self.assertRaises(MTTimeError,
                          self.mtime_obj.from_str, 
                          self.input_fail)
        
    def test_compare_dt(self):
        dt_01 = MTime()
        dt_02 = MTime()
        
        self.assertTrue(dt_01 == dt_02)
        self.assertTrue(dt_01 == dt_02.iso_str)
        self.assertTrue(dt_01 == dt_02.epoch_seconds)
        self.assertTrue(dt_01 >= dt_02)
        self.assertTrue(dt_01 <= dt_02)
        
        
# =============================================================================
# Run
# =============================================================================
if __name__ == '__main__':
    unittest.main()