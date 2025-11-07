# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:00:04 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.io.zen import zen_tools


# =============================================================================


class TestSplitStation(unittest.TestCase):
    def setUp(self):
        self.station = "cl01"
        self.name = "cl"
        self.number = "01"

    def test_split_station(self):
        name, number = zen_tools.split_station(self.station)

        with self.subTest("name"):
            self.assertEqual(name, self.name)

        with self.subTest("number"):
            self.assertEqual(number, self.number)


class TestZenSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.zs = zen_tools.ZenSchedule()

    def test_df_list(self):
        self.assertTupleEqual((4096, 256), self.zs.df_list)

    def test_df_time_list(self):
        self.assertTupleEqual(("00:10:00", "07:50:00"), self.zs.df_time_list)

    def test_resync_pause(self):
        self.assertEqual(20, self.zs._resync_pause)

    def test_add_time_day(self):
        self.assertEqual(
            "2020-01-02T00:00:00",
            self.zs.add_time("2020-01-01,00:00:00", add_days=1).isoformat(),
        )

    def test_add_time_hour(self):
        self.assertEqual(
            "2020-01-01T01:00:00",
            self.zs.add_time("2020-01-01,00:00:00", add_hours=1).isoformat(),
        )

    def test_add_time_minute(self):
        self.assertEqual(
            "2020-01-01T00:30:00",
            self.zs.add_time("2020-01-01,00:00:00", add_minutes=30).isoformat(),
        )

    def test_add_time_seconds(self):
        self.assertEqual(
            "2020-01-01T00:00:30",
            self.zs.add_time("2020-01-01,00:00:00", add_seconds=30).isoformat(),
        )

    def test_make_schedule(self):
        final = [
            {
                "dt": "2000-01-01,00:00:00",
                "df": 4096,
                "date": "2000-01-01",
                "time": "00:00:00",
                "sr": "4",
            },
            {
                "dt": "2000-01-01,00:10:00",
                "df": 256,
                "date": "2000-01-01",
                "time": "00:10:00",
                "sr": "0",
            },
            {
                "dt": "2000-01-01,08:00:00",
                "df": 4096,
                "date": "2000-01-01",
                "time": "08:00:00",
                "sr": "4",
            },
            {
                "dt": "2000-01-01,08:10:00",
                "df": 256,
                "date": "2000-01-01",
                "time": "08:10:00",
                "sr": "0",
            },
            {
                "dt": "2000-01-01,16:00:00",
                "df": 4096,
                "date": "2000-01-01",
                "time": "16:00:00",
                "sr": "4",
            },
        ]

        self.assertListEqual(
            final,
            self.zs.make_schedule(self.zs.df_list, self.zs.df_time_list, repeat=2),
        )

    def test_convert_time_to_seconds(self):
        self.assertEqual(43200.0, self.zs._convert_time_to_seconds("12:00:00"))


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
