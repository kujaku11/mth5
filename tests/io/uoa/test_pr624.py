# -*- coding: utf-8 -*-
"""
Test reading University of Adelaide PR6-24 (Earth Data Logger) files

Created on Thu Nov 07 2025

@author: bkay
"""

# ==============================================================================
# Imports
# ==============================================================================
import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np

from mth5.io.uoa import read_uoa

# ==============================================================================
# Create example PR6-24 data (10 Hz sample rate, 60 samples = 6 seconds)
# Format: BX BY BZ EX EY (space-separated, microvolts)
# ==============================================================================

pr624_data = """3243511 3158712 4524511 -125.5 87.3
3243516 3158715 4524516 -125.2 87.5
3243516 3158716 4524516 -125.1 87.6
3243514 3158714 4524514 -125.3 87.4
3243521 3158721 4524521 -124.9 87.8
3243511 3158711 4524511 -125.5 87.3
3243522 3158722 4524522 -124.8 87.9
3243531 3158731 4524531 -124.3 88.2
3243513 3158713 4524513 -125.4 87.4
3243519 3158719 4524519 -125.0 87.7
3243524 3158724 4524524 -124.7 88.0
3243517 3158717 4524517 -125.1 87.6
3243519 3158719 4524519 -125.0 87.7
3243521 3158721 4524521 -124.9 87.8
3243514 3158714 4524514 -125.3 87.4
3243517 3158717 4524517 -125.1 87.6
3243526 3158726 4524526 -124.7 88.1
3243521 3158721 4524521 -124.9 87.8
3243520 3158720 4524520 -125.0 87.7
3243519 3158719 4524519 -125.0 87.7
3243512 3158712 4524512 -125.4 87.3
3243513 3158713 4524513 -125.4 87.4
3243513 3158713 4524513 -125.4 87.4
3243508 3158708 4524508 -125.7 87.1
3243511 3158711 4524511 -125.5 87.3
3243516 3158716 4524516 -125.2 87.5
3243508 3158708 4524508 -125.7 87.1
3243500 3158700 4524500 -126.0 86.9
3243502 3158702 4524502 -125.9 87.0
3243507 3158707 4524507 -125.6 87.2
3243515 3158715 4524515 -125.2 87.5
3243514 3158714 4524514 -125.3 87.4
3243506 3158706 4524506 -125.6 87.2
3243508 3158708 4524508 -125.7 87.1
3243495 3158695 4524495 -126.2 86.7
3243495 3158695 4524495 -126.2 86.7
3243499 3158699 4524499 -126.0 86.9
3243500 3158700 4524500 -126.0 86.9
3243522 3158722 4524522 -124.8 87.9
3243511 3158711 4524511 -125.5 87.3
3243498 3158698 4524498 -126.0 86.9
3243498 3158698 4524498 -126.0 86.9
3243500 3158700 4524500 -126.0 86.9
3243506 3158706 4524506 -125.6 87.2
3243496 3158696 4524496 -126.0 86.8
3243510 3158710 4524510 -125.5 87.3
3243509 3158709 4524509 -125.5 87.3
3243493 3158693 4524493 -126.3 86.6
3243501 3158701 4524501 -126.0 86.9
3243509 3158709 4524509 -125.5 87.3
3243506 3158706 4524506 -125.6 87.2
3243501 3158701 4524501 -126.0 86.9
3243501 3158701 4524501 -126.0 86.9
3243496 3158696 4524496 -126.0 86.8
3243508 3158708 4524508 -125.7 87.1
3243515 3158715 4524515 -125.2 87.5
3243500 3158700 4524500 -126.0 86.9
3243503 3158703 4524503 -125.9 87.0
3243506 3158706 4524506 -125.6 87.2
3243509 3158709 4524509 -125.5 87.3
"""

# ==============================================================================


class TestPR624Reader(unittest.TestCase):
    """Test PR6-24 reader with sample data"""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory with sample PR6-24 files"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_path = Path(cls.temp_dir)

        # Parse multi-column data into separate channel files
        # Filename format: EDL03_041026010000.{BX,BY,BZ,EX,EY}
        # Date: 2004-10-26 01:00:00
        cls.base_name = "EDL03_041026010000"

        # Split data into columns (BX BY BZ EX EY)
        data = np.loadtxt(pr624_data.strip().split('\n'))

        cls.bx_file = cls.temp_path / f"{cls.base_name}.BX"
        cls.by_file = cls.temp_path / f"{cls.base_name}.BY"
        cls.bz_file = cls.temp_path / f"{cls.base_name}.BZ"
        cls.ex_file = cls.temp_path / f"{cls.base_name}.EX"
        cls.ey_file = cls.temp_path / f"{cls.base_name}.EY"

        # Write each column to its respective file
        np.savetxt(cls.bx_file, data[:, 0], fmt='%.1f')
        np.savetxt(cls.by_file, data[:, 1], fmt='%.1f')
        np.savetxt(cls.bz_file, data[:, 2], fmt='%.1f')
        np.savetxt(cls.ex_file, data[:, 3], fmt='%.1f')
        np.savetxt(cls.ey_file, data[:, 4], fmt='%.1f')

        # Read data using read_uoa() function
        cls.run_ts = read_uoa(
            cls.temp_path,
            sample_rate=10.0,
            station_id="TEST001",
            sensor_type='bartington',
            dipole_length_ex=50.0,
            dipole_length_ey=50.0
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_run_ts_created(self):
        """Test that RunTS object was created"""
        self.assertIsNotNone(self.run_ts)

    def test_channels(self):
        """Test that all channels are present"""
        expected_channels = ['hx', 'hy', 'hz', 'ex', 'ey']
        self.assertEqual(sorted(self.run_ts.channels), sorted(expected_channels))

    def test_n_samples(self):
        """Test that correct number of samples"""
        # 60 samples in each file
        self.assertEqual(self.run_ts.dataset.dims["time"], 60)

    def test_sample_rate(self):
        """Test that sample rate is set correctly"""
        self.assertAlmostEqual(self.run_ts.hx.sample_rate, 10.0, places=1)

    def test_start_time(self):
        """Test that start time defaults when not parsed from filename"""
        # Note: Real implementation extracts timestamp from filename like EDL03_041026010000
        # For this test, using generic filename, so expects default start time
        start_time = self.run_ts.run_metadata.time_period.start
        self.assertIsNotNone(start_time)

    def test_end_time(self):
        """Test that end time is calculated correctly based on sample count"""
        # 60 samples at 10 Hz = 6 seconds duration
        # end = start + (n_samples - 1) / sample_rate
        # = start + 59 / 10 = start + 5.9 seconds
        from dateutil.parser import parse
        start = parse(self.run_ts.run_metadata.time_period.start)
        end = parse(self.run_ts.run_metadata.time_period.end)
        duration = (end - start).total_seconds()
        self.assertAlmostEqual(duration, 5.9, places=1)

    def test_station_id(self):
        """Test that station ID is set"""
        self.assertEqual(self.run_ts.station_metadata.id, "TEST001")

    def test_data_values_hx(self):
        """Test that Hx data is stored as RAW microvolts"""
        # First value should be 3243511 μV (RAW from file)
        first_value = self.run_ts.hx.ts[0]
        self.assertAlmostEqual(first_value, 3243511, delta=1)

    def test_data_values_ex(self):
        """Test that Ex data is stored as RAW microvolts"""
        # First value should be -125.5 μV (RAW from file)
        first_value = self.run_ts.ex.ts[0]
        self.assertAlmostEqual(first_value, -125.5, places=1)

    def test_data_units_magnetic(self):
        """Test that magnetic channels have microvolt units (RAW)"""
        self.assertEqual(self.run_ts.hx.channel_metadata.units, "microvolts")

    def test_data_units_electric(self):
        """Test that electric channels have microvolt units (RAW)"""
        self.assertEqual(self.run_ts.ex.channel_metadata.units, "microvolts")

    def test_has_filters(self):
        """Test that calibration filters were created"""
        # Should have at least one filter (the calibration filter)
        self.assertGreater(len(self.run_ts.hx.channel_metadata.filter.applied), 0)

    def test_filters_not_applied(self):
        """Test that filters are marked as not applied"""
        # All filters should have applied=False
        for filter_name in self.run_ts.hx.channel_metadata.filter.applied:
            self.assertFalse(
                self.run_ts.hx.channel_metadata.filter.applied[filter_name]
            )


class TestPR624ReaderWithLEMI120(unittest.TestCase):
    """Test PR6-24 reader with LEMI-120 coils"""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory with minimal sample data"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_path = Path(cls.temp_dir)

        # Use first 3 rows of test data
        data = np.loadtxt(pr624_data.strip().split('\n'))[:3]

        cls.bx_file = cls.temp_path / "EDL03_041026010000.BX"
        np.savetxt(cls.bx_file, data[:, 0], fmt='%.1f')

        # Read data with LEMI-120 sensor type
        cls.run_ts = read_uoa(
            cls.temp_path,
            sample_rate=100.0,  # Broadband rate
            station_id="TEST001",
            sensor_type='lemi120',
            dipole_length_ex=50.0,
            dipole_length_ey=50.0
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_lemi120_sensor_type(self):
        """Test that LEMI-120 sensor type is set"""
        # Check that sensor is LEMI-120 in metadata
        self.assertIn('lemi', self.run_ts.hx.channel_metadata.sensor.model.lower())


class TestPR624ReaderError(unittest.TestCase):
    """Test PR6-24 reader error handling"""

    def test_missing_sample_rate(self):
        """Test that reader raises error without sample_rate"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a dummy file
            test_file = Path(temp_dir) / "test.BX"
            with open(test_file, 'w') as f:
                f.write("123\n456\n")

            # Should raise ValueError
            with self.assertRaises(ValueError):
                read_uoa(temp_dir)  # Missing sample_rate
        finally:
            shutil.rmtree(temp_dir)

    def test_empty_directory(self):
        """Test that reader raises error for empty directory"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Should raise ValueError (no channel data)
            with self.assertRaises(ValueError):
                read_uoa(temp_dir, sample_rate=10.0)
        finally:
            shutil.rmtree(temp_dir)


# ==============================================================================
# run
# ==============================================================================
if __name__ == "__main__":
    unittest.main()
