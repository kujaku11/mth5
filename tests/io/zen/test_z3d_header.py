# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 21:00:28 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

import numpy as np

from mth5.io.zen import Z3DHeader


try:
    import mth5_test_data

    z3d_test_path = mth5_test_data.get_test_data_path("zen")
except ImportError:
    z3d_test_path = None


# =============================================================================


@unittest.skipIf(z3d_test_path is None, "local files")
class TestZ3DHeader(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = z3d_test_path / "bm100_20220517_131017_256_EY.Z3D"
        self.z3d_obj = Z3DHeader(fn=self.fn)
        self.z3d_obj.read_header()

    def test_ad_gain(self):
        self.assertEqual(getattr(self.z3d_obj, "ad_gain"), 1.0)

    def test_ad_rate(self):
        self.assertEqual(getattr(self.z3d_obj, "ad_rate"), 256.0)

    def test_alt(self):
        self.assertEqual(getattr(self.z3d_obj, "alt"), 1456.3)

    def test_attenchannelsmask(self):
        self.assertEqual(getattr(self.z3d_obj, "attenchannelsmask"), "0x80")

    def test_box_number(self):
        self.assertEqual(getattr(self.z3d_obj, "box_number"), 24.0)

    def test_box_serial(self):
        self.assertEqual(getattr(self.z3d_obj, "box_serial"), "0x0000010013A20040")

    def test_ch_factor(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_factor"), 9.536743164062e-10)

    def test_channel(self):
        self.assertEqual(getattr(self.z3d_obj, "channel"), 5.0)

    def test_channelgain(self):
        self.assertEqual(getattr(self.z3d_obj, "channelgain"), 1.0)

    def test_channelserial(self):
        self.assertEqual(getattr(self.z3d_obj, "channelserial"), "0xD474777C")

    def test_data_logger(self):
        self.assertEqual(getattr(self.z3d_obj, "data_logger"), "ZEN024")

    def test_duty(self):
        self.assertEqual(getattr(self.z3d_obj, "duty"), 32767.0)

    def test_dutynormalized(self):
        self.assertEqual(getattr(self.z3d_obj, "dutynormalized"), np.inf)

    def test_dutyoff(self):
        self.assertEqual(getattr(self.z3d_obj, "dutyoff"), 1.0)

    def test_fn(self):
        self.assertEqual(getattr(self.z3d_obj, "fn"), self.fn)

    def test_fpga_buildnum(self):
        self.assertEqual(getattr(self.z3d_obj, "fpga_buildnum"), 1125.0)

    def test_gpsweek(self):
        self.assertEqual(getattr(self.z3d_obj, "gpsweek"), 2210.0)

    def test_header_str(self):
        self.assertEqual(
            getattr(self.z3d_obj, "header_str"),
            b"\n\n\nGPS Brd339 Logfile\nVersion = 4147\nMain.hex Buildnum = 5357\nChannelSerial = 0xD474777C\nFpga Buildnum = 1125\nBox Serial = 0x0000010013A20040\nBox number = 24\nChannel = 5\nA/D Rate = 256\nA/D Gain =  1\nPeriod = 4294967295\nDuty = 32767\nDutyOff = 1\nDutyNormalized = inf\nLogTerminal = N\nTx.Freq = 0.000000\nTx.Duty = inf\nLat = 0.706816081\nLong = -2.038914402\nAlt = 1456.300\nNumSats = 17\nGpsWeek = 2210\nAttenChannelsMask = 0x80\nChannelGain = 1.0000\nCh.Factor = 9.536743164062e-10\n\x00                                    \r\n\x00",
        )

    def test_lat(self):
        self.assertEqual(getattr(self.z3d_obj, "lat"), 40.49757833327694)

    def test_logterminal(self):
        self.assertEqual(getattr(self.z3d_obj, "logterminal"), "N")

    def test_long(self):
        self.assertEqual(getattr(self.z3d_obj, "long"), -116.8211900230401)

    def test_main_hex_buildnum(self):
        self.assertEqual(getattr(self.z3d_obj, "main_hex_buildnum"), 5357.0)

    def test_numsats(self):
        self.assertEqual(getattr(self.z3d_obj, "numsats"), 17.0)

    def test_old_version(self):
        self.assertEqual(getattr(self.z3d_obj, "old_version"), False)

    def test_period(self):
        self.assertEqual(getattr(self.z3d_obj, "period"), 4294967295.0)

    def test_tx_duty(self):
        self.assertEqual(getattr(self.z3d_obj, "tx_duty"), np.inf)

    def test_tx_freq(self):
        self.assertEqual(getattr(self.z3d_obj, "tx_freq"), 0.0)

    def test_version(self):
        self.assertEqual(getattr(self.z3d_obj, "version"), 4147.0)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
