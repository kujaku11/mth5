# -*- coding: utf-8 -*-

# =============================================================================
# Imports
# =============================================================================
import unittest

import numpy as np
from mth5.timeseries import ChannelTS, RunTS
from mt_metadata.timeseries import Electric, Magnetic, Auxiliary, Run, Station

# =============================================================================


class TestMakeRunTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.maxDiff = None
        channel_list = []
        self.common_start = "2020-01-01T00:00:00+00:00"
        self.sample_rate = 1.0
        self.n_samples = 4096
        t = np.arange(self.n_samples)
        data = np.sum(
            [
                np.cos(2 * np.pi * w * t + phi)
                for w, phi in zip(np.logspace(-3, 3, 20), np.random.rand(20))
            ],
            axis=0,
        )

        self.station_metadata = Station(id="mt001")
        self.run_metadata = Run(id="001")

        for component in ["hx", "hy", "hz"]:
            h_metadata = Magnetic(component=component)
            h_metadata.time_period.start = self.common_start
            h_metadata.sample_rate = self.sample_rate
            h_channel = ChannelTS(
                channel_type="magnetic",
                data=data,
                channel_metadata=h_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
            channel_list.append(h_channel)

        for component in ["ex", "ey"]:
            e_metadata = Electric(component=component)
            e_metadata.time_period.start = self.common_start
            e_metadata.sample_rate = self.sample_rate
            e_channel = ChannelTS(
                channel_type="electric",
                data=data,
                channel_metadata=e_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
            channel_list.append(e_channel)

        aux_metadata = Auxiliary(component="temperature")
        aux_metadata.time_period.start = self.common_start
        aux_metadata.sample_rate = self.sample_rate
        aux_channel = ChannelTS(
            channel_type="auxiliary",
            data=np.random.rand(self.n_samples) * 30,
            channel_metadata=aux_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
        )
        channel_list.append(aux_channel)

        self.run_ts = RunTS(channel_list)

    def test_station_metadata(self):
        with self.subTest("station id"):
            self.assertEqual(
                self.run_ts.station_metadata.id, self.station_metadata.id
            )
        with self.subTest("start"):
            self.assertEqual(
                self.run_ts.station_metadata.time_period.start,
                self.common_start,
            )
        with self.subTest("run list"):
            self.assertListEqual(
                self.run_ts.station_metadata.run_list,
                [self.run_metadata.id],
            )

    def test_run_metadata(self):
        with self.subTest("run id"):
            self.assertEqual(self.run_ts.run_metadata.id, self.run_metadata.id)
        with self.subTest("start"):
            self.assertEqual(
                self.run_ts.run_metadata.time_period.start,
                self.common_start,
            )
        with self.subTest("channels"):
            self.assertListEqual(
                self.run_ts.run_metadata.channels_recorded_all,
                ["hx", "hy", "hz", "ex", "ey", "temperature"],
            )

    def test_channels(self):
        for comp in self.run_ts.channels:
            ch = getattr(self.run_ts, comp)
            with self.subTest("start"):
                self.assertEqual(
                    ch.channel_metadata.time_period.start,
                    self.common_start,
                )
            with self.subTest("sample rate"):
                self.assertEqual(
                    ch.sample_rate,
                    self.sample_rate,
                )
            with self.subTest("n samples"):
                self.assertEqual(
                    ch.n_samples,
                    4096,
                )


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
