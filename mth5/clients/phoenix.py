#!/usr/bin/env python
# coding: utf-8

# # Make an MTH5 from Phoenix Data
#
# This example demonstrates how to read Phoenix data into an MTH5 file.  The data comes from example data in [PhoenixGeoPy](https://github.com/torresolmx/PhoenixGeoPy). Here I downloaded those data into a local folder on my computer by forking the main branch.

# ## Imports

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5.mth5 import MTH5
from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.io.phoenix import PhoenixCollection
from mth5.io.phoenix.readers.calibrations import PhoenixCalibration

# =============================================================================


class PhoenixClient(ClientBase):
    def __init__(
        self,
        data_path,
        sample_rates=[150, 24000],
        save_path=None,
        receiver_calibration_dict={},
        sensor_calibration_dict={},
        mth5_filename="from_phoenix.h5",
        **kwargs,
    ):
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates,
            mth5_filename=mth5_filename,
            **kwargs,
        )

        self.receiver_calibration_dict = receiver_calibration_dict
        self.sensor_calibration_dict = sensor_calibration_dict

        self.collection = PhoenixCollection(self.data_path)

    @property
    def receiver_calibration_dict(self):
        """receiver calibrations"""
        return self._receiver_calibration_dict

    @receiver_calibration_dict.setter
    def receiver_calibration_dict(self, value):
        if isinstance(value, dict):
            self._receiver_calibration_dict = value

        elif isinstance(value, (str, Path)):
            receiver_path = Path(value)
            if receiver_path.is_dir():
                self._receiver_calibration_dict = {}
                for fn in list(receiver_path.glob("*.rxcal.json")) + list(
                    receiver_path.glob("*.rx_cal.json")
                ):
                    self._receiver_calibration_dict[fn.stem.split("_")[0]] = fn
            elif receiver_path.is_file():
                self._receiver_calibration_dict = {}
                self._receiver_calibration_dict[fn.stem.split("_")[0]] = (
                    receiver_path
                )
        else:
            raise TypeError(f"type {type(value)} not supported.")

    @property
    def sensor_calibration_dict(self):
        """Path to calibration data"""
        return self._sensor_calibration_dict

    @sensor_calibration_dict.setter
    def sensor_calibration_dict(self, value):
        """

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(value, dict):
            self._sensor_calibration_dict = value
        elif isinstance(value, (str, Path)):
            self._sensor_calibration_dict = {}
            cal_path = Path(value)
            if cal_path.is_dir():
                for fn in cal_path.glob("*scal.json"):
                    self._sensor_calibration_dict[fn.stem.split("_")[0]] = (
                        PhoenixCalibration(fn)
                    )
            self._calibration_path = Path(value)
            if not self._calibration_path.exists():
                raise IOError(f"Could not find {self._calibration_path}")

        else:
            raise ValueError("calibration_path cannot be None")

    def make_mth5_from_phoenix(self, **kwargs):
        """
        Make an MTH5 from Phoenix files.  Split into runs, account for filters

        :param data_path: DESCRIPTION, defaults to None
        :type data_path: TYPE, optional
        :param sample_rates: DESCRIPTION, defaults to None
        :type sample_rates: TYPE, optional
        :param save_path: DESCRIPTION, defaults to None
        :type save_path: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        run_dict = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:

            m.open_mth5(self.save_path, "w")

            for station_id, station_dict in run_dict.items():
                collection_metadata = self.collection.metadata_dict[station_id]
                survey_metadata = collection_metadata.survey_metadata
                survey_group = m.add_survey(survey_metadata.id)

                station_metadata = self.collection.metadata_dict[
                    station_id
                ].station_metadata
                station_group = survey_group.stations_group.add_station(
                    station_metadata.id, station_metadata=station_metadata
                )
                for run_id, run_df in station_dict.items():
                    run_metadata = collection_metadata.run_metadata
                    run_metadata.id = run_id
                    run_metadata.sample_rate = float(
                        run_df.sample_rate.unique()[0]
                    )

                    run_group = station_group.add_run(
                        run_metadata.id, run_metadata=run_metadata
                    )
                    for row in run_df.itertuples():
                        try:
                            ch_ts = read_file(
                                row.fn,
                                **{
                                    "channel_map": collection_metadata.channel_map,
                                    "rxcal_fn": self.receiver_calibration_dict[
                                        collection_metadata.instrument_id
                                    ],
                                },
                            )
                        except OSError:
                            print(
                                f"OSError: skipping {row.fn.name} likely too small"
                            )
                            continue

                        if ch_ts.component in ["h1", "h2", "h3"]:

                            # for phx coils from generic response curves
                            if (
                                ch_ts.channel_metadata.sensor.id
                                in self.sensor_calibration_dict.keys()
                            ):
                                pc = self.sensor_calibration_dict[
                                    ch_ts.channel_metadata.sensor.id
                                ]
                                for key in pc.__dict__.keys():
                                    if key.startswith("h"):
                                        break
                                coil_fap = getattr(pc, key)

                            # add filter
                            ch_ts.channel_metadata.filter.name.append(
                                coil_fap.name
                            )
                            ch_ts.channel_metadata.filter.applied.append(True)
                            ch_ts.channel_response.filters_list.append(coil_fap)

                        # add channel to the run group
                        run_group.from_channel_ts(ch_ts)

                    run_group.update_metadata()

            station_group.update_metadata()
            survey_group.update_metadata()

        return self.save_path
