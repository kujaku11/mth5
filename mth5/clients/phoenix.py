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
from mth5.io.phoenix import ReceiverMetadataJSON, PhoenixCollection

# =============================================================================


class PhoenixClient:
    def __init__(
        self,
        data_path,
        sample_rates=[130, 24000],
        save_path=None,
        calibration_path=None,
    ):
        self.data_path = data_path
        self.sample_rates = sample_rates
        self.save_path = save_path
        self.mth5_filename = "from_phoenix.h5"
        self.calibration_path = calibration_path

        self.collection = PhoenixCollection(self.data_path)

    @property
    def data_path(self):
        """Path to phoenix data"""
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        """

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if value is not None:
            self._data_path = Path(value)
            if not self._data_path.exists():
                raise IOError(f"Could not find {self._data_path}")

            self.collection = PhoenixCollection(self.data_path)

        else:
            raise ValueError("data_path cannot be None")

    @property
    def calibration_path(self):
        """Path to calibration data"""
        return self._calibration_path

    @calibration_path.setter
    def calibration_path(self, value):
        """

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if value is not None:
            self._calibration_path = Path(value)
            if not self._calibration_path.exists():
                raise IOError(f"Could not find {self._calibration_path}")

        else:
            raise ValueError("calibration_path cannot be None")

    @property
    def sample_rates(self):
        """sample rates to look for"""
        return self._sample_rates

    @sample_rates.setter
    def sample_rates(self, value):
        """
        sample rates set to a list

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(value, (int, float)):
            self._value = [value]
        elif isinstance(value, str):
            self._value = [float(v) for v in value.split(",")]

        elif isinstance(value, (tuple, list)):
            self._value = [float(v) for v in value]
        else:
            raise TypeError(f"Cannot parse {type(value)}")

    @property
    def save_path(self):
        """Path to save mth5"""
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        """

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if value is not None:
            self._save_path = Path(value)
            if self._save_path.is_dir():
                self._save_path = self._save_path.joinpath(self.mth5_filename)

        else:
            self._save_path = self.data_path.joinpath(self.mth5_filename)

    def get_run_dict(self):
        """
        Get Run information

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(sample_rates=self.sample_rates)

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

        with MTH5() as m:
            m.open_mth5(self.save_path, "w")

            for station_id, station_dict in run_dict.items():
                survey_metadata = self.collection.metadata_dict[
                    station_id
                ].survey_metadata
                survey_group = m.add_survey(survey_metadata.id)

                station_metadata = self.collection.metadata_dict[
                    station_id
                ].station_metadata
                station_group = survey_group.stations_group.add_station(
                    station_metadata.id, station_metadata=station_metadata
                )
                for run_id, run_df in station_dict.items():
                    run_metadata = self.collection.metadata_dict[
                        station_id
                    ].run_metadata
                    run_metadata.id = run_id
                    run_metadata.sample_rate = float(
                        run_df.sample_rate.unique()[0]
                    )

                    run_group = station_group.add_run(
                        run_metadata.id, run_metadata=run_metadata
                    )
                    for row in run_df.itertuples():
                        ch_ts = read_file(row.fn)

                        # add channel to the run group
                        ch_dataset = run_group.from_channel_ts(ch_ts)

                    run_group.update_run_metadata()

            station_group.update_station_metadata()
            station_group.write_metadata()

            survey_group.update_survey_metadata()
            survey_group.write_metadata()
