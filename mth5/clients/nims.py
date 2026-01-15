#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.io.nims import NIMSCollection
from mth5.mth5 import MTH5


# =============================================================================


class NIMSClient(ClientBase):
    def __init__(
        self,
        data_path,
        sample_rates=[1, 8],
        save_path=None,
        calibration_path=None,
        mth5_filename="from_nims.h5",
        **kwargs,
    ):
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates,
            mth5_filename=mth5_filename,
            **kwargs,
        )

        self.calibration_path = calibration_path
        self.collection = NIMSCollection(self.data_path)

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
            self._calibration_path = None
            # raise ValueError("calibration_path cannot be None")

    def get_run_dict(self):
        """
        Get Run information

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(
            sample_rates=self.sample_rates,
            calibration_path=self.calibration_path,
        )

    def get_survey(self, station_dict):
        """
        get survey name from a dictionary of a single station of runs
        :param station_dict: DESCRIPTION
        :type station_dict: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return list(
            set([station_dict[k].survey.unique()[0] for k in station_dict.keys()])
        )[0]

    def make_mth5_from_nims(self, survey_id="default_survey", combine=True, **kwargs):
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

        if survey_id is None:
            self.collection.survey_id = "none"
        else:
            self.collection.survey_id = survey_id

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, "w")
            survey_group = m.add_survey(self.collection.survey_id)

            for station_id in runs.keys():
                station_group = survey_group.stations_group.add_station(station_id)
                for run_id, run_df in runs[station_id].items():
                    run_group = station_group.add_run(run_id)
                    # use iloc to avoid KeyError if the DataFrame index does not start at 0
                    run_ts = read_file(run_df["fn"].iloc[0])
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            # update survey metadata from input station
            survey_group.update_metadata()

        return self.save_path
