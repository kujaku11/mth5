# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:57:54 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from mth5.mth5 import MTH5
from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.io.lemi import LEMICollection

# =============================================================================


class LEMI424Client(ClientBase):
    def __init__(
        self,
        data_path,
        save_path=None,
        mth5_filename="from_lemi424.h5",
        **kwargs
    ):
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=[1],
            mth5_filename=mth5_filename,
            **kwargs
        )

        self.collection = LEMICollection(self.data_path)

    def make_mth5_from_lemi424(self, survey_id, station_id, **kwargs):
        """
        create an MTH5 file from LEMI 424 long period data.

        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        self.collection.survey_id = survey_id
        self.collection.station_id = station_id

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, "w")
            survey_group = m.add_survey(self.collection.survey_id)

            for station_id in runs.keys():
                station_group = survey_group.stations_group.add_station(
                    station_id
                )
                for run_id, run_df in runs[station_id].items():
                    run_group = station_group.add_run(run_id)
                    run_ts = read_file(run_df.fn.to_list())
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            # update survey metadata from input station
            survey_group.update_metadata()

        return self.save_path
