# -*- coding: utf-8 -*-
"""
Phoenix file collection

Created on Thu Aug  4 16:48:47 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import pandas as pd
from mth5.io.phoenix import open_file, ReceiverMetadataJSON
from mth5.io import Collection

# =============================================================================


class PhoenixCollection(Collection):
    """
    A class to collect the various files in a Phoenix file system and try
    to organize them into runs.
    """

    def __init__(self, file_path=None, **kwargs):

        super().__init__(file_path=file_path, **kwargs)

        self._file_extension_map = {
            30: "td_30",
            150: "td_150",
            2400: "td_2400",
            24000: "td_24k",
            96000: "td_96k",
        }

        self._default_channel_map = {
            0: "E1",
            1: "H3",
            2: "H2",
            3: "H1",
            4: "H4",
            5: "H5",
            6: "H6",
            7: "E2",
        }

        self._receiver_metadata_name = "recmeta.json"

    def _read_receiver_metadata_json(self):
        """
        read in metadata information from receiver metadata file

        :return: DESCRIPTION
        :rtype: TYPE

        """

        rec_fn = self.file_path.joinpath(self._receiver_metadata_name)
        if rec_fn.is_file():
            return ReceiverMetadataJSON(fn=rec_fn)
        else:
            self.logger.warning(
                f"Could not fine {self._receiver_metadata_name} in {self.file_path}"
            )
            return None

    def to_dataframe(self, sample_rates=[150, 24000]):
        """
        Get a data frame with columns of the specified
        :param sample_rates: DESCRIPTION, defaults to [150, 24000]
        :type sample_rates: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        receiver_metadata = self._read_receiver_metadata_json()
        if receiver_metadata is not None:
            station_id = receiver_metadata.station_metadata.id
            survey_id = receiver_metadata.survey_metadata.id
            channel_map = receiver_metadata.channel_map
        else:
            station_id = ""
            survey_id = ""
            channel_map = self._default_channel_map

        if not isinstance(sample_rates, (list, tuple)):
            sample_rates = [sample_rates]

        entries = []
        for sr in sample_rates:
            for fn in self._get_files(self._file_extension_map[int(sr)]):
                phx_obj = open_file(fn)
                entry = {
                    "survey": survey_id,
                    "station": station_id,
                    "run": None,
                    "start": phx_obj.segment_start_time.isoformat(),
                    "end": phx_obj.segment_end_time.isoformat(),
                    "channel": channel_map[phx_obj.channel_id],
                    "fn": fn,
                    "sample_rate": phx_obj.sample_rate,
                    "file_size": phx_obj.file_size,
                    "n_samples": phx_obj.max_samples,
                    "sequence_number": phx_obj.seq,
                }
                entries.append(entry)

        df = pd.DataFrame(entries)
        df.start = pd.to_datetime(df.start)
        df.end = pd.to_datetime(df.end)

        # sort by start time
        df.sort_values(by=["start"], inplace=True)

        return df


# =============================================================================
# test
# =============================================================================

pc = PhoenixCollection(
    r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\10291_2019-09-06-015630"
)
