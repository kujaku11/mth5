# -*- coding: utf-8 -*-
"""
Phoenix file collection

Created on Thu Aug  4 16:48:47 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
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

        super().__init__(file_path=file_path, **kwargs)

        self.station_id = None
        self.survey_id = None
        self.channel_map = self._default_channel_map

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

    def to_dataframe(self, sample_rates=[150, 24000], run_name_zeros=4):
        """
        Get a data frame with columns of the specified
        :param sample_rates: DESCRIPTION, defaults to [150, 24000]
        :type sample_rates: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        receiver_metadata = self._read_receiver_metadata_json()
        if receiver_metadata is not None:
            self.station_id = receiver_metadata.station_metadata.id
            self.survey_id = receiver_metadata.survey_metadata.id
            self.channel_map = receiver_metadata.channel_map

        if not isinstance(sample_rates, (list, tuple)):
            sample_rates = [sample_rates]

        entries = []
        for sr in sample_rates:
            for fn in self._get_files(self._file_extension_map[int(sr)]):
                phx_obj = open_file(fn)
                if hasattr(phx_obj, "read_segment"):
                    segment = phx_obj.read_segment(metadata_only=True)
                    start = segment.segment_start_time.isoformat()
                    end = segment.segment_end_time.isoformat()
                    n_samples = segment.n_samples

                else:
                    start = phx_obj.segment_start_time.isoformat()
                    end = phx_obj.segment_end_time.isoformat()
                    n_samples = phx_obj.max_samples
                entry = {
                    "survey": self.survey_id,
                    "station": self.station_id,
                    "run": None,
                    "start": start,
                    "end": end,
                    "channel_id": phx_obj.channel_id,
                    "component": self.channel_map[phx_obj.channel_id],
                    "fn": fn,
                    "sample_rate": phx_obj.sample_rate,
                    "file_size": phx_obj.file_size,
                    "n_samples": n_samples,
                    "sequence_number": phx_obj.seq,
                    "instrument_id": phx_obj.recording_id,
                }
                entries.append(entry)

        df = pd.DataFrame(entries)
        df.start = pd.to_datetime(df.start)
        df.end = pd.to_datetime(df.end)

        # sort by start time
        df.sort_values(by=["start"], inplace=True)

        df = self.assign_run_names(df, zeros=run_name_zeros)

        return df

    def assign_run_names(self, df, zeros=4):
        """
        Assign run names by looping through start times

        :return: DESCRIPTION
        :rtype: TYPE

        """

        rdf = df.copy()
        sample_rates = rdf.sample_rate.unique()

        for sr in sample_rates:
            run_stem = self._file_extension_map[int(sr)].split("_")[-1]
            # continuous data
            if sr < 1000:
                rdf = rdf.sort_values("sequence_number")
                starts = rdf.loc[rdf.sample_rate == sr].start.unique()
                ends = rdf.loc[rdf.sample_rate == sr].end.unique()

                # find any breaks in the data
                diff = ends[0:-1] - starts[1:]

                breaks = np.where(diff != np.timedelta64(0))
                count = 1
                # this logic probably needs some work.
                if len(breaks[0]) > 0:
                    start_breaks = starts[breaks[0]]
                    for ii in range(len(start_breaks)):
                        count += 1
                        rdf[
                            (rdf.start == start_breaks[ii])
                            & (rdf.start < start_breaks[ii + 1])
                        ].loc[
                            rdf.sample_rate == sr, "run"
                        ] = f"sr{run_stem}_{count:0{zeros}}"

                else:
                    rdf.loc[
                        rdf.sample_rate == sr, "run"
                    ] = f"sr{run_stem}_{count:0{zeros}}"

            # segmented data
            else:

                starts = rdf.loc[rdf.sample_rate == sr].start.unique()
                for ii, s in enumerate(starts, 1):
                    rdf.loc[
                        rdf.start == s, "run"
                    ] = f"sr{run_stem}_{ii:0{zeros}}"

        return rdf


# =============================================================================
# test
# =============================================================================

pc = PhoenixCollection(
    r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\10291_2019-09-06-015630"
)
