# -*- coding: utf-8 -*-
"""
Phoenix file collection

Created on Thu Aug  4 16:48:47 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from collections import OrderedDict
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
        self.receiver_metadata = None

        self._receiver_metadata_name = "recmeta.json"

    def _read_receiver_metadata_json(self):
        """
        read in metadata information from receiver metadata file into
        an `ReceiverMetadataJSON` object.

        :return: Receiver metadata
        :rtype: :class:`ReceiverMetadataJSON`

        """

        rec_fn = self.file_path.joinpath(self._receiver_metadata_name)
        if rec_fn.is_file():
            return ReceiverMetadataJSON(fn=rec_fn)
        else:
            self.logger.warning(
                f"Could not fine {self._receiver_metadata_name} in {self.file_path}"
            )
            return None

    def to_dataframe(
        self,
        sample_rates=[150, 24000],
        run_name_zeros=4,
        calibration_path=None,
    ):
        """
        Get a dataframe of all the files in a given directory with given
        columns.

        :param sample_rates: list of sample rates to read, defaults to [150, 24000]
        :type sample_rates: list of integers, optional
        :param run_name_zeros: Number of zeros in the run name, defaults to 4
        :type run_name_zeros: integer, optional
        :return: Dataframe with each row representing a single file
        :rtype: :class:`pandas.DataFrame`

        """

        self.receiver_metadata = self._read_receiver_metadata_json()
        if self.receiver_metadata is not None:
            self.station_id = self.receiver_metadata.station_metadata.id
            self.survey_id = self.receiver_metadata.survey_metadata.id
            self.channel_map = self.receiver_metadata.channel_map

        if not isinstance(sample_rates, (list, tuple)):
            sample_rates = [sample_rates]

        entries = []
        for sr in sample_rates:
            for fn in self.get_files(self._file_extension_map[int(sr)]):
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
                    "calibration_fn": None,
                }
                entries.append(entry)

        df = self._sort_df(
            self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
        )

        return df

    def assign_run_names(self, df, zeros=4):
        """
        Assign run names by looping through start times.

        For continous data a single run is assigned as long as the start and
        end times of each file align.  If there is a break a new run name is
        assigned.

        For segmented data a new run name is assigned to each segment

        :param df: Dataframe returned by `to_dataframe` method
        :type df: :class:`pandas.DataFrame`
        :param zeros: Number of zeros in the run name, defaults to 4
        :type zeros: integer, optional
        :return: Dataframe with run names
        :rtype: :class:`pandas.DataFrame`

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

    def get_runs(
        self,
        sample_rates,
        run_name_zeros=4,
        calibration_path=None,
    ):
        """
        Get a list of runs contained within the given folder.  First the
        dataframe will be developed from which the runs are extracted.

        For continous data all you need is the first file in the sequence. The
        reader will read in the entire sequence.

        For segmented data it will only read in the given segment, which is
        slightly different from the original reader.

        :param sample_rates: list of sample rates to read, defaults to [150, 24000]
        :param run_name_zeros: Number of zeros in the run name, defaults to 4
        :type run_name_zeros: integer, optional
        :return: List of run dataframes with only the first block of files
        :rtype: OrderedDict

        :Example:

            >>> from mth5.io.phoenix import PhoenixCollection
            >>> phx_collection = PhoenixCollection(r"/path/to/station")
            >>> run_dict = phx_collection.get_runs(sample_rates=[150, 24000])

        """

        df = self.to_dataframe(
            sample_rates=sample_rates,
            run_name_zeros=run_name_zeros,
            calibration_path=calibration_path,
        )

        run_dict = OrderedDict()

        for station in sorted(df.station.unique()):
            run_dict[station] = OrderedDict()

            for run_id in sorted(
                df[df.station == station].run.unique(),
                key=lambda x: x[-run_name_zeros:],
            ):
                run_df = df[(df.station == station) & (df.run == run_id)]
                run_dict[station][run_id] = run_df[
                    run_df.start == run_df.start.min()
                ]

        return run_dict
