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
from collections import OrderedDict
import numpy as np
import pandas as pd

from mth5.io.phoenix import open_phoenix, PhoenixReceiverMetadata
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

        self.metadata_dict = {}

        self._receiver_metadata_name = "recmeta.json"

    def _read_receiver_metadata_json(self, rec_fn):
        """
        read in metadata information from receiver metadata file into
        an `ReceiverMetadataJSON` object.

        :return: Receiver metadata
        :rtype: :class:`ReceiverMetadataJSON`

        """

        if Path(rec_fn).is_file():
            return PhoenixReceiverMetadata(fn=rec_fn)
        else:
            self.logger.warning(
                f"Could not fine {self._receiver_metadata_name} in {self.file_path}"
            )
            return None

    def _locate_station_folders(self):
        """
        Locate the station folder, the one that has the recmeta.json in it

        :param folder: DESCRIPTION
        :type folder: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        station_folders = []
        for folder in self.file_path.rglob("**/"):
            rec_fn = folder.joinpath("recmeta.json")
            if rec_fn.exists():
                station_folders.append(folder)

        return station_folders

    def to_dataframe(
        self,
        sample_rates=[150, 24000],
        run_name_zeros=4,
        calibration_path=None,
    ):
        """
        Get a dataframe of all the files in a given directory with given
        columns.  Loop over station folders.

        :param sample_rates: list of sample rates to read, defaults to [150, 24000]
        :type sample_rates: list of integers, optional
        :param run_name_zeros: Number of zeros in the run name, defaults to 4
        :type run_name_zeros: integer, optional
        :return: Dataframe with each row representing a single file
        :rtype: :class:`pandas.DataFrame`

        """

        if not isinstance(sample_rates, (list, tuple)):
            sample_rates = [sample_rates]

        station_folders = self._locate_station_folders()

        entries = []
        for folder in station_folders:
            rec_fn = folder.joinpath(self._receiver_metadata_name)
            receiver_metadata = self._read_receiver_metadata_json(rec_fn)
            self.metadata_dict[
                receiver_metadata.station_metadata.id
            ] = receiver_metadata

            for sr in sample_rates:
                for fn in folder.rglob(f"*{self._file_extension_map[int(sr)]}"):
                    try:
                        phx_obj = open_phoenix(fn)
                    except OSError:
                        self.logger.warning(f"Skipping {fn.name}")
                        continue
                    if hasattr(phx_obj, "read_segment"):
                        segment = phx_obj.read_segment(metadata_only=True)
                        try:
                            start = segment.segment_start_time.isoformat()
                        except IOError:
                            self.logger.warning(
                                f"Could not read file {fn}, SKIPPING"
                            )
                            continue
                        end = segment.segment_end_time.isoformat()
                        n_samples = segment.n_samples

                    else:
                        start = phx_obj.segment_start_time.isoformat()
                        end = phx_obj.segment_end_time.isoformat()
                        n_samples = phx_obj.max_samples

                    entry = self.get_empty_entry_dict()
                    entry["survey"] = receiver_metadata.survey_metadata.id
                    entry["station"] = receiver_metadata.station_metadata.id
                    entry["run"] = (None,)
                    entry["start"] = start
                    entry["end"] = end
                    entry["channel_id"] = phx_obj.channel_id
                    entry["component"] = receiver_metadata.channel_map[
                        phx_obj.channel_id
                    ]
                    entry["fn"] = fn
                    entry["sample_rate"] = phx_obj.sample_rate
                    entry["file_size"] = phx_obj.file_size
                    entry["n_samples"] = n_samples
                    entry["sequence_number"] = phx_obj.seq
                    entry["instrument_id"] = phx_obj.recording_id
                    entry["calibration_fn"] = None
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

        for station in df.station.unique():
            for sr in sample_rates:
                run_stem = self._file_extension_map[int(sr)].split("_")[-1]
                # continuous data
                if sr < 1000:
                    sdf = rdf.loc[
                        (rdf.station == station) & (rdf.sample_rate == sr)
                    ].sort_values("sequence_number")
                    starts = np.sort(
                        sdf.loc[sdf.sample_rate == sr].start.unique()
                    )
                    ends = np.sort(sdf.loc[sdf.sample_rate == sr].end.unique())

                    # find any breaks in the data
                    diff = ends[0:-1] - starts[1:]
                    diff = diff.astype("timedelta64[s]").astype(float)

                    breaks = np.nonzero(diff)[0]

                    # this logic probably needs some work.  Need to figure
                    # out how to set pandas values
                    count = 1
                    if len(breaks) > 0:

                        start_breaks = starts[breaks]
                        for ii in range(len(start_breaks)):
                            count += 1
                            rdf.loc[
                                (rdf.station == station)
                                & (rdf.start == start_breaks[ii])
                                & (rdf.sample_rate == sr),
                                "run",
                            ] = f"sr{run_stem}_{count:0{zeros}}"

                    else:
                        rdf.loc[
                            (rdf.station == station) & (rdf.sample_rate == sr),
                            "run",
                        ] = f"sr{run_stem}_{count:0{zeros}}"

                # segmented data
                else:

                    starts = rdf.loc[
                        (rdf.station == station) & (rdf.sample_rate == sr),
                        "start",
                    ].unique()
                    for ii, s in enumerate(starts, 1):
                        rdf.loc[
                            (rdf.start == s) & (rdf.sample_rate == sr), "run"
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
