# -*- coding: utf-8 -*-
"""
Phoenix file collection module for organizing and processing Phoenix MTU data files.

This module provides the PhoenixCollection class for discovering, organizing,
and managing Phoenix magnetotelluric receiver files within a directory structure.

Created on Thu Aug  4 16:48:47 2022

@author: jpeacock
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from mth5.io import Collection
from mth5.io.phoenix import open_phoenix, PhoenixReceiverMetadata


# =============================================================================


class PhoenixCollection(Collection):
    """
    Collection manager for Phoenix MTU data files.

    Organizes Phoenix magnetotelluric receiver files into runs based on
    timing and sample rates. Handles multiple sample rates (30, 150, 2400,
    24000, 96000 Hz) and manages receiver metadata.

    Parameters
    ----------
    file_path : str | Path | None, optional
        Path to the directory containing Phoenix data files. Can be the
        station folder or a parent folder containing multiple stations.
    **kwargs
        Additional keyword arguments passed to parent Collection class.

    Attributes
    ----------
    metadata_dict : dict[str, PhoenixReceiverMetadata]
        Dictionary mapping station IDs to their receiver metadata.

    Examples
    --------
    Create a collection from a station directory:

    >>> from mth5.io.phoenix import PhoenixCollection
    >>> collection = PhoenixCollection(r"/path/to/station")
    >>> runs = collection.get_runs(sample_rates=[150, 24000])
    >>> print(runs.keys())
    dict_keys(['MT001'])

    Process multiple sample rates:

    >>> df = collection.to_dataframe(sample_rates=[150, 2400, 24000])
    >>> print(df.columns)
    Index(['survey', 'station', 'run', 'start', 'end', ...])

    Notes
    -----
    The class automatically discovers station folders by locating
    'recmeta.json' files and organizes time series files by sample rate.

    File extensions are mapped as:

    - 30 Hz: td_30
    - 150 Hz: td_150
    - 2400 Hz: td_2400
    - 24000 Hz: td_24k
    - 96000 Hz: td_96k

    See Also
    --------
    mth5.io.Collection : Base collection class
    mth5.io.phoenix.PhoenixReceiverMetadata : Receiver metadata handler

    """

    def __init__(self, file_path: str | Path | None = None, **kwargs) -> None:
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

    def _read_receiver_metadata_json(
        self, rec_fn: str | Path
    ) -> PhoenixReceiverMetadata | None:
        """
        Read receiver metadata from JSON file.

        Loads and parses the recmeta.json file containing station and
        channel configuration information.

        Parameters
        ----------
        rec_fn : str | Path
            Path to the recmeta.json metadata file.

        Returns
        -------
        PhoenixReceiverMetadata | None
            Receiver metadata object if file exists, None otherwise.

        Examples
        --------
        >>> metadata = collection._read_receiver_metadata_json(
        ...     Path("/data/station/recmeta.json")
        ... )
        >>> print(metadata.station_metadata.id)
        'MT001'

        """

        if Path(rec_fn).is_file():
            return PhoenixReceiverMetadata(fn=rec_fn)
        else:
            self.logger.warning(
                f"Could not find {self._receiver_metadata_name} in {self.file_path}"
            )
            return None

    def _locate_station_folders(self) -> list[Path]:
        """
        Locate all station folders containing recmeta.json files.

        Recursively searches the collection path for directories containing
        the receiver metadata file (recmeta.json), which identifies a valid
        Phoenix station folder.

        Returns
        -------
        list[Path]
            List of Path objects pointing to station folders.

        Examples
        --------
        >>> folders = collection._locate_station_folders()
        >>> print([f.name for f in folders])
        ['MT001', 'MT002', 'MT003']

        Notes
        -----
        Each station folder must contain a recmeta.json file to be recognized.
        The search is recursive, allowing for nested directory structures.

        """
        station_folders = []
        for folder in self.file_path.rglob("**/"):
            rec_fn = folder.joinpath("recmeta.json")
            if rec_fn.exists():
                station_folders.append(folder)

        return station_folders

    def to_dataframe(
        self,
        sample_rates: list[int] | int = [150, 24000],
        run_name_zeros: int = 4,
        calibration_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Create a DataFrame cataloging all Phoenix files in the collection.

        Scans all station folders for time series files at specified sample
        rates and creates a comprehensive inventory with metadata for each file.

        Parameters
        ----------
        sample_rates : list[int] | int, optional
            Sample rate(s) to include in Hz. Valid values are 30, 150, 2400,
            24000, 96000. Can be a single integer or list (default is [150, 24000]).
        run_name_zeros : int, optional
            Number of zeros for zero-padding run names (default is 4).
            For example, 4 produces 'sr150_0001'.
        calibration_path : str | Path | None, optional
            Path to calibration files. Currently unused but reserved for
            future functionality.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per file containing columns:

            - survey: Survey ID from metadata
            - station: Station ID from metadata
            - run: Run ID (assigned by assign_run_names)
            - start: File start time (ISO format)
            - end: File end time (ISO format)
            - channel_id: Numeric channel identifier
            - component: Channel component name (e.g., 'Ex', 'Hy')
            - fn: Full file path
            - sample_rate: Sample rate in Hz
            - file_size: File size in bytes
            - n_samples: Number of samples in file
            - sequence_number: File sequence number for continuous data
            - instrument_id: Recording/receiver ID
            - calibration_fn: Path to calibration file (currently None)

        Examples
        --------
        Get DataFrame for standard sample rates:

        >>> df = collection.to_dataframe(sample_rates=[150, 24000])
        >>> print(df.shape)
        (245, 14)
        >>> print(df.station.unique())
        ['MT001']

        Process single sample rate:

        >>> df_150 = collection.to_dataframe(sample_rates=150)
        >>> print(df_150.sample_rate.unique())
        [150.]

        Check file coverage:

        >>> for comp in df.component.unique():
        ...     comp_df = df[df.component == comp]
        ...     print(f"{comp}: {len(comp_df)} files")
        Ex: 35 files
        Ey: 35 files
        Hx: 35 files

        Notes
        -----
        - Calibration files (identified by 'calibration' in filename) are
          automatically skipped
        - Files that cannot be opened are logged and skipped
        - The DataFrame is sorted by station, sample_rate, and start time
        - Run names must be assigned separately using assign_run_names()

        See Also
        --------
        assign_run_names : Assign run identifiers based on timing
        get_runs : Get organized runs directly

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
                    if "calibration" in fn.as_posix().lower():
                        self.logger.debug(f"skipping calibration time series {fn}")
                        continue
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
                            self.logger.warning(f"Could not read file {fn}, SKIPPING")
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

        df = self._sort_df(self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros)

        return df

    def assign_run_names(self, df: pd.DataFrame, zeros: int = 4) -> pd.DataFrame:
        """
        Assign run names based on temporal continuity.

        Analyzes file timing to group files into runs. For continuous data
        (< 1000 Hz), maintains a single run as long as files are contiguous.
        For segmented data (≥ 1000 Hz), assigns a unique run to each segment.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by `to_dataframe` method with file inventory.
        zeros : int, optional
            Number of zeros for zero-padding run names (default is 4).

        Returns
        -------
        pd.DataFrame
            DataFrame with 'run' column populated. Run names follow the
            format 'sr{rate}_{number:0{zeros}}', e.g., 'sr150_0001'.

        Examples
        --------
        Assign run names to a DataFrame:

        >>> df = collection.to_dataframe(sample_rates=[150, 24000])
        >>> df_with_runs = collection.assign_run_names(df, zeros=4)
        >>> print(df_with_runs.run.unique())
        ['sr150_0001', 'sr24k_0001', 'sr24k_0002', ...]

        Check for data gaps in continuous data:

        >>> df_150 = df_with_runs[df_with_runs.sample_rate == 150]
        >>> print(df_150.run.unique())
        ['sr150_0001', 'sr150_0002']  # Gap detected between runs

        Count segments in high-rate data:

        >>> df_24k = df_with_runs[df_with_runs.sample_rate == 24000]
        >>> n_segments = len(df_24k.run.unique())
        >>> print(f"Found {n_segments} segments at 24 kHz")
        Found 43 segments at 24 kHz

        Notes
        -----
        **Continuous Data (< 1000 Hz):**

        - Maintains single run ID while files are temporally contiguous
        - Detects gaps by comparing end time of file N with start time of
          file N+1
        - Increments run counter when gap > 0 seconds detected

        **Segmented Data (≥ 1000 Hz):**

        - Each unique start time receives a new run ID
        - Typically results in one run per segment/file

        The run naming scheme uses the sample rate in the identifier:

        - 30 Hz → 'sr30_NNNN'
        - 150 Hz → 'sr150_NNNN'
        - 2400 Hz → 'sr2400_NNNN'
        - 24000 Hz → 'sr24k_NNNN'
        - 96000 Hz → 'sr96k_NNNN'

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
                    starts = np.sort(sdf.loc[sdf.sample_rate == sr].start.unique())
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
        sample_rates: list[int] | int,
        run_name_zeros: int = 4,
        calibration_path: str | Path | None = None,
    ) -> OrderedDict[str, OrderedDict[str, pd.DataFrame]]:
        """
        Organize Phoenix files into runs ready for reading.

        Creates a nested dictionary structure organizing files by station and
        run. For each run, returns only the first file(s) needed to initialize
        reading, as continuous readers will automatically load sequences.

        Parameters
        ----------
        sample_rates : list[int] | int
            Sample rate(s) to include in Hz. Valid values are 30, 150, 2400,
            24000, 96000. Can be a single integer or list.
        run_name_zeros : int, optional
            Number of zeros for zero-padding run names (default is 4).
        calibration_path : str | Path | None, optional
            Path to calibration files. Currently unused but reserved for
            future functionality.

        Returns
        -------
        OrderedDict[str, OrderedDict[str, pd.DataFrame]]
            Nested OrderedDict with structure:

            - Keys: station IDs
            - Values: OrderedDict of runs

              - Keys: run IDs (e.g., 'sr150_0001')
              - Values: DataFrame with first file(s) for each channel

        Examples
        --------
        Get runs for standard sample rates:

        >>> from mth5.io.phoenix import PhoenixCollection
        >>> collection = PhoenixCollection(r"/path/to/station")
        >>> runs = collection.get_runs(sample_rates=[150, 24000])
        >>> print(runs.keys())
        odict_keys(['MT001'])

        Access specific station's runs:

        >>> station_runs = runs['MT001']
        >>> print(list(station_runs.keys()))
        ['sr150_0001', 'sr24k_0001', 'sr24k_0002', ...]

        Get first file for a specific run:

        >>> run_df = runs['MT001']['sr150_0001']
        >>> print(run_df[['component', 'fn', 'start']])
          component                           fn                 start
        0        Ex  /path/to/8441_2020...td_150  2020-06-02T19:00:00
        1        Ey  /path/to/8441_2020...td_150  2020-06-02T19:00:00

        Iterate over all runs:

        >>> for station_id, station_runs in runs.items():
        ...     for run_id, run_df in station_runs.items():
        ...         print(f"{station_id}/{run_id}: {len(run_df)} channels")
        MT001/sr150_0001: 5 channels
        MT001/sr24k_0001: 5 channels

        Get single sample rate:

        >>> runs_150 = collection.get_runs(sample_rates=150)
        >>> run_ids = list(runs_150['MT001'].keys())
        >>> print([r for r in run_ids if 'sr150' in r])
        ['sr150_0001']

        Notes
        -----
        **For Continuous Data (< 1000 Hz):**

        Returns only the first file in each sequence per channel. The Phoenix
        reader will automatically load the complete sequence when reading.

        **For Segmented Data (≥ 1000 Hz):**

        Returns the first file for each segment. Each segment must be read
        separately.

        **DataFrame Content:**

        Each DataFrame contains one row per channel component with the earliest
        file for that component in the run. This ensures all channels start from
        the same time.

        The method internally:

        1. Calls to_dataframe() to inventory all files
        2. Calls assign_run_names() to group files into runs
        3. Selects first file(s) for each run and component
        4. Returns organized structure for easy iteration

        See Also
        --------
        to_dataframe : Create complete file inventory
        assign_run_names : Group files into runs
        mth5.io.phoenix.read_phoenix : Read Phoenix files

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

                first_row_list = []
                for comp in run_df.component.unique():
                    comp_df = run_df[run_df.component == comp]
                    comp_df = comp_df[comp_df.start == comp_df.start.min()]
                    first_row_list.append(comp_df)

                # run_dict[station][run_id] = run_df[
                #     run_df.start == run_df.start.min()
                # ]
                # need to get the earliest file for each component separately
                run_dict[station][run_id] = pd.concat(first_row_list)

        return run_dict
