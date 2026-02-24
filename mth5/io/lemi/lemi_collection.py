# -*- coding: utf-8 -*-
"""
LEMI Collection
===============

Collection of LEMI files combined into runs.
Supports both LEMI-424 (.txt) and LEMI-423 (.B423) instruments.

Created on Wed Aug 31 10:32:44 2022

@author: jpeacock
"""

import pathlib
from pathlib import Path
from typing import List

# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mth5.io.collection import Collection
from mth5.io.lemi import LEMI423Reader, LEMI424


# =============================================================================


class LEMICollection(Collection):
    """
    Collection of LEMI files into runs based on start and end times.

    Supports both LEMI-424 (.txt) and LEMI-423 (.B423) instrument files.
    Will assign run names as 'sr{sample_rate}_{index:0{zeros}}' --> 'sr1_0001'.

    :param file_path: full path to single station LEMI directory
    :type file_path: string or :class`pathlib.Path`
    :param file_ext: extension of LEMI files, default is ['txt', 'TXT', 'B423', 'b423']
    :type file_ext: list of strings
    :param station_id: station id
    :type station_id: string
    :param survey_id: survey id
    :type survey_id: string

    LEMI data comes with little metadata about the station or survey,
    therefore you should assign `station_id` and `survey_id`.

    Parameters
    ----------
    file_path : str or pathlib.Path, optional
        Full path to single station LEMI424 directory, by default None
    file_ext : list of str, optional
        Extension of LEMI424 files, by default ["txt", "TXT"]
    **kwargs
        Additional keyword arguments passed to parent Collection class

    Attributes
    ----------
    station_id : str
        Station identification string, defaults to "mt001"
    survey_id : str
        Survey identification string, defaults to "mt"
    .. note:: LEMI data comes with little metadata about the station or survey,
     therefore you should assign `station_id` and `survey_id`. For LEMI-423,
     station_id can be auto-detected from folder name.

    :LEMI-423 Sample Rates:
        The LEMI-423 instrument supports the following sample rates:
        - 4000 Hz
        - 2000 Hz
        - 1000 Hz
        - 500 Hz
        - 250 Hz

        Note: The rate is auto-detected from the tick index during file reading.

    :LEMI-424 Sample Rates:
        The LEMI-424 instrument operates at:
        - 1 Hz (long period only)

    :Example:

    .. code-block:: python

        >>> from mth5.io.lemi import LEMICollection
        >>> # LEMI-424 files
        >>> lc = LEMICollection(r"/path/to/lemi424/station", file_ext=['txt'])
        >>> lc.station_id = "mt001"
        >>> lc.survey_id = "test_survey"
        >>> run_dict = lc.get_runs([1])
        >>>
        >>> # LEMI-423 files (auto-detects sample rate)
        >>> lc = LEMICollection(r"/path/to/lemi423/station", file_ext=['B423'])
        >>> lc.station_id = "mt002"  # Or auto-detected from folder name
        >>> # Include all possible rates - actual rate detected from files
        >>> run_dict = lc.get_runs([4000, 2000, 1000, 500, 250])
        >>>
        >>> # Or if you know the rate was 1000 Hz
        >>> run_dict = lc.get_runs([1000])

    Examples
    --------
    >>> from mth5.io.lemi import LEMICollection
    >>> lc = LEMICollection(r"/path/to/single/lemi/station")
    >>> lc.station_id = "mt001"
    >>> lc.survey_id = "test_survey"
    >>> run_dict = lc.get_runs(1)
    """

    def __init__(
        self,
        file_path: str | pathlib.Path | None = None,
        file_ext: List[str] | None = None,
        **kwargs,
    ) -> None:
        if file_ext is None:
            file_ext = ["txt", "TXT", "B423", "b423"]
        super().__init__(file_path=file_path, file_ext=file_ext, **kwargs)

        self.station_id = "mt001"
        self.survey_id = "mt"
        self.calibration_dict = {}

    def get_calibrations(self, calibration_path: str | Path) -> dict:
        """
        Get calibration dictionary for LEMI424 files.  This assumes that the
        calibrations files are in JSON format and named as
        'LEMI-424-<component>.json'

        Parameters
        ----------
        calibration_path : str or pathlib.Path
            Path to calibration files

        Returns
        -------
        dict
            Calibration dictionary for LEMI424 files

        Examples
        --------
        >>> from mth5.io.lemi import LEMICollection
        >>> lc = LEMICollection("/path/to/single/lemi/station")
        >>> cal_dict = lc.get_calibrations(Path("/path/to/calibrations"))
        """
        calibration_path = Path(calibration_path)

        calibration_dict = {}
        for fn in calibration_path.rglob("*.json"):
            comp = fn.stem.split("-")[-1].split(".", 1)[0]
            calibration_dict[comp] = fn

        return calibration_dict

    def to_dataframe(
        self,
        sample_rates: int | List[int] | None = None,
        run_name_zeros: int = 4,
        calibration_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Create a data frame of LEMI files (both .txt and .B423) in a directory.

        Automatically detects file type and uses appropriate reader. For LEMI-423
        files, sample rate is auto-detected from the tick counter (max_tick + 1).

        Notes
        -----
        This assumes the given directory contains a single station

        Parameters
        ----------
        sample_rates : int or list of int, optional
            - LEMI-424: Use [1] (always 1 Hz)
            - LEMI-423: List which rate(s) to include. Sample rate is auto-detected from each file's tick counter.
                - If you know the rate was 1000 Hz: [1000]
                - To include all possible rates: [4000, 2000, 1000, 500, 250]
        run_name_zeros : int, optional
            Number of zeros to assign to the run name, by default 4
        calibration_path : str or pathlib.Path, optional
            Path to calibration files, by default None

        Returns
        -------
        pd.DataFrame
            DataFrame with information of each TXT file in the given directory

        :Example:

            >>> from mth5.io.lemi import LEMICollection
            >>> # LEMI-424 files
            >>> lc = LEMICollection("/path/to/lemi424/station", file_ext=['txt'])
            >>> lemi_df = lc.to_dataframe(sample_rates=[1])
            >>>
            >>> # LEMI-423 files - sample rate auto-detected
            >>> lc = LEMICollection("/path/to/lemi423/station", file_ext=['B423'])
            >>> # Include all possible rates (will auto-detect actual rate)
            >>> lemi_df = lc.to_dataframe(sample_rates=[4000, 2000, 1000, 500, 250])
            >>>
            >>> # Or if you know it was recorded at 1000 Hz
            >>> lemi_df = lc.to_dataframe(sample_rates=[1000])

        Examples
        --------
        >>> from mth5.io.lemi import LEMICollection
        >>> lc = LEMICollection("/path/to/single/lemi/station")
        >>> lemi_df = lc.to_dataframe()
        """
        if sample_rates is None:
            sample_rates = [1]

        if calibration_path is None:
            calibration_path = Path(self.file_path)
        self.calibration_dict = self.get_calibrations(calibration_path)
        if not self.calibration_dict:
            self.logger.warning(
                f"No calibration files found in {calibration_path}, "
                "proceeding without calibrations."
            )

        entries = []
        for fn in self.get_files(self.file_ext):
            fn_path = pathlib.Path(fn)

            # Determine which reader to use based on file extension
            if fn_path.suffix.lower() in [".txt"]:
                # LEMI-424 reader
                lemi_obj = LEMI424(fn)
                lemi_obj.read_metadata()
                instrument_id = "LEMI-424"
                n_samples = int(lemi_obj.n_samples)
                sample_rate = lemi_obj.sample_rate
                file_size = lemi_obj.file_size
                start = lemi_obj.start.isoformat()
                end = lemi_obj.end.isoformat()
                components = ",".join(lemi_obj.run_metadata.channels_recorded_all)

            elif fn_path.suffix.lower() in [".b423"]:
                # LEMI-423 reader - read header only for metadata
                lemi_obj = LEMI423Reader([fn])
                # Read header to get metadata without loading full data
                df, hdr = lemi_obj._read_one(fn_path)
                lemi_obj.header = hdr
                lemi_obj.data = df

                # Use sample rate from header (auto-detected from tick counter in _read_one)
                # LEMI-423 supports: 4000, 2000, 1000, 500, 250 Hz
                sample_rate = hdr.get("sample_rate", None)

                # Skip file if sample rate couldn't be detected
                if sample_rate is None:
                    self.logger.warning(
                        f"Could not detect sample rate from {fn}, skipping"
                    )
                    continue

                instrument_id = f"LEMI-423 #{hdr.get('instrument_number', '')}"
                n_samples = len(df)
                file_size = fn_path.stat().st_size
                start = df.index[0].isoformat() if len(df) > 0 else None
                end = df.index[-1].isoformat() if len(df) > 0 else None
                components = "hx,hy,hz,ex,ey"  # LEMI-423 always has 5 channels
            else:
                self.logger.warning(f"Unknown file extension for {fn}, skipping")
                continue

            # Filter by sample rate
            if sample_rate not in sample_rates:
                continue

            entry = self.get_empty_entry_dict()
            entry["survey"] = self.survey_id
            entry["station"] = self.station_id
            entry["start"] = start
            entry["end"] = end
            entry["component"] = components
            entry["fn"] = fn
            entry["sample_rate"] = sample_rate
            entry["file_size"] = file_size
            entry["n_samples"] = n_samples
            entry["instrument_id"] = instrument_id

            entries.append(entry)

        # make pandas dataframe and set data types
        if len(entries) == 0:
            self.logger.warning("No entries found for LEMI collection")
            return pd.DataFrame()

        df = pd.DataFrame(entries)
        df.loc[:, "channel_id"] = 1
        df.loc[:, "sequence_number"] = 0

        df = self._sort_df(self._set_df_dtypes(df), run_name_zeros)

        return df

    def assign_run_names(self, df: pd.DataFrame, zeros: int = 4) -> pd.DataFrame:
        """
        Assign run names based on start and end times and sample rates.

        Checks if a file has the same start time as the last end time.
        Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the appropriate columns
        zeros : int, optional
            Number of zeros in run name, by default 4

        Returns
        -------
        pd.DataFrame
            DataFrame with run names assigned
        :Example:
            >>> # LEMI-424 at 1 Hz: sr1_0001, sr1_0002
            >>> # LEMI-423 at 1000 Hz: sr1000_0001, sr1000_0002
            >>> # LEMI-423 at 250 Hz: sr250_0001, sr250_0002

        """
        # Track counters and previous end times per sample rate
        run_counts = {}
        previous_ends = {}

        for row in df.itertuples():
            sr = int(row.sample_rate)

            # Initialize for this sample rate
            if sr not in run_counts:
                run_counts[sr] = 1
                previous_ends[sr] = None

            # Check if new run (time gap detected)
            if previous_ends[sr] is not None:
                gap = (row.start - previous_ends[sr]).total_seconds()
                if gap > 1.0 / sr:  # Gap > 1 sample period = new run
                    run_counts[sr] += 1

            # Assign run name
            df.loc[row.Index, "run"] = f"sr{sr}_{run_counts[sr]:0{zeros}}"
            previous_ends[sr] = row.end

        return df
