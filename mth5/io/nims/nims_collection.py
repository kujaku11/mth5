# -*- coding: utf-8 -*-
"""
NIMS Collection
===============

Collection of NIMS binary files combined into runs for magnetotelluric data processing.

Created on Wed Aug 31 10:32:44 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mth5.io.collection import Collection
from mth5.io.nims import NIMS


# =============================================================================


class NIMSCollection(Collection):
    """
    Collection of NIMS binary files into runs.

    This class provides functionality for organizing and processing multiple NIMS
    binary files into a structured format for magnetotelluric data analysis.

    Parameters
    ----------
    file_path : str | Path | None, optional
        Path to the directory containing NIMS binary files.
    **kwargs : dict
        Additional keyword arguments passed to the parent Collection class.

    Attributes
    ----------
    file_ext : str
        File extension for NIMS binary files ('bin').
    survey_id : str
        Survey identifier, defaults to 'mt'.

    Examples
    --------
    >>> from mth5.io.nims import NIMSCollection
    >>> nc = NIMSCollection(r"/path/to/nims/station")
    >>> nc.survey_id = "mt001"
    >>> df = nc.to_dataframe()

    See Also
    --------
    mth5.io.collection.Collection : Base collection class
    mth5.io.nims.NIMS : NIMS file reader
    """

    def __init__(self, file_path: str | Path | None = None, **kwargs: Any) -> None:
        """
        Initialize NIMSCollection instance.

        Parameters
        ----------
        file_path : str | Path | None, optional
            Path to the directory containing NIMS binary files.
        **kwargs : dict
            Additional keyword arguments passed to the parent Collection class.
        """
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext: str = "bin"
        self.survey_id: str = "mt"

    def to_dataframe(
        self,
        sample_rates: int | list[int] = [1],
        run_name_zeros: int = 2,
        calibration_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Create a DataFrame of each NIMS binary file in the collection directory.

        This method processes all NIMS binary files in the specified directory and
        extracts metadata to create a structured DataFrame suitable for further
        magnetotelluric data processing.

        Parameters
        ----------
        sample_rates : int | list[int], default [1]
            Sample rates to include in the DataFrame. Note that for NIMS data,
            this parameter is present for interface consistency but all files
            will be processed regardless of their sample rate.
        run_name_zeros : int, default 2
            Number of zeros to use when formatting run names in the output.
        calibration_path : str | Path | None, optional
            Path to calibration files. Currently not used in NIMS processing
            but included for interface consistency.

        Returns
        -------
        pd.DataFrame
            DataFrame containing metadata for each NIMS file with columns:
            - survey : Survey identifier
            - station : Station name from NIMS file
            - run : Run identifier from NIMS file
            - start : Start time in ISO format
            - end : End time in ISO format
            - fn : File path
            - sample_rate : Sampling rate
            - file_size : File size in bytes
            - n_samples : Number of samples
            - dipole : Electric dipole lengths [Ex, Ey]
            - channel_id : Channel identifier (always 1)
            - sequence_number : Sequence number (always 0)
            - component : Comma-separated component list
            - instrument_id : Instrument identifier (always 'NIMS')

        Notes
        -----
        This method assumes the directory contains files from a single station.
        Each NIMS file is read to extract header information including timing,
        station identification, and measurement parameters.

        Examples
        --------
        >>> from mth5.io.nims import NIMSCollection
        >>> nc = NIMSCollection("/path/to/nims/station")
        >>> df = nc.to_dataframe(run_name_zeros=3)
        >>> print(df[['station', 'run', 'start', 'sample_rate']])
        """
        entries = []
        for fn in self.get_files(
            [self.file_ext, self.file_ext.lower(), self.file_ext.upper()]
        ):
            nims_obj = NIMS(fn)
            nims_obj.read_header()
            entry = self.get_empty_entry_dict()
            entry["survey"] = self.survey_id
            entry["station"] = nims_obj.station
            entry["run"] = nims_obj.run_id
            entry["start"] = nims_obj.start_time.isoformat()
            entry["end"] = nims_obj.end_time.isoformat()
            entry["fn"] = fn
            entry["sample_rate"] = nims_obj.sample_rate
            entry["file_size"] = nims_obj.file_size
            entry["n_samples"] = nims_obj.n_samples
            entry["dipole"] = [nims_obj.ex_length, nims_obj.ey_length]

            entries.append(entry)

        # make pandas dataframe and set data types
        df = pd.DataFrame(entries)

        # If there are no entries, create an empty DataFrame with the
        # expected columns so subsequent scalar assignments and dtype
        # enforcement work without raising (pandas raises when assigning
        # scalars into an empty frame with no defined index).
        if df.empty:
            expected_cols = [
                "survey",
                "station",
                "run",
                "start",
                "end",
                "fn",
                "sample_rate",
                "file_size",
                "n_samples",
                "dipole",
                "channel_id",
                "sequence_number",
                "component",
                "instrument_id",
            ]
            df = pd.DataFrame(columns=expected_cols)

        # Populate/ensure scalar columns exist
        if "channel_id" not in df.columns:
            df["channel_id"] = 1
        else:
            # Explicitly coerce to numeric before filling to avoid future downcast warnings
            df.loc[:, "channel_id"] = (
                pd.to_numeric(df.loc[:, "channel_id"], errors="coerce")
                .fillna(1)
                .astype("int64")
            )

        if "sequence_number" not in df.columns:
            df["sequence_number"] = 0
        else:
            df.loc[:, "sequence_number"] = (
                pd.to_numeric(df.loc[:, "sequence_number"], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

        if "component" not in df.columns:
            df["component"] = ",".join(["hx", "hy", "hz", "ex", "ey", "temperature"])
        else:
            df.loc[:, "component"] = df.loc[:, "component"].fillna(
                ",".join(["hx", "hy", "hz", "ex", "ey", "temperature"])
            )

        if "instrument_id" not in df.columns:
            df["instrument_id"] = "NIMS"
        else:
            df.loc[:, "instrument_id"] = df.loc[:, "instrument_id"].fillna("NIMS")

        df = self._sort_df(self._set_df_dtypes(df), run_name_zeros)

        return df

    def assign_run_names(self, df: pd.DataFrame, zeros: int = 2) -> pd.DataFrame:
        """
        Assign standardized run names to DataFrame entries by station.

        This method assigns run names following the pattern 'sr{sample_rate}_{run_number}'
        where run_number is zero-padded according to the zeros parameter. Run names
        are assigned sequentially within each station, ordered by start time.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing NIMS file metadata with required columns:
            'station', 'start', 'run', 'sample_rate'. The DataFrame will be
            modified in-place.
        zeros : int, default 2
            Number of zeros to use for zero-padding the run number in the
            generated run names (e.g., zeros=2 gives '01', '02', etc.).

        Returns
        -------
        pd.DataFrame
            The input DataFrame with updated 'run' and 'sequence_number' columns.
            Run names follow the format 'sr{sample_rate}_{run_number:0{zeros}}'.

        Notes
        -----
        - Existing run names (non-None values) are preserved
        - Files are processed in chronological order within each station
        - Sequence numbers are assigned incrementally starting from 1
        - Only files with None run names receive new assignments

        Examples
        --------
        >>> import pandas as pd
        >>> from mth5.io.nims import NIMSCollection
        >>> # Assuming df has columns: station, start, run, sample_rate
        >>> nc = NIMSCollection()
        >>> df_updated = nc.assign_run_names(df, zeros=3)
        >>> print(df_updated['run'].tolist())
        ['sr8_001', 'sr8_002', 'sr1_001']
        """

        for station in df.station.unique():
            count = 1
            for row in df[df.station == station].sort_values("start").itertuples():
                if row.run is None:
                    df.loc[row.Index, "run"] = f"sr{row.sample_rate}_{count:0{zeros}}"
                df.loc[row.Index, "sequence_number"] = count
                count += 1

        return df
