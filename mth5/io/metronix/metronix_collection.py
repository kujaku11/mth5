# -*- coding: utf-8 -*-
"""
Metronix collection utilities for managing ATSS files.

This module provides classes for collecting and managing Metronix ATSS
(Audio Time Series System) files and creating pandas DataFrames with
metadata for processing workflows.

Classes
-------
MetronixCollection
    Collection class for managing Metronix ATSS files

Created on Fri Nov 22 13:22:44 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from typing import Any, Union

import pandas as pd

from mth5.io.collection import Collection
from mth5.io.metronix import ATSS


# =============================================================================


class MetronixCollection(Collection):
    """
    Collection class for managing Metronix ATSS files.

    This class extends the base Collection class to handle Metronix ATSS
    (Audio Time Series System) files and their associated JSON metadata files.
    It provides functionality to create pandas DataFrames with comprehensive
    metadata for processing workflows.

    Parameters
    ----------
    file_path : Union[str, Path, None], optional
        Path to directory containing Metronix ATSS files, by default None
    **kwargs
        Additional keyword arguments passed to parent Collection class

    Attributes
    ----------
    file_ext : list[str]
        List of file extensions to search for (["atss"])

    Examples
    --------
    >>> from mth5.io.metronix import MetronixCollection
    >>> collection = MetronixCollection("/path/to/metronix/files")
    >>> df = collection.to_dataframe(sample_rates=[128, 256])
    """

    def __init__(self, file_path: Union[str, Path, None] = None, **kwargs: Any) -> None:
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext: list[str] = ["atss"]

    def to_dataframe(
        self,
        sample_rates: list[int] = [128],
        run_name_zeros: int = 0,
        calibration_path: Union[str, Path, None] = None,
    ) -> pd.DataFrame:
        """
        Create DataFrame for Metronix timeseries ATSS + JSON file sets.

        Processes all ATSS files in the collection directory, extracts metadata,
        and creates a comprehensive pandas DataFrame with information about each
        channel including timing, location, and instrument details.

        Parameters
        ----------
        sample_rates : list[int], optional
            List of sample rates to include in Hz, by default [128]
        run_name_zeros : int, optional
            Number of zeros for zero-padding run names. If 0, run names
            are unchanged. If > 0, run names are formatted as
            'sr{sample_rate}_{run_number:0{zeros}d}', by default 0
        calibration_path : Union[str, Path, None], optional
            Path to calibration files (currently unused), by default None

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - survey: Survey ID
            - station: Station ID
            - run: Run ID
            - start: Start time (datetime)
            - end: End time (datetime)
            - channel_id: Channel number
            - component: Component name (ex, ey, hx, hy, hz)
            - fn: File path
            - sample_rate: Sample rate in Hz
            - file_size: File size in bytes
            - n_samples: Number of samples
            - sequence_number: Sequence number (always 0)
            - dipole: Dipole length (always 0)
            - coil_number: Coil serial number (magnetic channels only)
            - latitude: Latitude in decimal degrees
            - longitude: Longitude in decimal degrees
            - elevation: Elevation in meters
            - instrument_id: Instrument/system number
            - calibration_fn: Calibration file path (always None)

        Examples
        --------
        >>> collection = MetronixCollection("/path/to/files")
        >>> df = collection.to_dataframe(sample_rates=[128, 256])
        >>> df = collection.to_dataframe(run_name_zeros=4)  # Zero-pad run names
        """
        entries = []
        for atss_fn in set(self.get_files(self.file_ext)):
            atss_obj = ATSS(atss_fn)
            if not atss_obj.sample_rate in sample_rates:
                continue
            ch_metadata = atss_obj.channel_metadata

            entry = self.get_empty_entry_dict()
            entry["survey"] = atss_obj.survey_id
            entry["station"] = atss_obj.station_id
            entry["run"] = atss_obj.run_id
            entry["start"] = ch_metadata.time_period.start
            entry["end"] = ch_metadata.time_period.end
            entry["channel_id"] = atss_obj.channel_number
            entry["component"] = atss_obj.component
            entry["fn"] = atss_fn
            entry["sample_rate"] = ch_metadata.sample_rate
            entry["file_size"] = atss_obj.file_size
            entry["n_samples"] = atss_obj.n_samples
            entry["sequence_number"] = 0
            entry["dipole"] = 0
            if ch_metadata.type in ["magnetic"]:
                entry["coil_number"] = ch_metadata.sensor.id
                entry["latitude"] = ch_metadata.location.latitude
                entry["longitude"] = ch_metadata.location.longitude
                entry["elevation"] = ch_metadata.location.elevation
            else:
                entry["coil_number"] = None
                entry["latitude"] = ch_metadata.positive.latitude
                entry["longitude"] = ch_metadata.positive.longitude
                entry["elevation"] = ch_metadata.positive.elevation

            entry["instrument_id"] = atss_obj.system_number
            entry["calibration_fn"] = None
            entries.append(entry)
        # make pandas dataframe and set data types
        df = self._sort_df(self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros)

        return df

    def assign_run_names(self, df: pd.DataFrame, zeros: int = 0) -> pd.DataFrame:
        """
        Assign formatted run names based on sample rate and run number.

        If zeros is 0, run names are unchanged. Otherwise, run names are
        formatted as 'sr{sample_rate}_{run_number:0{zeros}d}' where the
        run number is extracted from the original run name after the first
        underscore.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing run information with 'run' and 'sample_rate' columns
        zeros : int, optional
            Number of zeros for zero-padding run numbers. If 0, run names
            are unchanged, by default 0

        Returns
        -------
        pd.DataFrame
            DataFrame with updated run names

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'run': ['run_1', 'run_2'],
        ...     'sample_rate': [128, 256]
        ... })
        >>> collection = MetronixCollection()
        >>> result = collection.assign_run_names(df, zeros=3)
        >>> print(result['run'].tolist())
        ['sr128_001', 'sr256_002']

        Notes
        -----
        The method expects run names to be in format 'prefix_number' where
        'number' can be extracted and converted to an integer for formatting.
        """
        if zeros == 0:
            return df

        for row in df.itertuples():
            df.loc[
                row.Index, "run"
            ] = f"sr{row.sample_rate:.0f}_{int(row.run.split('_')[1]):0{zeros}}"
        return df
