#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z3DCollection
=================

An object to hold Z3D file information to make processing easier.


Created on Sat Apr  4 12:40:40 2020

@author: peacock
"""
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from mt_metadata.timeseries import Station

from mth5.io.collection import Collection
from mth5.io.zen import Z3D
from mth5.io.zen.coil_response import CoilResponse


# =============================================================================
# Collection of Z3D Files
# =============================================================================


class Z3DCollection(Collection):
    """
    Collection manager for Z3D file operations and metadata processing.

    This class provides functionality to handle collections of Z3D files,
    including metadata extraction, station information management, and
    dataframe creation for analysis workflows.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to directory containing Z3D files, by default None
    **kwargs : dict
        Additional keyword arguments passed to parent Collection class

    Attributes
    ----------
    station_metadata_dict : dict[str, Station]
        Dictionary mapping station IDs to Station metadata objects
    file_ext : str
        File extension for Z3D files ("z3d")

    Examples
    --------
    >>> zc = Z3DCollection("/path/to/z3d/files")
    >>> df = zc.to_dataframe(sample_rates=[256, 4096])
    >>> print(df.head())
    """

    def __init__(self, file_path: str | Path | None = None, **kwargs: Any) -> None:
        """
        Initialize Z3DCollection with optional file path.

        Parameters
        ----------
        file_path : str or Path, optional
            Path to directory containing Z3D files, by default None
        **kwargs : dict
            Additional keyword arguments passed to parent Collection class
        """
        super().__init__(file_path=file_path, **kwargs)
        self.station_metadata_dict: dict[str, Station] = {}
        self.file_ext: str = "z3d"

    def get_calibrations(self, antenna_calibration_file: str | Path) -> CoilResponse:
        """
        Load coil calibration data from antenna calibration file.

        Parameters
        ----------
        antenna_calibration_file : str or Path
            Path to the antenna.cal file containing coil calibration data

        Returns
        -------
        CoilResponse
            CoilResponse object containing calibration information for
            various coil serial numbers

        Examples
        --------
        >>> zc = Z3DCollection("/path/to/z3d/files")
        >>> cal_obj = zc.get_calibrations("/path/to/antenna.cal")
        >>> print(cal_obj.has_coil_number("2324"))
        """
        return CoilResponse(antenna_calibration_file)

    def _sort_station_metadata(
        self, station_list: list[dict[str, Any]]
    ) -> dict[str, Station]:
        """
        Process and consolidate station metadata from multiple Z3D files.

        Takes a list of station metadata dictionaries and consolidates them
        by station ID, computing median values for coordinates when multiple
        measurements exist for the same station.

        Parameters
        ----------
        station_list : list of dict
            List of station metadata dictionaries, each containing station
            information with keys like 'id', 'location.latitude', etc.

        Returns
        -------
        dict[str, Station]
            Dictionary mapping station IDs to Station metadata objects
            with consolidated location information

        Notes
        -----
        For stations with multiple coordinate measurements, this method
        computes the median latitude, longitude, and elevation values
        to provide a robust central estimate.

        Examples
        --------
        >>> station_data = [
        ...     {'id': '001', 'location.latitude': 40.5, 'location.longitude': -116.8},
        ...     {'id': '001', 'location.latitude': 40.6, 'location.longitude': -116.9}
        ... ]
        >>> zc = Z3DCollection()
        >>> stations = zc._sort_station_metadata(station_data)
        >>> print(stations['001'].location.latitude)  # Median value
        """
        sdf = pd.DataFrame(station_list)
        info: dict[str, Station] = {}
        for station in sdf.id.unique():
            station_df = sdf[sdf.id == station]
            station_metadata = Station()
            station_metadata.id = station
            station_metadata.location.latitude = station_df[
                "location.latitude"
            ].median()
            station_metadata.location.longitude = station_df[
                "location.longitude"
            ].median()
            station_metadata.location.elevation = station_df[
                "location.elevation"
            ].median()

            info[station] = station_metadata

        return info

    def to_dataframe(
        self,
        sample_rates: list[int] = [256, 4096],
        run_name_zeros: int = 4,
        calibration_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Extract Z3D file information and create analysis-ready dataframe.

        Processes all Z3D files in the collection, extracting metadata and
        file information to create a comprehensive dataframe suitable for
        magnetotelluric data analysis workflows.

        Parameters
        ----------
        sample_rates : list of int, default [256, 4096]
            Allowed sampling rates in Hz. Files with sample rates not in
            this list will trigger a warning and early return
        run_name_zeros : int, default 4
            Number of zero-padding digits for run names in dataframe sorting
        calibration_path : str or Path, optional
            Path to antenna calibration file. If None, calibration information
            will not be included, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe containing Z3D file information with columns:
            - survey: Survey/job name from Z3D metadata
            - station: Station identifier
            - run: Automatically assigned run names based on start times
            - start/end: ISO format timestamps for data recording period
            - channel_id: Channel number from Z3D file
            - component: Measurement component (ex, ey, hx, hy, hz)
            - fn: Path to Z3D file
            - sample_rate: Sampling frequency in Hz
            - file_size: Size of Z3D file in bytes
            - n_samples: Number of data samples in file
            - sequence_number: Sequential numbering within station
            - dipole: Dipole length in meters (for electric channels)
            - coil_number: Coil serial number (for magnetic channels)
            - latitude/longitude/elevation: Station coordinates
            - instrument_id: ZEN box identifier
            - calibration_fn: Path to calibration file if available

        Raises
        ------
        AttributeError
            If Z3D files contain invalid or missing required metadata
        FileNotFoundError
            If calibration_path is specified but file doesn't exist

        Examples
        --------
        >>> zc = Z3DCollection("/path/to/z3d/files")
        >>> df = zc.to_dataframe(sample_rates=[256, 4096],
        ...                      calibration_path="/path/to/antenna.cal")
        >>> print(df[['station', 'component', 'sample_rate']].head())
        >>> df.to_csv("/path/output/z3d_inventory.csv")

        Notes
        -----
        This method also populates the `station_metadata_dict` attribute
        with consolidated station metadata derived from all processed files.
        """
        station_metadata: list[dict[str, Any]] = []

        # Handle optional calibration path
        cal_obj: CoilResponse | None = None
        if calibration_path is not None:
            cal_obj = self.get_calibrations(calibration_path)

        entries: list[dict[str, Any]] = []

        for z3d_fn in set(
            self.get_files(
                [self.file_ext, self.file_ext.lower(), self.file_ext.upper()]
            )
        ):
            z3d_obj = Z3D(z3d_fn)
            z3d_obj.read_all_info()
            station_metadata.append(z3d_obj.station_metadata.to_dict(single=True))

            # Validate sample rate
            if (
                z3d_obj.sample_rate is not None
                and int(z3d_obj.sample_rate) not in sample_rates
            ):
                self.logger.warning(f"{z3d_obj.sample_rate} not in {sample_rates}")
                return pd.DataFrame()  # Return empty dataframe instead of None

            entry = self.get_empty_entry_dict()
            entry["survey"] = z3d_obj.metadata.job_name
            entry["station"] = z3d_obj.station
            entry["run"] = None
            entry["start"] = z3d_obj.start.isoformat()
            entry["end"] = (
                z3d_obj.end.isoformat()
                if hasattr(z3d_obj.end, "isoformat")
                else str(z3d_obj.end)
            )
            entry["channel_id"] = z3d_obj.channel_number
            entry["component"] = z3d_obj.component
            entry["fn"] = z3d_fn
            entry["sample_rate"] = z3d_obj.sample_rate
            entry["file_size"] = z3d_obj.file_size
            entry["n_samples"] = z3d_obj.n_samples
            entry["sequence_number"] = 0
            entry["dipole"] = z3d_obj.dipole_length
            entry["coil_number"] = z3d_obj.coil_number
            entry["latitude"] = z3d_obj.latitude
            entry["longitude"] = z3d_obj.longitude
            entry["elevation"] = z3d_obj.elevation
            entry["instrument_id"] = f"ZEN_{int(z3d_obj.header.box_number):03}"

            # Handle calibration file assignment
            if (
                cal_obj is not None
                and z3d_obj.coil_number
                and cal_obj.has_coil_number(z3d_obj.coil_number)
            ):
                entry["calibration_fn"] = cal_obj.calibration_file
            else:
                entry["calibration_fn"] = None

            entries.append(entry)

        # Create and process dataframe
        df = self._sort_df(self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros)

        # Store consolidated station metadata
        self.station_metadata_dict = self._sort_station_metadata(station_metadata)

        return df

    def assign_run_names(self, df: pd.DataFrame, zeros: int = 3) -> pd.DataFrame:
        """
        Assign standardized run names to dataframe based on start times.

        Creates run names using the pattern 'sr{sample_rate}_{block_number}'
        where block_number is assigned sequentially based on unique start
        times within each station.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing Z3D file information with at least
            'station', 'start', and 'sample_rate' columns
        zeros : int, default 3
            Number of zero-padding digits for block numbers in run names

        Returns
        -------
        pd.DataFrame
            Modified dataframe with updated 'run' and 'sequence_number'
            columns assigned based on temporal ordering within each station

        Examples
        --------
        >>> zc = Z3DCollection()
        >>> df = pd.DataFrame({
        ...     'station': ['001', '001', '002'],
        ...     'start': ['2022-01-01T10:00:00', '2022-01-01T12:00:00', '2022-01-01T10:00:00'],
        ...     'sample_rate': [256, 256, 4096]
        ... })
        >>> df_with_runs = zc.assign_run_names(df, zeros=3)
        >>> print(df_with_runs['run'].tolist())
        ['sr256_001', 'sr256_002', 'sr4096_001']

        Notes
        -----
        This method modifies the input dataframe in-place by updating the
        'run' and 'sequence_number' columns. Start times are used to
        determine temporal ordering within each station.
        """
        # Assign run names based on station and start time
        for station in df.station.unique():
            starts = sorted(df[df.station == station].start.unique())
            for block_num, start in enumerate(starts, 1):
                sample_rate = df[
                    (df.station == station) & (df.start == start)
                ].sample_rate.unique()[0]

                df.loc[
                    (df.station == station) & (df.start == start), "run"
                ] = f"sr{sample_rate:.0f}_{block_num:0{zeros}}"
                df.loc[
                    (df.station == station) & (df.start == start),
                    "sequence_number",
                ] = block_num
        return df
