# -*- coding: utf-8 -*-
from __future__ import annotations


"""Station-level HDF5 helpers for MTH5."""

# =============================================================================
# Imports
# =============================================================================
import inspect
from typing import Any

import h5py
import numpy as np
import pandas as pd
from mt_metadata import timeseries as metadata
from mt_metadata.common.mttime import MTime

from mth5.groups import (
    BaseGroup,
    MasterFCGroup,
    MasterFeaturesGroup,
    RunGroup,
    TransferFunctionsGroup,
)
from mth5.helpers import read_attrs_to_dict
from mth5.utils.exceptions import MTH5Error


meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
# =============================================================================
# Standards Group
# =============================================================================


class MasterStationGroup(BaseGroup):
    """Collection helper for all stations in a survey.

    The group lives at ``/Survey/Stations`` and offers convenience accessors
    to add, fetch, or remove stations along with a summary table.

    Examples
    --------
    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> _ = mth5_obj.open_mth5("/tmp/example.mth5", mode="a")
    >>> stations = mth5_obj.stations_group
    >>> _ = stations.add_station("MT001")
    >>> stations.station_summary.head()  # doctest: +SKIP
    """

    def __init__(self, group: h5py.Group, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)

    @property
    def station_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all stations in the file.

        Returns
        -------
        pandas.DataFrame
            Columns include ``station``, ``start``, ``end``, ``latitude``,
            and ``longitude``. Empty if no stations are present.

        Notes
        -----
        Timestamps are parsed to pandas ``datetime64[ns]`` when possible.

        Examples
        --------
        >>> summary = stations.station_summary
        >>> list(summary.columns)
        ['station', 'start', 'end', 'latitude', 'longitude']
        """

        def _get_entry(group: h5py.Group) -> dict[str, Any]:
            return {
                "station": group.attrs["id"],
                "start": group.attrs["time_period.start"],
                "end": group.attrs["time_period.end"],
                "latitude": group.attrs["location.latitude"],
                "longitude": group.attrs["location.longitude"],
            }

        def _recursive_get_station_entry(
            group: h5py.Group,
            entry_list: list[dict[str, Any]] | None = None,
        ) -> list[dict[str, Any]]:
            """Collect station entries recursively from nested groups."""

            if entry_list is None:
                entry_list = []

            if isinstance(group, h5py._hl.group.Group):
                try:
                    group_type = group.attrs["mth5_type"].lower()
                    if group_type in ["station"]:
                        entry_list.append(_get_entry(group))
                    elif group_type in ["masterstation"]:
                        for node in group.values():
                            entry_list = _recursive_get_station_entry(node, entry_list)
                except KeyError:
                    pass
            return entry_list

        st_list: list[dict[str, Any]] = []
        st_list = _recursive_get_station_entry(self.hdf5_group, st_list)
        df = pd.DataFrame(st_list)
        if len(df):
            try:
                df.start = pd.to_datetime(df.start, format="mixed")
                df.end = pd.to_datetime(df.end, format="mixed")
            except ValueError:
                df.start = pd.to_datetime(df.start)
                df.end = pd.to_datetime(df.end)

        return df

    def add_station(
        self, station_name: str, station_metadata: metadata.Station | None = None
    ) -> "StationGroup":
        """Add or fetch a station group at ``/Survey/Stations/<name>``.

        Parameters
        ----------
        station_name : str
            Station identifier, typically matches ``metadata.id``.
        station_metadata : mt_metadata.timeseries.Station, optional
            Metadata container to seed the station attributes.

        Returns
        -------
        StationGroup
            Convenience wrapper for the created or existing station.

        Raises
        ------
        ValueError
            If ``station_name`` is empty.

        Examples
        --------
        >>> station = stations.add_station("MT001")
        >>> station.metadata.id
        'MT001'
        """
        if not station_name:
            raise ValueError("station name is None, do not know what to name it")

        return self._add_group(station_name, StationGroup, station_metadata, match="id")

    def get_station(self, station_name: str) -> "StationGroup":
        """Return an existing station by name.

        Parameters
        ----------
        station_name : str
            Name of the station to retrieve.

        Returns
        -------
        StationGroup
            Wrapper for the requested station.

        Raises
        ------
        MTH5Error
            If the station does not exist.

        Examples
        --------
        >>> existing = stations.get_station("MT001")
        >>> existing.name
        'MT001'
        """
        return self._get_group(station_name, StationGroup)

    def remove_station(self, station_name: str) -> None:
        """Delete a station group reference from the file.

        Parameters
        ----------
        station_name : str
            Existing station name.

        Notes
        -----
        HDF5 deletion removes the reference only; underlying storage is not
        reclaimed.

        Examples
        --------
        >>> stations.remove_station("MT001")
        """

        self._remove_group(station_name)


# =============================================================================
# Station Group
# =============================================================================
class StationGroup(BaseGroup):
    """Utility wrapper for a single station at ``/Survey/Stations/<id>``.

    Station groups manage run collections, metadata propagation, and provide
    summary utilities for quick inspection.

    Examples
    --------
    >>> from mth5 import mth5
    >>> m5 = mth5.MTH5()
    >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
    >>> station = m5.stations_group.add_station("MT001")
    >>> _ = station.add_run("MT001a")
    >>> station.run_summary.shape[0] >= 1
    True
    """

    def __init__(
        self,
        group: h5py.Group,
        station_metadata: metadata.Station | None = None,
        **kwargs: Any,
    ) -> None:
        self._default_subgroup_names = [
            "Transfer_Functions",
            "Fourier_Coefficients",
            "Features",
        ]
        super().__init__(group, group_metadata=station_metadata, **kwargs)

    def initialize_group(self, **kwargs: Any) -> None:
        """Create default subgroups and write metadata.

        Parameters
        ----------
        **kwargs
            Additional attributes to set on the instance before initialization.

        Examples
        --------
        >>> station.initialize_group()
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.write_metadata()

        for group_name in self._default_subgroup_names:
            try:
                self.hdf5_group.create_group(f"{group_name}")
                m5_grp = getattr(self, f"{group_name.lower()}_group")
                m5_grp.initialize_group()
            except ValueError as value_error:
                if "Unable to synchronously create group" in str(value_error):
                    self.logger.warning("File is in write mode, cannot create group.")
                else:
                    raise ValueError(value_error)

    @property
    def master_station_group(self) -> MasterStationGroup:
        """Shortcut to the containing master station group."""
        return MasterStationGroup(self.hdf5_group.parent)

    @property
    def transfer_functions_group(self) -> TransferFunctionsGroup:
        """Convenience accessor for ``/Station/Transfer_Functions``."""
        return TransferFunctionsGroup(
            self.hdf5_group["Transfer_Functions"], **self.dataset_options
        )

    @property
    def fourier_coefficients_group(self) -> MasterFCGroup:
        """Convenience accessor for ``/Station/Fourier_Coefficients``."""
        return MasterFCGroup(
            self.hdf5_group["Fourier_Coefficients"], **self.dataset_options
        )

    @property
    def features_group(self) -> MasterFeaturesGroup:
        """Convenience accessor for ``/Station/Features``."""
        return MasterFeaturesGroup(self.hdf5_group["Features"], **self.dataset_options)

    @property
    def survey_metadata(self) -> metadata.Survey:
        """Return survey metadata with this station appended."""

        meta_dict = read_attrs_to_dict(
            dict(self.hdf5_group.parent.parent.attrs), metadata.Survey()
        )
        survey_metadata = metadata.Survey()
        survey_metadata.from_dict({"survey": meta_dict})
        survey_metadata.add_station(self.metadata)
        return survey_metadata

    @BaseGroup.metadata.getter
    def metadata(self) -> metadata.Station:
        """Station metadata enriched with run information."""

        if not self._has_read_metadata:
            self.read_metadata()
            self._has_read_metadata = True

        for key in self.groups_list:
            if key.lower() in [name.lower() for name in self._default_subgroup_names]:
                continue
            try:
                key_group = self.get_run(key)
                if key_group.metadata.mth5_type.lower() in ["run"]:
                    self._metadata.add_run(key_group.metadata)
            except MTH5Error:
                self.logger.warning(f"Could not find run {key}")
        return self._metadata

    @property
    def name(self) -> str:
        return self.metadata.id

    @name.setter
    def name(self, name: str) -> None:
        self.metadata.id = name

    @property
    def run_summary(self) -> pd.DataFrame:
        """Return a summary of runs belonging to the station.

        Returns
        -------
        pandas.DataFrame
            Columns include ``id``, ``start``, ``end``, ``components``,
            ``measurement_type``, ``sample_rate``, and ``hdf5_reference``.

        Notes
        -----
        Channel lists stored as byte arrays or JSON strings are normalized
        before summarization.

        Examples
        --------
        >>> station.run_summary.head()  # doctest: +SKIP
        """

        run_list = []
        for key, group in self.hdf5_group.items():
            if group.attrs["mth5_type"].lower() in ["run"]:
                # Helper function to handle both array and string cases
                def get_channel_list(attr_value):
                    if hasattr(attr_value, "tolist"):
                        # If it's an array, use tolist()
                        return attr_value.tolist()
                    elif isinstance(attr_value, str):
                        # If it's a string, try to parse as JSON list
                        try:
                            import json

                            parsed = json.loads(attr_value)
                            if isinstance(parsed, list):
                                return parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                        # If JSON parsing fails, treat as empty list
                        return []
                    else:
                        # For other types, convert to list if possible
                        try:
                            return list(attr_value)
                        except (TypeError, ValueError):
                            return []

                # Get channel lists, handling both string and array formats
                aux_channels = get_channel_list(
                    group.attrs["channels_recorded_auxiliary"]
                )
                elec_channels = get_channel_list(
                    group.attrs["channels_recorded_electric"]
                )
                mag_channels = get_channel_list(
                    group.attrs["channels_recorded_magnetic"]
                )

                comps = ",".join(
                    [
                        ii.decode() if isinstance(ii, bytes) else str(ii)
                        for ii in aux_channels + elec_channels + mag_channels
                    ]
                )
                run_list.append(
                    (
                        group.attrs["id"],
                        group.attrs["time_period.start"].split("+")[0],
                        group.attrs["time_period.end"].split("+")[0],
                        comps,
                        group.attrs["data_type"],
                        group.attrs["sample_rate"],
                        group.ref,
                    )
                )
        run_summary = np.array(
            run_list,
            dtype=np.dtype(
                [
                    ("id", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("components", "U100"),
                    ("measurement_type", "U12"),
                    ("sample_rate", float),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(run_summary)

    def make_run_name(self, alphabet: bool = False) -> str | None:
        """Generate the next run name using an alphabetic or numeric suffix.

        Parameters
        ----------
        alphabet : bool, default False
            If ``True`` use letters (``a``, ``b``, ...); otherwise use
            numeric suffixes (``001``).

        Returns
        -------
        str or None
            Proposed run name or ``None`` if generation fails.

        Examples
        --------
        >>> station.metadata.id = "MT001"
        >>> station.make_run_name()
        'MT001a'
        """

        run_list = sorted(
            [group[-1:] for group in self.groups_list if self.name in group]
        )

        next_letter = None
        if len(run_list) == 0:
            if alphabet:
                next_letter = "a"
            else:
                next_letter = "001"
        else:
            try:
                next_letter = chr(ord(run_list[-1]) + 1)
            except TypeError:
                try:
                    next_letter = f"{int(run_list[-1]) + 1}"
                except ValueError:
                    self.logger.info("Could not create a new run name")
        return next_letter

    def locate_run(self, sample_rate: float, start: str | MTime) -> pd.DataFrame | None:
        """Locate runs matching a sample rate and start time.

        Parameters
        ----------
        sample_rate : float
            Sample rate in samples per second.
        start : str or MTime
            Start time string or ``MTime`` instance.

        Returns
        -------
        pandas.DataFrame or None
            Matching rows from ``run_summary`` or ``None`` when no match exists.

        Examples
        --------
        >>> station.locate_run(256.0, "2020-01-01T00:00:00")  # doctest: +SKIP
        """

        if not isinstance(start, MTime):
            start = MTime(time_stamp=start)

        run_summary = self.run_summary.copy()
        if run_summary.size < 1:
            return None
        sr_find = run_summary[
            (run_summary.sample_rate == sample_rate) & (run_summary.start == start)
        ]
        if sr_find.size < 1:
            return None
        return sr_find

    def add_run(
        self, run_name: str, run_metadata: metadata.Run | None = None
    ) -> RunGroup:
        """Add a run under this station.

        Parameters
        ----------
        run_name : str
            Run identifier (for example ``id`` + suffix).
        run_metadata : mt_metadata.timeseries.Run, optional
            Metadata container to seed the run attributes.

        Returns
        -------
        RunGroup
            Wrapper for the created or existing run.

        Examples
        --------
        >>> run = station.add_run("MT001a")
        >>> run.metadata.id
        'MT001a'
        """

        return self._add_group(
            run_name, RunGroup, group_metadata=run_metadata, match="id"
        )

    def get_run(self, run_name: str) -> RunGroup:
        """Return a run by name.

        Parameters
        ----------
        run_name : str
            Existing run name.

        Returns
        -------
        RunGroup
            Wrapper for the requested run.

        Raises
        ------
        MTH5Error
            If the run does not exist.

        Examples
        --------
        >>> existing_run = station.get_run("MT001a")
        >>> existing_run.name
        'MT001a'
        """

        return self._get_group(run_name, RunGroup)

    def remove_run(self, run_name: str) -> None:
        """Remove a run from this station.

        Parameters
        ----------
        run_name : str
            Existing run name.

        Notes
        -----
        Deleting removes the reference only; storage is not reclaimed.

        Examples
        --------
        >>> station.remove_run("MT001a")
        """

        self._remove_group(run_name)

    def update_station_metadata(self) -> None:
        """Deprecated alias for :py:meth:`update_metadata`.

        Raises
        ------
        DeprecationWarning
            Always raised to direct callers to ``update_metadata``.

        Examples
        --------
        >>> station.update_station_metadata()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        DeprecationWarning: 'update_station_metadata' has been deprecated use 'update_metadata()'
        """
        raise DeprecationWarning(
            "'update_station_metadata' has been deprecated use 'update_metadata()'"
        )

    def update_metadata(self) -> None:
        """Synchronize station metadata from contained runs.

        Notes
        -----
        The station ``time_period`` is set to the min/max of all runs, and
        ``channels_recorded`` combines all recorded components.

        Examples
        --------
        >>> _ = station.update_metadata()
        >>> station.metadata.time_period.start  # doctest: +SKIP
        '2020-01-01T00:00:00'
        """

        run_summary = self.run_summary.copy()
        self._metadata.time_period.start = run_summary.start.min().isoformat()
        self._metadata.time_period.end = run_summary.end.max().isoformat()
        self._metadata.channels_recorded = list(
            set(",".join(run_summary.components.to_list()).split(","))
        )

        self.write_metadata()
