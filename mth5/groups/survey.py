# -*- coding: utf-8 -*-
from __future__ import annotations


"""Survey-level HDF5 helpers for MTH5."""

from typing import Any

import h5py

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from mt_metadata.timeseries import Survey

from mth5.groups import (
    BaseGroup,
    FiltersGroup,
    MasterStationGroup,
    ReportsGroup,
    StandardsGroup,
)
from mth5.helpers import to_numpy_type, validate_name
from mth5.utils.exceptions import MTH5Error


# =============================================================================
# Survey Group
# =============================================================================
class MasterSurveyGroup(BaseGroup):
    """Collection helper for surveys under ``Experiment/Surveys``.

    Provides helpers to add, fetch, or remove surveys and to summarize all
    channels in the experiment.

    Examples
    --------
    >>> from mth5 import mth5
    >>> m5 = mth5.MTH5()
    >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
    >>> surveys = m5.surveys_group
    >>> _ = surveys.add_survey("survey_01")
    >>> surveys.channel_summary.head()  # doctest: +SKIP
    """

    def __init__(self, group: h5py.Group, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)

    @property
    def channel_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all channels across surveys.

        Returns
        -------
        pandas.DataFrame
            Columns include survey, station, run, location, component,
            start/end, sample info, orientation, units, and HDF5 reference.

        Examples
        --------
        >>> summary = surveys.channel_summary
        >>> set(summary.columns) >= {"survey", "station", "run", "component"}
        True
        """
        ch_list = []
        for survey in self.groups_list:
            survey_group = self.get_survey(survey)
            for station in survey_group.stations_group.groups_list:
                station_group = survey_group.stations_group.get_station(station)
                for run in station_group.groups_list:
                    run_group = station_group.get_run(run)
                    for ch in run_group.groups_list:
                        ch_dataset = run_group.get_channel(ch)
                        entry = np.array(
                            [
                                (
                                    survey_group.metadata.id,
                                    station_group.metadata.id,
                                    run_group.metadata.id,
                                    station_group.metadata.location.latitude,
                                    station_group.metadata.location.longitude,
                                    station_group.metadata.location.elevation,
                                    ch_dataset.metadata.component,
                                    ch_dataset.metadata.time_period.start,
                                    ch_dataset.metadata.time_period.end,
                                    ch_dataset.hdf5_dataset.size,
                                    ch_dataset.metadata.sample_rate,
                                    ch_dataset.metadata.type,
                                    ch_dataset.metadata.measurement_azimuth,
                                    ch_dataset.metadata.measurement_tilt,
                                    ch_dataset.metadata.units,
                                    ch_dataset.hdf5_dataset.ref,
                                )
                            ],
                            dtype=np.dtype(
                                [
                                    ("survey", "U10"),
                                    ("station", "U10"),
                                    ("run", "U11"),
                                    ("latitude", float),
                                    ("longitude", float),
                                    ("elevation", float),
                                    ("component", "U20"),
                                    ("start", "datetime64[ns]"),
                                    ("end", "datetime64[ns]"),
                                    ("n_samples", int),
                                    ("sample_rate", float),
                                    ("measurement_type", "U12"),
                                    ("azimuth", float),
                                    ("tilt", float),
                                    ("units", "U25"),
                                    ("hdf5_reference", h5py.ref_dtype),
                                ]
                            ),
                        )
                        ch_list.append(entry)
        ch_list = np.array(ch_list)
        return pd.DataFrame(ch_list.flatten())

    def add_survey(
        self, survey_name: str, survey_metadata: Survey | None = None
    ) -> "SurveyGroup":
        """Add or fetch a survey at ``/Experiment/Surveys/<name>``.

        Parameters
        ----------
        survey_name : str
            Survey identifier; validated with ``validate_name``.
        survey_metadata : Survey, optional
            Metadata container used to seed the survey attributes.

        Returns
        -------
        SurveyGroup
            Wrapper for the created or existing survey.

        Raises
        ------
        ValueError
            If ``survey_name`` is empty.
        MTH5Error
            If the provided metadata id conflicts with the group name.

        Examples
        --------
        >>> survey = surveys.add_survey("survey_01")
        >>> survey.metadata.id
        'survey_01'
        """
        if not survey_name:
            raise ValueError("survey name is None, do not know what to name it")
        survey_name = validate_name(survey_name)
        try:
            survey_group = self.hdf5_group.create_group(survey_name)
            self.logger.debug(f"Created group {survey_group.name}")

            if survey_metadata is None:
                survey_metadata = Survey(id=survey_name)
            else:
                if validate_name(survey_metadata.id) != survey_name:
                    msg = (
                        f"survey group name {survey_name} must be same as "
                        f"survey id {survey_metadata.id.replace(' ', '_')}"
                    )
                    self.logger.error(msg)
                    raise MTH5Error(msg)
            survey_obj = SurveyGroup(
                survey_group,
                survey_metadata=survey_metadata,
                **self.dataset_options,
            )
            survey_obj.initialize_group()
        except ValueError:
            msg = f"survey {survey_name} already exists, returning existing group."
            self.logger.info(msg)
            survey_obj = self.get_survey(survey_name)
        return survey_obj

    def get_survey(self, survey_name: str) -> "SurveyGroup":
        """Return an existing survey by name.

        Parameters
        ----------
        survey_name : str
            Existing survey name.

        Returns
        -------
        SurveyGroup
            Wrapper for the requested survey.

        Raises
        ------
        MTH5Error
            If the survey does not exist.

        Examples
        --------
        >>> existing = surveys.get_survey("survey_01")
        >>> existing.metadata.id
        'survey_01'
        """

        survey_name = validate_name(survey_name)

        try:
            return SurveyGroup(self.hdf5_group[survey_name], **self.dataset_options)
        except KeyError:
            msg = (
                f"{survey_name} does not exist, "
                + "check survey_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)

    def remove_survey(self, survey_name: str) -> None:
        """Delete a survey reference from the file.

        Parameters
        ----------
        survey_name : str
            Existing survey name.

        Notes
        -----
        HDF5 deletion removes the reference only; storage is not reclaimed.

        Examples
        --------
        >>> surveys.remove_survey("survey_01")
        """

        survey_name = validate_name(survey_name)

        try:
            del self.hdf5_group[survey_name]
            self.logger.info(
                "Deleting a survey does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = f"{survey_name} does not exist, check survey_list for existing names"
            self.logger.exception(msg)
            raise MTH5Error(msg)


class SurveyGroup(BaseGroup):
    """Wrapper for a single survey at ``Experiment/Surveys/<id>``.

    Handles survey-level metadata, child groups (stations, reports, filters,
    standards), and synchronization utilities.

    Examples
    --------
    >>> survey = surveys.add_survey("survey_01")
    >>> survey.metadata.id
    'survey_01'
    """

    def __init__(
        self,
        group: h5py.Group,
        survey_metadata: Survey | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(group, group_metadata=survey_metadata, **kwargs)

        self._default_subgroup_names = [
            "Stations",
            "Reports",
            "Filters",
            "Standards",
        ]

    def initialize_group(self, **kwargs: Any) -> None:
        """Create default subgroups and write survey metadata.

        Parameters
        ----------
        **kwargs
            Additional attributes to set on the instance before initialization.

        Examples
        --------
        >>> survey.initialize_group()
        """
        # need to make groups first because metadata pulls from them.
        for group_name in self._default_subgroup_names:
            self.hdf5_group.create_group(f"{group_name}")
            m5_grp = getattr(self, f"{group_name.lower()}_group")
            m5_grp.initialize_group()

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.write_metadata()

    @BaseGroup.metadata.getter
    def metadata(self) -> Survey:
        """Survey metadata enriched with station and filter information."""

        if not self._has_read_metadata:
            self.read_metadata()
            self._has_read_metadata = True

        try:
            if self.stations_group.groups_list != self._metadata.station_names:
                for key in self.stations_group.groups_list:
                    try:
                        key_group = self.stations_group.get_station(key)
                        if key_group.metadata.id in self._metadata.stations.keys():
                            continue
                        # skip non-station groups like Features, FCs, TransferFunction
                        elif key_group.metadata.mth5_type.lower() not in ["station"]:
                            continue
                        self._metadata.add_station(key_group.metadata)
                    except MTH5Error:
                        self.logger.warning(f"Could not find station {key}")
        except KeyError:
            self.logger.debug(
                "Stations Group does not exists yet. Metadata contains no station information"
            )

        try:
            filters_group = self.filters_group
            if list(filters_group.filter_dict.keys()) != list(
                self._metadata.filters.keys()
            ):
                for key in self.filters_group.filter_dict.keys():
                    try:
                        if key in self._metadata.filters.keys():
                            continue
                        filter_obj = filters_group.to_filter_object(key)
                        self._metadata.filters[key] = filter_obj
                    except MTH5Error:
                        self.logger.warning(f"Could not find filter {key}")

        except KeyError:
            self.logger.debug(
                "Filters Group does not exists yet. Metadata contains no filter information"
            )
        return self._metadata

    def write_metadata(self) -> None:
        """Write HDF5 attributes from the survey metadata object."""

        try:
            for key, value in self._metadata.to_dict(single=True).items():
                value = to_numpy_type(value)
                self.logger.debug(f"wrote metadata {key} = {value}")
                self.hdf5_group.attrs.create(key, value)
            self._has_read_metadata = True
        except KeyError as key_error:
            if "no write intent" in str(key_error):
                self.logger.warning("File is in read-only mode, cannot write metadata.")
            else:
                raise KeyError(key_error)
        except ValueError as value_error:
            if "Unable to synchronously create group" in str(value_error):
                self.logger.warning("File is in read-only mode, cannot write metadata.")
            else:
                raise ValueError(value_error)

    @property
    def stations_group(self) -> MasterStationGroup:
        return MasterStationGroup(self.hdf5_group["Stations"])

    @property
    def filters_group(self) -> FiltersGroup:
        """Convenience accessor for ``/Survey/Filters`` group."""
        return FiltersGroup(self.hdf5_group["Filters"], **self.dataset_options)

    @property
    def reports_group(self) -> ReportsGroup:
        """Convenience accessor for ``/Survey/Reports`` group."""
        return ReportsGroup(self.hdf5_group["Reports"], **self.dataset_options)

    @property
    def standards_group(self) -> StandardsGroup:
        """Convenience accessor for ``/Survey/Standards`` group."""
        return StandardsGroup(self.hdf5_group["Standards"], **self.dataset_options)

    def update_survey_metadata(self, survey_dict: dict[str, Any] | None = None) -> None:
        """Deprecated alias for :py:meth:`update_metadata`.

        Raises
        ------
        DeprecationWarning
            Always raised to direct callers to ``update_metadata``.

        Examples
        --------
        >>> survey.update_survey_metadata()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        DeprecationWarning: 'update_survey_metadata' has been deprecated use 'update_metadata()'
        """

        raise DeprecationWarning(
            "'update_survey_metadata' has been deprecated use 'update_metadata()'"
        )

    def update_metadata(self, survey_dict: dict[str, Any] | None = None) -> None:
        """Synchronize survey metadata from station summaries.

        Parameters
        ----------
        survey_dict : dict, optional
            Additional metadata values to merge before synchronization.

        Notes
        -----
        Updates survey start/end dates and bounding box from station summaries,
        then writes metadata to HDF5.

        Examples
        --------
        >>> _ = survey.update_metadata()
        >>> survey.metadata.time_period.start_date  # doctest: +SKIP
        '2020-01-01'
        """

        station_summary = self.stations_group.station_summary.copy()
        self.logger.debug("Updating survey metadata from stations summary table")

        if survey_dict:
            self.metadata.from_dict(survey_dict, skip_none=True)

        if not len(station_summary):  # if station info is empty df, skip parsing
            self.write_metadata()
            return

        self._metadata.time_period.start_date = (
            station_summary.start.min().isoformat().split("T")[0]
        )
        self._metadata.time_period.end_date = (
            station_summary.end.max().isoformat().split("T")[0]
        )
        self._metadata.northwest_corner.latitude = station_summary.latitude.max()
        self._metadata.northwest_corner.longitude = station_summary.longitude.min()
        self._metadata.southeast_corner.latitude = station_summary.latitude.min()
        self._metadata.southeast_corner.longitude = station_summary.longitude.max()

        # metadata by default comes with stations and runs, need to remove those
        # before writing the metadata.
        self.write_metadata()
