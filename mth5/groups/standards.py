# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:05:33 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np
from mt_metadata import timeseries
from mt_metadata.base import BaseDict
from mt_metadata.timeseries import filters
from mt_metadata.utils.summarize import summarize_standards
from mt_metadata.utils.validators import validate_attribute

from mth5 import STANDARDS_DTYPE
from mth5.groups.base import BaseGroup
from mth5.tables import MTH5Table
from mth5.utils.exceptions import MTH5TableError


ts_classes = dict(inspect.getmembers(timeseries, inspect.isclass))
flt_classes = dict(inspect.getmembers(filters, inspect.isclass))


# =============================================================================
# Summarize standards
# =============================================================================
def summarize_metadata_standards() -> BaseDict:
    """
    Summarize metadata standards into a dictionary.

    Aggregates metadata standard definitions from timeseries and filter
    classes, creating a flattened dictionary suitable for storage in
    the standards summary table.

    Returns
    -------
    BaseDict
        Flattened dictionary containing metadata standards for all supported
        classes (Survey, Station, Run, Electric, Magnetic, Auxiliary,
        and various Filter types).

    Notes
    -----
    Creates copies of attribute dictionaries to avoid mutations to the
    original class definitions.

    Examples
    --------
    >>> standards = summarize_metadata_standards()
    >>> 'survey' in standards
    True
    >>> 'electric' in standards
    True
    """

    # need to be sure to make copies otherwise things will get
    # added in not great places.
    summary_dict = BaseDict()
    for key in [
        "survey",
        "station",
        "run",
        "electric",
        "magnetic",
        "auxiliary",
    ]:
        obj = ts_classes[key.capitalize()]()
        summary_dict.add_dict(obj._attr_dict.copy(), key)
    for key in [
        "Coefficient",
        "FIR",
        "FrequencyResponseTable",
        "PoleZero",
        "TimeDelay",
    ]:
        key += "Filter"
        obj = flt_classes[key]()
        summary_dict.add_dict(obj._attr_dict.copy(), validate_attribute(key))
    return summary_dict


# =============================================================================
# Standards Group
# =============================================================================


class StandardsGroup(BaseGroup):
    """
    Container for metadata standards documentation stored in the HDF5 file.

    Stores metadata standards used throughout the survey in a standardized
    summary table. This enables users to understand metadata directly from
    the file without requiring external documentation.

    The standards are organized in a summary table at ``/Survey/Standards/summary``
    with columns for attribute name, type, requirements, style, units, and
    descriptions.

    Attributes
    ----------
    summary_table : MTH5Table
        The standards summary table with metadata definitions.

    Notes
    -----
    Standards include definitions for:

    - Survey, Station, Run, Electric, Magnetic, Auxiliary metadata
    - Filter types: Coefficient, FIR, FrequencyResponseTable, PoleZero, TimeDelay
    - Processing standards from aurora and fourier_coefficients modules

    Examples
    --------
    >>> with MTH5('survey.mth5') as mth5_obj:
    ...     standards = mth5_obj.standards_group
    ...     summary = standards.summary_table
    ...     print(summary.array.dtype.names)
    ('attribute', 'type', 'required', 'style', 'units', 'description', ...)

    Get information about a specific attribute:

    >>> standards.get_attribute_information('survey.release_license')
    survey.release_license
    --------------------------
            type          : string
            required      : True
            style         : controlled vocabulary
            ...
    """

    def __init__(self, group: Any, **kwargs: Any) -> None:
        """
        Initialize StandardsGroup.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to manage standards data.
        **kwargs : Any
            Additional keyword arguments passed to BaseGroup.
        """
        super().__init__(group, **kwargs)

        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (1000,),
            "dtype": STANDARDS_DTYPE,
        }

        self._modules = [
            "common",
            "timeseries",
            "timeseries.filters",
            "transfer_functions.tf",
            "features",
            "features.weights",
            "processing",
            "processing.fourier_coefficients",
            "processing.aurora",
        ]

    @property
    def summary_table(self) -> MTH5Table:
        return self._get_summary_table()

    def _get_summary_table(self) -> MTH5Table:
        """
        Get the standards summary table from HDF5.

        Returns
        -------
        MTH5Table
            The MTH5Table object wrapping the standards summary dataset.
        """
        return MTH5Table(self.hdf5_group["summary"], STANDARDS_DTYPE)

    def get_attribute_information(self, attribute_name: str) -> None:
        """
        Print detailed information about a metadata attribute.

        Retrieves and displays all metadata standards information for
        the specified attribute from the standards summary table.

        Parameters
        ----------
        attribute_name : str
            Name of the attribute to describe (e.g., 'survey.release_license').

        Raises
        ------
        MTH5TableError
            If the attribute is not found in the standards summary table.

        Notes
        -----
        Prints formatted output including:

        - Data type
        - Whether attribute is required
        - Style (e.g., controlled vocabulary)
        - Units
        - Description
        - Valid options
        - Aliases
        - Example values
        - Default value

        Examples
        --------
        >>> standards = mth5_obj.standards_group
        >>> standards.get_attribute_information('survey.release_license')
        survey.release_license
        --------------------------
                type          : string
                required      : True
                style         : controlled vocabulary
                units         :
                description   : How the data can be used. The options are based on
                         Creative Commons licenses.
                options       : CC-0,CC-BY,CC-BY-SA,CC-BY-ND,CC-BY-NC-SA
                alias         :
                example       : CC-0
                default       : CC-0
        """
        find = self.summary_table.locate("attribute", attribute_name)
        if len(find) == 0:
            msg = f"Could not find {attribute_name} in standards."
            self.logger.error(msg)
            raise MTH5TableError(msg)
        meta_item = self.summary_table.array[find]
        lines = ["", attribute_name, "-" * (len(attribute_name) + 4)]
        for name, value in zip(meta_item.dtype.names[1:], meta_item.item()[1:]):
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode()
            lines.append("\t{0:<14} {1}".format(name + ":", value))
        print("\n".join(lines))

    def summary_table_from_dict(self, summary_dict: dict[str, Any]) -> None:
        """
        Populate summary table from a dictionary of metadata standards.

        Converts a flattened dictionary of metadata standards into rows
        in the HDF5 summary table.

        Parameters
        ----------
        summary_dict : dict[str, Any]
            Flattened dictionary of all metadata standards. Keys are
            attribute names, values are dictionaries with type, required,
            style, units, description, etc.

        Notes
        -----
        Processes dictionary values:

        - Lists are converted to comma-separated strings
        - None values become empty strings
        - Bytes are decoded to UTF-8

        TODO
        ----
        Adapt method to accept pandas.DataFrame as alternative input.

        Examples
        --------
        >>> standards = StandardsGroup(group)
        >>> metadata = summarize_metadata_standards()
        >>> standards.summary_table_from_dict(metadata)
        """

        for key, v_dict in summary_dict.items():
            key_list = [key]
            for dkey in self.summary_table.dtype.names[1:]:
                value = v_dict[dkey]

                if isinstance(value, list):
                    if len(value) == 0:
                        value = ""
                    else:
                        value = ",".join(["{0}".format(ii) for ii in value])
                if value is None:
                    value = ""
                key_list.append(value)
            key_list = np.array([tuple(key_list)], self.summary_table.dtype)
            index = self.summary_table.add_row(key_list)
        self.logger.debug(f"Added {index} rows to Standards Group")

    def get_standards_summary(self, modules: Optional[list[str]] = None) -> np.ndarray:
        """
        Get standards for specified metadata modules.

        Retrieves and concatenates standards arrays from one or more
        metadata modules for inclusion in the standards table.

        Parameters
        ----------
        modules : list[str], optional
            List of module names to include (e.g., 'timeseries', 'filters').
            If None, uses default modules: common, timeseries, timeseries.filters,
            transfer_functions.tf, features, features.weights, processing,
            processing.fourier_coefficients, processing.aurora.
            Default is None.

        Returns
        -------
        np.ndarray
            Concatenated numpy structured array containing standards for all
            requested modules with dtype matching STANDARDS_DTYPE.

        Examples
        --------
        >>> standards = StandardsGroup(group)
        >>> ts_standards = standards.get_standards_summary(['timeseries'])
        >>> print(ts_standards.shape)
        (45,)

        Get all default modules:

        >>> all_standards = standards.get_standards_summary()
        """
        if modules is None:
            modules = self._modules

        summaries = []
        for module in modules:
            summaries.append(
                summarize_standards(module, output_type="array", dtype=STANDARDS_DTYPE)
            )

        return np.concatenate(summaries)

    def summary_table_from_array(self, array: np.ndarray) -> None:
        """
        Populate summary table from a numpy structured array.

        Converts a structured numpy array into rows in the HDF5 summary table.

        Parameters
        ----------
        array : np.ndarray
            Structured numpy array with dtype matching STANDARDS_DTYPE.
            Each row represents one metadata attribute definition.

        Notes
        -----
        Iterates through all rows of the structured array and adds them
        sequentially to the summary table using add_row().

        Examples
        --------
        >>> standards = StandardsGroup(group)
        >>> standards_array = standards.get_standards_summary()
        >>> standards.summary_table_from_array(standards_array)
        """
        summary_table = self._get_summary_table()

        for index, row in enumerate(np.nditer(array)):
            index = summary_table.add_row(row)
        self.logger.debug(f"Added {index} rows to Standards Group")

    def initialize_group(self) -> None:
        """
        Initialize the standards group and create the summary table.

        Creates the summary table dataset in the HDF5 file and populates it
        with metadata standards from all default modules. Sets appropriate
        HDF5 attributes and writes the group metadata.

        Notes
        -----
        Initialization process:

        1. Creates HDF5 dataset for summary table with maximum expandable shape
        2. Applies compression if configured in dataset_options
        3. Sets HDF5 attributes: type, last_updated, reference
        4. Populates table with standards from all default modules
        5. Writes group metadata to HDF5

        The summary table uses STANDARDS_DTYPE and supports up to 1000 rows.

        Examples
        --------
        >>> mth5_obj.initialize_group()
        >>> summary_table = mth5_obj.standards_group.summary_table
        >>> print(summary_table.array.shape)
        (342,)
        """
        if self.dataset_options["compression"] is None:
            summary_dataset = self.hdf5_group.create_dataset(
                self._defaults_summary_attrs["name"],
                (0,),
                maxshape=self._defaults_summary_attrs["max_shape"],
                dtype=self._defaults_summary_attrs["dtype"],
            )
        else:
            summary_dataset = self.hdf5_group.create_dataset(
                self._defaults_summary_attrs["name"],
                (0,),
                maxshape=self._defaults_summary_attrs["max_shape"],
                dtype=self._defaults_summary_attrs["dtype"],
                **self.dataset_options,
            )
        summary_dataset.attrs.update(
            {
                "type": "summary table",
                "last_updated": "date_time",
                "reference": summary_dataset.ref,
            }
        )

        self.logger.debug(
            f"Created {self._defaults_summary_attrs['name']} table with "
            f"max_shape = {self._defaults_summary_attrs['max_shape']}, "
            "dtype={self._defaults_summary_attrs['dtype']}"
        )
        self.logger.debug(
            "used options: "
            "; ".join([f"{k} = {v}" for k, v in self.dataset_options.items()])
        )

        self.summary_table_from_array(self.get_standards_summary())

        self.write_metadata()
