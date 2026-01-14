# -*- coding: utf-8 -*-
"""
Filter groups manager for handling multiple filter types in MTH5.

This module provides a unified interface for managing different types of filters
including zeros-poles-gain (ZPK), coefficients, time delays, frequency-amplitude-phase (FAP),
and finite impulse response (FIR) filters.

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from typing import Any

import h5py

from mth5.groups.base import BaseGroup
from mth5.groups.filter_groups import (
    CoefficientGroup,
    FAPGroup,
    FIRGroup,
    TimeDelayGroup,
    ZPKGroup,
)


# =============================================================================
# Filters Group
# =============================================================================


class FiltersGroup(BaseGroup):
    """
    Container for managing all filter types in MTH5 format.

    This class provides a unified interface for organizing and accessing filters
    of different types. It automatically creates and manages subgroups for each
    filter type (ZPK, Coefficient, Time Delay, FAP, and FIR) within the HDF5
    file structure.

    Filter Types
    -----------
    - **zpk**: Zeros, Poles, and Gain representation
    - **coefficient**: FIR coefficient filter
    - **time_delay**: Time delay filter
    - **fap**: Frequency-Amplitude-Phase (FAP) lookup table
    - **fir**: Finite Impulse Response filter

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for the filters container.
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Attributes
    ----------
    zpk_group : ZPKGroup
        Subgroup for zeros-poles-gain filters.
    coefficient_group : CoefficientGroup
        Subgroup for coefficient filters.
    time_delay_group : TimeDelayGroup
        Subgroup for time delay filters.
    fap_group : FAPGroup
        Subgroup for frequency-amplitude-phase filters.
    fir_group : FIRGroup
        Subgroup for FIR filters.

    Examples
    --------
    >>> import h5py
    >>> from mth5.groups.filters import FiltersGroup
    >>> with h5py.File('data.h5', 'r') as f:
    ...     filters = FiltersGroup(f['Filters'])
    ...     all_filters = filters.filter_dict
    ...     zpk_filter = filters.to_filter_object('my_zpk_filter')
    """

    def __init__(self, group: h5py.Group, **kwargs) -> None:
        super().__init__(group, **kwargs)

        try:
            self.zpk_group = ZPKGroup(self.hdf5_group.create_group("zpk"))
        except ValueError:
            self.zpk_group = ZPKGroup(self.hdf5_group["zpk"])
        try:
            self.coefficient_group = CoefficientGroup(
                self.hdf5_group.create_group("coefficient")
            )
        except ValueError:
            self.coefficient_group = CoefficientGroup(self.hdf5_group["coefficient"])
        try:
            self.time_delay_group = TimeDelayGroup(
                self.hdf5_group.create_group("time_delay")
            )
        except ValueError:
            self.time_delay_group = TimeDelayGroup(self.hdf5_group["time_delay"])
        try:
            self.fap_group = FAPGroup(self.hdf5_group.create_group("fap"))
        except ValueError:
            self.fap_group = FAPGroup(self.hdf5_group["fap"])
        try:
            self.fir_group = FIRGroup(self.hdf5_group.create_group("fir"))
        except ValueError:
            self.fir_group = FIRGroup(self.hdf5_group["fir"])

    @property
    def filter_dict(self) -> dict[str, Any]:
        """
        Get a dictionary of all filters across all filter type groups.

        Aggregates filters from all subgroups (ZPK, Coefficient, Time Delay, FAP, FIR)
        into a single dictionary for convenient access and querying.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping filter names to filter metadata dictionaries.
            Each entry contains filter information including type and HDF5 reference.

        Examples
        --------
        >>> filters = FiltersGroup(h5_group)
        >>> all_filters = filters.filter_dict
        >>> print(list(all_filters.keys()))
        ['my_zpk_filter', 'lowpass_coefficient', 'time_delay_1', ...]
        >>> print(all_filters['my_zpk_filter']['type'])
        'zpk'
        """
        filter_dict = {}
        filter_dict.update(self.zpk_group.filter_dict)
        filter_dict.update(self.coefficient_group.filter_dict)
        filter_dict.update(self.time_delay_group.filter_dict)
        filter_dict.update(self.fap_group.filter_dict)
        filter_dict.update(self.fir_group.filter_dict)

        return filter_dict

    def add_filter(self, filter_object: object) -> object:
        """
        Add a filter dataset based on its type.

        Automatically detects the filter type and routes the filter to the
        appropriate subgroup. Filter names are normalized to lowercase and
        forward slashes are replaced with " per " for consistency.

        Parameters
        ----------
        filter_object : mt_metadata.timeseries.filters
            An MT metadata filter object with a 'type' attribute.
            Supported types:

            - 'zpk', 'poles_zeros': Zeros-Poles-Gain filter
            - 'coefficient': Coefficient filter
            - 'time_delay', 'time delay': Time delay filter
            - 'fap', 'frequency response table': Frequency-Amplitude-Phase filter
            - 'fir': Finite Impulse Response filter

        Returns
        -------
        object
            Filter group object from the appropriate subgroup.

        Notes
        -----
        If a filter with the same name already exists, the existing filter
        is returned instead of creating a duplicate.

        Examples
        --------
        >>> from mt_metadata.timeseries.filters import ZPK
        >>> filters = FiltersGroup(h5_group)
        >>> zpk_filter = ZPK(name='my_filter')
        >>> added_filter = filters.add_filter(zpk_filter)

        Add coefficient filter:

        >>> from mt_metadata.timeseries.filters import Coefficient
        >>> coeff_filter = Coefficient(name='lowpass')
        >>> filters.add_filter(coeff_filter)
        """
        self.logger.debug(f"Type of filter {type(filter_object)}")
        # make everything lower case for consistency
        filter_object.name = filter_object.name.replace("/", " per ").lower()

        if filter_object.type in ["zpk", "poles_zeros"]:
            try:
                return self.zpk_group.from_object(filter_object)
            except ValueError:
                self.logger.debug(f"group {filter_object.name} already exists")
                return self.zpk_group.get_filter(filter_object.name)
        elif filter_object.type in ["coefficient"]:
            try:
                return self.coefficient_group.from_object(filter_object)
            except ValueError:
                self.logger.debug(f"group {filter_object.name} already exists")
                return self.coefficient_group.get_filter(filter_object.name)
        elif filter_object.type in ["time_delay", "time delay"]:
            try:
                return self.time_delay_group.from_object(filter_object)
            except ValueError:
                self.logger.debug(f"group {filter_object.name} already exists")
                return self.time_delay_group.get_filter(filter_object.name)
        elif filter_object.type in ["fap", "frequency response table"]:
            try:
                return self.fap_group.from_object(filter_object)
            except ValueError:
                self.logger.debug(f"group {filter_object.name} already exists")
                return self.fap_group.get_filter(filter_object.name)
        elif filter_object.type in ["fir"]:
            try:
                return self.fir_group.from_object(filter_object)
            except ValueError:
                self.logger.debug(f"group {filter_object.name} already exists")
                return self.fir_group.get_filter(filter_object.name)

    def get_filter(self, name: str) -> h5py.Dataset | h5py.Group:
        """
        Retrieve a filter dataset by name.

        Looks up the filter by name in the aggregated filter dictionary and
        returns the HDF5 dataset or group object.

        Parameters
        ----------
        name : str
            Name of the filter to retrieve.

        Returns
        -------
        h5py.Dataset or h5py.Group
            HDF5 dataset or group object for the requested filter.

        Raises
        ------
        KeyError
            If the filter name is not found in the filter dictionary.

        Examples
        --------
        >>> filters = FiltersGroup(h5_group)
        >>> filter_dataset = filters.get_filter('my_zpk_filter')
        >>> print(filter_dataset.attrs)
        """

        try:
            hdf5_ref = self.filter_dict[name]["hdf5_ref"]
        except KeyError:
            msg = f"Could not find {name} in the filter dictionary"
            self.logger.error(msg)
            raise KeyError(msg)
        return self.hdf5_group[hdf5_ref]

    def to_filter_object(self, name: str) -> object:
        """
        Convert a filter HDF5 dataset to an MT metadata filter object.

        Retrieves the filter metadata from the HDF5 file and converts it to
        the appropriate MT metadata filter class based on filter type.

        Parameters
        ----------
        name : str
            Name of the filter to convert.

        Returns
        -------
        object
            MT metadata filter object (ZPK, Coefficient, TimeDelay, FAP, or FIR).

        Raises
        ------
        KeyError
            If the filter name is not found in the filter dictionary.

        Examples
        --------
        >>> filters = FiltersGroup(h5_group)
        >>> zpk_filter = filters.to_filter_object('my_zpk_filter')
        >>> print(zpk_filter.name)
        'my_zpk_filter'
        >>> print(type(zpk_filter))
        <class 'mt_metadata.timeseries.filters.ZPK'>

        Get different filter types:

        >>> coeff_filter = filters.to_filter_object('lowpass_coefficient')
        >>> fap_filter = filters.to_filter_object('frequency_response_1')
        """

        try:
            f_type = self.filter_dict[name]["type"]
        except KeyError:
            msg = f"Could not find {name} in the filter dictionary"
            self.logger.error(msg)
            raise KeyError(msg, name)
        if f_type in ["zpk"]:
            return self.zpk_group.to_object(name)
        elif f_type in ["coefficient"]:
            return self.coefficient_group.to_object(name)
        elif f_type in ["time_delay", "time delay"]:
            return self.time_delay_group.to_object(name)
        elif f_type in ["fap", "frequency response table"]:
            return self.fap_group.to_object(name)
        elif f_type in ["fir"]:
            return self.fir_group.to_object(name)
