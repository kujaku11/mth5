# -*- coding: utf-8 -*-
"""
MTH5 table utilities.

This module provides the `MTH5Table` base class which wraps an HDF5 dataset
and offers convenience methods for row management, locating entries, and
exporting to `pandas.DataFrame`.

Notes
-----
- Designed as a thin layer on top of NumPy/HDF5; for complex querying, prefer
    converting to a DataFrame via `to_dataframe()`.
- Datatypes are validated and kept consistent with the underlying dataset.

"""
from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import weakref
from typing import Any, cast, Literal

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from mth5.utils.exceptions import MTH5TableError


# =============================================================================
# MTH5 Table Class
# =============================================================================


class MTH5Table:
    """
    Base wrapper around an HDF5 dataset representing a typed table.

    Provides simple NumPy-based operations including row insertion/removal,
    basic locating utilities, and conversion to `pandas.DataFrame`.

    Parameters
    ----------
    hdf5_dataset : h5py.Dataset
        The HDF5 dataset that stores the table.
    default_dtype : numpy.dtype
        The default dtype schema for the table entries.

    Raises
    ------
    MTH5TableError
        If `hdf5_dataset` is not an instance of `h5py.Dataset`.

    Examples
    --------
    Create a simple table and add a row::

        >>> import h5py, numpy as np
        >>> f = h5py.File('example.h5', 'w')
        >>> dtype = np.dtype([('name', 'S16'), ('value', 'f8')])
        >>> ds = f.create_dataset('table', (1,), maxshape=(None,), dtype=dtype)
        >>> from mth5.tables.mth5_table import MTH5Table
        >>> t = MTH5Table(ds, dtype)
        >>> row = np.array([('alpha'.encode('utf-8'), 1.23)], dtype=dtype)
        >>> t.add_row(row)
        1
        >>> df = t.to_dataframe()
        >>> df.head()

    """

    def __init__(self, hdf5_dataset: h5py.Dataset, default_dtype: np.dtype) -> None:
        self.logger = logger
        self._default_dtype = self._validate_dtype(default_dtype)

        # validate dtype with dataset
        if isinstance(hdf5_dataset, h5py.Dataset):
            # Use a weak reference to the dataset and ensure it's valid
            _ref = weakref.ref(hdf5_dataset)()
            if _ref is None:
                raise MTH5TableError("Dataset reference is not available.")
            self.array: h5py.Dataset = cast(h5py.Dataset, _ref)
            if self.array.dtype != self._default_dtype:
                self.update_dtype(self._default_dtype)
        else:
            msg = f"Input must be a h5py.Dataset not {type(hdf5_dataset)}"
            self.logger.error(msg)
            raise MTH5TableError(msg)

    def __str__(self) -> str:
        """
        Return a string representation of the table contents.

        Returns
        -------
        str
            A string representation of the table's DataFrame contents or an
            empty string if the table is empty.
        """
        # if the array is empty
        if getattr(self.array, "size", 0) > 0:
            df = self.to_dataframe()

            return df.__str__()
        return ""

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: MTH5Table | h5py.Dataset | object) -> bool:
        if isinstance(other, MTH5Table):
            return self.array == other.array
        elif isinstance(other, h5py.Dataset):
            return self.array == other
        else:
            msg = f"Cannot compare type={type(other)}"
            self.logger.error(msg)
            raise TypeError(msg)

    def __ne__(self, other: MTH5Table | h5py.Dataset | object) -> bool:
        return not self.__eq__(other)

    def __len__(self) -> int:
        return self.array.shape[0]

    @property
    def hdf5_reference(self) -> object:
        return getattr(self.array, "ref", None)

    @property
    def dtype(self) -> np.dtype:
        return self._default_dtype

    @dtype.setter
    def dtype(self, value: np.dtype) -> None:
        """
        Set the table dtype, updating the underlying dataset if it differs.

        Parameters
        ----------
        value : numpy.dtype
            New dtype to apply. Must match the existing field names.

        Raises
        ------
        TypeError
            If `value` is not an instance of `numpy.dtype`.
        """

        if not isinstance(value, np.dtype):
            raise TypeError(f"Input dtype must be np.dtype not type {type(value)}")

        if value != self._default_dtype:
            self.update_dtype(value)

    def _validate_dtype(self, value: np.dtype) -> np.dtype:
        """
        Validate that `value` is a `numpy.dtype`.

        Parameters
        ----------
        value : numpy.dtype
            Dtype to validate.

        Returns
        -------
        numpy.dtype
            The validated dtype.

        Raises
        ------
        TypeError
            If `value` is not a `numpy.dtype`.
        """
        if not isinstance(value, np.dtype):
            msg = f"Input dtype must be np.dtype not type {type(value)}"
            self.logger.exception(msg)
            raise TypeError(msg)
        return value

    def _validate_dtype_names(self, value: np.dtype) -> np.dtype:
        if self.dtype.names != value.names:
            msg = f"New dtype must have the same names: {self.dtype.names}"
            self.logger.exception(msg)
            raise TypeError(msg)

        return value

    def check_dtypes(self, other_dtype: np.dtype) -> bool:
        """
        Check that dtypes match the table's dtype (including field names).

        Parameters
        ----------
        other_dtype : numpy.dtype
            The dtype to compare against the table's dtype.

        Returns
        -------
        bool
            True if the dtypes match; otherwise False.
        """
        other_dtype = self._validate_dtype(other_dtype)
        try:
            other_dtype = self._validate_dtype_names(other_dtype)
        except TypeError:
            return False
        if self.dtype == other_dtype:
            return True
        return False

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def nrows(self) -> int:
        return self.array.shape[0]

    def locate(
        self,
        column: str,
        value: Any,
        test: Literal["eq", "lt", "le", "gt", "ge", "be", "bt"] = "eq",
    ) -> np.ndarray:
        """
        Locate row indices where a column satisfies a comparison.

        Parameters
        ----------
        column : str
            Name of the column to test.
        value : Any
            Value to compare against. For string columns, a `str` is converted
            to a `numpy.bytes_`. For time columns (`start`, `end`,
            `start_date`, `end_date`), values are coerced to `numpy.datetime64`.
        test : {'eq','lt','le','gt','ge','be','bt'}, default 'eq'
            Type of comparison to perform.
            - 'eq': equals
            - 'lt': less than
            - 'le': less than or equal to
            - 'gt': greater than
            - 'ge': greater than or equal to
            - 'be': strictly between
            - 'bt': alias for 'be'

        Returns
        -------
        numpy.ndarray
            Array of matching row indices.

        Raises
        ------
        ValueError
            If `test` is 'be'/'bt' and `value` is not a 2-length iterable.

        Examples
        --------
        Find rows with value greater than 10::

            >>> idx = t.locate('value', 10, test='gt')
        """
        if isinstance(value, str):
            value = np.bytes_(value)
        # use numpy datetime for testing against time.
        if column in ["start", "end", "start_date", "end_date"]:
            test_array = self.array[column].astype(np.datetime64)
            value = np.datetime64(value)
        else:
            test_array = self.array[column]
        if test == "eq":
            index_values = np.where(test_array == value)[0]
        elif test == "lt":
            index_values = np.where(test_array < value)[0]
        elif test == "le":
            index_values = np.where(test_array <= value)[0]
        elif test == "gt":
            index_values = np.where(test_array > value)[0]
        elif test == "ge":
            index_values = np.where(test_array >= value)[0]
        elif test == "be":
            if not isinstance(value, (list, tuple, np.ndarray)):
                msg = "If testing for between value must be an iterable of length 2."
                self.logger.error(msg)
                raise ValueError(msg)
            index_values = np.where((test_array > value[0]) & (test_array < value[1]))[
                0
            ]
        else:
            raise ValueError("Test {0} not understood".format(test))
        return index_values

    def add_row(self, row: np.ndarray, index: int | None = None) -> int:
        """
        Add a row to the table.

        Parameters
        ----------
        row : numpy.ndarray
            Row to insert. Must have the same dtype (or same field names,
            allowing safe casting) as the table.
        index : int, optional
            Index at which to insert the row. If None, appends to the end.

        Returns
        -------
        int
            Index of the inserted row.

        Raises
        ------
        TypeError
            If `row` is not a `numpy.ndarray`.
        ValueError
            If the dtype is incompatible with the table.
        """

        if not isinstance(row, (np.ndarray)):
            msg = f"Input must be an numpy.ndarray not {type(row)}"
            self.logger.exception(msg)
            raise TypeError(msg)
        if isinstance(row, np.ndarray):
            if not self.check_dtypes(row.dtype):
                if row.dtype.names == self.dtype.names:
                    row = row.astype(self.dtype)
                else:
                    msg = (
                        f"Data types are not equal. Input dtypes: "
                        f"{row.dtype} Table dtypes: {self.dtype}"
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
        if index is None:
            index = self.nrows
            if self.nrows == 1:
                match = True
                null_array = np.zeros(1, dtype=self.dtype)
                if self.dtype.names is None:
                    raise TypeError("Table dtype must have named fields.")
                for name in self.dtype.names:
                    if "reference" in name:
                        continue
                    if self.array[name][0] != null_array[name][0]:
                        match = False
                        break
                if match:
                    index = 0
                else:
                    new_shape = tuple([self.nrows + 1] + [ii for ii in self.shape[1:]])
                    self.array.resize(new_shape)
            else:
                new_shape = tuple([self.nrows + 1] + [ii for ii in self.shape[1:]])
                self.array.resize(new_shape)
        # add the row
        self.array[index] = row
        self.logger.debug(f"Added row as index {index} with values {row}")

        return index

    def update_row(self, entry: np.ndarray) -> int:
        """
        Update a row by locating its index and rewriting the entry.

        Parameters
        ----------
        entry : numpy.ndarray
            Entry to update, with the same dtype as the table.

        Returns
        -------
        int
            Row index that was updated, or the new row index if not found.

        Notes
        -----
        Matching by `hdf5_reference` is not reliable; this uses `add_row`
        and will append if the original row cannot be located.
        """
        try:
            row_index = self.locate("hdf5_reference", entry["hdf5_reference"])[0]
            return self.add_row(entry, index=row_index)
        except IndexError:
            self.logger.debug("Could not find row, adding a new one")
            return self.add_row(entry)

    def remove_row(self, index: int) -> int:
        """
        Remove a row by replacing it with a null entry.

        Parameters
        ----------
        index : int
            Index of the row to remove.

        Returns
        -------
        int
            Index that was updated with a null row.

        Raises
        ------
        IndexError
            If the index is out of bounds for the current shape.

        Notes
        -----
        - There is no intrinsic index stored within the array; indexing is
          on-the-fly. Prefer using the HDF5 reference column for robust
          identification.
        - The current approach inserts a null row at the specified index.
        """
        null_array = np.empty((1,), dtype=self.dtype)
        try:
            return self.add_row(null_array, index=index)
        except IndexError as error:
            msg = f"Could not find index {index} in shape {self.shape}"
            self.logger.exception(msg)
            raise IndexError(f"{error}\n{msg}")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the table into a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with decoded string columns where applicable.

        Examples
        --------
        Convert and preview::

            >>> df = t.to_dataframe()
            >>> df.head()
        """

        df = pd.DataFrame(self.array[()])
        if self.dtype.names is None:
            raise TypeError("Table dtype must have named fields.")
        fields = self.dtype.fields or {}
        for key in self.dtype.names:
            field_info = fields.get(cast(Any, key))
            if field_info is None:
                continue
            dtype_kind = field_info[0].kind
            if dtype_kind in ["S", "U"]:
                setattr(df, key, getattr(df, key).str.decode("utf-8"))

        return df

    def clear_table(self) -> None:
        """
        Reset the table by recreating the dataset with a single null row.

        Notes
        -----
        Deletes the current dataset and replaces it with a new dataset with
        the same compression/options and `dtype`, but shape `(1,)`.
        """

        root = self.array.parent
        if not isinstance(root, (h5py.Group, h5py.File)):
            raise TypeError("Unexpected parent type; expected Group or File.")
        name = str(self.array.name).split("/")[-1]
        ds_options = {
            "compression": self.array.compression,
            "compression_opts": self.array.compression_opts,
            "shuffle": self.array.shuffle,
            "fletcher32": self.array.fletcher32,
        }

        del root[name]

        self.array = root.create_dataset(
            name, (1,), maxshape=(None,), dtype=self.dtype, **ds_options
        )

    def update_dtype(self, new_dtype: np.dtype) -> None:
        """
        Update the dataset's dtype while preserving data and field names.

        Parameters
        ----------
        new_dtype : numpy.dtype
            New dtype to apply. Must have identical field names.

        Notes
        -----
        Performs a manual copy into a new array to avoid unsafe casting
        errors, then recreates the dataset with the new dtype and same
        dataset options.
        """

        try:
            new_dtype = self._validate_dtype_names(self._validate_dtype(new_dtype))

            # need to do this manually otherwise get an error of not safe
            new_array = np.ones(self.array.shape, dtype=new_dtype)
            for key in self.array.dtype.fields.keys():
                new_array[key] = self.array[key][()]

            root = self.array.parent
            if not isinstance(root, (h5py.Group, h5py.File)):
                raise TypeError("Unexpected parent type; expected Group or File.")
            name = str(self.array.name).split("/")[-1]
            ds_options = {
                "compression": self.array.compression,
                "compression_opts": self.array.compression_opts,
                "shuffle": self.array.shuffle,
                "fletcher32": self.array.fletcher32,
            }

            del root[name]

            self.array = root.create_dataset(
                name,
                data=new_array,
                maxshape=(None,),
                dtype=new_dtype,
                **ds_options,
            )

            self._default_dtype = new_dtype
        except:
            self.logger.info(
                "Could not update table dtype, likely an older file.  Clearing table."
            )
            self.clear_table()
