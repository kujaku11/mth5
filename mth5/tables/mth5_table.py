# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:53:55 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
import weakref
import copy

import h5py
import numpy as np
import pandas as pd

from mth5.utils.exceptions import MTH5TableError
from mth5.utils.mth5_logger import setup_logger

# =============================================================================
# MTH5 Table Class
# =============================================================================


class MTH5Table:
    """
    Use the underlying NumPy basics, there are simple actions in this table,
    if a user wants to use something more sophisticated for querying they
    should try using a pandas table.  In this case entries in the table
    are more difficult to change and datatypes need to be kept track of.

    """

    def __init__(self, hdf5_dataset):
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

        self.hdf5_reference = None
        if isinstance(hdf5_dataset, h5py.Dataset):
            self.array = weakref.ref(hdf5_dataset)()
            self.hdf5_reference = hdf5_dataset.ref
        else:
            msg = "Input must be a h5py.Dataset not {0}".format(type(hdf5_dataset))
            self.logger.error(msg)
            raise MTH5TableError(msg)

    def __str__(self):
        """
        return a string that shows the table in text form

        :return: text representation of the table
        :rtype: string

        """
        # if the array is empty
        if self.array.size == 0:
            length_dict = dict([(key, len(str(key))) for key in list(self.dtype.names)])
            lines = [
                " | ".join(
                    ["index"]
                    + [
                        "{0:^{1}}".format(name, length_dict[name])
                        for name in list(self.dtype.names)
                    ]
                )
            ]
            lines.append("-" * len(lines[0]))
            return "\n".join(lines)

        length_dict = dict(
            [
                (key, max([len(str(b)) for b in self.array[key]]))
                for key in list(self.dtype.names)
            ]
        )
        lines = [
            " | ".join(
                ["index"]
                + [
                    "{0:^{1}}".format(name, length_dict[name])
                    for name in list(self.dtype.names)
                ]
            )
        ]
        lines.append("-" * len(lines[0]))

        for ii, row in enumerate(self.array):
            line = ["{0:^5}".format(ii)]
            for element, key in zip(row, list(self.dtype.names)):
                if isinstance(element, (np.bytes_)):
                    element = element.decode()
                try:
                    line.append("{0:^{1}}".format(element, length_dict[key]))

                except TypeError as error:
                    if isinstance(element, h5py.h5r.Reference):
                        msg = "{0}: Cannot represent h5 reference as a string"
                        self.logger.debug(msg.format(error))
                        line.append(
                            "{0:^{1}}".format(
                                "<HDF5 object reference>", length_dict[key]
                            )
                        )
                    else:
                        self.logger.exception(f"{error}")

            lines.append(" | ".join(line))
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, MTH5Table):
            return self.array == other.array
        elif isinstance(other, h5py.Dataset):
            return self.array == other
        else:
            msg = "Cannot compare type={0}".format(type(other))
            self.logger.error(msg)
            raise TypeError(msg)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return self.array.shape[0]

    @property
    def dtype(self):
        try:
            return self.array.dtype
        except AttributeError as error:
            msg = "{0}, dataframe is not initiated yet".format(error)
            self.logger.warning(msg)
            return None

    def check_dtypes(self, other_dtype):
        """
        Check to make sure datatypes match
        """

        if self.dtype == other_dtype:
            return True

        return False

    @property
    def shape(self):
        return self.array.shape

    @property
    def nrows(self):
        return self.array.shape[0]

    def locate(self, column, value, test="eq"):
        """

        locate index where column is equal to value
        :param column: DESCRIPTION
        :type column: TYPE
        :param value: DESCRIPTION
        :type value: TYPE
        :type test: type of test to try
        * 'eq': equals
        * 'lt': less than
        * 'le': less than or equal to
        * 'gt': greater than
        * 'ge': greater than or equal to.
        * 'be': between or equal to
        * 'bt': between

        If be or bt input value as a list of 2 values

        :return: DESCRIPTION
        :rtype: TYPE

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

    def add_row(self, row, index=None):
        """
        Add a row to the table.

        row must be of the same data type as the table


        :param row: row entry for the table
        :type row: TYPE

        :param index: index of row to add
        :type index: integer, if None is given then the row is added to the
                     end of the array

        :return: index of the row added
        :rtype: integer

        """

        if not isinstance(row, (np.ndarray)):
            msg = "Input must be an numpy.ndarray" + "not {0}".format(type(row))
        if isinstance(row, np.ndarray):
            if not self.check_dtypes(row.dtype):
                msg = "{0}\nInput dtypes:\n{1}\n\nTable dtypes:\n{2}".format(
                    "Data types are not equal:", row.dtype, self.dtype
                )
                self.logger.error(msg)
                raise ValueError(msg)

        if index is None:
            index = self.nrows
            if self.nrows == 1:
                match = True
                null_array = np.empty(1, dtype=self.dtype)
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
        self.logger.debug("Added row as index {0} with values {1}".format(index, row))

        return index

    def update_row(self, entry):
        """
        Update an entry by first locating the index and then rewriting the entry.

        :param entry: numpy array with same datatype as the table
        :type entry: np.ndarray

        :return: row index.

        This doesn't work because you cannot test for hdf5_reference, should use
        add row and locate by index.

        """
        try:
            row_index = self.locate("hdf5_reference", entry["hdf5_reference"])[0]
            return self.add_row(entry, index=row_index)

        except IndexError:
            self.logger.debug("Could not find row, adding a new one")
            return self.add_row(entry)

    def remove_row(self, index):
        """
        Remove a row

        .. note:: that there is not index value within the array, so the
                  indexing is on the fly.  A user should use the HDF5
                  reference instead of index number that is the safest and
                  most robust method.

        :param index: DESCRIPTION
        :type index: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        This isn't as easy as just deleteing an element.
        Need to delete the element from the weakly referenced array and then
        set the summary table dataset to the new array.

        So set to a null array for now until a more clever option is found.

        """
        null_array = np.empty((1,), dtype=self.dtype)
        try:
            return self.add_row(null_array, index=index)

        except IndexError as error:
            msg = "Could not find index {0} in shape {1}".format(index, self.shape())
            self.logger.exception(msg)
            raise IndexError(f"{error}\n{msg}")

    def to_dataframe(self):
        """
        Convert the table into a :class:`pandas.DataFrame` object.

        :return: convert table into a :class:`pandas.DataFrame` with the
                 appropriate data types.
        :rtype: :class:`pandas.DataFrame`

        """

        df = pd.DataFrame(self.array[()])
        for key in [
            "station",
            "run",
            "component",
            "measurement_type",
            "units",
        ]:
            setattr(df, key, getattr(df, key).str.decode("utf-8"))
        df.start = pd.to_datetime(df.start.str.decode("utf-8"))
        df.end = pd.to_datetime(df.end.str.decode("utf-8"))

        return df
    
    def clear_table(self):
        """
        clear a table, 
        
        Basically delete the table and start over
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        dtype = copy.deepcopy(self.dtype)
        
        root = self.array.parent
        name = self.array.name.split("/")[-1]
        ds_options = {
            "compression": self.array.compression,
            "compression_opts": self.array.compression_opts,
            "shuffle": self.array.shuffle,
            "fletcher32": self.array.fletcher32,
        }
        
        del root[name]
        
        self.array = root.create_dataset(name, (1, ), maxshape=(None, ),
                                         dtype=dtype, **ds_options) 
        
