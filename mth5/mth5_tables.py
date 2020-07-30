# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:15:27 2020

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import h5py
import logging
from pathlib import Path
from mth5.utils.exceptions import MTH5Error

# =============================================================================
# Main API
# =============================================================================
class MTH5(object):
    """
    container for MT data in HDF5 format
    """

    def __init__(self, **kwargs):
        self.mth5_fn = None

    def open_mth5_file(self, mth5_fn):
        """
        

        Parameters
        ----------
        mth5_fn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if Path(mth5_fn).exists():
            self.mth5_fn = mth5_fn
            self.mth5 = tables.open_file(mth5_fn, mode="r+")
        else:
            msg = "File {0} does not exist \n".format(mth5_fn)
            msg += "Use open_mth5_file to make new file."
            raise MTH5Error(msg)

        print("INFO: Opened {0}".format(mth5_fn))

    def new_mth5_file(self, mth5_fn, title="MTH5 File"):
        """
        Make new MTH5 file

        Parameters
        ----------
        mth5_fn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mth5_fn = mth5_fn
        self.mth5 = tables.open_file(mth5_fn, mode="w", title=title)

    def _is_write(self):
        """
        check to see if the hdf5 file is open and writeable
        """
        if isinstance(self.mth5, tables.file.File):
            try:
                if "w" in self.mth5.mode or "+" in self.mth5.mode:
                    return True
                elif self.mth5.mode == "r":
                    return False
            except ValueError:
                return False
        return False

    def close_mth5_file(self):
        """
        Close file
        
        """
        self.mth5.flush()
        self.mth5.close()

        print("INFO: Closed {0}".format(self.mth5_fn))

    def _initiate_file_structure(self):
        """
        build the general file structure
        
        Groups
            * Receivers
            * Sources
            * Reports
            * Filters
            
        Tables
            * Survey 
            * Receivers/Receivers_table
            * Sources/Sources_table
            * Reports/Reports_table
        """

        pass
