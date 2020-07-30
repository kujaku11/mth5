# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:33:00 2020

@author: jpeacock
"""
import os
import datetime
import time
import json
import dateutil


import h5py
import pandas as pd
import numpy as np

# # =============================================================================
# # Calibrations
# # =============================================================================
# class Calibration(Generic):
#     """
#     container for insturment calibrations

#     Each instrument should be a separate class

#     Metadata should be:
#         * instrument_id
#         * calibration_date
#         * calibration_person
#         * units
#     """

#     def __init__(self, name=None):
#         super(Calibration, self).__init__()
#         self.name = name
#         self.instrument_id = None
#         self.units = None
#         self.calibration_date = None
#         self.calibration_person = Person()
#         self.frequency = None
#         self.real = None
#         self.imaginary = None
#         self._col_list = ["frequency", "real", "imaginary"]
#         self._attrs_list = [
#             "name",
#             "instrument_id",
#             "units",
#             "calibration_date",
#             "calibration_person",
#         ]

#     def from_dataframe(self, cal_dataframe, name=None):
#         """
#         updated attributes from a pandas DataFrame

#         :param cal_dataframe: dataframe with columns frequency, real, imaginary
#         :type cal_dataframe: pandas.DataFrame

#         """
#         assert isinstance(cal_dataframe, pd.DataFrame) is True

#         if name is not None:
#             self.name = name

#         for col in cal_dataframe.columns:
#             setattr(self, col, cal_dataframe[col])

#     def from_numpy_array(self, cal_np_array, name=None):
#         """
#         update attributes from a numpy array

#         :param cal_np_array: array of values for calibration, see below
#         :type cal_np_array: numpy.ndarray

#         if array is a numpy structured array names need to be:
#             * frequency
#             * real
#             * imaginary

#         if array is just columns, needs to be ordered:
#             * frequency (index 0)
#             * real (index 1)
#             * imaginary (index 2)

#         """
#         if name is not None:
#             self.name = name

#         ### assume length of 1 is a structured array
#         if len(cal_np_array.shape) == 1:
#             assert cal_np_array.dtype.names == ("frequency", "real", "imaginary")
#             for key in cal_np_array.dtype.names:
#                 setattr(self, key, cal_np_array[key])

#         ### assume an unstructured array (f, r, i)
#         if len(cal_np_array.shape) == 2 and cal_np_array.shape[0] == 3:
#             for ii, key in enumerate(["frequency", "real", "imaginary"]):
#                 setattr(self, key, cal_np_array[ii, :])

#         return

#     def from_mth5(self, mth5_obj, name):
#         """
#         update attribues from mth5 file
#         """
#         self.name = name
#         for key in mth5_obj["/calibrations/{0}".format(self.name)].keys():
#             setattr(self, key, mth5_obj["/calibrations/{0}/{1}".format(self.name, key)])

#         ### read in attributes
#         self.from_json(
#             mth5_obj["/calibrations/{0}".format(self.name)].attrs["metadata"]
#         )

#     def from_csv(self, cal_csv, name=None, header=False):
#         """
#         Read a csv file that is in the format frequency,real,imaginary
        
#         :param cal_csv: full path to calibration csv file
#         :type cal_csv: string
        
#         :param name: instrument id
#         :type name: string
        
#         :param header: boolean if there is a header in the csv file
#         :type header: [ True | False ]
        
#         """
#         if not header:
#             cal_df = pd.read_csv(cal_csv, header=None, names=self._col_list)
#         else:
#             cal_df = pd.read_csv(cal_csv, names=self._col_list)

#         if name is not None:
#             self.name
#         self.from_dataframe(cal_df)
