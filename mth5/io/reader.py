# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:32:45 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from mth5.io import zen, nims

# =============================================================================
# generic reader for any file type
# =============================================================================
readers = {'zen':{'file_types':['z3d'], 'reader':zen.read_z3d},
           'nims':{'file_types':['bin'], 'reader':nims.read_nims}}

def get_reader(extension):
    """
    
    get the proper reader for file extension
    
    :param extension: DESCRIPTION
    :type extension: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    
    for key, vdict in readers.items():
        if extension.lower() in vdict['file_types']:
            return key, vdict['reader']
    
    raise ValueError(f"Could not find a reader for file type {extension}")    
    

def read_file(fn):
    """
    
    :param fn: full path to file
    :type fn: string or :class:`pathlib.Path`
    :return: MT time series object
    :rtype: :class:`mth5.timeseries.MTTS`

    """
    
    if not isinstance(fn, Path):
        fn = Path(fn)
        
    file_type, file_reader = get_reader(fn.suffix.replace('.', ''))
    
    return file_reader(fn)       
    
        