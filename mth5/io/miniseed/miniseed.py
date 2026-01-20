# -*- coding: utf-8 -*-
"""

Created on Wed Sep 30 10:20:12 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from obspy import read as obspy_read

from mth5.timeseries import RunTS


# =============================================================================
# read seismic file
# =============================================================================
def read_miniseed(fn):
    """
    Read a miniseed file into a :class:`mth5.timeseries.RunTS` object. Uses
    `Obspy` to read the miniseed.

    :param fn: full path to the miniseed file
    :type fn: string
    :return: RunTS object
    :rtype: :class:`mth5.timeseries.RunTS`

    """

    # obspy does not use Path objects for file names
    if isinstance(fn, Path):
        fn = fn.as_posix()
    obs_stream = obspy_read(fn)
    run_obj = RunTS()
    run_obj.from_obspy_stream(obs_stream)

    return run_obj
