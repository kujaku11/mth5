# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 06:55:21 2020

@author: kkappler

.. note:: DATA_DIR is the mth5 test data repository, either the repo or a symink to it.
It needs to be on the same level as the git cloned folder of mth5.
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import absolute_import, division, print_function

import datetime
import inspect
from pathlib import Path

import mth5


def get_test_data_path():
    try:
        from mth5_test_data.util import MTH5_TEST_DATA_DIR

        return MTH5_TEST_DATA_DIR
    except (ImportError):
        print(
            "Need to install mth5_test_data repo at https://github.com/kujaku11/mth5_test_data"
        )
        # raise Exception
        return None


# =============================================================================
# global variables
# =============================================================================
DATA_DIR = get_test_data_path()


def ensure_is_path(directory):
    if not isinstance(directory, Path):
        directory = Path(directory)
    return directory


def my_function():
    """"""
    pass


def main():
    """"""
    my_function()
    print("finito {}".format(datetime.datetime.now()))


if __name__ == "__main__":
    main()
