# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 06:55:21 2020

@author: kkappler
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import absolute_import, division, print_function

import datetime
import inspect
from pathlib import Path

import mth5

# =============================================================================
# global variables
# =============================================================================
mth5_dir = Path(inspect.getfile(mth5)).parent
mth5_git_dir = mth5_dir.parent
DATA_DIR = mth5_git_dir.parent.joinpath("mth5_test_data")
DATA_REPO_ROOT_PATH = DATA_DIR

if not DATA_DIR.exists():
    print("Need to install mth5_test_data repo at https://github.com/kujaku11/mth5_test_data")

def my_function():
    """
    """
    pass

def main():
    """
    """
    my_function()
    print("finito {}".format(datetime.datetime.now()))

if __name__ == "__main__":
    main()
