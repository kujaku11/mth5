# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 06:55:21 2020

@author: kkappler
"""

from __future__ import absolute_import, division, print_function


import datetime
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb

from pathlib import Path

import mth5


mth5_dir = os.path.dirname(inspect.getfile(mth5))
mth5_git_dir = os.path.dirname(mth5_dir)
DATA_DIR = os.path.join(mth5_git_dir, 'data')
DATA_REPO_ROOT_PATH = Path(DATA_DIR)

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
