# -*- coding: utf-8 -*-
"""

Filter object

Created on Wed Sep 30 12:55:58 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================

import pandas as pd

from mth5 import metadata

# =============================================================================
#
# =============================================================================
class Filter:
    """
    
    Object to hold a filter.
    
    """

    def __init__(self, **kwargs):
        self.metadata = metadata.Filter()
        self.filter = None
