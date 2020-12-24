# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:34:32 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.utils.helpers import inherit_doc_string
from mth5.datasets import ChannelDataset

# =============================================================================
# Auxiliary Channel
# =============================================================================
@inherit_doc_string
class AuxiliaryDataset(ChannelDataset):
    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)
