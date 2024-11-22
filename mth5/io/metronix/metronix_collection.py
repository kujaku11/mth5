# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:22:44 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5.io.collection import Collection

# =============================================================================


class MetronixCollection(Collection):
    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.file_ext = "atss"
