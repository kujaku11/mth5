# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:48:24 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.clients.base import ClientBase

# =============================================================================


class TestClientBase(unittest.TestCase):
    def setUp(self):
        self.base = ClientBase({"h5_mode": "w", "h5_driver": "sec2"})

    def test_h5_kwargs(self):
        keys = []


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
