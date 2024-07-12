"""
Proof of concept for issue #209 mulitstation FCs

"""
from loguru import logger
from mth5.utils.fc_tools import FCRunChunk
import unittest


class TestFCRunChunk(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_initialize(unitte):
        fcrc = FCRunChunk()
        fcrc = FCRunChunk(
            station_id="mt01",
            run_id="001",
            decimation_level_id="0",
            # start="2023-10-05T20:03:00",
            start="",
            end="",
            channels=[],
        )
