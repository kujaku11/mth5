"""
Proof of concept for issue #209 mulitstation FCs

"""
from loguru import logger
from mth5.timeseries.spectre import FCRunChunk
from mth5.timeseries.spectre import MultivariateLabelScheme
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


class TestMultivariateLabelScheme(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_initialize(unitte):
        mvls_default = MultivariateLabelScheme()
        mvls = MultivariateLabelScheme(
            label_elements = ("station", "component",),
            join_char = "_",
        )
        assert mvls_default.id == mvls.id
        mvls = MultivariateLabelScheme(
            label_elements=("foo", "bar",),
            join_char="_",
        )
        assert mvls_default.id != mvls.id

        # print(mvls)



# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()

