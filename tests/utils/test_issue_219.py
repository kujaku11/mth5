"""
Proof of concept for issue #219

"""

from loguru import logger

from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.paths import SyntheticTestPaths
from mth5.utils.extract_subset_mth5 import extract_subset
from mth5.utils.helpers import get_channel_summary


synthetic_test_paths = SyntheticTestPaths()
MTH5_PATH = synthetic_test_paths.mth5_path
FILE_VERSION = "0.1.0"


def _select_source_file():
    """

    Returns
    -------

    """
    # Get a list of mth5 files available

    # If this list is empty, you may need to run `make_mth5_from_asc.py`
    synthetic_test_paths = SyntheticTestPaths()
    MTH5_PATH = synthetic_test_paths.mth5_path
    logger.info(f"MTH5_PATH = {MTH5_PATH}")
    assert MTH5_PATH.exists()
    mth5_files = list(MTH5_PATH.glob("*h5"))
    logger.info(f"mth5_files: {mth5_files}")
    source_file = MTH5_PATH.joinpath("test3.h5")
    try:
        assert source_file.exists()
    except AssertionError:
        create_test3_h5(file_version=FILE_VERSION)

    return source_file


def _select_target_file():
    synthetic_test_paths = SyntheticTestPaths()
    MTH5_PATH = synthetic_test_paths.mth5_path
    target_file = MTH5_PATH.joinpath("subset.h5")
    return target_file


def _select_data_subset():
    source_file = _select_source_file()
    df = get_channel_summary(source_file)
    selected_df = df[df.run.isin(["002", "004"])]
    selected_df
    return selected_df


def test_extract_subset():
    source_file = _select_source_file()
    target_file = _select_target_file()
    subset_df = _select_data_subset()
    extract_subset(source_file, target_file, subset_df)
    df2 = get_channel_summary(target_file)
    compare_columns = [x for x in df2.columns if "hdf5_reference" not in x]
    subset_df.reset_index(inplace=True, drop=True)  # allow comparison
    assert (df2[compare_columns] == subset_df[compare_columns]).all().all()
    print(df2)


def main():
    test_extract_subset()


if __name__ == "__main__":
    main()
