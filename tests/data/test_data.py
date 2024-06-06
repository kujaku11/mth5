import mth5
import pathlib
import unittest

# from loguru import logger


class TestDataFolder(unittest.TestCase):
    """ """

    def setUp(self):
        init_file = pathlib.Path(mth5.__file__)
        self.data_folder = init_file.parent.joinpath("data")

    def test_ascii_data_paths(self):
        """
        Make sure that the ascii data are where we think they are.

        """

        self.assertTrue(self.data_folder.exists())
        file_paths = list(self.data_folder.glob("*asc"))
        file_names = [x.name for x in file_paths]

        assert "test1.asc" in file_names
        assert "test2.asc" in file_names


def main():
    unittest.main()


if __name__ == "__main__":
    main()
