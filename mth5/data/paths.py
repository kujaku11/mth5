"""
Sets up paths for synthetic data testing.

"""
import pathlib

from loguru import logger

DEFAULT_SANDBOX_PATH = pathlib.Path(__file__).parent.resolve()


class SyntheticTestPaths:
    """
    sandbox path must be a place that has write access.  This class was created because on some
    installations we only have read access.  Originally there was a data/ folder with the synthetic
    ascii data stored there, and we created the mth5 and other data products in the same place.

    Here we have a ascii_data_path which points at the ascii files (which may be read only), but we
    also accept a kwarg for "sandbox_path" which is writable and this is where the mth5 and etc. will
    get built.

    """

    def __init__(self, sandbox_path=None, ascii_data_path=None):
        """

        Parameters
        ----------
        sandbox_path: pathlib.Path
            A writable path where test results are stored

        ivars:
        - ascii_data_path: where the ascii data are stored
        - mth5_path: this is where the mth5 files get written to.

        """
        # READ ONLY OK
        if ascii_data_path is None:
            self.ascii_data_path = pathlib.Path(__file__).parent.resolve()

        # NEED WRITE ACCESS
        # Consider using an environment variable for sandbox_path
        if sandbox_path is None:
            logger.debug(
                f"synthetic sandbox path is being set to {DEFAULT_SANDBOX_PATH}"
            )
            self._sandbox_path = DEFAULT_SANDBOX_PATH
        else:
            self._sandbox_path = sandbox_path

        self.mth5_path = self._sandbox_path.joinpath("mth5")
        self.writability_check()
        # assert self.ascii_data_path.exists()

    def writability_check(self):
        """
        Placeholder
        Should check that sandbox and dirs below have write access
        If dirs are not writeable, consider
        HOME = pathlib.Path().home()
        workaround_sandbox = HOME.joinpath(".cache", "aurora", "sandbox")
        """
        pass

    def mkdirs(self):
        """
        Makes the directories that the tests will write results to.

        """
        self.mth5_path.mkdir(parents=True, exist_ok=True)


# def main():
#     print(DEFAULT_SANDBOX_PATH.absolute())
#
# if __name__ =="__main__":
#     main()
