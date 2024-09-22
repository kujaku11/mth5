"""
Sets up paths for synthetic data testing.

"""
import pathlib
import tempfile
import errno

from loguru import logger
from typing import Optional, Union

DEFAULT_SANDBOX_PATH = pathlib.Path(__file__).parent.resolve()


class SyntheticTestPaths:
    """
    This class was created to workaround installations with read-only access to the folder containing mth5.
    Normally, the mth5 data/ folder can be used to store mth5 test data generated when running tests or examples.
    If data/ is read-only, then this class allows setting "sandbox_path", a writable folder for tests or examples.

    """

    def __init__(
        self,
        sandbox_path: Optional[Union[pathlib.Path, None]] = None,
        ascii_data_path: Optional[Union[pathlib.Path, None]] = None
    ):
        """

        :type sandbox_path: Union[pathlib.Path, None]
        :param sandbox_path: A writable path where test results are stored.
        :type ascii_data_path: Union[pathlib.Path, None]
        :param ascii_data_path: This is where the synthetic ascii data are loaded from.


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
        self.mkdirs()
        self.writability_check()

    def writability_check(self) -> None:
        """

        Check if the path is writable, and Placeholder

        Tried adding the second solution from here:
        https://stackoverflow.com/questions/2113427/determining-whether-a-directory-is-writeable

        If dirs are not writeable, consider
        HOME = pathlib.Path().home()
        workaround_sandbox = HOME.joinpath(".cache", "aurora", "sandbox")
        """
        if not _is_writable(self.mth5_path):
            msg = f"mth5_path {self.mth5_path} is not writable -- cannot make test data"
            raise IOError(msg)


    def mkdirs(self) -> None:
        """
        Makes the directories that the tests will write results to.

        """
        try:
            self.mth5_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            msg = "unable to create mth5 data folder -- check write access!"
            raise FileNotFoundError(msg)


def _is_writable(path: pathlib.Path) -> bool:
    """
    Checks a path to see if you can write to it.

    :type path: pathlib.Path
    :param path: a place you want to write data
    :rtype: bool
    :return: True if path is writable, else False

    """
    try:
        testfile = tempfile.TemporaryFile(dir = path)
        testfile.close()
    except OSError as e:
        if e.errno == errno.EACCES:  # 13
            return False
        e.filename = path
        raise
    return True

# def main():
#     print(DEFAULT_SANDBOX_PATH.absolute())
#
# if __name__ =="__main__":
#     main()
