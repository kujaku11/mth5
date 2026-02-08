mth5.data.paths
===============

.. py:module:: mth5.data.paths

.. autoapi-nested-parse::

   Sets up paths for synthetic data testing.



Attributes
----------

.. autoapisummary::

   mth5.data.paths.DEFAULT_SANDBOX_PATH


Classes
-------

.. autoapisummary::

   mth5.data.paths.SyntheticTestPaths


Module Contents
---------------

.. py:data:: DEFAULT_SANDBOX_PATH

.. py:class:: SyntheticTestPaths(sandbox_path: Optional[Union[pathlib.Path, None]] = None, ascii_data_path: Optional[Union[pathlib.Path, None]] = None)

   This class was created to workaround installations with read-only access to the folder containing mth5.
   Normally, the mth5 data/ folder can be used to store mth5 test data generated when running tests or examples.
   If data/ is read-only, then this class allows setting "sandbox_path", a writable folder for tests or examples.



   .. py:attribute:: mth5_path


   .. py:method:: writability_check() -> None

      Check if the path is writable, and Placeholder

      Tried adding the second solution from here:
      https://stackoverflow.com/questions/2113427/determining-whether-a-directory-is-writeable

      If dirs are not writeable, consider
      HOME = pathlib.Path().home()
      workaround_sandbox = HOME.joinpath(".cache", "aurora", "sandbox")



   .. py:method:: mkdirs() -> None

      Makes the directories that the tests will write results to.




