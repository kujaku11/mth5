# History

0.1.0 (2021-06-30)
------------------

* First release on PyPI.

0.2.0 (2021-10-31)
-------------------

* Updated the structure of MTH5 to have ``Experiment`` as the top level
* Updated tests
* Backwards compatibility works
* Updated Docs

0.2.5 (2022-04-07)
----------------------
* fixed bugs
* Added TransferFunctions and TransferFunction groups at the station level that can now hold transfer functions
* Added channel_summary and tf_summary tables that are updated upon close if the file is in 'w' mode
* Updated tests
* Updated documentation
* Note: tests for make_mth5 from IRIS are currently not working as there has been some reorganization with data at the DMC

