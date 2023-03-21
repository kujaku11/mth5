History
=========

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

0.2.6 (2022-07-01)
-----------------------

* Added calibration functions
* minor bug fixes
* updated tests
* updated documentation

0.2.7 (2022-09-14)
------------------------

* Rebased IO module to make a module for each data logger type
* Updated tests
* Updated documentation
* Factored `make_mth5`

0.3.0 (2022-09-25)
------------------------

* change default initialize_mth5 to use append mode, issue #92 by @kkappler in #94
* Fix issue 105 by @kkappler in PR #106
* adding in parallel mth5 tutorial by @nre900 in #110
* adding in new tutorial and modifications to mth5_in_parallel.ipynb by @nre900 in issue #112
* Add phoenix reader by @kujaku11 in PR #103
* Remove response by @kujaku11 in PR #100 

0.3.1 (2023-01-18)
------------------------

* Speed up station and survey validataion by 
* Tutorial updates by @nre900 
* remove kwarg specifying default value 
* update initialize_mth5 
* Have a single metadata object for ChannelTS and RunTS 
* Use h5 Paths to get groups and datasets
* Bump wheel from 0.33.6 to 0.38.1