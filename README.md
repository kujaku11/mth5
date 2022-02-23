====
MTH5
====

[![PyPi Version](https://img.shields.io/pypi/v/mth5.svg)](https://pypi.python.org/pypi/mth5)
[![Conda Version](https://img.shields.io/conda/v/conda-forge/mth5.svg)](https://anaconda.org/conda-forge/mth5)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://code.chs.usgs.gov/jpeacock/mth5/-/new/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/mth5/badge/?version=latest)](https://mth5.readthedocs.io/en/latest/?badge=latest)		
[![](https://codecov.io/gh/kujaku11/mth5/branch/master/graph/badge.svg?token=XU5QSRM1ZO)](https://codecov.io/gh/kujaku11/mth5)\
[![](https://zenodo.org/badge/283883448.svg)](https://zenodo.org/badge/latestdoi/283883448)
[![](https://mybinder.org/badge_logo.svg)] (https://mybinder.org/v2/gh/kujaku11/mth5/master)

MTH5 is an HDF5 data container for magnetotelluric time series data, but could be extended to other data types.  This package provides tools for reading/writing/manipulating MTH5 files.

MTH5 uses `h5py <https://www.h5py.org/>`__  to interact with the HDF5 file, `xarray <http://xarray.pydata.org/en/stable/>`__ to interact with the data in a nice way, and all metadata use `mt_metadata <https://github.com/kujaku11/mt_metadata>`__. 

This project is in cooperation with the Incorporated Research Institutes of Seismology, the U.S. Geological Survey, and other collaborators.  Facilities of the IRIS Consortium are supported by the National Science Foundation’s Seismological Facilities for the Advancement of Geoscience (SAGE) Award under Cooperative Support Agreement EAR-1851048.  USGS is partially funded through the Community for Data Integration and IMAGe through the Minerals Resources Program.  


* **Version**: 0.2.4
* **Free software**: MIT license
* **Documentation**: https://mth5.readthedocs.io.
* **Examples**: Click the `Binder` badge above and Jupyter Notebook examples are in **docs/examples/notebooks**


Features
--------

* Read and write HDF5 files formated for magnetotelluric time series.
* From MTH5 a user can create an MTH5 file, get/add/remove stations, runs, channels and filters and all associated metadata.
* Data is contained as an `xarray <http://xarray.pydata.org/en/stable/index.html>`_ which can house the data and metadata together, and data is indexed by time.
* Readers for some data types are included as plugins, namely
    - Z3D
    - NIMS BIN
    - USGS ASCII
    - LEMI
    - StationXML + miniseed

Introduction
-------------

The goal of **MTH5** is to provide a self describing heirarchical data format for working, sharing, and archiving.  **MTH5** was cooperatively developed with community input and follows logically how magnetotelluric data are collected.  This module provides open-source tools to interact with an **MTH5** file.  


The metadata follows the standards proposed by the `IRIS-PASSCAL MT
Software working
group <https://www.iris.edu/hq/about_iris/governance/mt_soft>`__ and
documented in `MT Metadata
Standards <https://doi.org/10.5066/P9AXGKEV>`__.

.. note:: If you would like to comment or contribute checkout `Issues <https://github.com/kujaku11/mth5/issues>`__ or `Slack <simpeg.slack.com>`__.   

MTH5 Format
-----------

-  The basic format of MTH5 is illustrated below, where metadata is
   attached at each level.

MTH5 File Version 0.1.0
~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: docs/source/images/example_mt_file_structure.png
   :alt: MTH5 Format version 0.1.0
   :align: center

|
   
MTH5 file version **0.1.0** was the original file version where `Survey` was the highest level of the file.  This has some limitations in that only one `Survey` could be saved in a single file, but if you have mulitple `Surveys` that you would like to store we need to add a higher level `Experiment`.  

.. important:: Some MTH5 **0.1.0** files have already been archived on `ScienceBase <https://www.sciencebase.gov/catalog/>`__ and has been used as the working format for Aurora and is here for reference.  Moving forward the new format will be **0.2.0** as described below.
   
   
MTH5 File Version 0.2.0
~~~~~~~~~~~~~~~~~~~~~~~~
   
.. figure:: docs/source/images/example_mt_file_structure_v2.svg
   :alt: MTH5 Format version 0.2.0
   :align: center

|
   
MTH5 file version **0.2.0** has `Experiment` as the top level.  This allows for multiple `Surveys` to be included in a single file and therefore allows for more flexibility.  For example if you would like to remote reference stations in a local survey with stations from a different survey collected at the same time you can have all those surveys and stations in the same file and make it easier for processing.

.. hint:: MTH5 is comprehensively logged, therefore if any problems arise you can always check the mth5_debug.log (if you are in debug mode, change the mode in the mth5.__init__) and the mth5_error.log, which will be written to your current working directory.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
