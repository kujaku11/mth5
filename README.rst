====
MTH5
====


.. image:: https://img.shields.io/pypi/v/mth5.svg
        :target: https://pypi.python.org/pypi/mth5

.. image:: https://img.shields.io/travis/kujaku11/mth5.svg
        :target: https://travis-ci.com/kujaku11/mth5

.. image:: https://readthedocs.org/projects/mth5/badge/?version=latest
        :target: https://mth5.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/kujaku11/mth5/branch/master/graph/badge.svg?token=XU5QSRM1ZO
        :target: https://codecov.io/gh/kujaku11/mth5


The goal of MTH5 is to develop an archivable and exchangeable format for magnetotelluric time series data.  MTH5 provides tools to read/write MTH5 files and tools to interact with MTH5 files.  All metadata is based on the `mt_metadata <https://github.com/kujaku11/mt_metadata>`_   


* Free software: MIT license
* Documentation: https://mth5.readthedocs.io.


Features
--------

* Read and write HDF5 files formated for magnetotelluric time series.
* From MTH5 a user can create an MTH5 file, get/add/remove stations, runs, channels and filters and all associated metadata.
* Data is contained as an `xarray <http://xarray.pydata.org/en/stable/index.html>`_ which can house the data and metadata together, and data is indexed by time.
* Readers for some data types are included as plugins, namely
	- Z3D
	- NIMS BIN
	- USGS ASCII
	- StationXML + miniseed

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
