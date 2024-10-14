.. highlight:: shell

============
Installation
============


Stable release
--------------

To install MTH5, run this command in your terminal:

.. code-block:: console

    $ pip install mth5

PIP will always install the latest release, however for better package managment use Conda.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Conda Forge
-------------

To install MTH5 from Conda run this command:

.. code-block:: console

    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict
    $ conda install mth5

This should be the same as installing with `pip` that pulls from PyPi and should work better if you are using an Anaconda environment.

From sources
------------

The sources for MTH5 can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/kujaku11/mth5

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/kujaku11/mth5/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install
	
Or you can install it in editing mode and be able to adjust the code as needed:

.. code-block:: console

    $ python setup.py -e install


.. _Github repo: https://github.com/kujaku11/mth5
.. _tarball: https://github.com/kujaku11/mth5/tarball/master
