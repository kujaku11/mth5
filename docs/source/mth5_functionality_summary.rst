MTH5 Class Functionality Summary
=================================

The `MTH5` class provides an object-oriented interface for managing magnetotelluric (MT) data in the HDF5 format. It supports creating, reading, and manipulating hierarchical MT data structures, as well as validating and summarizing the data.

Initialization
--------------

.. code-block:: python

    from mth5 import mth5
    mth5_obj = mth5.MTH5(file_version='0.2.0')

Attributes
----------

- **dataset_options**: Returns a dictionary of dataset options such as compression and shuffle settings.
- **file_attributes**: Provides file-level metadata attributes.
- **filename**: The name of the HDF5 file.
- **file_type**: The type of file (e.g., 'mth5').
- **file_version**: The version of the MTH5 file format.
- **software_name**: The name of the software used to create the file.
- **data_level**: The data level (e.g., raw, processed).

Methods
-------

### File Operations

**open_mth5**
Opens an existing MTH5 file or creates a new one.

.. code-block:: python

    mth5_obj.open_mth5('test.mth5', 'w')

**close_mth5**
Closes the MTH5 file, flushing data to disk and updating summary tables.

.. code-block:: python

    mth5_obj.close_mth5()

**validate_file**
Validates the structure and metadata of the MTH5 file.

.. code-block:: python

    is_valid = mth5_obj.validate_file()

### Group and Metadata Management

**add_survey**
Adds a survey to the file.

.. code-block:: python

    survey = mth5_obj.add_survey('survey_001')

**get_survey**
Retrieves a survey by name.

.. code-block:: python

    survey = mth5_obj.get_survey('survey_001')

**remove_survey**
Removes a survey from the file.

.. code-block:: python

    mth5_obj.remove_survey('survey_001')

**add_station**
Adds a station to a survey.

.. code-block:: python

    station = mth5_obj.add_station('MT001', survey='survey_001')

**get_station**
Retrieves a station by name.

.. code-block:: python

    station = mth5_obj.get_station('MT001', survey='survey_001')

**remove_station**
Removes a station from a survey.

.. code-block:: python

    mth5_obj.remove_station('MT001', survey='survey_001')

**add_run**
Adds a run to a station.

.. code-block:: python

    run = mth5_obj.add_run('MT001', 'run_001', survey='survey_001')

**get_run**
Retrieves a run by name.

.. code-block:: python

    run = mth5_obj.get_run('MT001', 'run_001', survey='survey_001')

**remove_run**
Removes a run from a station.

.. code-block:: python

    mth5_obj.remove_run('MT001', 'run_001', survey='survey_001')

**add_channel**
Adds a channel to a run.

.. code-block:: python

    channel = mth5_obj.add_channel(
        station_name='MT001',
        run_name='run_001',
        channel_name='ex',
        channel_type='electric',
        data=[1, 2, 3],
        survey='survey_001'
    )

**get_channel**
Retrieves a channel by name.

.. code-block:: python

    channel = mth5_obj.get_channel('MT001', 'run_001', 'ex', survey='survey_001')

**remove_channel**
Removes a channel from a run.

.. code-block:: python

    mth5_obj.remove_channel('MT001', 'run_001', 'ex', survey='survey_001')

### Data Conversion

**to_experiment**
Converts the MTH5 file to an `Experiment` object.

.. code-block:: python

    experiment = mth5_obj.to_experiment()

**from_experiment**
Fills the MTH5 file from an `Experiment` object.

.. code-block:: python

    mth5_obj.from_experiment(experiment)

### Summary Tables

**channel_summary**
Provides a summary of all channels in the file.

**fc_summary**
Provides a summary of Fourier coefficients.

**tf_summary**
Provides a summary of transfer functions.

Examples
--------

Creating a new MTH5 file and adding data using the `with` statement for automatic resource management:

.. code-block:: python

    from mth5 import mth5

    with MTH5(file_version='0.2.0') as mth5_obj:
        mth5_obj.open_mth5('example.mth5', 'w')

        survey = mth5_obj.add_survey('survey_001')
        station = mth5_obj.add_station('MT001', survey='survey_001')
        run = mth5_obj.add_run('MT001', 'run_001', survey='survey_001')
        channel = mth5_obj.add_channel(
            station_name='MT001',
            run_name='run_001',
            channel_name='ex',
            channel_type='electric',
            data=[1, 2, 3],
            survey='survey_001'
        )


For more details, refer to the MTH5 documentation and examples.