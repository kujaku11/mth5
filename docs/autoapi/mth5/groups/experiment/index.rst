mth5.groups.experiment
======================

.. py:module:: mth5.groups.experiment

.. autoapi-nested-parse::

   Created on Wed Dec 23 16:59:45 2020

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license:
       MIT



Classes
-------

.. autoapisummary::

   mth5.groups.experiment.ExperimentGroup


Module Contents
---------------

.. py:class:: ExperimentGroup(group, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Utility class to hold general information about the experiment and
   accompanying metadata for an MT experiment.

   To access the hdf5 group directly use `ExperimentGroup.hdf5_group`.

   >>> experiment = ExperimentGroup(hdf5_group)
   >>> experiment.hdf5_group.ref
   <HDF5 Group Reference>

   .. note:: All attributes should be input into the metadata object, that
            way all input will be validated against the metadata standards.
            If you change attributes in metadata object, you should run the
            `ExperimentGroup.write_metadata()` method.  This is a temporary
            solution, working on an automatic updater if metadata is changed.

   >>> experiment.metadata.existing_attribute = 'update_existing_attribute'
   >>> experiment.write_metadata()

   If you want to add a new attribute this should be done using the
   `metadata.add_base_attribute` method.

   >>> experiment.metadata.add_base_attribute('new_attribute',
   >>> ...                                'new_attribute_value',
   >>> ...                                {'type':str,
   >>> ...                                 'required':True,
   >>> ...                                 'style':'free form',
   >>> ...                                 'description': 'new attribute desc.',
   >>> ...                                 'units':None,
   >>> ...                                 'options':[],
   >>> ...                                 'alias':[],
   >>> ...                                 'example':'new attribute

   .. tip:: If you want ot add stations, reports, etc to the experiment this
             should be done from the MTH5 object.  This is to avoid
             duplication, at least for now.

   To look at what the structure of ``/Experiment`` looks like:

       >>> experiment
       /Experiment:
       ====================
           |- Group: Surveys
           -----------------
           |- Group: Reports
           -----------------
           |- Group: Standards
           -------------------
           |- Group: Stations
           ------------------



   .. py:method:: metadata()

      Overwrite get metadata to include station information



   .. py:property:: surveys_group


