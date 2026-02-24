# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:59:45 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT

"""

# =============================================================================
# Imports
# =============================================================================
from mth5.groups import BaseGroup, MasterSurveyGroup


# =============================================================================
# Experiment Group
# =============================================================================


class ExperimentGroup(BaseGroup):
    """
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

    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    @BaseGroup.metadata.getter
    def metadata(self):
        """Overwrite get metadata to include station information"""

        # need the try statement for when the file is initiated there is no
        # /Station group yet
        try:
            self._metadata.surveys = []
            for key in self.surveys_group.groups_list:
                key_group = self.surveys_group.get_survey(key)
                self._metadata.surveys.append(key_group.metadata)

        except KeyError:
            pass

        return self._metadata

    @property
    def surveys_group(self):
        return MasterSurveyGroup(self.hdf5_group["Surveys"])
