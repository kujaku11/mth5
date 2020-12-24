# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:19:07 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
from mth5.metadata import Base
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards
from mth5.utils.exceptions import MTSchemaError

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# filter
# =============================================================================
class Filtered(Base):
    __doc__ = write_lines(ATTR_DICT["filtered"])

    def __init__(self, **kwargs):
        self._name = []
        self._applied = []
        self.name = None
        self.applied = None
        self.comments = None
        super().__init__(attr_dict=ATTR_DICT["filtered"], **kwargs)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, names):
        if names is None:
            self._name = ["none"]
            return

        if isinstance(names, str):
            self._name = [ss.strip().lower() for ss in names.split(",")]
        elif isinstance(names, list):
            self._name = [ss.strip().lower() for ss in names]
        elif isinstance(names, np.ndarray):
            names = names.astype(np.unicode_)
            self._name = [ss.strip().lower() for ss in names]
        else:
            msg = "names must be a string or list of strings not {0}, type {1}"
            self.logger.error(msg.format(names, type(names)))
            raise MTSchemaError(msg.format(names, type(names)))

        check = self._check_consistency()
        if not check:
            self.logger.debug(
                "Filter names and applied lists are not the "
                + "same size. Be sure to check the inputs."
                + " names = {0}, applied = {1}".format(self._name, self._applied)
            )

    @property
    def applied(self):
        return self._applied

    @applied.setter
    def applied(self, applied):
        if not isinstance(applied, (list, tuple, np.ndarray)):
            if applied in [None, "none", "None", "NONE", "null", 0, "0"]:
                self._applied = [False]
                return
        if applied == []:
            self.applied = [False]
            return

        if isinstance(applied, str):
            if applied.find("[") >= 0:
                applied = applied.replace("[", "").replace("]", "")

            if applied.count(",") > 0:
                applied_list = [ss.strip().lower() for ss in applied.split(",")]
            else:
                applied_list = [ss.lower() for ss in applied.split()]
        elif isinstance(applied, list):
            applied_list = applied
        elif isinstance(applied, bool):
            applied_list = [applied]
        # the returned type from a hdf5 dataset is a numpy array.
        elif isinstance(applied, np.ndarray):
            applied_list = list(applied)
            if applied_list == []:
                applied_list = [False]
        else:
            msg = "applied must be a string or list of strings not {0}"
            self.logger.error(msg.format(applied))
            raise MTSchemaError(msg.format(applied))

        bool_list = []
        for app_bool in applied_list:
            if app_bool is None:
                bool_list.append(False)
            if isinstance(app_bool, str):
                if app_bool.lower() in ["false", "0"]:
                    bool_list.append(False)
                elif app_bool.lower() in ["true", "1"]:
                    bool_list.append(True)
                else:
                    msg = "Filter.applied must be [ True | False ], not {0}"
                    self.logger.error(msg.format(app_bool))
                    raise MTSchemaError(msg.format(app_bool))
            elif isinstance(app_bool, bool):
                bool_list.append(app_bool)
            elif isinstance(app_bool, np.bool_):
                bool_list.append(bool(app_bool))
            else:
                msg = "Filter.applied must be [True | False], not {0}"
                self.logger.error(msg.format(app_bool))
        self._applied = bool_list

        # check for consistency
        check = self._check_consistency()
        if not check:
            self.logger.debug(
                "Filter names and applied lists are not the "
                + "same size. Be sure to check the inputs."
                + ". name = {0}, applied = {1}".format(self._name, self._applied)
            )

    def _check_consistency(self):
        # check for consistency
        if self._name is not None:
            if self._applied is None:
                self.logger.warning("Need to input filter.applied")
                return False
            if len(self._name) == 1:
                if len(self._applied) == 1:
                    return True
            elif len(self._name) > 1:
                if len(self._applied) == 1:
                    self.logger.debug(
                        "Assuming all filters have been "
                        + "applied as {0}".format(self._applied[0])
                    )
                    return True
                elif len(self._applied) > 1:
                    if len(self._applied) != len(self._name):
                        self.logger.waring(
                            "Applied and filter names "
                            + "should be the same length. "
                            + "Appied={0}, names={1}".format(
                                len(self._applied), len(self._name)
                            )
                        )
                        return False
        else:
            return False
