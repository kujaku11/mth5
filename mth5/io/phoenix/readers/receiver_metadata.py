# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:06:08 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from loguru import logger

from mt_metadata.timeseries import Survey, Station, Run, Electric, Magnetic

from .helpers import read_json_to_object

# =============================================================================


class PhoenixReceiverMetadata:
    """
    A container for the recmeta.json file used to control the recording

    """

    def __init__(self, fn=None, **kwargs):

        self.fn = fn
        self.obj = None

        self._e_map = {
            "tag": "component",
            "ty": "type",
            "ga": "gain",
            "sampleRate": "sample_rate",
            "pot_p": "contact_resistance.start",
            "pot_n": "contact_resistance.end",
        }

        self._h_map = {
            "tag": "component",
            "ty": "type",
            "ga": "gain",
            "sampleRate": "sample_rate",
            "type_name": "sensor.model",
            "type": "sensor.type",
            "serial": "sensor.id",
        }
        self.logger = logger

        if self.fn is not None:
            self.read()

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, fn):
        if fn is None:
            self._fn = None
        else:
            fn = Path(fn)
            if fn.exists():
                self._fn = Path(fn)
            else:
                raise ValueError(f"Could not find {fn}")

    @property
    def instrument_id(self):
        if self.has_obj():
            return self.obj.instid

    def read(self, fn=None):
        """
        read a config.json file that is in the Phoenix format

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if fn is not None:
            self.fn = fn
        self.obj = read_json_to_object(self.fn)

    def has_obj(self):
        if self.obj is not None:
            return True
        return False

    @property
    def channel_map(self):
        return dict(
            [(d.idx, d.tag.lower()) for d in self.obj.channel_map.mapping]
        )

    @property
    def lp_filter_base_name(self):
        if self.has_obj():
            return (
                f"{self.obj.receiver_commercial_name}_"
                f"{self.obj.receiver_model}_"
                f"{self.obj.instid}"
            ).lower()

    def get_ch_index(self, tag):
        if self.has_obj():
            for item in self.obj.channel_map.mapping:
                if item.tag.lower() == tag.lower():
                    return item.idx
            raise ValueError(f"Could not find {tag} in channel map.")

    def get_ch_tag(self, index):
        if self.has_obj():
            for item in self.obj.channel_map.mapping:
                if item.idx == index:
                    return item.tag
            raise ValueError(f"Could not find {index} in channel map.")

    def _to_electric_metadata(self, tag):
        c = Electric()

        if self.has_obj():
            ch = self.obj.chconfig.chans[self.get_ch_index(tag)]

            for p_key, m_value in self._e_map.items():
                if p_key == "ty":
                    m_value = "electric"
                c.set_attr_from_name(m_value, getattr(ch, p_key))
            c.channel_number = self.get_ch_index(tag)
            c.dipole_length = ch.length1 + ch.length2
            c.units = "volts"
            c.time_period.start = self.obj.start
            c.time_period.end = self.obj.stop
            c.filter.name = [
                f"{self.lp_filter_base_name}_{int(ch.lp)}hz_low_pass",
                f"dipole_{int(c.dipole_length)}m",
            ]
            c.filter.applied = [False, False]
        return c

    def _to_magnetic_metadata(self, tag):
        c = Magnetic()

        if self.has_obj():
            ch = self.obj.chconfig.chans[self.get_ch_index(tag)]

            for p_key, m_value in self._h_map.items():
                if p_key == "ty":
                    m_value = "magnetic"
                try:
                    c.set_attr_from_name(m_value, getattr(ch, p_key))
                except AttributeError:
                    self.logger.error(
                        f"recmeta.json does not contain attribute '{p_key}' for "
                        f"channel '{ch.tag}'."
                    )
            c.channel_number = self.get_ch_index(tag)
            c.sensor.manufacturer = "Phoenix Geophysics"
            c.units = "volts"
            c.time_period.start = self.obj.start
            c.time_period.end = self.obj.stop
            c.filter.name = [
                f"{self.lp_filter_base_name}_{int(ch.lp)}hz_low_pass"
            ]
            c.filter.applied = [False]
            if c.sensor.id is not None:
                c.filter.name.append(f"coil_{c.sensor.id}_response")
                c.filter.applied.append(False)

        return c

    ### should think about putting this part in get_attr
    @property
    def e1_metadata(self):
        return self._to_electric_metadata("e1")

    @property
    def e2_metadata(self):
        return self._to_electric_metadata("e2")

    @property
    def h1_metadata(self):
        return self._to_magnetic_metadata("h1")

    @property
    def h2_metadata(self):
        return self._to_magnetic_metadata("h2")

    @property
    def h3_metadata(self):
        return self._to_magnetic_metadata("h3")

    @property
    def h4_metadata(self):
        return self._to_magnetic_metadata("h4")

    @property
    def h5_metadata(self):
        return self._to_magnetic_metadata("h5")

    @property
    def h6_metadata(self):
        return self._to_magnetic_metadata("h6")

    def get_ch_metadata(self, index):
        """
        get channel metadata from index
        """

        tag = self.get_ch_tag(index)

        return getattr(self, f"{tag.lower()}_metadata")

    ### need to add station and run metadata objects and should be good to go.
    @property
    def run_metadata(self):
        r = Run()
        if self.has_obj():
            r.data_logger.type = self.obj.receiver_model
            r.data_logger.model = self.obj.receiver_commercial_name
            r.data_logger.firmware.version = self.obj.motherboard.mb_fw_ver
            r.data_logger.timing_system.drift = self.obj.timing.tm_drift
        return r

    @property
    def station_metadata(self):
        s = Station()
        if self.has_obj():
            s.id = self.obj.layout.Station_Name
            s.comments = self.obj.layout.Notes
            try:
                s.acquired_by.organization = self.obj.layout.Company_Name
            except AttributeError:
                pass

            s.acquired_by.name = self.obj.layout.Operator
            s.location.latitude = self.obj.timing.gps_lat
            s.location.longitude = self.obj.timing.gps_lon
            s.location.elevation = self.obj.timing.gps_alt
        return s

    @property
    def survey_metadata(self):
        s = Survey()
        if self.has_obj():
            s.id = self.obj.layout.Survey_Name
        return s
