# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:55:34 2020

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
from mth5.metadata import Declination

ATTR_DICT = Standards().ATTR_DICT
# ============================================================================
# location
# ============================================================================
class Location(Base):
    __doc__ = write_lines(ATTR_DICT["location"])

    def __init__(self, **kwargs):

        self.comments = None
        self.datum = "WGS84"
        self.declination = Declination()

        self._elevation = 0.0
        self._latitude = 0.0
        self._longitude = 0.0

        super(Location, self).__init__(attr_dict=ATTR_DICT["location"], **kwargs)

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, lat):
        self._latitude = self._assert_lat_value(lat)

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, lon):
        self._longitude = self._assert_lon_value(lon)

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, elev):
        self._elevation = self._assert_elevation_value(elev)

    def _assert_lat_value(self, latitude):
        """
        Make sure the latitude value is in decimal degrees, if not change it.
        And that the latitude is within -90 < lat > 90.

        :param latitude: latitude in decimal degrees or other format
        :type latitude: float or string
        """
        if latitude in [None, "None", "none", "unknown"]:
            self.logger.debug("Latitude is None, setting to 0")
            return 0.0
        try:
            lat_value = float(latitude)

        except TypeError:
            self.logger.debug("Could not convert {0} setting to 0".format(latitude))
            return 0.0

        except ValueError:
            self.logger.debug("Latitude is a string {0}".format(latitude))
            lat_value = self._convert_position_str2float(latitude)

        if abs(lat_value) >= 90:
            msg = (
                "latitude value = {0} is unacceptable!".format(lat_value)
                + ".  Must be |Latitude| > 90"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return lat_value

    def _assert_lon_value(self, longitude):
        """
        Make sure the longitude value is in decimal degrees, if not change it.
        And that the latitude is within -180 < lat > 180.

        :param latitude: longitude in decimal degrees or other format
        :type latitude: float or string
        """
        if longitude in [None, "None", "none", "unknown"]:
            self.logger.debug("Longitude is None, setting to 0")
            return 0.0
        try:
            lon_value = float(longitude)

        except TypeError:
            self.logger.debug("Could not convert {0} setting to 0".format(longitude))
            return 0.0

        except ValueError:
            self.logger.debug("Longitude is a string {0}".format(longitude))
            lon_value = self._convert_position_str2float(longitude)

        if abs(lon_value) >= 180:
            msg = (
                "longitude value = {0} is unacceptable!".format(lon_value)
                + ".  Must be |longitude| > 180"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return lon_value

    def _assert_elevation_value(self, elevation):
        """
        make sure elevation is a floating point number

        :param elevation: elevation as a float or string that can convert
        :type elevation: float or str
        """

        try:
            elev_value = float(elevation)
        except (ValueError, TypeError):
            msg = "Could not convert {0} to a number setting to 0".format(elevation)
            self.logger.debug(msg)
            elev_value = 0.0

        return elev_value

    def _convert_position_float2str(self, position):
        """
        Convert position float to a string in the format of DD:MM:SS.

        :param position: decimal degrees of latitude or longitude
        :type position: float

        :returns: latitude or longitude in format of DD:MM:SS.ms
        """

        assert type(position) is float, "Given value is not a float"

        deg = int(position)
        sign = 1
        if deg < 0:
            sign = -1

        deg = abs(deg)
        minutes = (abs(position) - deg) * 60.0
        # need to round seconds to 4 decimal places otherwise machine precision
        # keeps the 60 second roll over and the string is incorrect.
        sec = np.round((minutes - int(minutes)) * 60.0, 4)
        if sec >= 60.0:
            minutes += 1
            sec = 0

        if int(minutes) == 60:
            deg += 1
            minutes = 0

        position_str = "{0}:{1:02.0f}:{2:05.2f}".format(
            sign * int(deg), int(minutes), sec
        )
        self.logger.debug("Converted {0} to {1}".format(position, position_str))

        return position_str

    def _convert_position_str2float(self, position_str):
        """
        Convert a position string in the format of DD:MM:SS to decimal degrees

        :param position: latitude or longitude om DD:MM:SS.ms
        :type position: float

        :returns: latitude or longitude as a float
        """

        if position_str in [None, "None"]:
            return None

        p_list = position_str.split(":")
        if len(p_list) != 3:
            msg = "{0} not correct format, should be DD:MM:SS".format(position_str)
            self.logger.error(msg)
            raise ValueError(msg)

        deg = float(p_list[0])
        minutes = self._assert_minutes(float(p_list[1]))
        sec = self._assert_seconds(float(p_list[2]))

        # get the sign of the position so that when all are added together the
        # position is in the correct place
        sign = 1
        if deg < 0:
            sign = -1

        position_value = sign * (abs(deg) + minutes / 60.0 + sec / 3600.0)

        self.logger.debug("Converted {0} to {1}".format(position_str, position_value))

        return position_value

    def _assert_minutes(self, minutes):
        if not 0 <= minutes < 60.0:
            msg = (
                "minutes should be 0 < > 60, currently {0:.0f}".format(minutes)
                + " conversion will account for non-uniform"
                + "timne. Be sure to check accuracy."
            )
            self.logger.warning(msg)

        return minutes

    def _assert_seconds(self, seconds):
        if not 0 <= seconds < 60.0:
            msg = (
                "seconds should be 0 < > 60, currently {0:.0f}".format(seconds)
                + " conversion will account for non-uniform"
                + "timne. Be sure to check accuracy."
            )
            self.logger.warning(msg)

        return seconds

