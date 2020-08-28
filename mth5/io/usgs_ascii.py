# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:54:09 2020

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================

import os
import time
import datetime
import sys
import glob
import logging
from io import StringIO
import collections

import gzip
import urllib as url
import xml.etree.ElementTree as ET

import numpy as np
import scipy.signal as sps
import pandas as pd

from mth5 import timeseries
from mth5.utils.mttime import MTime

# =============================================================================
#  Metadata for usgs ascii file
# =============================================================================
class AsciiMetadata:
    """
    Container for all the important metadata in a USGS ascii file.

    ========================= =================================================
    Attributes                Description
    ========================= =================================================
    SurveyID                  Survey name
    SiteID                    Site name
    RunID                     Run number
    SiteLatitude              Site latitude in decimal degrees WGS84
    SiteLongitude             Site longitude in decimal degrees WGS84
    SiteElevation             Site elevation according to national map meters
    AcqStartTime              Start time of station YYYY-MM-DDThh:mm:ss UTC
    AcqStopTime               Stop time of station YYYY-MM-DDThh:mm:ss UTC
    AcqSmpFreq                Sampling rate samples/second
    AcqNumSmp                 Number of samples
    Nchan                     Number of channels
    CoordinateSystem          [ Geographic North | Geomagnetic North ]
    ChnSettings               Channel settings, see below
    MissingDataFlag           Missing data value
    ========================= =================================================

    *ChnSettings*
    ========================= =================================================
    Keys                      Description
    ========================= =================================================
    ChnNum                    SiteID+channel number
    ChnID                     Component [ ex | ey | hx | hy | hz ]
    InstrumentID              Data logger + sensor number
    Azimuth                   Setup angle of componet in degrees relative to
                              CoordinateSystem
    Dipole_Length             Dipole length in meters
    ========================= =================================================


    """
    def __init__(self, fn=None, **kwargs):
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.fn = fn
        self.SurveyID = None
        self.RunID = None
        self.MissingDataFlag = np.NaN
        self.CoordinateSystem = None
        self._metadata_len = 30
        self.declination = 0.0
        self._latitude = None
        self._longitude = None
        self._start = MTime()
        self._end = MTime()
        self._station = None

        self._key_list = ['SurveyID',
                          'SiteID',
                          'RunID',
                          'SiteLatitude',
                          'SiteLongitude',
                          'SiteElevation',
                          'AcqStartTime',
                          'AcqStopTime',
                          'AcqSmpFreq',
                          'AcqNumSmp',
                          'Nchan',
                          'CoordinateSystem',
                          'ChnSettings',
                          'MissingDataFlag',
                          'DataSet']

        self._chn_settings = ['ChnNum',
                              'ChnID',
                              'InstrumentID',
                              'Azimuth',
                              'Dipole_Length']
        self._chn_fmt = {'ChnNum':'<8',
                         'ChnID':'<6',
                         'InstrumentID':'<12',
                         'Azimuth':'>7.1f',
                         'Dipole_Length':'>14.1f'}
        
        self.channel_dict = dict([(comp, dict([(key, None) for key in self._chn_settings]))
                                   for comp in ['ex', 'ey', 'hx', 'hy', 'hz']])

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    @property
    def SiteID(self):
        return self._station
    @SiteID.setter
    def SiteID(self, station):
        self._station = station

    @property
    def SiteLatitude(self):
        return self._latitude
        #return gis_tools.convert_position_float2str(self._latitude)

    @SiteLatitude.setter
    def SiteLatitude(self, lat):
        self._latitude = lat

    @property
    def SiteLongitude(self):
        return self._longitude
        #return gis_tools.convert_position_float2str(self._longitude)

    @SiteLongitude.setter
    def SiteLongitude(self, lon):
        self._longitude = lon

    @property
    def SiteElevation(self):
        """
        get elevation from national map
        """
        # the url for national map elevation query
        nm_url = r"https://nationalmap.gov/epqs/pqs.php?x={0:.5f}&y={1:.5f}&units=Meters&output=xml"

        # call the url and get the response
        try:
            response = url.request.urlopen(nm_url.format(self._longitude, self._latitude))
        except url.error.HTTPError:
            self.logger.error("could not connect to get elevation from national map.")
            self.logger.debug(nm_url.format(self._longitude, self._latitude))
            return -666

        # read the xml response and convert to a float
        info = ET.ElementTree(ET.fromstring(response.read()))
        info = info.getroot()
        for elev in info.iter('Elevation'):
            nm_elev = float(elev.text)
        return nm_elev

    @property
    def AcqStartTime(self):
        return self._start.iso_str

    @AcqStartTime.setter
    def AcqStartTime(self, time_string):
        self._start.from_str(time_string)

    @property
    def AcqStopTime(self):
        return self._end.iso_str

    @AcqStopTime.setter
    def AcqStopTime(self, time_string):
        self._end.from_str(time_string)

    @property
    def Nchan(self):
        return self._chn_num

    @Nchan.setter
    def Nchan(self, n_channel):
        try:
            self._chn_num = int(n_channel)
        except ValueError:
            self.logger.warning(f"{n_channel} is not a number, setting Nchan to 0")

    @property
    def AcqSmpFreq(self):
        return self._sampling_rate
    @AcqSmpFreq.setter
    def AcqSmpFreq(self, df):
        self._sampling_rate = float(df)

    @property
    def AcqNumSmp(self):
        return self._n_samples

    @AcqNumSmp.setter
    def AcqNumSmp(self, n_samples):
        self._n_samples = int(n_samples)
        
    def get_component_info(self, comp):
        """
        
        :param comp: DESCRIPTION
        :type comp: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        for key, kdict in self.channel_dict.items():
            if kdict['ChnID'].lower() == comp.lower():
                return kdict
            
        return None

    def read_metadata(self, fn=None, meta_lines=None):
        """
        Read in a meta from the raw string or file.  Populate all metadata
        as attributes.

        :param fn: full path to USGS ascii file
        :type fn: string

        :param meta_lines: lines of metadata to read
        :type meta_lines: list
        """
        chn_find = False
        comp = 0
        self.channel_dict = {}
        if fn is not None:
            self.fn = fn
        if self.fn is not None:
            with open(self.fn, 'r') as fid:
                meta_lines = [fid.readline() for ii in range(self._metadata_len)]
        for ii, line in enumerate(meta_lines):
            if line.find(':') > 0:
                key, value = line.strip().split(':', 1)
                value = value.strip()
                if len(value) < 1 and key == 'DataSet':
                    chn_find = False
                    # return the line that the data starts on that way can
                    # read in as a numpy object or pandas
                    return ii+1
                elif len(value) < 1:
                    chn_find = True
                if 'elev' in key.lower():
                    pass
                else:
                    setattr(self, key, value)
            elif 'coordinate' in line:
                self.CoordinateSystem = ' '.join(line.strip().split()[-2:])
            else:
                if chn_find is True:
                    if 'chnnum' in line.lower():
                        ch_key = line.strip().split()
                    else:
                        line_list = line.strip().split()
                        if len(line_list) == 5:
                            comp += 1
                            self.channel_dict[comp] = {}
                            for key, value in zip(ch_key, line_list):
                                if key.lower() in ['azimuth', 'dipole_length']:
                                    value = float(value)
                                self.channel_dict[comp][key] = value
                        else:
                            self.logger.warning('Not sure what line this is')

    def write_metadata(self, chn_list=['Ex', 'Ey', 'Hx', 'Hy', 'Hz']):
        """
        Write out metadata in the format of USGS ascii.

        :return: list of metadate lines.

        .. note:: meant to use '\n'.join(lines) to write out in a file.
        """

        lines = []
        for key in self._key_list:
            if key in ['ChnSettings']:
                lines.append('{0}:'.format(key))
                lines.append(' '.join(self._chn_settings))
                for chn_key in chn_list:
                    chn_line = []
                    try:
                        for comp_key in self._chn_settings:
                            chn_line.append('{0:{1}}'.format(self.channel_dict[chn_key][comp_key],
                                            self._chn_fmt[comp_key]))
                        lines.append(''.join(chn_line))
                    except KeyError:
                        pass
            elif key in ['DataSet']:
                lines.append('{0}:'.format(key))
                return lines
            else:
                if key in ['SiteLatitude', 'SiteLongitude']:
                    lines.append('{0}: {1:.5f}'.format(key, getattr(self, key)))
                else:
                    lines.append('{0}: {1}'.format(key, getattr(self, key)))

        return lines


# =============================================================================
# Class for the asc file
# =============================================================================
class USGSasc(AsciiMetadata):
    """
    Read and write USGS ascii formatted time series.

    =================== =======================================================
    Attributes          Description
    =================== =======================================================
    ts                  Pandas dataframe holding the time series data
    fn                  Full path to .asc file
    station_dir         Full path to station directory
    meta_notes          Notes of how the station was collected
    =================== =======================================================

    ============================== ============================================
    Methods                        Description
    ============================== ============================================
    get_z3d_db                     Get Pandas dataframe for schedule block
    locate_mtft24_cfg_fn           Look for a mtft24.cfg file in station_dir
    get_metadata_from_mtft24       Get metadata from mtft24.cfg file
    get_metadata_from_survey_csv   Get metadata from survey csv file
    fill_metadata                  Fill Metadata container from a meta_array
    read_asc_file                  Read in USGS ascii file
    convert_electrics              Convert electric channels to mV/km
    write_asc_file                 Write an USGS ascii file
    write_station_info_metadata    Write metadata to a .cfg file
    ============================== ============================================

    :Example: ::

        >>> zc = Z3DCollection()
        >>> fn_list = zc.get_time_blocks(z3d_path)
        >>> zm = USGSasc()
        >>> zm.SurveyID = 'iMUSH'
        >>> zm.get_z3d_db(fn_list[0])
        >>> zm.read_mtft24_cfg()
        >>> zm.CoordinateSystem = 'Geomagnetic North'
        >>> zm.SurveyID = 'MT'
        >>> zm.write_asc_file(str_fmt='%15.7e')
        >>> zm.write_station_info_metadata()
    """

    def __init__(self, fn=None, **kwargs):
        super(USGSasc, self).__init__(fn) 
        self.ts = None
        self.station_dir = os.getcwd()
        self.meta_notes = None
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
            
    @property
    def hx(self):
        """HX"""
        if self.ts is not None:
            comp_dict = self.get_component_info('hx')
            if comp_dict is None: 
                return None
            meta_dict = {
                "channel_number": comp_dict['ChnNum'],
                "component": "hx",
                "measurement_azimuth": comp_dict['Azimuth'],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "magnetic",
                "units": "nanotesla",
                "sensor.id": comp_dict['InstrumentID'],
            }

            return timeseries.MTTS(
                "magnetic",
                data=self.ts.hx.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
            )
        return None


    @property
    def hy(self):
        """hy"""
        if self.ts is not None:
            comp_dict = self.get_component_info('hy')
            if comp_dict is None: 
                return None
            meta_dict = {
                "channel_number": comp_dict['ChnNum'],
                "component": "hy",
                "measurement_azimuth": comp_dict['Azimuth'],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "magnetic",
                "units": "nanotesla",
                "sensor.id": comp_dict['InstrumentID'],
            }

            return timeseries.MTTS(
                "magnetic",
                data=self.ts.hy.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
            )
        return None
    
    @property
    def hz(self):
        """hz"""
        if self.ts is not None:
            comp_dict = self.get_component_info('hz')
            if comp_dict is None: 
                return None
            meta_dict = {
                "channel_number": comp_dict['ChnNum'],
                "component": "hz",
                "measurement_azimuth": comp_dict['Azimuth'],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "magnetic",
                "units": "nanotesla",
                "sensor.id": comp_dict['InstrumentID'],
            }

            return timeseries.MTTS(
                "magnetic",
                data=self.ts.hz.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
            )
        return None
    
    @property
    def ex(self):
        """ex"""
        if self.ts is not None:
            comp_dict = self.get_component_info('ex')
            if comp_dict is None: 
                return None
            meta_dict = {
                "channel_number": comp_dict['ChnNum'],
                "component": "ex",
                "measurement_azimuth": comp_dict['Azimuth'],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "electric",
                "units":  "millivolts per kilometer",
                "sensor.id": comp_dict['InstrumentID'],
                "dipole_length": comp_dict['Dipole_Length'],
            }

            return timeseries.MTTS(
                "electric",
                data=self.ts.ex.to_numpy(),
                channel_metadata={"electric": meta_dict},
            )
        return None
    
    @property
    def ey(self):
        """ey"""
        if self.ts is not None:
            comp_dict = self.get_component_info('ey')
            if comp_dict is None: 
                return None
            meta_dict = {
                "channel_number": comp_dict['ChnNum'],
                "component": "ey",
                "measurement_azimuth": comp_dict['Azimuth'],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "electric",
                "units": "millivolts per kilometer",
                "sensor.id": comp_dict['InstrumentID'],
                "dipole_length": comp_dict['Dipole_Length'],
            }

            return timeseries.MTTS(
                "electric",
                data=self.ts.ey.to_numpy(),
                channel_metadata={"electric": meta_dict},
            )
        return None
    
    @property
    def electric_channels(self):
        electrics = []
        for key, kdict in self.channel_dict.items():
            if 'e' in kdict['ChnID'].lower():
                electrics.append(kdict['ChnID'].lower())

        return ', '.join(electrics)
    
    @property
    def magnetic_channels(self):
        magnetics = []
        for key, kdict in self.channel_dict.items():
            if 'h' in kdict['ChnID'].lower() or 'b' in kdict['ChnID'].lower():
                magnetics.append(kdict['ChnID'].lower())

        return ', '.join(magnetics)
    
    @property
    def run_xarray(self):
        """ Get xarray for run """
        if self.ts is not None:
            meta_dict = {
                "run": {
                    "channels_recorded_electric": self.electric_channels,
                    "channels_recorded_magnetic": self.magnetic_channels,
                    "channels_recorded_auxiliary": None,
                    "comments": self.comments,
                    "id": self.SiteID,
                    "sample_rate": self.sample_rate,
                    "time_period.end": self.AcqStartTime,
                    "time_period.start": self.AcqStopTime,
                }
            }
    
            return timeseries.RunTS(
                array_list=[self.hx, self.hy, self.hz, self.ex, self.ey],
                run_metadata=meta_dict,
            )
        
        return None

    def fill_metadata(self, meta_arr):
        """
        Fill in metadata from time array made by
        Z3DCollection.check_time_series.

        :param meta_arr: structured array of metadata for the Z3D files to be
                         combined.
        :type meta_arr: np.ndarray
        """
        try:
            self.AcqNumSmp = self.ts.shape[0]
        except AttributeError:
            pass
        self.AcqSmpFreq = meta_arr['df'].mean()
        self.AcqStartTime = meta_arr['start'].max()
        self.AcqStopTime = meta_arr['stop'].min()
        try:
            self.Nchan = self.ts.shape[1]
        except AttributeError:
            self.Nchan = meta_arr.shape[0]
        self.RunID = 1
        self.SiteLatitude = np.median(meta_arr['lat'])
        self.SiteLongitude = np.median(meta_arr['lon'])
        self.SiteID = os.path.basename(meta_arr['fn'][0]).split('_')[0]
        self.station_dir = os.path.dirname(meta_arr['fn'][0])

        # if geographic coordinates add in declination
        if 'geographic' in self.CoordinateSystem.lower():
            meta_arr['ch_azimuth'][np.where(meta_arr['comp'] != 'hz')] += self.declination

        # fill channel dictionary with appropriate values
        self.channel_dict = dict([(comp.capitalize(),
                                   {'ChnNum':'{0}{1}'.format(self.SiteID, ii+1),
                                    'ChnID':meta_arr['comp'][ii].capitalize(),
                                    'InstrumentID':meta_arr['ch_box'][ii],
                                    'Azimuth':meta_arr['ch_azimuth'][ii],
                                    'Dipole_Length':meta_arr['ch_length'][ii],
                                    'n_samples':meta_arr['n_samples'][ii],
                                    'n_diff':meta_arr['t_diff'][ii],
                                    'std':meta_arr['std'][ii],
                                    'start':meta_arr['start'][ii]})
                                   for ii, comp in enumerate(meta_arr['comp'])])
        for ii, comp in enumerate(meta_arr['comp']):
            if 'h' in comp.lower():
                self.channel_dict[comp.capitalize()]['InstrumentID'] += '-{0}'.format(meta_arr['ch_num'])

    def read_asc_file(self, fn=None):
        """
        Read in a USGS ascii file and fill attributes accordingly.

        :param fn: full path to .asc file to be read in
        :type fn: string
        """
        if fn is not None:
            self.fn = fn
        st = datetime.datetime.now()
        data_line = self.read_metadata()
        self.ts = pd.read_csv(self.fn,
                         delim_whitespace=True,
                         skiprows=data_line,
                         dtype=np.float32)
        dt_freq = '{0:.0f}N'.format(1./(self.AcqSmpFreq)*1E9)
        dt_index = pd.date_range(start=self.AcqStartTime,
                                 periods=self.AcqNumSmp,
                                 freq=dt_freq)
        self.ts.index = dt_index
        self.ts.columns = self.ts.columns.str.lower()
            
        et = datetime.datetime.now()
        read_time = et-st
        self.logger.info('Reading took {0}'.format(read_time.total_seconds()))

    def _make_file_name(self, save_path=None, compression=True,
                        compress_type='zip'):
        """
        get the file name to save to

        :param save_path: full path to directory to save file to
        :type save_path: string

        :param compression: compress file
        :type compression: [ True | False ]

        :return: save_fn
        :rtype: string

        """
        # make the file name to save to
        if save_path is not None:
            save_fn = os.path.join(save_path,
                                   '{0}_{1}T{2}_{3:.0f}.asc'.format(self.SiteID,
                                    self._start_time.strftime('%Y-%m-%d'),
                                    self._start_time.strftime('%H%M%S'),
                                    self.AcqSmpFreq))
        else:
            save_fn = os.path.join(self.station_dir,
                                   '{0}_{1}T{2}_{3:.0f}.asc'.format(self.SiteID,
                                    self._start_time.strftime('%Y-%m-%d'),
                                    self._start_time.strftime('%H%M%S'),
                                    self.AcqSmpFreq))

        if compression:
            if compress_type == 'zip':
                save_fn = save_fn + '.zip'
            elif compress_type == 'gzip':
                save_fn = save_fn + '.gz'

        return save_fn

    def write_asc_file(self, save_fn=None, chunk_size=1024, str_fmt='%15.7e',
                       full=True, compress=False, save_dir=None,
                       compress_type='zip', convert_electrics=True):
        """
        Write an ascii file in the USGS ascii format.

        :param save_fn: full path to file name to save the merged ascii to
        :type save_fn: string

        :param chunck_size: chunck size to write file in blocks, larger numbers
                            are typically slower.
        :type chunck_size: int

        :param str_fmt: format of the data as written
        :type str_fmt: string

        :param full: write out the complete file, mostly for testing.
        :type full: boolean [ True | False ]

        :param compress: compress file
        :type compress: boolean [ True | False ]

        :param compress_type: compress file using zip or gzip
        :type compress_type: boolean [ zip | gzip ]
        """
        # get the filename to save to
        save_fn = self._make_file_name(save_path=save_dir,
                                       compression=compress,
                                       compress_type=compress_type)
        # get the number of characters in the desired string
        s_num = int(str_fmt[1:str_fmt.find('.')])

        # convert electric fields into mV/km
        if convert_electrics:
            self.convert_electrics()

        print('==> {0}'.format(save_fn))
        print('START --> {0}'.format(time.ctime()))
        st = datetime.datetime.now()

        # write meta data first
        # sort channel information same as columns
        meta_lines = self.write_metadata(chn_list=[c.capitalize() for c in self.ts.columns])
        if compress == True and compress_type == 'gzip':
            with gzip.open(save_fn, 'wb') as fid:
                h_line = [''.join(['{0:>{1}}'.format(c.capitalize(), s_num)
                          for c in self.ts.columns])]
                fid.write('\n'.join(meta_lines+h_line) + '\n')

                # write out data
                if full is False:
                    out = np.array(self.ts[0:chunk_size])
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = '\n'.join([''.join(out[ii, :]) for ii in range(out.shape[0])])
                    fid.write(lines+'\n')
                    print('END --> {0}'.format(time.ctime()))
                    et = datetime.datetime.now()
                    write_time = et-st
                    print('Writing took: {0} seconds'.format(write_time.total_seconds()))
                    return

                for chunk in range(int(self.ts.shape[0]/chunk_size)):
                    out = np.array(self.ts[chunk*chunk_size:(chunk+1)*chunk_size])
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = '\n'.join([''.join(out[ii, :]) for ii in range(out.shape[0])])
                    fid.write(lines+'\n')

        else:
            if compress == True and compress_type == 'zip':
                print('ZIPPING')
                save_fn = save_fn[0:-4]
                zip_file = True
                print(zip_file)
            with open(save_fn, 'w') as fid:
                h_line = [''.join(['{0:>{1}}'.format(c.capitalize(), s_num)
                          for c in self.ts.columns])]
                fid.write('\n'.join(meta_lines+h_line) + '\n')

                # write out data
                if full is False:
                    out = np.array(self.ts[0:chunk_size])
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = '\n'.join([''.join(out[ii, :]) for ii in range(out.shape[0])])
                    fid.write(lines+'\n')
                    print('END --> {0}'.format(time.ctime()))
                    et = datetime.datetime.now()
                    write_time = et-st
                    print('Writing took: {0} seconds'.format(write_time.total_seconds()))
                    return

                for chunk in range(int(self.ts.shape[0]/chunk_size)):
                    out = np.array(self.ts[chunk*chunk_size:(chunk+1)*chunk_size])
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = '\n'.join([''.join(out[ii, :]) for ii in range(out.shape[0])])
                    fid.write(lines+'\n')

        # for some fucking reason, all interal variables don't exist anymore
        # and if you try to do the zipping nothing happens, so have to do
        # it externally.  WTF
        print('END -->   {0}'.format(time.ctime()))
        et = datetime.datetime.now()
        write_time = et-st
        print('Writing took: {0} seconds'.format(write_time.total_seconds()))

    def write_station_info_metadata(self, save_dir=None, mtft_bool=False):
        """
        write out station info that can later be put into a data base

        the data we need is
            - site name
            - site id number
            - lat
            - lon
            - national map elevation
            - hx azimuth
            - ex azimuth
            - hy azimuth
            - hz azimuth
            - ex length
            - ey length
            - start date
            - end date
            - instrument type (lp, bb)
            - number of channels

        """
        if save_dir is not None:
            save_fn = os.path.join(save_dir,
                                   '{0}_{1}T{2}_{3:.0f}.cfg'.format(self.SiteID,
                                    self._start_time.strftime('%Y-%m-%d'),
                                    self._start_time.strftime('%H%M%S'),
                                    self.AcqSmpFreq))
        else:
            save_fn = os.path.join(self.station_dir,
                                       '{0}_{1}T{2}_{3:.0f}.cfg'.format(self.SiteID,
                                        self._start_time.strftime('%Y-%m-%d'),
                                        self._start_time.strftime('%H%M%S'),
                                        self.AcqSmpFreq))
        meta_dict = {}
        key = '{0}_{1}T{2}_{3:.0f}'.format(self.SiteID,
                                    self._start_time.strftime('%Y-%m-%d'),
                                    self._start_time.strftime('%H%M%S'),
                                    self.AcqSmpFreq)
        meta_dict[key] = {}
        meta_dict[key]['site'] = self.SiteID
        meta_dict[key]['lat'] = self._latitude
        meta_dict[key]['lon'] = self._longitude
        meta_dict[key]['elev'] = self.SiteElevation
        meta_dict[key]['mtft_file'] = mtft_bool
        try:
            meta_dict[key]['hx_azimuth'] = self.channel_dict['Hx']['Azimuth']
            meta_dict[key]['hx_id'] = self.channel_dict['Hx']['InstrumentID'].split('-')[1]
            meta_dict[key]['hx_nsamples'] = self.channel_dict['Hx']['n_samples']
            meta_dict[key]['hx_ndiff'] = self.channel_dict['Hx']['n_diff']
            meta_dict[key]['hx_std'] = self.channel_dict['Hx']['std']
            meta_dict[key]['hx_start'] = self.channel_dict['Hx']['start']
            meta_dict[key]['zen_num'] = self.channel_dict['Hx']['InstrumentID'].split('-')[0]
            meta_dict[key]['hx_num'] = self.channel_dict['Hx']['ChnNum'][-1]
        except KeyError:
            meta_dict[key]['hx_azimuth'] = None
            meta_dict[key]['hx_id'] = None
            meta_dict[key]['hx_nsamples'] = None
            meta_dict[key]['hx_ndiff'] = None
            meta_dict[key]['hx_std'] = None
            meta_dict[key]['hx_start'] = None
            meta_dict[key]['hx_num'] = None

        try:
            meta_dict[key]['hy_azimuth'] = self.channel_dict['Hy']['Azimuth']
            meta_dict[key]['hy_id'] = self.channel_dict['Hy']['InstrumentID'].split('-')[1]
            meta_dict[key]['hy_nsamples'] = self.channel_dict['Hy']['n_samples']
            meta_dict[key]['hy_ndiff'] = self.channel_dict['Hy']['n_diff']
            meta_dict[key]['hy_std'] = self.channel_dict['Hy']['std']
            meta_dict[key]['hy_start'] = self.channel_dict['Hy']['start']
            meta_dict[key]['zen_num'] = self.channel_dict['Hy']['InstrumentID'].split('-')[0]
            meta_dict[key]['hy_num'] = self.channel_dict['Hy']['ChnNum'][-1:]
        except KeyError:
            meta_dict[key]['hy_azimuth'] = None
            meta_dict[key]['hy_id'] = None
            meta_dict[key]['hy_nsamples'] = None
            meta_dict[key]['hy_ndiff'] = None
            meta_dict[key]['hy_std'] = None
            meta_dict[key]['hy_start'] = None
            meta_dict[key]['hy_num'] = None
        try:
            meta_dict[key]['hz_azimuth'] = self.channel_dict['Hz']['Azimuth']
            meta_dict[key]['hz_id'] = self.channel_dict['Hz']['InstrumentID'].split('-')[1]
            meta_dict[key]['hz_nsamples'] = self.channel_dict['Hz']['n_samples']
            meta_dict[key]['hz_ndiff'] = self.channel_dict['Hz']['n_diff']
            meta_dict[key]['hz_std'] = self.channel_dict['Hz']['std']
            meta_dict[key]['hz_start'] = self.channel_dict['Hz']['start']
            meta_dict[key]['zen_num'] = self.channel_dict['Hz']['InstrumentID'].split('-')[0]
            meta_dict[key]['hz_num'] = self.channel_dict['Hz']['ChnNum'][-1:]
        except KeyError:
            meta_dict[key]['hz_azimuth'] = None
            meta_dict[key]['hz_id'] = None
            meta_dict[key]['hz_nsamples'] = None
            meta_dict[key]['hz_ndiff'] = None
            meta_dict[key]['hz_std'] = None
            meta_dict[key]['hz_start'] = None
            meta_dict[key]['hz_num'] = None

        try:
            meta_dict[key]['ex_azimuth'] = self.channel_dict['Ex']['Azimuth']
            meta_dict[key]['ex_id'] = self.channel_dict['Ex']['InstrumentID']
            meta_dict[key]['ex_len'] = self.channel_dict['Ex']['Dipole_Length']
            meta_dict[key]['ex_nsamples'] = self.channel_dict['Ex']['n_samples']
            meta_dict[key]['ex_ndiff'] = self.channel_dict['Ex']['n_diff']
            meta_dict[key]['ex_std'] = self.channel_dict['Ex']['std']
            meta_dict[key]['ex_start'] = self.channel_dict['Ex']['start']
            meta_dict[key]['zen_num'] = self.channel_dict['Ex']['InstrumentID']
            meta_dict[key]['ex_num'] = self.channel_dict['Ex']['ChnNum'][-1:]
        except KeyError:
            meta_dict[key]['ex_azimuth'] = None
            meta_dict[key]['ex_id'] = None
            meta_dict[key]['ex_len'] = None
            meta_dict[key]['ex_nsamples'] = None
            meta_dict[key]['ex_ndiff'] = None
            meta_dict[key]['ex_std'] = None
            meta_dict[key]['ex_start'] = None
            meta_dict[key]['ex_num'] = None
        try:
            meta_dict[key]['ey_azimuth'] = self.channel_dict['Ey']['Azimuth']
            meta_dict[key]['ey_id'] = self.channel_dict['Ey']['InstrumentID']
            meta_dict[key]['ey_len'] = self.channel_dict['Ey']['Dipole_Length']
            meta_dict[key]['ey_nsamples'] = self.channel_dict['Ey']['n_samples']
            meta_dict[key]['ey_ndiff'] = self.channel_dict['Ey']['n_diff']
            meta_dict[key]['ey_std'] = self.channel_dict['Ey']['std']
            meta_dict[key]['ey_start'] = self.channel_dict['Ey']['start']
            meta_dict[key]['zen_num'] = self.channel_dict['Ey']['InstrumentID']
            meta_dict[key]['ey_num'] = self.channel_dict['Ey']['ChnNum'][-1:]
        except KeyError:
            meta_dict[key]['ey_azimuth'] = None
            meta_dict[key]['ey_id'] = None
            meta_dict[key]['ey_len'] = None
            meta_dict[key]['ey_nsamples'] = None
            meta_dict[key]['ey_ndiff'] = None
            meta_dict[key]['ey_std'] = None
            meta_dict[key]['ey_start'] = None
            meta_dict[key]['ey_num'] = None

        meta_dict[key]['start_date'] = self.AcqStartTime
        meta_dict[key]['stop_date'] = self.AcqStopTime
        meta_dict[key]['sampling_rate'] = self.AcqSmpFreq
        meta_dict[key]['n_samples'] = self.AcqNumSmp
        meta_dict[key]['n_chan'] = self.Nchan


        if meta_dict[key]['zen_num'] in [24, 25, 26, 46, '24', '25', '26', '46',
                                        'ZEN24', 'ZEN25', 'ZEN26', 'ZEN46']:
            meta_dict[key]['collected_by'] = 'USGS'
        else:
            meta_dict[key]['collected_by'] = 'OSU'

        # in the old OSU z3d files there are notes in the metadata section
        # pass those on
        meta_dict[key]['notes'] = self.meta_notes

        mtcfg.write_dict_to_configfile(meta_dict, save_fn)

        return save_fn
