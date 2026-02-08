mth5.io.zen.z3d_metadata
========================

.. py:module:: mth5.io.zen.z3d_metadata

.. autoapi-nested-parse::

   Created on Wed Aug 24 11:35:59 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.zen.z3d_metadata.Z3DMetadata


Module Contents
---------------

.. py:class:: Z3DMetadata(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None, **kwargs: Any)

   Read metadata information from a Z3D file and make each metadata entry an attribute.

   The attributes are left in capitalization of the Z3D file format.

   :param fn: Full path to Z3D file.
   :type fn: str | pathlib.Path, optional
   :param fid: File object (e.g., open(Z3Dfile, 'rb')).
   :type fid: BinaryIO, optional
   :param \*\*kwargs: Additional keyword arguments to set as attributes.
   :type \*\*kwargs: dict

   .. attribute:: _header_length

      Length of header in bits (512).

      :type: int

   .. attribute:: _metadata_length

      Length of metadata blocks (512).

      :type: int

   .. attribute:: _schedule_metadata_len

      Length of schedule meta data (512).

      :type: int

   .. attribute:: board_cal

      Board calibration array with frequency, rate, amplitude, phase.

      :type: np.ndarray | None

   .. attribute:: cal_ant

      Antenna calibration information.

      :type: str | None

   .. attribute:: cal_board

      Board calibration dictionary.

      :type: dict | None

   .. attribute:: cal_ver

      Calibration version.

      :type: str | None

   .. attribute:: ch_azimuth

      Channel azimuth.

      :type: str | None

   .. attribute:: ch_cmp

      Channel component.

      :type: str | None

   .. attribute:: ch_length

      Channel length (or number of coils).

      :type: str | None

   .. attribute:: ch_number

      Channel number on the ZEN board.

      :type: str | None

   .. attribute:: ch_xyz1

      Channel xyz location.

      :type: str | None

   .. attribute:: ch_xyz2

      Channel xyz location.

      :type: str | None

   .. attribute:: ch_cres

      Channel resistance.

      :type: str | None

   .. attribute:: coil_cal

      Coil calibration array (frequency, amplitude, phase).

      :type: np.ndarray | None

   .. attribute:: fid

      File object.

      :type: BinaryIO | None

   .. attribute:: find_metadata

      Boolean flag for finding metadata.

      :type: bool

   .. attribute:: fn

      Full path to Z3D file.

      :type: str | pathlib.Path | None

   .. attribute:: gdp_operator

      Operator of the survey.

      :type: str | None

   .. attribute:: gdp_progver

      Program version.

      :type: str | None

   .. attribute:: gdp_temp

      GDP temperature.

      :type: str | None

   .. attribute:: gdp_volt

      GDP voltage.

      :type: str | None

   .. attribute:: job_by

      Job performed by.

      :type: str | None

   .. attribute:: job_for

      Job for.

      :type: str | None

   .. attribute:: job_name

      Job name.

      :type: str | None

   .. attribute:: job_number

      Job number.

      :type: str | None

   .. attribute:: line_name

      Survey line name.

      :type: str | None

   .. attribute:: m_tell

      Location in the file where the last metadata block was found.

      :type: int

   .. attribute:: notes

      Additional notes from metadata.

      :type: str | None

   .. attribute:: rx_aspace

      Electrode spacing.

      :type: str | None

   .. attribute:: rx_sspace

      Receiver spacing.

      :type: str | None

   .. attribute:: rx_xazimuth

      X azimuth of electrode.

      :type: str | None

   .. attribute:: rx_xyz0

      Receiver xyz coordinates.

      :type: str | None

   .. attribute:: rx_yazimuth

      Y azimuth of electrode.

      :type: str | None

   .. attribute:: rx_zpositive

      Z positive direction (default 'down').

      :type: str

   .. attribute:: station

      Station name.

      :type: str | None

   .. attribute:: survey_type

      Type of survey.

      :type: str | None

   .. attribute:: unit_length

      Length units (m).

      :type: str | None

   .. attribute:: count

      Counter for metadata blocks read.

      :type: int

   .. rubric:: Examples

   >>> from mth5.io.zen import Z3DMetadata
   >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
   >>> header_obj = Z3DMetadata(fn=Z3Dfn)
   >>> header_obj.read_metadata()


   .. py:attribute:: logger


   .. py:attribute:: fn
      :type:  str | pathlib.Path | None
      :value: None



   .. py:attribute:: fid
      :type:  BinaryIO | None
      :value: None



   .. py:attribute:: find_metadata
      :type:  bool
      :value: True



   .. py:attribute:: board_cal
      :type:  list | numpy.ndarray | None
      :value: None



   .. py:attribute:: coil_cal
      :type:  list | numpy.ndarray | None
      :value: None



   .. py:attribute:: m_tell
      :type:  int
      :value: 0



   .. py:attribute:: cal_ant
      :type:  str | None
      :value: None



   .. py:attribute:: cal_board
      :type:  dict[str, Any] | None
      :value: None



   .. py:attribute:: cal_ver
      :type:  str | None
      :value: None



   .. py:attribute:: ch_azimuth
      :type:  str | None
      :value: None



   .. py:attribute:: ch_cmp
      :type:  str | None
      :value: None



   .. py:attribute:: ch_length
      :type:  str | None
      :value: None



   .. py:attribute:: ch_number
      :type:  str | None
      :value: None



   .. py:attribute:: ch_xyz1
      :type:  str | None
      :value: None



   .. py:attribute:: ch_xyz2
      :type:  str | None
      :value: None



   .. py:attribute:: ch_cres
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_operator
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_progver
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_volt
      :type:  str | None
      :value: None



   .. py:attribute:: gdp_temp
      :type:  str | None
      :value: None



   .. py:attribute:: job_by
      :type:  str | None
      :value: None



   .. py:attribute:: job_for
      :type:  str | None
      :value: None



   .. py:attribute:: job_name
      :type:  str | None
      :value: None



   .. py:attribute:: job_number
      :type:  str | None
      :value: None



   .. py:attribute:: rx_aspace
      :type:  str | None
      :value: None



   .. py:attribute:: rx_sspace
      :type:  str | None
      :value: None



   .. py:attribute:: rx_xazimuth
      :type:  str | None
      :value: None



   .. py:attribute:: rx_xyz0
      :type:  str | None
      :value: None



   .. py:attribute:: rx_yazimuth
      :type:  str | None
      :value: None



   .. py:attribute:: rx_zpositive
      :type:  str
      :value: 'down'



   .. py:attribute:: line_name
      :type:  str | None
      :value: None



   .. py:attribute:: survey_type
      :type:  str | None
      :value: None



   .. py:attribute:: unit_length
      :type:  str | None
      :value: None



   .. py:attribute:: station
      :type:  str | None
      :value: None



   .. py:attribute:: count
      :type:  int
      :value: 0



   .. py:attribute:: notes
      :type:  str | None
      :value: None



   .. py:method:: read_metadata(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None) -> None

      Read metadata from Z3D file.

      Parses the metadata blocks in a Z3D file and populates the object's
      attributes with the extracted values. Also reads calibration data
      for both board and coil calibrations.

      :param fn: Full path to file. If None, uses the instance's fn attribute.
      :type fn: str | pathlib.Path, optional
      :param fid: Open file object. If None, uses the instance's fid attribute or
                  opens the file specified by fn.
      :type fid: BinaryIO, optional

      :raises UnicodeDecodeError: If metadata blocks cannot be decoded as text.

      .. rubric:: Notes

      This method reads metadata blocks sequentially from the Z3D file,
      starting after the header and schedule metadata sections. It processes:

      - Standard metadata records with key=value pairs
      - Board calibration data (cal.brd format)
      - Coil calibration data (cal.ant format)
      - Calibration data blocks (caldata format)

      The method automatically determines the station name from available
      metadata fields in the following priority:
      1. line_name + rx_xyz0 (first coordinate)
      2. rx_stn
      3. ch_stn



