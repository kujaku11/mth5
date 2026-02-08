mth5.io.zen.coil_response
=========================

.. py:module:: mth5.io.zen.coil_response

.. autoapi-nested-parse::

   Read an amtant.cal file provided by Zonge.


   Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
   is a fancy way of saying f x 6 and f x 8.




Classes
-------

.. autoapisummary::

   mth5.io.zen.coil_response.CoilResponse


Module Contents
---------------

.. py:class:: CoilResponse(calibration_file: str | pathlib.Path | None = None, angular_frequency: bool = False)

   Read ANT4 coil calibration files from Zonge (``amtant.cal``).

   This class parses a Zonge antenna calibration file and exposes a
   :class:`mt_metadata.timeseries.filters.FrequencyResponseTableFilter` for a
   specified coil number.

   :param calibration_file: Path to the antenna calibration file. If provided the file will be
                            read during initialization, by default None.
   :type calibration_file: str | Path | None, optional
   :param angular_frequency: If True, reported frequencies will be converted to angular frequency
                             (rad/s), by default False.
   :type angular_frequency: bool, optional

   .. attribute:: coil_calibrations

      Mapping of coil serial numbers to a structured numpy array containing
      frequency, amplitude, and phase columns.

      :type: dict[str, numpy.ndarray]

   .. rubric:: Examples

   >>> from mth5.mth5.io.zen.coil_response import CoilResponse
   >>> cr = CoilResponse('amtant.cal')
   >>> fap = cr.get_coil_response_fap(1234)
   >>> print(fap.name)


   .. py:attribute:: logger


   .. py:attribute:: coil_calibrations
      :type:  dict[str, numpy.ndarray]


   .. py:property:: calibration_file


   .. py:attribute:: angular_frequency
      :type:  bool
      :value: False



   .. py:method:: file_exists() -> bool

      Check to make sure the file exists

      :returns: True if the file exists, False if it does not
      :rtype: bool



   .. py:method:: read_antenna_file(antenna_calibration_file: str | pathlib.Path | None = None) -> None

      Read a Zonge antenna calibration file and parse coil responses.

      The expected file format contains blocks starting with an "antenna"
      header line that provides the base frequency followed by lines with
      coil serial number and amplitude/phase values for the 6th and 8th
      harmonics.

      :param antenna_calibration_file: Optional path to the antenna calibration file. If provided, it
                                       overrides the instance ``calibration_file``.
      :type antenna_calibration_file: str | Path | None, optional

      .. rubric:: Notes

      Phase values in the file are expected in milliradians and are
      converted to radians.



   .. py:method:: get_coil_response_fap(coil_number: int | str, extrapolate: bool = True) -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter

      Read an amtant.cal file provided by Zonge.


      Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
      is a fancy way of saying f * 6 and f * 8.

      :param coil_number: ANT4 4 digit serial number
      :type coil_number: int or str
      :param extrapolate: If True, extrapolate the frequency response to low and high frequencies,
                          by default True
      :type extrapolate: bool, optional

      :returns: Frequency look up table for the specified coil number.
      :rtype: FrequencyResponseTableFilter

      :raises KeyError: If the coil number is not found in the calibration file.

      .. rubric:: Notes

      Ensure that the antenna calibration file has been read prior to calling
      this method. This can be done by providing the calibration file during
      initialization or by calling :meth:`read_antenna_file`.



   .. py:method:: extrapolate(fap: mt_metadata.timeseries.filters.FrequencyResponseTableFilter) -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter

      Extrapolate a frequency/amplitude/phase table using log-linear pads.

      :param fap: Frequency response object to extrapolate.
      :type fap: FrequencyResponseTableFilter

      :returns: A copy of ``fap`` with low- and high-frequency extrapolated
                values appended.
      :rtype: FrequencyResponseTableFilter



   .. py:method:: has_coil_number(coil_number: int | str | None) -> bool

      Test if coil number is in the antenna file

      :param coil_number: ANT4 serial number
      :type coil_number: int or str or None

      :returns: True if the coil is found, False if it is not
      :rtype: bool



