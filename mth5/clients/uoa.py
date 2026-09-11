# -*- coding: utf-8 -*-
"""
UoA Client
==========

Client for creating MTH5 files from University of Adelaide instrument data.
Supports both PR6-24 (Earth Data Logger) and Orange Box instruments.

Created on Thu Nov 7 15:35:00 2025

@author: bkay
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mt_io.uoa import UoACollection, read_orange, read_uoa

from mth5.clients.base import ClientBase
from mth5.mth5 import MTH5


# =============================================================================


class UoAClient(ClientBase):
    """
    Client for creating MTH5 files from University of Adelaide instrument data.

    Supports:
    - **PR6-24 (Earth Data Logger)**: ASCII files (.BX, .BY, .BZ, .EX, .EY)
    - **Orange Box**: Binary files (.BIN)

    :param data_path: Path to directory or file(s)
    :type data_path: str or pathlib.Path
    :param save_path: Path to save MTH5 file, defaults to data_path
    :type save_path: str or pathlib.Path, optional
    :param instrument_type: 'pr624' or 'orange', defaults to 'pr624'
    :type instrument_type: str, optional
    :param mth5_filename: Name of output MTH5 file, defaults to "from_uoa.h5"
    :type mth5_filename: str, optional

    :Example:

        >>> from mth5.clients import UoAClient
        >>> # PR6-24 data
        >>> uc = UoAClient("/path/to/pr624/data", instrument_type='pr624')
        >>> mth5_path = uc.make_mth5_from_uoa(
        ...     survey_id="Survey01",
        ...     station_id="MT001",
        ...     sample_rate=10.0,
        ...     dipole_length_ex=50.0,
        ...     dipole_length_ey=50.0
        ... )
        >>>
        >>> # Orange Box data
        >>> uc = UoAClient("/path/to/orange/data", instrument_type='orange')
        >>> mth5_path = uc.make_mth5_from_uoa(
        ...     survey_id="Survey01",
        ...     station_id="ST61",
        ...     dipole_length_ex=100.0,
        ...     dipole_length_ey=100.0
        ... )

    """

    # reader arguments the collection also needs
    _collection_keys = (
        "sample_rate",
        "dipole_length_ex",
        "dipole_length_ey",
        "latitude",
        "longitude",
        "elevation",
    )

    def __init__(
        self,
        data_path,
        save_path=None,
        instrument_type="pr624",
        sample_rates=None,
        mth5_filename="from_uoa.h5",
        **kwargs,
    ):
        # EDL carries no header, so the rate usually arrives as sample_rate
        self.sample_rates_given = sample_rates is not None
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates if sample_rates is not None else [1],
            mth5_filename=mth5_filename,
            **kwargs,
        )

        self.instrument_type = instrument_type.lower()
        if self.instrument_type not in ["pr624", "orange"]:
            raise ValueError(
                f"instrument_type must be 'pr624' or 'orange', got '{instrument_type}'"
            )

        # the Orange Box writes one file per deployment and has no collection
        if self.instrument_type == "pr624":
            self.collection = UoACollection(self.data_path)

    def make_mth5_from_uoa(self, survey_id, station_id, run_id="001", **kwargs):
        """
        Create an MTH5 file from UoA instrument data.

        :param survey_id: Survey identifier
        :type survey_id: str
        PR6-24 stations are split into runs on the gaps. The station id comes
        from the file names, so a mid-deployment rename gives two stations.

        :param station_id: Station identifier
        :type station_id: str
        :param run_id: Run identifier for the Orange Box, which records one
         run per deployment. PR6-24 run names come from the collection.
        :type run_id: str, optional
        :param kwargs: Additional keyword arguments for reader:

            **PR6-24 arguments:**
                - sample_rate (float): REQUIRED - sample rate in Hz
                - sensor_type (str): 'bartington' or 'lemi120'
                - dipole_length_ex (float): Ex dipole length in meters
                - dipole_length_ey (float): Ey dipole length in meters
                - calibration_fn_bx (str): LEMI-120 .rsp file for Bx
                - calibration_fn_by (str): LEMI-120 .rsp file for By
                - calibration_fn_bz (str): LEMI-120 .rsp file for Bz
                - latitude (float): Station latitude
                - longitude (float): Station longitude
                - elevation (float): Station elevation

            **Orange Box arguments:**
                - dipole_length_ex (float): Ex dipole length in meters
                - dipole_length_ey (float): Ey dipole length in meters
                - latitude (float): Station latitude
                - longitude (float): Station longitude
                - elevation (float): Station elevation

        :type kwargs: dict
        :return: Path to created MTH5 file
        :rtype: pathlib.Path

        :Example:

            >>> from mth5.clients import UoAClient
            >>> # PR6-24 long-period data
            >>> uc = UoAClient("/data/pr624", instrument_type='pr624')
            >>> mth5_file = uc.make_mth5_from_uoa(
            ...     survey_id="MySurvey",
            ...     station_id="MT001",
            ...     sample_rate=10.0,
            ...     sensor_type='bartington',
            ...     dipole_length_ex=50.0,
            ...     dipole_length_ey=50.0
            ... )
            >>>
            >>> # Orange Box data
            >>> uc = UoAClient("/data/orange", instrument_type='orange')
            >>> mth5_file = uc.make_mth5_from_uoa(
            ...     survey_id="MySurvey",
            ...     station_id="ST61",
            ...     dipole_length_ex=100.0,
            ...     dipole_length_ey=100.0
            ... )

        """

        if self.instrument_type == "pr624":
            return self._make_mth5_from_pr624(survey_id, station_id, **kwargs)

        # For Orange Box, handle both single file and directory
        data_path = Path(self.data_path)
        if data_path.is_dir():
            # Get all .BIN files in directory
            bin_files = sorted(data_path.glob("*.BIN"))
            if not bin_files:
                raise FileNotFoundError(f"No .BIN files found in {data_path}")
            run_ts = read_orange(bin_files, station_id=station_id, **kwargs)
        else:
            run_ts = read_orange(self.data_path, station_id=station_id, **kwargs)

        # Set run ID
        run_ts.run_metadata.id = run_id

        # Create MTH5 file
        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, "w")
            survey_group = m.add_survey(survey_id)
            station_group = survey_group.stations_group.add_station(station_id)
            run_group = station_group.add_run(run_id)
            run_group.from_runts(run_ts)
            station_group.metadata.update(run_ts.station_metadata)
            station_group.write_metadata()
            survey_group.update_metadata()

        return self.save_path

    def _make_mth5_from_pr624(self, survey_id, station_id, **kwargs):
        """
        Build the MTH5 run by run, using the collection to find the runs.

        The collection knows where a recording stops and starts again.

        :param survey_id: Survey identifier
        :type survey_id: str
        :param station_id: Station identifier
        :type station_id: str
        :param kwargs: Keyword arguments for :func:`mt_io.uoa.read_uoa`
        :type kwargs: dict
        :return: Path to created MTH5 file
        :rtype: pathlib.Path
        """

        self.collection.survey_id = survey_id
        for key in self._collection_keys:
            if kwargs.get(key) is not None:
                setattr(self.collection, key, kwargs[key])

        # the collection selects on sample rate
        if not self.sample_rates_given and kwargs.get("sample_rate") is not None:
            self.sample_rates = [kwargs["sample_rate"]]

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, "w")
            survey_group = m.add_survey(self.collection.survey_id)

            for found_station, run_dict in runs.items():
                station_group = survey_group.stations_group.add_station(
                    found_station
                )
                for run_id, run_df in run_dict.items():
                    run_group = station_group.add_run(run_id)
                    run_ts = read_uoa(
                        run_df.fn.to_list(), station_id=found_station, **kwargs
                    )
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)

                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            survey_group.update_metadata()

        return self.save_path
