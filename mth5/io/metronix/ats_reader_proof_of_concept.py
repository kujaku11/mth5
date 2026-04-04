"""
Proof of concept for reading Metronix ATS files with XML metadata.

>>> from mth5.io.metronix.metronix_atss import ATSS, read_atss
>>>
>>> # Using the ATSS class directly
>>> atss = ATSS('data/station001.atss')
>>> data = atss.read_atss()
>>> channel_ts = atss.to_channel_ts()
>>>
>>> # Using the convenience function
>>> channel_ts = read_atss('data/station001.atss')
"""

import pathlib


HOME = pathlib.Path.home()
METRONIX_EXAMPLE_DATA_PATH = (
    HOME / "software" / "irismt" / "mth5_test_data" / "mth5_test_data" / "metronix"
)

# One-off run from Mayotte
# Note we do not have an xml for this run, but we at least have the .ats file to test reading the timeseries
MAYOTTE_RUN_DATA_PATH = METRONIX_EXAMPLE_DATA_PATH / "meas_2024-12-01_15-31-21"
EXAMPLE_ATS_FILE = MAYOTTE_RUN_DATA_PATH / "104_V01_C00_R124_TEx_BL_4H.ats"

# Data from MTHotel
EXAMPLE_SURVEY = METRONIX_EXAMPLE_DATA_PATH / "Northern_Mining"
EXAMPLE_STATION = EXAMPLE_SURVEY / "ts" / "Kocatepe"
EXAMPLE_RUN = EXAMPLE_STATION / "meas_2009-08-20_13-22-00"
EXAMPLE_XML_FILE = (
    EXAMPLE_RUN / "085_2009-08-20_13-22-00_2009-08-21_07-00-00_R001_128H.xml"
)


def test_metronix_run_xml():
    """Test the MetronixRunXML class with the example data."""
    from mth5.io.metronix import MetronixRunXML

    assert EXAMPLE_XML_FILE.exists(), f"XML file {EXAMPLE_XML_FILE} does not exist"

    # Parse the XML file
    print(f"\n{'='*60}")
    print(f"Testing MetronixRunXML with: {EXAMPLE_XML_FILE.name}")
    print(f"{'='*60}\n")

    xml = MetronixRunXML(EXAMPLE_XML_FILE)
    print(xml)

    # Test properties
    print(f"\n--- Run Properties ---")
    print(f"Sample Rate: {xml.sample_rate} Hz")
    print(f"N Channels:  {xml.n_channels}")
    print(f"Start Time:  {xml.start_time}")
    print(f"Stop Time:   {xml.stop_time}")
    print(f"Site Name:   {xml.site_name}")
    print(f"Survey ID:   {xml.survey_id}")

    # Test channel access
    print(f"\n--- Channel Details ---")
    for ch_id in xml.channel_ids:
        ch_info = xml.get_channel_info(ch_id)
        print(f"\nChannel {ch_id}:")
        print(f"  Type:        {xml.get_channel_type(ch_id)}")
        print(f"  Component:   {ch_info.get('channel_type', '?')}")
        print(f"  ATS File:    {xml.get_ats_filename(ch_id)}")
        print(f"  N Samples:   {xml.get_n_samples(ch_id):,}")
        print(f"  LSB Scale:   {xml.get_scaling_factor(ch_id)}")
        print(f"  Chopper:     {xml.get_chopper_state(ch_id)}")
        print(f"  Sensor:      {ch_info.get('sensor_type', '?')}")
        if "dipole_length" in ch_info:
            print(f"  Dipole:      {ch_info['dipole_length']} m")

    # Test mt_metadata object creation
    print(f"\n--- mt_metadata Objects ---")
    for ch_id in xml.channel_ids:
        ch_meta = xml.get_channel_metadata(ch_id)
        if ch_meta is not None:
            print(f"\nChannel {ch_id} ({type(ch_meta).__name__}):")
            print(f"  component:   {ch_meta.component}")
            print(f"  sample_rate: {ch_meta.sample_rate}")
            print(
                f"  time_period: {ch_meta.time_period.start} to {ch_meta.time_period.end}"
            )
            if hasattr(ch_meta, "sensor"):
                print(f"  sensor.type: {ch_meta.sensor.type}")

    # Test sensor response filter
    print(f"\n--- Sensor Response Filters ---")
    for ch_id in xml.channel_ids:
        fap = xml.get_sensor_response_filter(ch_id)
        if fap is not None:
            print(f"Channel {ch_id}: {fap.name}")
            print(
                f"  Frequencies: {len(fap.frequencies)} points, {min(fap.frequencies):.3f} - {max(fap.frequencies):.1f} Hz"
            )
        else:
            print(f"Channel {ch_id}: No sensor response filter")

    # Test Run metadata
    print(f"\n--- Run Metadata ---")
    run_meta = xml.get_run_metadata()
    print(f"Run sample_rate: {run_meta.sample_rate}")
    print(
        f"Run time_period: {run_meta.time_period.start} to {run_meta.time_period.end}"
    )
    print(f"N channels in run: {len(run_meta.channels)}")

    print(f"\n{'='*60}")
    print("MetronixRunXML test completed successfully!")
    print(f"{'='*60}\n")


def test_ats_reader():
    """Test reading actual ATS binary data from Mayotte files."""
    import numpy as np

    from mth5.io.metronix import ATS

    assert EXAMPLE_ATS_FILE.exists(), f"ATS file {EXAMPLE_ATS_FILE} does not exist"

    print(f"\n{'='*60}")
    print(f"Testing ATS Reader with: {EXAMPLE_ATS_FILE.name}")
    print(f"{'='*60}\n")

    # Create ATS reader (no XML available for this data)
    ats = ATS(EXAMPLE_ATS_FILE)

    print(f"--- File Info (from filename parsing) ---")
    print(f"System Number:  {ats.system_number}")
    print(f"System Name:    {ats.system_name}")
    print(f"Channel Number: {ats.channel_number}")
    print(f"Component:      {ats.component}")
    print(f"Sample Rate:    {ats.sample_rate} Hz")
    print(f"Channel Type:   {ats.channel_type}")
    print(f"File Size:      {ats.fn.stat().st_size:,} bytes")
    print(f"N Samples:      {ats.n_samples:,}")
    print(f"Has XML:        {ats.has_metadata_file()}")

    # Read the actual data
    print(f"\n--- Reading Binary Data ---")
    data = ats.read_ats(apply_scaling=False)  # Raw counts (no XML for scaling)
    print(f"Data shape:     {data.shape}")
    print(f"Data dtype:     {data.dtype}")
    print(f"Data range:     [{data.min():.1f}, {data.max():.1f}]")
    print(f"Data mean:      {data.mean():.2f}")
    print(f"Data std:       {data.std():.2f}")

    # Read a subset
    print(f"\n--- Reading Data Subset ---")
    data_subset = ats.read_ats(start=1000, stop=2000, apply_scaling=False)
    print(f"Subset shape:   {data_subset.shape}")
    print(f"Subset matches full: {np.allclose(data[1000:2000], data_subset)}")

    # Test reading all channels in the directory
    print(f"\n--- All Channels in Run ---")
    for ats_file in sorted(MAYOTTE_RUN_DATA_PATH.glob("*.ats")):
        ats_ch = ATS(ats_file)
        data_ch = ats_ch.read_ats(apply_scaling=False)
        print(
            f"  {ats_ch.component}: {data_ch.shape[0]:,} samples, "
            f"range [{data_ch.min():.0f}, {data_ch.max():.0f}]"
        )

    print(f"\n{'='*60}")
    print("ATS Reader test completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_metronix_run_xml()
    test_ats_reader()
