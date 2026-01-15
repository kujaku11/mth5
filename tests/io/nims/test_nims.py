import numpy as np
import pytest
from mth5.io.nims.header import NIMSHeader

@pytest.fixture
def nims_instance():
    return NIMS()

def test_unwrap_sequence(nims_instance):
    wrapped = np.array([0, 1, 2, 255, 0, 1, 2], dtype=np.uint8)
    unwrapped = nims_instance.unwrap_sequence(wrapped)
    expected = np.array([0, 1, 2, 255, 256, 257, 258])
    assert np.array_equal(unwrapped, expected)

def test_locate_duplicate_blocks(nims_instance):
    sequence = np.array([0, 1, 2, 2, 3, 4])
    duplicates = nims_instance._locate_duplicate_blocks(sequence)
    assert duplicates[0]["sequence_index"] == 2
def test_find_sequence(nims_instance):
    data = np.array([0, 1, 131, 0, 1, 131, 0])
    data = data.reshape(-1, 1)
    result = nims_instance.find_sequence(data, block_sequence=[1, 131])
    assert isinstance(result, np.ndarray)

def test_make_dt_index(nims_instance):
    start_time = "2023-01-01T00:00:00"
    sample_rate = 2
    n_samples = 4
    dt_index = nims_instance.make_dt_index(start_time, sample_rate, n_samples=n_samples)
    assert len(dt_index) == 4
