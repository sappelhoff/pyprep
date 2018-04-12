"""Test the noisy module."""
from pyprep.noisy import (mad,
                          iqr,
                          find_bad_by_nan,
                          find_bad_by_flat,
                          find_bad_by_deviation,
                          find_bad_by_hf_noise,
                          find_bad_by_correlation,
                          find_bad_by_ransac,
                          find_noisy_channels)


def test_bogus():
    """Test bogus."""
    assert True
