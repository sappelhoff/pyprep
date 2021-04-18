"""Test various helper functions."""
import numpy as np

from pyprep.utils import _mat_round, _mat_quantile, _mat_iqr, _get_random_subset


def test_mat_round():
    """Test the MATLAB-compatible rounding function."""
    # Test normal rounding behaviour
    assert _mat_round(1.5) == 2
    assert _mat_round(0.4) == 0
    assert _mat_round(0.6) == 1

    # Test MATLAB-specific rounding behaviour
    assert _mat_round(0.5) == 1


def test_mat_quantile_iqr():
    """Test MATLAB-compatible quantile and IQR functions.

    MATLAB code used to generate the comparison results:

    .. code-block:: matlab

       % Generate test data
       rng(435656);
       tst = rand(100, 3);

       % Calculate IQR and 0.98 quantile for test data
       quantile(tst, 0.98);
       iqr(tst);

    """
    # Generate test data
    np.random.seed(435656)
    tst = np.transpose(np.random.rand(3, 100))

    # Create arrays containing MATLAB results
    quantile_expected = np.asarray([0.9710, 0.9876, 0.9802])
    iqr_expected = np.asarray([0.4776, 0.5144, 0.4851])

    # Test quantile equivalence with MATLAB
    quantile_actual = _mat_quantile(tst, 0.98, axis=0)
    assert all(np.isclose(quantile_expected, quantile_actual, atol=0.001))

    # Test IQR equivalence with MATLAB
    iqr_actual = _mat_iqr(tst, axis=0)
    assert all(np.isclose(iqr_expected, iqr_actual, atol=0.001))


def test_get_random_subset():
    """Test the function for getting random channel subsets."""
    # Generate test data
    rng = np.random.RandomState(435656)
    chans = range(1, 61)

    # Compare random subset equivalence with MATLAB
    expected_picks = [6, 47, 55, 31, 29, 44, 36, 15]
    actual_picks = _get_random_subset(chans, size=8, rand_state=rng)
    assert all(np.equal(expected_picks, actual_picks))
