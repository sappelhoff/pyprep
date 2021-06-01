"""Test various helper functions."""
import numpy as np
import pytest
from numpy.random import RandomState

from pyprep.utils import (
    _correlate_arrays,
    _eeglab_create_highpass,
    _get_random_subset,
    _mat_iqr,
    _mat_quantile,
    _mat_round,
    print_progress,
)


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

       % Add NaNs to input and re-test
       tst(1, :) = nan;
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

    # Add NaNs to test data
    tst_nan = tst.copy()
    tst_nan[0, :] = np.NaN

    # Create arrays containing MATLAB results for NaN test case
    quantile_expected = np.asarray([0.9712, 0.9880, 0.9807])
    iqr_expected = np.asarray([0.4764, 0.5188, 0.5044])

    # Test quantile equivalence with MATLAB for array with NaN
    quantile_actual = _mat_quantile(tst_nan, 0.98, axis=0)
    assert all(np.isclose(quantile_expected, quantile_actual, atol=0.001))

    # Test IQR equivalence with MATLAB for array with NaN
    iqr_actual = _mat_iqr(tst_nan, axis=0)
    assert all(np.isclose(iqr_expected, iqr_actual, atol=0.001))

    # Test quantile behaviour in special cases
    assert _mat_quantile([0.3], 0.98) == 0.3
    assert np.isnan(_mat_quantile([], 0.98))

    # Test error with 3 or more dimensional input
    tst_3d = tst.reshape(3, 5, -1)
    with pytest.raises(ValueError):
        _mat_quantile(tst_3d, 0.98, axis=0)


def test_get_random_subset():
    """Test the function for getting random channel subsets."""
    # Generate test data
    rng = RandomState(435656)
    chans = range(1, 61)

    # Compare random subset equivalence with MATLAB
    expected_picks = [6, 47, 55, 31, 29, 44, 36, 15]
    actual_picks = _get_random_subset(chans, size=8, rand_state=rng)
    assert all(np.equal(expected_picks, actual_picks))


def test_correlate_arrays():
    """Test MATLAB PREP-compatible array correlation function.

    MATLAB code used to generate the comparison results:

    .. code-block:: matlab

       % Generate test data
       rng(435656);
       a = rand(100, 3) - 0.5;
       b = rand(100, 3) - 0.5;

       % Calculate correlations
       correlations = sum(a.*b)./(sqrt(sum(a.^2)).*sqrt(sum(b.^2)));

    """
    # Generate test data
    np.random.seed(435656)
    a = np.random.rand(3, 100) - 0.5
    b = np.random.rand(3, 100) - 0.5

    # Test regular Pearson correlation
    corr_expected = np.asarray([-0.0898, 0.0340, -0.1068])
    corr_actual = _correlate_arrays(a, b)
    assert all(np.isclose(corr_expected, corr_actual, atol=0.001))

    # Test correlation equivalence with MATLAB PREP
    corr_expected = np.asarray([-0.0898, 0.0327, -0.1140])
    corr_actual = _correlate_arrays(a, b, matlab_strict=True)
    assert all(np.isclose(corr_expected, corr_actual, atol=0.001))


def test_eeglab_create_highpass():
    """Test EEGLAB-equivalent high-pass filter creation.

    NOTE: EEGLAB values were obtained using breakpoints in ``pop_eegfiltnew``,
    since filter creation and data filtering are both done in the same function.
    Values here are first 4 values of the array ``b`` which contains the FIR
    filter coefficents used by the function.

    """
    # Compare initial FIR filter coefficents with EEGLAB
    expected_vals = [5.3691e-5, 5.4165e-5, 5.4651e-5, 5.5149e-5]
    actual_vals = _eeglab_create_highpass(cutoff=1.0, srate=256)[:4]
    assert all(np.isclose(expected_vals, actual_vals, atol=0.001))

    # Compare middle FIR filter coefficent with EEGLAB
    vals = _eeglab_create_highpass(cutoff=1.0, srate=256)
    expected_val = 0.9961
    actual_val = vals[len(vals) // 2]
    assert np.isclose(expected_val, actual_val, atol=0.001)


def test_print_progress(capsys):
    """Test the function for printing progress updates within a loop."""
    # Test printing start value
    print_progress(1, 20)
    captured = capsys.readouterr()
    assert captured.out == "Progress: "

    # Test printing end values
    iterations = 27
    for i in range(iterations):
        print_progress(i + 1, iterations, every=0.2)
    captured = capsys.readouterr()
    assert captured.out == "Progress: 20%... 40%... 60%... 80%... 100%\n"

    # Test printing of updates at right times
    iterations = 176
    for i in range(iterations):
        print_progress(i + 1, iterations)
        if (i + 1) == 17:
            captured = capsys.readouterr()
            assert captured.out == "Progress: "
        elif (i + 1) == 18:
            captured = capsys.readouterr()
            assert captured.out == "10%... "
            break

    # Test shifted start value
    iterations = 25
    start = 5
    for i in range(start, iterations + 1):
        print_progress(i, iterations, start=start)
        if i == 6:
            captured = capsys.readouterr()
            assert captured.out == "Progress: "
        elif i == 7:
            captured = capsys.readouterr()
            assert captured.out == "10%... "
            break
