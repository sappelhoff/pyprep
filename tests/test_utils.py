"""Test various helper functions."""
import numpy as np

from pyprep.utils import _mat_quantile, _mat_iqr


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
