"""Tests for PyPREP GPU acceleration backend and PyTorch device selector."""

from unittest.mock import patch

import numpy as np
import pytest

import pyprep
import pyprep.gpu as gpu
import pyprep.gpu.core as gpu_core

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# HAS_TORCH=False safety guards (always run, no PyTorch required)
# ---------------------------------------------------------------------------


def test_lazy_gpu_import_when_no_torch():
    """Test lazy import behavior when PyTorch is not available."""
    with patch.object(gpu_core, "HAS_TORCH", False):
        with pytest.raises(ImportError, match="PyTorch is required"):
            gpu.get_device("auto")

        with pytest.raises(ImportError, match="PyTorch is required"):
            gpu_core._to_tensor(np.ones((2, 2)), "cpu")

        with pytest.raises(ImportError, match="PyTorch is required"):
            gpu.correlate_windows_gpu(np.ones((2, 100)), sfreq=100.0)

        with pytest.raises(ImportError, match="PyTorch is required"):
            gpu.find_bad_by_deviation_gpu(np.ones((2, 100)))


def test_top_level_pyprep_gpu_attribute():
    """Test that pyprep.gpu is accessible as a top-level attribute."""
    assert hasattr(pyprep, "gpu")


# ---------------------------------------------------------------------------
# Device selection (requires PyTorch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_get_device_auto():
    """Test that auto device selection returns a valid torch.device."""
    dev = gpu.get_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "xpu", "hpu", "cpu")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_get_device_none_equivalent_to_auto():
    """Test that device=None behaves identically to 'auto'."""
    assert gpu.get_device(None).type == gpu.get_device("auto").type


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_get_device_explicit_cpu():
    """Test explicit CPU device selection."""
    dev = gpu.get_device("cpu")
    assert dev.type == "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_get_device_torch_device_passthrough():
    """Test that a torch.device instance is returned unchanged."""
    dev_in = torch.device("cpu")
    dev_out = gpu.get_device(dev_in)
    assert dev_out.type == "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_get_device_invalid_raises():
    """Test that an invalid device string raises RuntimeError."""
    with pytest.raises(RuntimeError):
        gpu.get_device("invalid_device_name_xyz")


# ---------------------------------------------------------------------------
# _to_tensor helper
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_to_tensor_from_numpy():
    """Test conversion from numpy array to tensor."""
    arr = np.random.randn(5, 500).astype(np.float32)
    t = gpu_core._to_tensor(arr, torch.device("cpu"))
    assert isinstance(t, torch.Tensor)
    assert t.shape == (5, 500)
    assert t.device.type == "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_to_tensor_accepts_negative_stride_numpy_array():
    """Test conversion of reversed NumPy data produced by filtering operations."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]

    t = gpu_core._to_tensor(arr, torch.device("cpu"))

    np.testing.assert_array_equal(t.numpy(), arr)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_to_tensor_from_tensor_passthrough():
    """Test that an existing torch.Tensor is moved correctly."""
    t_in = torch.randn(5, 500)
    t_out = gpu_core._to_tensor(t_in, torch.device("cpu"))
    assert isinstance(t_out, torch.Tensor)
    assert t_out.device.type == "cpu"


# ---------------------------------------------------------------------------
# find_bad_by_deviation_gpu
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_find_bad_by_deviation_shape():
    """Test output shape of find_bad_by_deviation_gpu."""
    data = np.random.randn(10, 1000)
    z = gpu.find_bad_by_deviation_gpu(data, device="cpu")
    assert z.shape == (10,)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_find_bad_by_deviation_detects_bad_channel():
    """Test that an artificially large-amplitude channel gets the highest Z-score."""
    np.random.seed(42)
    data = np.random.randn(10, 1000)
    data[0] *= 50.0

    z = gpu.find_bad_by_deviation_gpu(data, device="cpu")
    assert np.argmax(z) == 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_find_bad_by_deviation_cpu_auto_match():
    """Test that CPU and auto-device paths produce identical results."""
    np.random.seed(7)
    data = np.random.randn(12, 2000)

    z_cpu = gpu.find_bad_by_deviation_gpu(data, device="cpu")
    z_auto = gpu.find_bad_by_deviation_gpu(data, device="auto")

    # Both paths use float64 on non-MPS hardware; on MPS they share float32.
    # Either way the ordering of bad channels must be identical.
    assert np.array_equal(np.argsort(z_cpu), np.argsort(z_auto))


# ---------------------------------------------------------------------------
# correlate_windows_gpu
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_correlate_windows_output_shape():
    """Test that output shape is (n_windows, n_chans, n_chans)."""
    np.random.seed(42)
    n_chans, sfreq, duration = 8, 500.0, 4.0
    data = np.random.randn(n_chans, int(sfreq * duration))
    corrs = gpu.correlate_windows_gpu(data, sfreq=sfreq, device="auto")

    assert corrs.ndim == 3
    assert corrs.shape[1] == n_chans
    assert corrs.shape[2] == n_chans


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_correlate_windows_small_batch():
    """Test batched processing produces the same result as single-pass."""
    np.random.seed(42)
    data = np.random.randn(8, 2000)
    corrs_big = gpu.correlate_windows_gpu(
        data, sfreq=500.0, device="cpu", max_batch_windows=1000
    )
    corrs_small = gpu.correlate_windows_gpu(
        data, sfreq=500.0, device="cpu", max_batch_windows=1
    )
    np.testing.assert_allclose(corrs_big, corrs_small, atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_correlate_windows_short_data_returns_identity():
    """Test fallback on data shorter than one window returns identity shape."""
    data = np.random.randn(8, 100)  # 0.2 s at 500 Hz — less than 1 window
    corrs = gpu.correlate_windows_gpu(data, sfreq=500.0, device="auto")
    assert corrs.ndim == 3
    assert corrs.shape == (1, 8, 8)
    np.testing.assert_array_equal(corrs[0], np.eye(8))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_correlate_windows_zero_variance_no_nan():
    """Test NaN prevention on zero-variance flat channels."""
    data = np.random.randn(8, 1000)
    data[2, :] = 0.0
    corrs = gpu.correlate_windows_gpu(data, sfreq=500.0, device="auto")
    assert not np.isnan(corrs).any()
