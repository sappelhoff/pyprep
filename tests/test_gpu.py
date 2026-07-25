"""Tests for PyPREP GPU acceleration backend, edge cases, device selector, and OOM chunking."""

import pytest
import numpy as np
import pyprep.gpu as gpu

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_get_device():
    dev_auto = gpu.get_device("auto")
    assert isinstance(dev_auto, torch.device)

    dev_cpu = gpu.get_device("cpu")
    assert dev_cpu.type == "cpu"

    dev_obj = gpu.get_device(torch.device("cpu"))
    assert dev_obj.type == "cpu"

    with pytest.raises(RuntimeError):
        gpu.get_device("invalid_device_name_xyz")

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_find_bad_by_deviation_gpu():
    np.random.seed(42)
    data = np.random.randn(10, 1000)
    data[0] *= 50.0  # Bad channel

    z_scores_cpu = gpu.core.find_bad_by_deviation_gpu(data, device="cpu")
    z_scores_auto = gpu.core.find_bad_by_deviation_gpu(data, device="auto")

    assert z_scores_cpu.shape == (10,)
    assert z_scores_auto.shape == (10,)
    assert np.argmax(z_scores_cpu) == 0
    assert np.argmax(z_scores_auto) == 0

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_correlate_windows_gpu():
    np.random.seed(42)
    data = np.random.randn(8, 2000)
    corrs = gpu.core.correlate_windows_gpu(data, sfreq=500.0, device="auto", max_batch_windows=2)

    assert corrs.ndim == 3
    assert corrs.shape[1:] == (8, 8)

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_short_data_edge_case():
    data = np.random.randn(8, 100)  # Shorter than 1 sec at 500Hz
    corrs = gpu.core.correlate_windows_gpu(data, sfreq=500.0, device="auto")
    assert corrs.shape == (8, 8)
    assert (corrs == 1.0).all()

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_zero_variance_flat_channel_edge_case():
    data = np.random.randn(8, 1000)
    data[2, :] = 0.0  # Completely flat zero-variance channel
    corrs = gpu.core.correlate_windows_gpu(data, sfreq=500.0, device="auto")
    assert not np.isnan(corrs).any()
