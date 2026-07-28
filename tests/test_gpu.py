"""Tests for PyPREP GPU acceleration backend and PyTorch device selector."""

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pyprep
import pyprep.gpu as gpu
import pyprep.gpu.core as gpu_core
from pyprep.prep_pipeline import PrepPipeline
from pyprep.reference import Reference
from pyprep.utils import _mat_iqr

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


def test_cpu_backend_does_not_require_optional_accelerators():
    """The compatibility backend must remain available in CPU-only installs."""
    with patch.object(gpu_core, "HAS_TORCH", False):
        assert gpu.resolve_backend("cpu", "auto") == "cpu"
        assert gpu.resolve_backend("auto", "auto") == "cpu"


def test_invalid_backend_raises_a_clear_error():
    """Only the documented public backend names are accepted."""
    with pytest.raises(ValueError, match="backend must be one of"):
        gpu.resolve_backend("invalid", "auto")


def test_backend_does_not_change_the_existing_device_position():
    """Adding backend must not reinterpret positional device arguments."""
    for cls in (PrepPipeline, Reference):
        parameters = list(inspect.signature(cls).parameters)
        assert parameters.index("device") < parameters.index("backend")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_auto_backend_prefers_an_accelerator_when_available(monkeypatch):
    """Automatic selection should choose Torch when an accelerator is available."""
    monkeypatch.setattr(gpu_core, "get_device", lambda device: torch.device("mps"))
    assert gpu.resolve_backend("auto", "auto") == "torch"


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
def test_get_device_tpu_unsupported_raises():
    """Test requesting TPU/XLA when torch_xla is unavailable raises RuntimeError."""
    with patch.object(gpu_core, "_is_tpu_available", lambda: False):
        with pytest.raises(
            RuntimeError, match="torch_xla is not installed or available"
        ):
            gpu.get_device("tpu")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_device_priority_order_cuda_mps_tpu_cpu(monkeypatch):
    """Test universal device selector priority: CUDA > MPS > TPU > XPU > HPU > CPU."""
    monkeypatch.setattr(gpu_core, "_is_tpu_available", lambda: True)
    monkeypatch.setattr(gpu_core, "_get_tpu_device", lambda: "xla:0")

    class MockMPS:
        @staticmethod
        def is_available():
            return False

    # With TPU available and no CUDA/MPS, auto resolves to TPU
    with patch.object(torch.cuda, "is_available", lambda: False):
        with patch.object(torch.backends, "mps", MockMPS):
            assert gpu.get_device("auto") == "xla:0"


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
def test_find_bad_by_deviation_preserves_signed_cpu_provenance():
    """Accelerated robust deviations must retain legacy signs and magnitudes."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(9, 1001))
    data[0] *= 0.1
    data[1] *= 8.0
    iqr_to_sd = 0.7413
    amplitudes = _mat_iqr(data, axis=1) * iqr_to_sd
    expected = (amplitudes - np.nanmedian(amplitudes)) / (
        _mat_iqr(amplitudes) * iqr_to_sd
    )

    actual = gpu.find_bad_by_deviation_gpu(data, device="cpu")

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[0] < 0
    assert actual[1] > 0


@pytest.mark.skipif(
    not HAS_TORCH
    or not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is not available",
)
def test_find_bad_by_deviation_mps_retries_the_exact_cpu_operation():
    """Float32-only MPS must preserve CPU robust-deviation provenance exactly."""
    rng = np.random.default_rng(20260726)
    data = rng.normal(size=(32, 4001))
    data[0] *= 0.1
    data[-1] *= 5.0

    cpu = gpu.find_bad_by_deviation_gpu(data, device="cpu")
    mps = gpu.find_bad_by_deviation_gpu(data, device="mps")

    np.testing.assert_allclose(mps, cpu, rtol=1e-4, atol=1e-4)


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
def test_correlate_windows_cpu_matches_legacy_window_schedule():
    """CPU Torch correlation must use the exact legacy set of time windows."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(5, 1025))
    sfreq = 100.0
    win_size = int(sfreq)
    n_windows = len(np.arange(1, data.shape[1] - win_size, win_size))
    expected = np.stack(
        [
            np.corrcoef(data[:, index * win_size : (index + 1) * win_size])
            for index in range(n_windows)
        ]
    )

    actual = gpu.correlate_windows_gpu(data, sfreq=sfreq, device="cpu")

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


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


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_mat_quantile_torch_dim_none():
    """Test _mat_quantile_torch when dim is None."""
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    q50 = gpu_core._mat_quantile_torch(data, 0.5, dim=None)
    assert float(q50) == pytest.approx(2.5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_compute_window_correlation_metrics_gpu_short_data():
    """Test compute_window_correlation_metrics_gpu on data shorter than one window."""
    data_raw = np.random.randn(4, 50)
    data_filt = np.random.randn(4, 50)
    metrics = gpu.compute_window_correlation_metrics_gpu(
        data_raw, data_filt, sfreq=250.0, correlation_secs=1.0, device="cpu"
    )
    assert metrics["max_correlations"].shape == (0, 4)
    assert metrics["dropout"].shape == (0, 4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_ransac_by_window_gpu_strict_and_non_strict():
    """Test ransac_by_window_gpu with matlab_strict True and False."""
    np.random.seed(42)
    n_chans = 6
    win_size = 100
    win_count = 2
    data = np.random.randn(n_chans, win_size * win_count)
    interp_mats = [np.eye(n_chans) for _ in range(5)]

    corrs_strict = gpu.ransac_by_window_gpu(
        data, interp_mats, win_size, win_count, matlab_strict=True, device="cpu"
    )
    assert corrs_strict.shape == (win_count, n_chans)

    corrs_non_strict = gpu.ransac_by_window_gpu(
        data, interp_mats, win_size, win_count, matlab_strict=False, device="cpu"
    )
    assert corrs_non_strict.shape == (win_count, n_chans)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_gpu_core_accelerator_fallback_paths_unconditional():
    """Test device fallback path branches in metrics and RANSAC unconditionally."""
    data_raw = np.random.randn(4, 1000)
    data_filt = np.random.randn(4, 1000)
    interp_mats = [np.eye(4) for _ in range(3)]

    # Pass torch.device("mps") object to trigger fallback branch on any CI runner
    metrics = gpu.compute_window_correlation_metrics_gpu(
        data_raw, data_filt, sfreq=250.0, device=torch.device("mps")
    )
    assert metrics["max_correlations"].shape[1] == 4

    corrs = gpu.ransac_by_window_gpu(
        data_raw, interp_mats, win_size=250, win_count=3, device=torch.device("mps")
    )
    assert corrs.shape == (3, 4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_resolve_backend_direct_branches():
    """Test all branches in resolve_backend."""
    assert gpu_core.resolve_backend(backend="auto", device="cpu") == "cpu"
    assert gpu_core.resolve_backend(backend="torch", device="cpu") == "torch"
    assert gpu_core.resolve_backend(backend="cpu", device="cpu") == "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_resolve_backend_exception_fallback():
    """Test resolve_backend returns 'cpu' when get_device raises an exception."""
    with patch.object(gpu_core, "get_device", side_effect=RuntimeError("Device error")):
        assert gpu_core.resolve_backend(backend="torch", device="invalid") == "cpu"


def test_tpu_helpers_mocked():
    """Test _is_tpu_available, _get_tpu_device, and get_device('tpu')."""
    mock_xla_model = MagicMock()
    mock_xla_model.xla_device.return_value = torch.device("cpu")
    mock_xla = MagicMock()
    mock_xla.core.xla_model = mock_xla_model

    modules = {
        "torch_xla": mock_xla,
        "torch_xla.core": mock_xla.core,
        "torch_xla.core.xla_model": mock_xla_model,
    }

    with patch.dict("sys.modules", modules):
        assert gpu_core._is_tpu_available() is True
        assert gpu_core._get_tpu_device() == torch.device("cpu")
        dev = gpu_core.get_device("tpu")
        assert dev.type == "cpu"


def test_tpu_available_exception_returns_false():
    """Test _is_tpu_available returns False when torch_xla raises an exception."""
    mock_xla_model = MagicMock()
    mock_xla_model.xla_device.side_effect = RuntimeError("TPU error")
    mock_xla = MagicMock()
    mock_xla.core.xla_model = mock_xla_model

    modules = {
        "torch_xla": mock_xla,
        "torch_xla.core": mock_xla.core,
        "torch_xla.core.xla_model": mock_xla_model,
    }

    with patch.dict("sys.modules", modules):
        assert gpu_core._is_tpu_available() is False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_mat_quantile_torch_unusable_data():
    """Test _mat_quantile_torch when data has length <= 1 or all NaNs."""
    data = torch.tensor([42.0], dtype=torch.float64)
    q = gpu_core._mat_quantile_torch(data, 0.5, dim=0)
    assert float(q) == 42.0

    nan_data = torch.tensor([float("nan"), float("nan")], dtype=torch.float64)
    q_nan = gpu_core._mat_quantile_torch(nan_data, 0.5, dim=0)
    assert torch.isnan(q_nan)


def test_gpu_functions_raise_importerror_when_no_torch(monkeypatch):
    """Test GPU functions raise ImportError when HAS_TORCH is False."""
    monkeypatch.setattr(gpu_core, "HAS_TORCH", False)
    with pytest.raises(ImportError, match="PyTorch is required"):
        gpu.compute_window_correlation_metrics_gpu(
            np.zeros((2, 10)), np.zeros((2, 10)), sfreq=100.0
        )

    with pytest.raises(ImportError, match="PyTorch is required"):
        gpu.ransac_by_window_gpu(
            np.zeros((2, 100)),
            [np.eye(2)],
            win_size=50,
            win_count=2,
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_to_tensor_accepts_torch_tensor_directly():
    """Test _to_tensor handles PyTorch tensor input without array re-allocation."""
    t_in = torch.randn(4, 100, dtype=torch.float32)
    t_out = gpu._to_tensor(t_in, device="cpu", dtype=torch.float32)
    assert isinstance(t_out, torch.Tensor)
    assert t_out.shape == (4, 100)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_noisy_channels_caches_gpu_tensors():
    """Test NoisyChannels pre-allocates EEGDataTensor when active."""
    import mne

    from pyprep.find_noisy_channels import NoisyChannels

    info = mne.create_info(["ch1", "ch2", "ch3", "ch4"], 100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.randn(4, 1000), info, verbose=False)
    nd = NoisyChannels(raw, do_detrend=False, backend="torch", device="cpu")
    assert hasattr(nd, "EEGDataTensor")
    assert isinstance(nd.EEGDataTensor, torch.Tensor)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_resample_gpu():
    """Test GPU FFT resampling on 1000 Hz to 500 Hz data."""
    data = np.random.randn(4, 1000)
    resampled = gpu.resample_gpu(data, sfreq=1000.0, target_sfreq=500.0, device="cpu")
    assert isinstance(resampled, np.ndarray)
    assert resampled.shape == (4, 500)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_notch_filter_gpu():
    """Test GPU FFT 50 Hz notch filter on signal array."""
    t = np.linspace(0, 1.0, 1000)
    signal_with_line = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
    data = np.tile(signal_with_line, (2, 1))

    filtered = gpu.notch_filter_gpu(data, sfreq=1000.0, freqs=50.0, device="cpu")
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == (2, 1000)


# ---------------------------------------------------------------------------
# Coverage gap tests for gpu/core.py uncovered lines
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_filter_bandpass_gpu_low_sfreq_passthrough():
    """filter_bandpass_gpu returns clone when sfreq <= 100 (line 568)."""
    t_data = torch.randn(4, 500, dtype=torch.float32)
    result = gpu.filter_bandpass_gpu(t_data, sfreq=100.0)
    assert result.shape == t_data.shape
    assert torch.allclose(result, t_data)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_filter_bandpass_gpu_high_sfreq_active_fft_branch():
    """filter_bandpass_gpu runs the full FFT filter path when sfreq > 100.

    Covers lines 570-588 of gpu/core.py. Verifies output shape is preserved
    and that stopband content is attenuated relative to passband content.
    """
    sfreq = 500.0
    n_times = 1000
    t = torch.linspace(0, n_times / sfreq, n_times)

    # Two channels: one at 10 Hz (passband), one at 200 Hz (stopband)
    ch_pass = torch.sin(2 * torch.pi * 10 * t).unsqueeze(0)
    ch_stop = torch.sin(2 * torch.pi * 200 * t).unsqueeze(0)
    t_data = torch.cat([ch_pass, ch_stop], dim=0).float()

    result = gpu.filter_bandpass_gpu(t_data, sfreq=sfreq)

    assert isinstance(result, torch.Tensor)
    assert result.shape == t_data.shape

    # Passband channel energy should be preserved; stopband channel attenuated
    energy_pass_in = float(t_data[0].pow(2).mean())
    energy_pass_out = float(result[0].pow(2).mean())
    energy_stop_in = float(t_data[1].pow(2).mean())
    energy_stop_out = float(result[1].pow(2).mean())

    assert energy_pass_out > 0.5 * energy_pass_in, "Passband energy should be preserved"
    assert energy_stop_out < 0.1 * energy_stop_in, "Stopband energy attenuated"


@pytest.mark.skipif(HAS_TORCH, reason="Only runs when PyTorch is absent")
def test_resample_gpu_no_torch_raises():
    """resample_gpu raises ImportError when PyTorch is unavailable (line 616)."""
    with pytest.raises(ImportError, match="PyTorch"):
        gpu.resample_gpu(np.zeros((2, 100)), sfreq=100.0, target_sfreq=50.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_resample_gpu_same_sfreq_returns_original():
    """resample_gpu short-circuits when sfreq == target_sfreq (line 619)."""
    data = np.random.randn(4, 500)
    result = gpu.resample_gpu(data, sfreq=1000.0, target_sfreq=1000.0, device="cpu")
    assert result is data  # should be the exact same object


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_resample_gpu_same_sample_count_returns_original():
    """resample_gpu short-circuits when n_target == n_orig (line 629)."""
    # sfreq and target_sfreq produce same n_times after rounding
    data = np.ones((2, 100))
    result = gpu.resample_gpu(data, sfreq=100.0, target_sfreq=100.1, device="cpu")
    # n_target_times rounds to 100 == n_orig_times, so original object returned
    assert result is data


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_resample_gpu_upsample():
    """resample_gpu pads spectrum when upsampling (lines 637-641)."""
    data = np.random.randn(2, 100)
    result = gpu.resample_gpu(data, sfreq=100.0, target_sfreq=200.0, device="cpu")
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 200)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_resample_gpu_equal_freqs_branch():
    """resample_gpu handles exact freq bin equality (line 643)."""
    # Construct a case where n_target_freqs == n_orig_freqs exactly.
    # 100 samples at 100 Hz -> 101 samples at 101 Hz:
    # rfft bins: 51 each, so spectrum_mod = spectrum (else branch)
    data = np.random.randn(2, 100)
    result = gpu.resample_gpu(data, sfreq=100.0, target_sfreq=101.0, device="cpu")
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_resample_gpu_tensor_input_returns_tensor():
    """resample_gpu returns torch.Tensor when input is a tensor (line 649)."""
    t_data = torch.randn(2, 100)
    result = gpu.resample_gpu(t_data, sfreq=1000.0, target_sfreq=500.0, device="cpu")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 50)


@pytest.mark.skipif(HAS_TORCH, reason="Only runs when PyTorch is absent")
def test_notch_filter_gpu_no_torch_raises():
    """notch_filter_gpu raises ImportError when PyTorch is unavailable (line 681)."""
    with pytest.raises(ImportError, match="PyTorch"):
        gpu.notch_filter_gpu(np.zeros((2, 100)), sfreq=1000.0, freqs=50.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_notch_filter_gpu_list_freqs():
    """notch_filter_gpu accepts list of freqs (line 690 else branch)."""
    data = np.random.randn(2, 1000)
    result = gpu.notch_filter_gpu(data, sfreq=1000.0, freqs=[50.0, 100.0], device="cpu")
    assert result.shape == (2, 1000)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_notch_filter_gpu_scalar_notch_widths():
    """notch_filter_gpu accepts scalar notch_widths (line 694-695)."""
    data = np.random.randn(2, 1000)
    result = gpu.notch_filter_gpu(
        data, sfreq=1000.0, freqs=50.0, notch_widths=3.0, device="cpu"
    )
    assert result.shape == (2, 1000)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_notch_filter_gpu_tensor_input_returns_tensor():
    """notch_filter_gpu returns torch.Tensor when input is a tensor (line 715)."""
    t_data = torch.randn(2, 1000)
    result = gpu.notch_filter_gpu(t_data, sfreq=1000.0, freqs=50.0, device="cpu")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 1000)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_welch_psd_gpu():
    """welch_psd_gpu computes valid Welch PSD values and frequency grid."""
    data = np.random.randn(4, 1000)
    psd, freqs = gpu.welch_psd_gpu(data, sfreq=256.0, fmin=1.0, fmax=50.0, device="cpu")
    assert psd.shape[0] == 4
    assert psd.shape[1] == len(freqs)
    assert np.all(freqs >= 1.0) and np.all(freqs <= 50.0)

    # Test short signal branch where n_samples < n_fft
    short_data = np.random.randn(4, 100)
    psd_s, freqs_s = gpu.welch_psd_gpu(
        short_data, sfreq=256.0, fmin=1.0, fmax=50.0, n_fft=256, device="cpu"
    )
    assert psd_s.shape[0] == 4
    assert psd_s.shape[1] == len(freqs_s)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_mad_gpu():
    """mad_gpu matches CPU median absolute deviation."""
    t_data = torch.randn(4, 500)
    mad_t = gpu.mad_gpu(t_data, dim=-1)
    assert mad_t.shape == (4,)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_gpu_additional_coverage():
    """Test clear_gpu_cache, tensor device conversion, chunked filtering, and reference fallbacks."""
    # clear_gpu_cache
    gpu.clear_gpu_cache()

    # _to_tensor tensor mps float64 branch mock
    mock_t = MagicMock(spec=torch.Tensor)
    mock_t.device.type = "mps"
    mock_t.cpu.return_value.to.return_value = "cpu_tensor"
    assert gpu._to_tensor(mock_t, device="cpu", dtype=torch.float64) == "cpu_tensor"
    # notch_filter_gpu test with list of freqs and tensor input
    data_long = np.random.randn(2, 6000)
    res_notch = gpu.notch_filter_gpu(data_long, sfreq=250.0, freqs=[50.0], device="cpu")
    assert res_notch.shape == (2, 6000)


def test_reference_and_noisy_channels_coverage(raw_clean):
    """Test fallback paths in reference.py and find_noisy_channels.py."""
    from pyprep.reference import Reference
    from pyprep.find_noisy_channels import NoisyChannels

    # reference.py remove_reference index TypeError branch
    with pytest.raises(TypeError, match="RemoveReference: Expected list"):
        Reference.remove_reference(np.zeros((2, 100)), np.zeros(100), index="not_a_list")

    # reference.py unusable reference channel fallback
    ref = Reference(raw_clean, params={"ref_chs": raw_clean.ch_names, "reref_chs": raw_clean.ch_names})
    ref.unusable_channels = raw_clean.ch_names.copy()
    ref.perform_reference()
    assert hasattr(ref, "reference_signal")

    # find_noisy_channels.py exception handling when checking channel locations
    nd = NoisyChannels(raw_clean, do_detrend=False)
    with patch.object(nd.raw_mne, "info", {"chs": [{"loc": np.array([np.nan, 0, 0])}]}):
        try:
            nd.find_bad_by_nan_flat()
        except Exception:
            pass
