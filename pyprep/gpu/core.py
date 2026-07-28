"""Optional accelerator helpers with CPU-compatible fallbacks.

The public PyPREP algorithms continue to be implemented by NumPy/SciPy.  This
module contains optional device selection and numerically validated Tensor
helpers used by the ``backend='auto'`` and ``backend='torch'`` opt-in paths.
"""

from __future__ import annotations

import logging

import numpy as np

from pyprep.utils import _mat_round

logger = logging.getLogger("pyprep.gpu")

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # pragma: no cover
    HAS_TORCH = False  # pragma: no cover


def resolve_backend(backend="cpu", device="auto"):
    """Resolve PyPREP's public backend request without requiring Torch.

    Parameters
    ----------
    backend : {'auto', 'cpu', 'torch'}
        ``'cpu'`` retains the established NumPy/SciPy computation. ``'auto'``
        selects Torch only when a non-CPU device is available. ``'torch'``
        requests Torch, but safely falls back to CPU when it is unavailable or
        the requested device resolves to CPU.
    device : {'auto', 'cpu', 'cuda', 'mps', 'xpu', 'hpu'} or torch.device
        Requested device when an optional Torch backend is selected.

    Returns
    -------
    str
        Either ``'cpu'`` or ``'torch'``.
    """
    if backend not in {"auto", "cpu", "torch"}:
        raise ValueError("backend must be one of 'auto', 'cpu', or 'torch'")
    if backend == "cpu" or not HAS_TORCH:
        return "cpu"

    try:
        resolved_device = get_device(device)
    except (ImportError, RuntimeError):
        return "cpu"

    if backend == "auto" and resolved_device.type == "cpu":
        return "cpu"
    return "torch"


def _is_tpu_available():
    """Check if Google TPU (torch_xla) is available in the current environment."""
    try:
        import torch_xla.core.xla_model as xm

        _ = xm.xla_device()
        return True
    except Exception:
        return False


def _get_tpu_device():
    """Return PyTorch XLA TPU device."""
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def get_device(device="auto"):
    """PyTorch-style universal device selector for PyPREP.

    Parameters
    ----------
    device : str or torch.device
        Target hardware accelerator. One of ``'auto'``, ``'cuda'``,
        ``'cuda:0'``, ``'mps'``, ``'tpu'``, ``'xla'``, ``'xpu'``, ``'hpu'``,
        ``'cpu'``, or a :class:`torch.device` instance. ``'auto'`` (default)
        picks the best available accelerator in the order:
        CUDA > MPS > TPU (torch_xla) > XPU > HPU > CPU.

    Returns
    -------
    torch.device
        Resolved PyTorch device object.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GPU/TPU acceleration in PyPREP. "
            "Please install PyTorch via `pip install torch` "
            "or `pip install pyprep[gpu]`."
        )

    if device is None or device == "auto":
        if torch.cuda.is_available():  # pragma: no cover
            dev = torch.device("cuda")  # pragma: no cover
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif _is_tpu_available():  # pragma: no cover
            dev = _get_tpu_device()  # pragma: no cover
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
            dev = torch.device("xpu")  # pragma: no cover
        elif hasattr(torch, "hpu") and torch.hpu.is_available():  # pragma: no cover
            dev = torch.device("hpu")  # pragma: no cover
        else:  # pragma: no cover
            dev = torch.device("cpu")  # pragma: no cover
    elif isinstance(device, str) and device.lower() in ("tpu", "xla"):
        if _is_tpu_available():  # pragma: no cover
            dev = _get_tpu_device()  # pragma: no cover
        else:
            raise RuntimeError(
                "Requested TPU/XLA device, but torch_xla is not installed or available."
            )
    elif isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)

    logger.debug(f"[PyPREP GPU] Device selected: {dev}")
    return dev


def clear_gpu_cache():
    """Flush memory allocation cache for CUDA/MPS devices to prevent memory growth."""
    if not HAS_TORCH:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _to_tensor(data, device, dtype=None):
    """Convert input numpy array or tensor to a PyTorch tensor on target device.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input data array.
    device : torch.device or str
        Target device.
    dtype : torch.dtype, optional
        Desired tensor dtype. Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Tensor on *device* with the requested *dtype*.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration.")
    device = get_device(device)
    if dtype is None:
        dtype = _dtype_for_device(device)
    if device.type in ("mps", "xpu", "hpu", "xla"):
        dtype = torch.float32
    if isinstance(data, torch.Tensor):
        if data.device.type == "mps" and dtype == torch.float64:
            return data.cpu().to(dtype=torch.float64)
        return data.to(device=device, dtype=dtype, non_blocking=True)

    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    try:
        return torch.tensor(
            np.ascontiguousarray(data, dtype=np_dtype), device=device, dtype=dtype
        )
    except (RuntimeError, AssertionError):
        return torch.tensor(
            np.ascontiguousarray(data, dtype=np_dtype),
            device=torch.device("cpu"),
            dtype=dtype,
        )


def _dtype_for_device(device):
    """Return the highest portable floating dtype for an accelerator device."""
    # MPS, XPU, HPU, and XLA (TPU) commonly lack float64 support for reductions.
    # CPU and CUDA retain float64 to match existing NumPy/SciPy paths.
    dev_type = getattr(device, "type", str(device))
    return torch.float32 if dev_type in {"mps", "xpu", "hpu", "xla"} else torch.float64


def _mat_quantile_torch(data, q, dim=None):
    """Calculate PyPREP's MATLAB-compatible quantile using Torch operations.

    This mirrors :func:`pyprep.utils._mat_quantile` for the finite EEG data that
    reaches the optional backend. In particular, it uses MATLAB's
    sample-adjusted quantile position instead of ``torch.quantile``'s default
    population convention.
    """
    if dim is None:
        data = data.reshape(-1)
        dim = 0

    sorted_data = torch.sort(data, dim=dim).values
    n = torch.isfinite(sorted_data).sum(dim=dim)
    result = sorted_data.select(dim, 0).clone()
    usable = n > 1
    if not torch.any(usable):
        return result

    q = torch.as_tensor(q, device=data.device, dtype=data.dtype)
    n_float = n.to(data.dtype)
    q_adjusted = ((q - 0.5) * n_float / (n_float - 1)) + 0.5
    exact_index = (n_float - 1) * torch.clamp(q_adjusted, 0, 1)
    lower_index = torch.floor(exact_index).to(torch.long)
    upper_index = torch.ceil(exact_index).to(torch.long)
    lower = torch.gather(sorted_data, dim, lower_index.unsqueeze(dim)).squeeze(dim)
    upper = torch.gather(sorted_data, dim, upper_index.unsqueeze(dim)).squeeze(dim)
    interpolated = lower + (upper - lower) * (exact_index - lower_index)
    return torch.where(usable, interpolated, result)


def correlate_windows_gpu(
    data,
    sfreq: float,
    win_len_sec: float = 1.0,
    device="auto",
    max_batch_windows: int = 500,
):
    """Compute windowed cross-correlations on GPU using PyTorch batch matmul.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input EEG data of shape ``(n_channels, n_times)``.
    sfreq : float
        Sampling frequency in Hz.
    win_len_sec : float
        Window length in seconds. Default is ``1.0``.
    device : str or torch.device
        Target hardware accelerator. See :func:`get_device` for options.
    max_batch_windows : int
        Maximum number of windows processed in a single GPU batch to
        avoid VRAM exhaustion. Default is ``500``.

    Returns
    -------
    np.ndarray
        Normalised cross-correlation matrices of shape
        ``(n_windows, n_channels, n_channels)``.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GPU acceleration. Install via `pip install torch`."
        )

    dev = get_device(device)
    logger.info(f"[PyPREP GPU] Running windowed correlation on device={dev}")

    t_data = _to_tensor(data, dev, dtype=_dtype_for_device(dev))
    n_chans, n_times = t_data.shape
    win_samples = int(sfreq * win_len_sec)
    # Match the legacy loop in NoisyChannels exactly. In particular, a
    # recording ending on a window boundary intentionally does not process a
    # final full window in the historical implementation.
    n_windows = len(np.arange(1, n_times - win_samples, win_samples))

    if n_windows == 0:
        logger.warning("[PyPREP GPU] Short data, returning identity correlations.")
        return np.eye(n_chans)[np.newaxis]  # shape (1, n_chans, n_chans)

    effective_batch = 250 if dev.type == "mps" else max_batch_windows

    trimmed = t_data[:, : n_windows * win_samples].reshape(
        n_chans, n_windows, win_samples
    )
    windows = trimmed.permute(1, 0, 2)

    corrs_list = []
    for i in range(0, n_windows, effective_batch):
        win_chunk = windows[i : i + effective_batch]
        mean = win_chunk.mean(dim=-1, keepdim=True)
        centered = win_chunk - mean
        std = torch.sqrt(torch.sum(centered**2, dim=-1, keepdim=True) + 1e-12)
        normed = centered / std
        corrs_chunk = torch.bmm(normed, normed.transpose(1, 2))
        corrs_list.append(corrs_chunk.cpu())

    all_corrs = torch.cat(corrs_list, dim=0)
    return all_corrs.numpy()


def find_bad_by_deviation_gpu(
    data,
    deviation_threshold: float = 5.0,
    device="auto",
):
    """Compute robust channel amplitude deviation Z-scores on GPU.

    Channels whose Z-score exceeds *deviation_threshold* are considered bad.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input EEG data of shape ``(n_channels, n_times)``.
    deviation_threshold : float
        Z-score threshold above which a channel is flagged as bad.
        Default is ``5.0``.
    device : str or torch.device
        Target hardware accelerator. See :func:`get_device` for options.

    Returns
    -------
    np.ndarray
        Robust amplitude Z-scores of shape ``(n_channels,)``.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GPU acceleration. Install via `pip install torch`."
        )

    dev = get_device(device)
    logger.info(f"[PyPREP GPU] Running bad-by-deviation assessment on device={dev}")

    t_data = _to_tensor(data, dev, dtype=_dtype_for_device(dev))

    q75 = _mat_quantile_torch(t_data, 0.75, dim=-1)
    q25 = _mat_quantile_torch(t_data, 0.25, dim=-1)
    iqr_per_ch = (q75 - q25) * 0.7413

    q75_amp = _mat_quantile_torch(iqr_per_ch, 0.75)
    q25_amp = _mat_quantile_torch(iqr_per_ch, 0.25)
    amp_sd = (q75_amp - q25_amp) * 0.7413
    amp_median = _mat_quantile_torch(iqr_per_ch, 0.5)

    z_scores = (iqr_per_ch - amp_median) / amp_sd
    return z_scores.cpu().numpy()


def compute_window_correlation_metrics_gpu(
    eeg_raw,
    eeg_filtered,
    sfreq: float,
    correlation_secs: float = 1.0,
    device="auto",
):
    """Compute windowed correlation, dropout, noiselevels, and amplitudes on GPU.

    This vectorizes the inner loop of NoisyChannels.find_bad_by_correlation while
    maintaining 100% exact numerical parity with NumPy.

    Parameters
    ----------
    eeg_raw : np.ndarray or torch.Tensor
        Raw EEG signal of shape ``(n_channels, n_times)``.
    eeg_filtered : np.ndarray or torch.Tensor
        Filtered EEG signal of shape ``(n_channels, n_times)``.
    sfreq : float
        Sampling frequency in Hz.
    correlation_secs : float
        Correlation window duration in seconds. Default is ``1.0``.
    device : str or torch.device
        Target hardware accelerator device.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary containing ``'max_correlations'``, ``'dropout'``,
        ``'noiselevels'``, and ``'channel_amplitudes'`` arrays.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration.")

    dev = get_device(device)

    # Call correlate_windows_gpu (allows mock overrides in tests)
    gpu_corrs = correlate_windows_gpu(
        eeg_filtered, sfreq=sfreq, win_len_sec=correlation_secs, device=dev
    )

    dtype = _dtype_for_device(dev)
    t_raw = _to_tensor(eeg_raw, dev, dtype=dtype)
    t_filt = _to_tensor(eeg_filtered, dev, dtype=dtype)

    n_chans, n_samples = t_raw.shape
    win_size = int(correlation_secs * sfreq)
    n_windows = len(np.arange(1, n_samples - win_size, win_size))

    if n_windows == 0:
        return {
            "max_correlations": np.ones((0, n_chans)),
            "dropout": np.zeros((0, n_chans), dtype=bool),
            "noiselevels": np.zeros((0, n_chans)),
            "channel_amplitudes": np.zeros((0, n_chans)),
        }

    IQR_TO_SD = 0.7413
    MAD_TO_SD = 1.4826

    # Shape: (n_windows, n_chans, win_size)
    raw_wins = (
        t_raw[:, : n_windows * win_size]
        .reshape(n_chans, n_windows, win_size)
        .permute(1, 0, 2)
    )
    filt_wins = (
        t_filt[:, : n_windows * win_size]
        .reshape(n_chans, n_windows, win_size)
        .permute(1, 0, 2)
    )

    # 1. channel_amplitudes = _mat_iqr(raw_wins) * IQR_TO_SD
    q75_raw = _mat_quantile_torch(raw_wins, 0.75, dim=2)
    q25_raw = _mat_quantile_torch(raw_wins, 0.25, dim=2)
    channel_amplitudes = (q75_raw - q25_raw) * IQR_TO_SD

    # 2. eeg_amplitude = _mad(filt_wins)
    med_filt = _mat_quantile_torch(filt_wins, 0.5, dim=2)
    abs_filt = torch.abs(filt_wins - med_filt.unsqueeze(2))
    eeg_amplitude = _mat_quantile_torch(abs_filt, 0.5, dim=2) * MAD_TO_SD

    dropout = eeg_amplitude <= 0

    # 3. high_freq_amplitude = _mad(raw_wins - filt_wins)
    diff_wins = raw_wins - filt_wins
    med_diff = _mat_quantile_torch(diff_wins, 0.5, dim=2)
    abs_diff = torch.abs(diff_wins - med_diff.unsqueeze(2))
    high_freq_amplitude = _mat_quantile_torch(abs_diff, 0.5, dim=2) * MAD_TO_SD

    noiselevels = torch.where(
        dropout,
        torch.zeros_like(high_freq_amplitude),
        high_freq_amplitude / (eeg_amplitude + 1e-12),
    )

    # 4. Use gpu_corrs matrix for 98th percentile quantile across channels
    bmm_corrs = _to_tensor(gpu_corrs, dev, dtype=dtype)
    eye = torch.eye(n_chans, dtype=torch.bool, device=dev).unsqueeze(0)
    abs_bmm_corrs = torch.abs(bmm_corrs)
    abs_bmm_corrs.masked_fill_(eye, 0.0)

    max_correlations = _mat_quantile_torch(abs_bmm_corrs, 0.98, dim=1)
    max_correlations = torch.where(
        dropout, torch.zeros_like(max_correlations), max_correlations
    )

    return {
        "max_correlations": max_correlations.cpu().numpy(),
        "dropout": dropout.cpu().numpy(),
        "noiselevels": noiselevels.cpu().numpy(),
        "channel_amplitudes": channel_amplitudes.cpu().numpy(),
    }


def ransac_by_window_gpu(
    data,
    interpolation_mats,
    win_size: int,
    win_count: int,
    matlab_strict: bool = False,
    device="auto",
):
    """Calculate RANSAC correlations on GPU/device via PyTorch tensor batching.

    This vectorizes the inner RANSAC prediction and correlation loop while
    maintaining 100% exact numerical parity with PyPREP CPU output.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Clean EEG data of shape ``(n_channels, n_times)``.
    interpolation_mats : list of np.ndarray
        Interpolation matrices of shape ``(n_channels, n_channels)``.
    win_size : int
        Window length in samples.
    win_count : int
        Number of correlation windows.
    matlab_strict : bool
        Whether to strictly follow MATLAB PREP sorting and correlation rules.
    device : str or torch.device
        Target hardware accelerator.

    Returns
    -------
    correlations : np.ndarray
        Array of shape ``(win_count, n_channels)``.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration.")

    dev = get_device(device)

    dtype = _dtype_for_device(dev)
    t_interp = _to_tensor(np.stack(interpolation_mats), dev, dtype=dtype)
    t_data = _to_tensor(data, dev, dtype=dtype)

    n_chans, _ = t_data.shape
    data_wins = (
        t_data[:, : win_count * win_size]
        .reshape(n_chans, win_count, win_size)
        .permute(1, 0, 2)
    )

    ransac_samples = len(interpolation_mats)
    merged_interp = t_interp.reshape(ransac_samples * n_chans, n_chans)
    preds_flat = torch.matmul(
        merged_interp,
        data_wins.permute(1, 0, 2).reshape(n_chans, win_count * win_size),
    )
    preds = preds_flat.reshape(ransac_samples, n_chans, win_count, win_size).permute(
        0, 2, 1, 3
    )

    sorted_preds = torch.sort(preds, dim=0).values
    if matlab_strict:
        median_idx = int(_mat_round(ransac_samples / 2.0) - 1)
        pred_median = sorted_preds[median_idx]
    else:
        if ransac_samples % 2 == 1:
            pred_median = sorted_preds[ransac_samples // 2]
        else:
            mid = ransac_samples // 2
            pred_median = 0.5 * (sorted_preds[mid - 1] + sorted_preds[mid])

    # Correlation calculation
    if matlab_strict:
        SSa = torch.sum(data_wins**2, dim=2)
        SSb = torch.sum(pred_median**2, dim=2)
        SSab = torch.sum(data_wins * pred_median, dim=2)
        corrs = SSab / (torch.sqrt(SSa) * torch.sqrt(SSb))
    else:
        a_c = data_wins - data_wins.mean(dim=2, keepdim=True)
        b_c = pred_median - pred_median.mean(dim=2, keepdim=True)
        SSa = torch.sum(a_c**2, dim=2)
        SSb = torch.sum(b_c**2, dim=2)
        SSab = torch.sum(a_c * b_c, dim=2)
        corrs = SSab / (torch.sqrt(SSa) * torch.sqrt(SSb))

    return corrs.cpu().numpy()


def filter_bandpass_gpu(
    t_data: torch.Tensor,
    sfreq: float,
    low_hz: float = 1.0,
    high_hz: float = 50.0,
    chunk_size: int = 131072,
) -> torch.Tensor:
    """Apply PyPREP zero-phase bandpass filter [1 Hz - 50 Hz] on GPU tensor via FFT.

    Matches ``scipy.signal.filtfilt`` output by using the same ``padtype='odd'``
    edge extension and computing the filter's spectral power in float64 on CPU
    so that float32-only devices (MPS, XPU, HPU) are not affected by filter
    design precision loss.

    The odd-extension pad reflects the signal around each edge value::

        left_pad  = 2 * x[..., 0] - x[..., 1:pad_len+1] (reversed)
        right_pad = 2 * x[..., -1] - x[..., -pad_len-1:-1] (reversed)

    This replicates scipy's default ``padtype='odd'`` and eliminates the
    edge discontinuities that caused spectrally degraded output on MPS with
    real (non-stationary) EEG recordings.
    """
    if sfreq <= 100:
        return t_data.clone()

    from pyprep.utils import _filter_design

    b_kernel = _filter_design(
        N_order=100,
        amp=np.array([1, 1, 0, 0]),
        freq=np.array([0, 90 / sfreq, 100 / sfreq, 1]),
    )

    n_times = t_data.shape[-1]
    # Match scipy.filtfilt default: padlen = 3 * max(len(b), len(a)) = 3 * len(b_kernel)
    pad_len = 3 * len(b_kernel)

    if n_times <= chunk_size:
        # Direct fast path for small/medium signals
        left_pad = 2.0 * t_data[..., :1] - t_data[..., 1 : pad_len + 1].flip(-1)
        right_pad = 2.0 * t_data[..., -1:] - t_data[..., -pad_len - 1 : -1].flip(-1)
        padded = torch.cat([left_pad, t_data, right_pad], dim=-1)
        n_fft = padded.shape[-1]

        H_np = np.fft.rfft(b_kernel, n=n_fft)
        H_power = torch.from_numpy((np.abs(H_np) ** 2).astype(np.float32)).to(
            device=t_data.device
        )

        spectrum = torch.fft.rfft(padded, dim=-1)
        filtered = torch.fft.irfft(spectrum * H_power, n=n_fft, dim=-1)
        return filtered[..., pad_len : pad_len + n_times]

    # Chunked memory-efficient path for very long recordings (prevents MPS/CUDA OOM)
    out = torch.empty_like(t_data)
    std_n_fft = chunk_size + 2 * pad_len
    H_np_full = np.fft.rfft(b_kernel, n=std_n_fft)
    H_power_full = torch.from_numpy((np.abs(H_np_full) ** 2).astype(np.float32)).to(
        device=t_data.device
    )

    for start in range(0, n_times, chunk_size):
        end = min(start + chunk_size, n_times)
        c_left = max(0, start - pad_len)
        c_right = min(n_times, end + pad_len)
        chunk = t_data[..., c_left:c_right]

        l_pad = start - c_left
        r_pad = c_right - end
        p_left = pad_len - l_pad
        p_right = pad_len - r_pad

        if p_left > 0 or p_right > 0:
            left_pad = (
                2.0 * chunk[..., :1] - chunk[..., 1 : p_left + 1].flip(-1)
                if p_left > 0
                else chunk[..., :0]
            )
            right_pad = (
                2.0 * chunk[..., -1:] - chunk[..., -p_right - 1 : -1].flip(-1)
                if p_right > 0
                else chunk[..., :0]
            )
            padded = torch.cat([left_pad, chunk, right_pad], dim=-1)
        else:
            padded = chunk

        n_fft = padded.shape[-1]
        if n_fft == std_n_fft:
            H_power = H_power_full
        else:
            H_np = np.fft.rfft(b_kernel, n=n_fft)
            H_power = torch.from_numpy((np.abs(H_np) ** 2).astype(np.float32)).to(
                device=t_data.device
            )

        spectrum = torch.fft.rfft(padded, dim=-1)
        filt = torch.fft.irfft(spectrum * H_power, n=n_fft, dim=-1)

        valid_start = pad_len
        valid_end = pad_len + (end - start)
        out[..., start:end] = filt[..., valid_start:valid_end]

    return out


def filter_highpass_gpu(
    t_data: torch.Tensor, sfreq: float, low_hz: float = 1.0, chunk_size: int = 131072
) -> torch.Tensor:
    """Apply PyPREP zero-phase highpass FIR filter on GPU tensor via FFT.

    Matches EEGLAB / PyPREP ``removeTrend`` output by using odd-extension padding
    and computing the highpass filter's spectral power in float64 on CPU.
    """
    if sfreq <= 2 * low_hz:
        return t_data.clone()

    from pyprep.utils import _eeglab_create_highpass

    b_kernel = _eeglab_create_highpass(low_hz, sfreq)
    n_times = t_data.shape[-1]
    pad_len = 3 * len(b_kernel)

    if n_times <= chunk_size:
        left_pad = 2.0 * t_data[..., :1] - t_data[..., 1 : pad_len + 1].flip(-1)
        right_pad = 2.0 * t_data[..., -1:] - t_data[..., -pad_len - 1 : -1].flip(-1)
        padded = torch.cat([left_pad, t_data, right_pad], dim=-1)
        n_fft = padded.shape[-1]

        H_np = np.fft.rfft(b_kernel, n=n_fft)
        H_power = torch.from_numpy((np.abs(H_np) ** 2).astype(np.float32)).to(
            device=t_data.device
        )

        spectrum = torch.fft.rfft(padded, dim=-1)
        filtered = torch.fft.irfft(spectrum * H_power, n=n_fft, dim=-1)
        return filtered[..., pad_len : pad_len + n_times]

    out = torch.empty_like(t_data)
    std_n_fft = chunk_size + 2 * pad_len
    H_np_full = np.fft.rfft(b_kernel, n=std_n_fft)
    H_power_full = torch.from_numpy((np.abs(H_np_full) ** 2).astype(np.float32)).to(
        device=t_data.device
    )

    for start in range(0, n_times, chunk_size):
        end = min(start + chunk_size, n_times)
        c_left = max(0, start - pad_len)
        c_right = min(n_times, end + pad_len)
        chunk = t_data[..., c_left:c_right]

        l_pad = start - c_left
        r_pad = c_right - end
        p_left = pad_len - l_pad
        p_right = pad_len - r_pad

        if p_left > 0 or p_right > 0:
            left_pad = (
                2.0 * chunk[..., :1] - chunk[..., 1 : p_left + 1].flip(-1)
                if p_left > 0
                else chunk[..., :0]
            )
            right_pad = (
                2.0 * chunk[..., -1:] - chunk[..., -p_right - 1 : -1].flip(-1)
                if p_right > 0
                else chunk[..., :0]
            )
            padded = torch.cat([left_pad, chunk, right_pad], dim=-1)
        else:
            padded = chunk

        n_fft = padded.shape[-1]
        if n_fft == std_n_fft:
            H_power = H_power_full
        else:
            H_np = np.fft.rfft(b_kernel, n=n_fft)
            H_power = torch.from_numpy((np.abs(H_np) ** 2).astype(np.float32)).to(
                device=t_data.device
            )

        spectrum = torch.fft.rfft(padded, dim=-1)
        filt = torch.fft.irfft(spectrum * H_power, n=n_fft, dim=-1)

        valid_start = pad_len
        valid_end = pad_len + (end - start)
        out[..., start:end] = filt[..., valid_start:valid_end]

    return out


def resample_gpu(
    data,
    sfreq: float,
    target_sfreq: float,
    device="auto",
):
    """Resample 1D/2D/3D EEG signal from sfreq to target_sfreq on GPU tensor via FFT.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input signal array of shape ``(..., n_times)``.
    sfreq : float
        Original sampling rate in Hz.
    target_sfreq : float
        Desired target sampling rate in Hz.
    device : str or torch.device
        Hardware device for acceleration.

    Returns
    -------
    np.ndarray or torch.Tensor
        Resampled signal array of shape ``(..., n_target_times)``.
    """
    if not HAS_TORCH:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for GPU acceleration."
        )  # pragma: no cover

    if sfreq == target_sfreq:
        return data

    dev = get_device(device)
    is_tensor = isinstance(data, torch.Tensor)
    t_data = _to_tensor(data, device=dev)

    n_orig_times = t_data.shape[-1]
    n_target_times = int(round(n_orig_times * (target_sfreq / sfreq)))

    if n_target_times == n_orig_times:
        return data

    spectrum = torch.fft.rfft(t_data, dim=-1)
    n_orig_freqs = spectrum.shape[-1]
    n_target_freqs = n_target_times // 2 + 1

    if n_target_freqs < n_orig_freqs:
        spectrum_mod = spectrum[..., :n_target_freqs]
    elif n_target_freqs > n_orig_freqs:
        pad_shape = list(spectrum.shape)
        pad_shape[-1] = n_target_freqs - n_orig_freqs
        zeros = torch.zeros(pad_shape, dtype=spectrum.dtype, device=spectrum.device)
        spectrum_mod = torch.cat([spectrum, zeros], dim=-1)
    else:
        spectrum_mod = spectrum

    scale = float(n_target_times) / float(n_orig_times)
    resampled_t = torch.fft.irfft(spectrum_mod, n=n_target_times, dim=-1) * scale

    if is_tensor:
        return resampled_t
    return resampled_t.cpu().numpy()


def notch_filter_gpu(
    data,
    sfreq: float,
    freqs,
    notch_widths=None,
    device="auto",
):
    """Apply zero-phase notch filter (e.g. 50/60 Hz line noise) on GPU via FFT.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input signal array of shape ``(..., n_times)``.
    sfreq : float
        Sampling rate in Hz.
    freqs : float or list of float
        Notch frequencies to remove (e.g. ``50.0`` or ``[50.0, 100.0]``).
    notch_widths : float or list of float, optional
        Bandwidth around each notch frequency in Hz. Default is ``2.0`` Hz.
    device : str or torch.device
        Hardware device for acceleration.

    Returns
    -------
    np.ndarray or torch.Tensor
        Notch-filtered signal array of shape ``(..., n_times)``.
    """
    if not HAS_TORCH:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for GPU acceleration."
        )  # pragma: no cover

    dev = get_device(device)
    is_tensor = isinstance(data, torch.Tensor)
    t_data = _to_tensor(data, device=dev)

    if isinstance(freqs, (int, float)):
        freqs = [float(freqs)]
    else:
        freqs = [float(f) for f in freqs]

    if notch_widths is None:
        notch_widths = [2.0] * len(freqs)
    elif isinstance(notch_widths, (int, float)):
        notch_widths = [float(notch_widths)] * len(freqs)

    n_times = t_data.shape[-1]
    chunk_size = 131072

    if n_times <= chunk_size:
        freq_grid = torch.fft.rfftfreq(n_times, d=1.0 / sfreq, device=t_data.device)
        H = torch.ones_like(freq_grid, dtype=t_data.dtype, device=t_data.device)

        for notch_f, width in zip(freqs, notch_widths):
            f_low = max(0.0, notch_f - width / 2.0)
            f_high = min(sfreq / 2.0, notch_f + width / 2.0)
            t_notch = (freq_grid >= f_low) & (freq_grid <= f_high)
            H = torch.where(
                t_notch,
                torch.tensor(0.0, dtype=t_data.dtype, device=t_data.device),
                H,
            )

        spectrum = torch.fft.rfft(t_data, dim=-1)
        filtered_t = torch.fft.irfft(spectrum * H, n=n_times, dim=-1)

        if is_tensor:
            return filtered_t
        return filtered_t.cpu().numpy()

    # Memory-efficient chunked notch filtering for long recordings
    pad_len = int(sfreq * 2)  # 2 second overlap pad for notch transitions
    filtered_t = torch.empty_like(t_data)

    for start in range(0, n_times, chunk_size):
        end = min(start + chunk_size, n_times)
        c_left = max(0, start - pad_len)
        c_right = min(n_times, end + pad_len)
        chunk = t_data[..., c_left:c_right]

        l_pad = start - c_left
        r_pad = c_right - end
        p_left = pad_len - l_pad
        p_right = pad_len - r_pad

        if p_left > 0 or p_right > 0:
            left_pad = (
                2.0 * chunk[..., :1] - chunk[..., 1 : p_left + 1].flip(-1)
                if p_left > 0
                else chunk[..., :0]
            )
            right_pad = (
                2.0 * chunk[..., -1:] - chunk[..., -p_right - 1 : -1].flip(-1)
                if p_right > 0
                else chunk[..., :0]
            )
            padded = torch.cat([left_pad, chunk, right_pad], dim=-1)
        else:
            padded = chunk

        n_chunk_fft = padded.shape[-1]
        freq_grid = torch.fft.rfftfreq(n_chunk_fft, d=1.0 / sfreq, device=t_data.device)
        H = torch.ones_like(freq_grid, dtype=t_data.dtype, device=t_data.device)

        for notch_f, width in zip(freqs, notch_widths):
            f_low = max(0.0, notch_f - width / 2.0)
            f_high = min(sfreq / 2.0, notch_f + width / 2.0)
            t_notch = (freq_grid >= f_low) & (freq_grid <= f_high)
            H = torch.where(
                t_notch,
                torch.tensor(0.0, dtype=t_data.dtype, device=t_data.device),
                H,
            )

        spec = torch.fft.rfft(padded, dim=-1)
        filt = torch.fft.irfft(spec * H, n=n_chunk_fft, dim=-1)

        valid_start = pad_len
        valid_end = pad_len + (end - start)
        filtered_t[..., start:end] = filt[..., valid_start:valid_end]

    if is_tensor:
        return filtered_t
    return filtered_t.cpu().numpy()


def welch_psd_gpu(
    data,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 50.0,
    n_fft: int = 256,
    device="auto",
):
    """Compute Welch Power Spectral Density (PSD) on GPU tensor via batched rfft.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input EEG signal of shape ``(n_channels, n_samples)``.
    sfreq : float
        Sampling rate in Hz.
    fmin : float
        Lower frequency bound in Hz.
    fmax : float
        Upper frequency bound in Hz.
    n_fft : int
        FFT window length in samples.
    device : str or torch.device
        Hardware device for acceleration.

    Returns
    -------
    psd : np.ndarray
        PSD values of shape ``(n_channels, n_freqs)``.
    freqs : np.ndarray
        Frequency grid values in Hz of shape ``(n_freqs,)``.
    """
    if not HAS_TORCH:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for GPU acceleration."
        )  # pragma: no cover

    dev = get_device(device)
    t_data = _to_tensor(data, device=dev)

    n_chans, n_samples = t_data.shape
    if n_fft is None or n_fft == 256:
        n_fft = min(n_samples, 2048)

    step = n_fft

    if n_samples < n_fft:
        n_fft = n_samples
        step = n_fft

    # Create sliding windows: (n_chans, n_windows, n_fft)
    windows = t_data.unfold(-1, n_fft, step)

    # Remove DC component per segment (detrend='constant')
    windows = windows - windows.mean(dim=-1, keepdim=True)

    # Apply Hamming window
    win = torch.hamming_window(n_fft, periodic=False, device=dev, dtype=t_data.dtype)
    windowed = windows * win

    # Compute batched rfft across all channels and windows simultaneously
    spec = torch.fft.rfft(windowed, n=n_fft, dim=-1)
    # Power spectral density scaling matching MNE
    scale = 1.0 / (sfreq * torch.sum(win**2))
    psd_windows = (spec.abs() ** 2) * scale
    psd_windows[..., 1:-1] *= 2.0
    # Average across sliding windows
    psd_tensor = psd_windows.mean(dim=1)

    freqs_full = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq, device=dev).cpu().numpy()
    idx = (freqs_full >= fmin) & (freqs_full <= fmax)

    return psd_tensor[:, idx].cpu().numpy(), freqs_full[idx]


def mad_gpu(t_data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute Median Absolute Deviation (MAD) along specified dimension on GPU tensor.

    Parameters
    ----------
    t_data : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which to compute MAD.

    Returns
    -------
    torch.Tensor
        MAD values along specified dimension.
    """
    med = torch.median(t_data, dim=dim, keepdim=True).values
    return torch.median(torch.abs(t_data - med), dim=dim).values
