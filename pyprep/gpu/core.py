"""GPU-accelerated PyPREP core implementations with PyTorch-style device selection."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("pyprep.gpu")

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # pragma: no cover
    HAS_TORCH = False  # pragma: no cover


def get_device(device="auto"):
    """PyTorch-style universal device selector for PyPREP.

    Parameters
    ----------
    device : str or torch.device
        Target hardware accelerator. One of ``'auto'``, ``'cuda'``,
        ``'cuda:0'``, ``'mps'``, ``'xpu'``, ``'hpu'``, ``'cpu'``, or a
        :class:`torch.device` instance.  ``'auto'`` (default) picks the best
        available accelerator in the order CUDA > MPS > XPU > HPU > CPU.

    Returns
    -------
    torch.device
        Resolved PyTorch device object.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GPU acceleration in PyPREP. "
            "Please install PyTorch via `pip install torch` "
            "or `pip install pyprep[gpu]`."
        )

    if device is None or device == "auto":
        if torch.cuda.is_available():  # pragma: no cover
            dev = torch.device("cuda")  # pragma: no cover
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
            dev = torch.device("xpu")  # pragma: no cover
        elif hasattr(torch, "hpu") and torch.hpu.is_available():  # pragma: no cover
            dev = torch.device("hpu")  # pragma: no cover
        else:  # pragma: no cover
            dev = torch.device("cpu")  # pragma: no cover
    elif isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)

    logger.debug(f"[PyPREP GPU] Device selected: {dev}")
    return dev


def _to_tensor(data, device, dtype=None):
    """Convert input numpy array or tensor to a PyTorch tensor on target device.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input data array.
    device : torch.device or str
        Target device.
    dtype : torch.dtype, optional
        Desired tensor dtype.  Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Tensor on *device* with the requested *dtype*.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration.")
    if dtype is None:
        dtype = torch.float32
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype, non_blocking=True)
    return torch.tensor(np.ascontiguousarray(data), device=device, dtype=dtype)


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

    t_data = _to_tensor(data, dev, dtype=torch.float32)
    n_chans, n_times = t_data.shape
    win_samples = int(sfreq * win_len_sec)
    n_windows = n_times // win_samples

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

        if dev.type == "cuda":  # pragma: no cover
            torch.cuda.empty_cache()  # pragma: no cover
        elif dev.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

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

    # Use float64 on CPU and for non-MPS GPU paths for full numerical precision.
    # MPS only supports float32 quantile operations.
    use_double = dev.type != "mps"
    dtype = torch.float64 if use_double else torch.float32

    t_data = _to_tensor(data, dev, dtype=dtype)

    q75 = torch.quantile(t_data, 0.75, dim=-1)
    q25 = torch.quantile(t_data, 0.25, dim=-1)
    iqr_per_ch = (q75 - q25) * 0.7413

    q75_amp = torch.quantile(iqr_per_ch, 0.75)
    q25_amp = torch.quantile(iqr_per_ch, 0.25)
    amp_sd = (q75_amp - q25_amp) * 0.7413
    amp_median = torch.median(iqr_per_ch)

    z_scores = torch.abs(iqr_per_ch - amp_median) / (amp_sd + 1e-12)
    return z_scores.cpu().numpy()
