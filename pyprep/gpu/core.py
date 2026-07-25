"""GPU-accelerated PyPREP module with optional PyTorch import safety."""

from __future__ import annotations
import logging

logger = logging.getLogger("pyprep.gpu")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

def get_device(device: str | torch.device = "auto"):
    """
    PyTorch-style universal device selector for PyPREP.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GPU acceleration in PyPREP. "
            "Please install PyTorch via `pip install torch` or `pip install pyprep[gpu]`."
        )

    if device is None or device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = torch.device("xpu")
        elif hasattr(torch, "hpu") and torch.hpu.is_available():
            dev = torch.device("hpu")
        else:
            dev = torch.device("cpu")
    elif isinstance(device, torch.device):
        dev = device
    else:
        dev = torch.device(device)
    
    logger.debug(f"[PyPREP GPU] Device selected: {dev}")
    return dev

def _to_tensor(data, device, dtype=None):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration.")
    if dtype is None:
        dtype = torch.float32
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype, non_blocking=True)
    return torch.tensor(data, device=device, dtype=dtype)

def correlate_windows_gpu(
    data,
    sfreq: float,
    win_len_sec: float = 1.0,
    device = "auto",
    max_batch_windows: int = 500,
):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration. Install via `pip install torch`.")
    import numpy as np

    dev = get_device(device)
    logger.info(f"[PyPREP GPU] Running windowed correlation on device={dev}")
    
    t_data = _to_tensor(data, dev, dtype=torch.float32)
    n_chans, n_times = t_data.shape
    win_samples = int(sfreq * win_len_sec)
    n_windows = n_times // win_samples

    if n_windows == 0:
        logger.warning("[PyPREP GPU] Data duration shorter than 1 window, returning identity correlations.")
        return np.ones((n_chans, n_chans))

    effective_batch = 250 if dev.type == "mps" else max_batch_windows

    trimmed = t_data[:, :n_windows * win_samples].reshape(n_chans, n_windows, win_samples)
    windows = trimmed.permute(1, 0, 2)

    corrs_list = []
    for i in range(0, n_windows, effective_batch):
        win_chunk = windows[i : i + effective_batch]
        mean = win_chunk.mean(dim=-1, keepdim=True)
        centered = win_chunk - mean
        std = torch.sqrt(torch.sum(centered ** 2, dim=-1, keepdim=True) + 1e-12)
        normed = centered / std
        corrs_chunk = torch.bmm(normed, normed.transpose(1, 2))
        corrs_list.append(corrs_chunk.cpu())
        
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        elif dev.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    all_corrs = torch.cat(corrs_list, dim=0)
    return all_corrs.numpy()

def find_bad_by_deviation_gpu(
    data,
    deviation_threshold: float = 5.0,
    device = "auto",
):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration. Install via `pip install torch`.")
    import numpy as np

    dev = get_device(device)
    logger.info(f"[PyPREP GPU] Running bad-by-deviation assessment on device={dev}")

    if dev.type == "cpu":
        t_data = _to_tensor(data, dev, dtype=torch.float64)
        q75 = torch.quantile(t_data, 0.75, dim=-1)
        q25 = torch.quantile(t_data, 0.25, dim=-1)
        iqr_per_ch = (q75 - q25) * 0.7413

        q75_amp = torch.quantile(iqr_per_ch, 0.75)
        q25_amp = torch.quantile(iqr_per_ch, 0.25)
        amp_sd = (q75_amp - q25_amp) * 0.7413
        amp_median = torch.median(iqr_per_ch)

        z_scores = torch.abs(iqr_per_ch - amp_median) / (amp_sd + 1e-12)
        return z_scores.numpy()

    t_data = _to_tensor(data, dev)
    d = t_data.float() if dev.type == "mps" else t_data.double()
    q75 = torch.quantile(d, 0.75, dim=-1)
    q25 = torch.quantile(d, 0.25, dim=-1)
    iqr_per_ch = (q75 - q25) * 0.7413

    q75_amp = torch.quantile(iqr_per_ch, 0.75)
    q25_amp = torch.quantile(iqr_per_ch, 0.25)
    amp_sd = (q75_amp - q25_amp) * 0.7413
    amp_median = torch.median(iqr_per_ch)

    z_scores = torch.abs(iqr_per_ch - amp_median) / (amp_sd + 1e-12)
    return z_scores.cpu().numpy()
