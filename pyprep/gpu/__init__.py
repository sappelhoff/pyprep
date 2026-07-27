"""GPU-accelerated PyPREP module initialization."""

from pyprep.gpu.core import (
    HAS_TORCH,
    _dtype_for_device,
    _to_tensor,
    clear_gpu_cache,
    compute_window_correlation_metrics_gpu,
    correlate_windows_gpu,
    filter_bandpass_gpu,
    filter_highpass_gpu,
    find_bad_by_deviation_gpu,
    get_device,
    mad_gpu,
    notch_filter_gpu,
    ransac_by_window_gpu,
    resample_gpu,
    resolve_backend,
    welch_psd_gpu,
)

__all__ = [
    "HAS_TORCH",
    "get_device",
    "resolve_backend",
    "_to_tensor",
    "_dtype_for_device",
    "clear_gpu_cache",
    "correlate_windows_gpu",
    "compute_window_correlation_metrics_gpu",
    "filter_bandpass_gpu",
    "filter_highpass_gpu",
    "find_bad_by_deviation_gpu",
    "ransac_by_window_gpu",
    "resample_gpu",
    "notch_filter_gpu",
    "welch_psd_gpu",
    "mad_gpu",
]
