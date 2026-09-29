"""
features.py - Feature Extraction Engine for PS52_DANC

Extracts 12 short-time spectral and temporal acoustic features per frame (20 ms frame / 10 ms hop at 16 kHz):
1. RMS Energy
2. Log Energy (dB)
3. Zero Crossing Rate (ZCR)
4. Spectral Centroid (Hz)
5. Spectral Bandwidth (Hz)
6. Spectral Flux (Normalized L2 spectral difference in [0, 2])
7. Spectral Flatness (Wiener entropy)
8. Low-Band Energy Ratio (0 - 500 Hz)
9. Mid-Band Energy Ratio (500 - 3000 Hz)
10. High-Band Energy Ratio (3000 - 8000 Hz)
11. Temporal Energy Variance (Rolling RMS variance)
12. Crest Factor (Peak-to-RMS energy ratio)

All functions use numerically safe bounds to prevent divide-by-zero, NaN, or Inf.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np

TARGET_SAMPLE_RATE = 16000

FEATURE_NAMES = [
    "rms_energy",
    "log_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flux",
    "spectral_flatness",
    "low_band_energy_ratio",
    "mid_band_energy_ratio",
    "high_band_energy_ratio",
    "temporal_energy_variance",
    "crest_factor"
]

def extract_frame_features(
    frame: np.ndarray,
    prev_magnitude_spectrum: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    eps: float = 1e-10
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Extracts 12 acoustic features from a single 1D audio frame.
    
    Returns:
      (features_dict, current_magnitude_spectrum)
    """
    if frame.ndim != 1 or len(frame) == 0:
        raise ValueError(f"Expected 1D non-empty frame array, got shape {frame.shape}")
        
    N = len(frame)
    
    # 1. RMS Energy & 2. Log Energy
    rms_energy = float(np.sqrt(np.mean(frame**2)))
    log_energy = float(10.0 * np.log10(rms_energy**2 + eps))
    
    # 3. Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame))))
    zcr = float(zero_crossings / max(1, N - 1))
    
    # 12. Crest Factor (Peak to RMS ratio)
    peak_val = float(np.max(np.abs(frame)))
    crest_factor = float(peak_val / (rms_energy + eps))
    
    # Windowed FFT for Spectral Features
    windowed = frame * np.hanning(N)
    fft_complex = np.fft.rfft(windowed)
    mag_spec = np.abs(fft_complex).astype(np.float32)
    freqs = np.fft.rfftfreq(N, 1.0 / sample_rate)
    
    total_mag = float(np.sum(mag_spec)) + eps
    
    # 4. Spectral Centroid
    spectral_centroid = float(np.sum(freqs * mag_spec) / total_mag)
    
    # 5. Spectral Bandwidth
    bandwidth_sq = np.sum(((freqs - spectral_centroid)**2) * mag_spec) / total_mag
    spectral_bandwidth = float(np.sqrt(max(0.0, bandwidth_sq)))
    
    # 6. Normalized Spectral Flux (L2 norm between normalized magnitude spectra, bounded in [0, 2])
    mag_normed = mag_spec / (float(np.linalg.norm(mag_spec)) + eps)
    if prev_magnitude_spectrum is not None and len(prev_magnitude_spectrum) == len(mag_spec):
        prev_normed = prev_magnitude_spectrum / (float(np.linalg.norm(prev_magnitude_spectrum)) + eps)
        spectral_flux = float(np.linalg.norm(mag_normed - prev_normed))
    else:
        spectral_flux = 0.0
        
    # 7. Spectral Flatness (Geometric Mean / Arithmetic Mean of Power Spectrum)
    power_spec = mag_spec**2 + eps
    log_mean = float(np.mean(np.log(power_spec)))
    arith_mean = float(np.mean(power_spec))
    spectral_flatness = float(np.exp(log_mean) / (arith_mean + eps))
    spectral_flatness = float(np.clip(spectral_flatness, 0.0, 1.0))
    
    # Band Energy Ratios: Low (0-500 Hz), Mid (500-3000 Hz), High (3000-8000 Hz)
    total_power = float(np.sum(power_spec)) + eps
    
    low_mask = (freqs >= 0.0) & (freqs < 500.0)
    mid_mask = (freqs >= 500.0) & (freqs < 3000.0)
    high_mask = (freqs >= 3000.0) & (freqs <= sample_rate / 2.0)
    
    # 8, 9, 10. Band Ratios
    low_band_ratio = float(np.sum(power_spec[low_mask]) / total_power)
    mid_band_ratio = float(np.sum(power_spec[mid_mask]) / total_power)
    high_band_ratio = float(np.sum(power_spec[high_mask]) / total_power)
    
    # 11. Temporal Energy Variance (Populated across sequences)
    temporal_energy_var = 0.0
    
    features = {
        "rms_energy": rms_energy,
        "log_energy": log_energy,
        "zero_crossing_rate": zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_flux": spectral_flux,
        "spectral_flatness": spectral_flatness,
        "low_band_energy_ratio": low_band_ratio,
        "mid_band_energy_ratio": mid_band_ratio,
        "high_band_energy_ratio": high_band_ratio,
        "temporal_energy_variance": temporal_energy_var,
        "crest_factor": crest_factor
    }
    
    return features, mag_spec

def extract_sequence_features(
    signal: np.ndarray,
    frame_length: int = 320,
    hop_length: int = 160,
    sample_rate: int = TARGET_SAMPLE_RATE,
    variance_window_frames: int = 10
) -> Dict[str, np.ndarray]:
    """
    Slices signal into frames and extracts sequence-level 1D feature arrays across all frames.
    Fast vectorized implementation.
    """
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("Signal must be a non-empty 1D array.")
        
    n_samples = len(signal)
    if n_samples < frame_length:
        padded = np.pad(signal, (0, frame_length - n_samples), mode="constant")
        frames = padded.reshape(1, frame_length)
    else:
        num_frames = 1 + (n_samples - frame_length) // hop_length
        shape = (num_frames, frame_length)
        strides = (signal.strides[0] * hop_length, signal.strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
        
    num_frames = len(frames)
    seq_features = {name: np.zeros(num_frames, dtype=np.float32) for name in FEATURE_NAMES}
    
    # Vectorized 2D STFT magnitude
    window = np.hanning(frame_length).astype(np.float32)
    win_frames = frames * window
    mag = np.abs(np.fft.rfft(win_frames, n=frame_length, axis=-1)).astype(np.float32)  # (num_frames, 161)
    
    # 1. RMS Energy & Log Energy
    rms_energy = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=-1)).astype(np.float32)
    log_energy = (10.0 * np.log10(rms_energy.astype(np.float64)**2 + 1e-10)).astype(np.float32)
    seq_features["rms_energy"] = rms_energy
    seq_features["log_energy"] = log_energy
    
    # 2. Zero Crossing Rate
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = np.mean(np.abs(np.diff(signs, axis=-1)) > 0, axis=-1).astype(np.float32)
    seq_features["zero_crossing_rate"] = zcr
    
    # 3. Spectral Centroid & Bandwidth
    n_bins = mag.shape[-1]
    freqs = np.linspace(0, sample_rate / 2.0, n_bins, dtype=np.float32)
    mag_sum = np.sum(mag, axis=-1) + 1e-10
    centroid = (np.sum(mag * freqs, axis=-1) / mag_sum).astype(np.float32)
    bandwidth = np.sqrt(np.maximum(0.0, np.sum(mag * (freqs - centroid[:, None])**2, axis=-1) / mag_sum)).astype(np.float32)
    seq_features["spectral_centroid"] = centroid
    seq_features["spectral_bandwidth"] = bandwidth
    
    # 4. Spectral Flux
    mag_norm = mag / (np.linalg.norm(mag, axis=-1, keepdims=True) + 1e-10)
    flux = np.zeros(num_frames, dtype=np.float32)
    if num_frames > 1:
        diff = mag_norm[1:] - mag_norm[:-1]
        flux[1:] = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)
    seq_features["spectral_flux"] = flux
    
    # 5. Spectral Flatness
    log_pwr = np.log(mag.astype(np.float64)**2 + 1e-12)
    geom_mean = np.exp(np.mean(log_pwr, axis=-1))
    arith_mean = np.mean(mag.astype(np.float64)**2, axis=-1) + 1e-10
    flatness = np.clip(geom_mean / arith_mean, 0.0, 1.0).astype(np.float32)
    seq_features["spectral_flatness"] = flatness
    
    # 6. Sub-band energy ratios
    bin_500 = max(1, int(500.0 / (sample_rate / 2.0) * (n_bins - 1)))
    bin_3k = max(bin_500 + 1, int(3000.0 / (sample_rate / 2.0) * (n_bins - 1)))
    pwr = mag**2
    tot_pwr = np.sum(pwr, axis=-1) + 1e-10
    seq_features["low_band_energy_ratio"] = (np.sum(pwr[:, :bin_500], axis=-1) / tot_pwr).astype(np.float32)
    seq_features["mid_band_energy_ratio"] = (np.sum(pwr[:, bin_500:bin_3k], axis=-1) / tot_pwr).astype(np.float32)
    seq_features["high_band_energy_ratio"] = (np.sum(pwr[:, bin_3k:], axis=-1) / tot_pwr).astype(np.float32)
    
    # 7. Crest Factor
    peak = np.max(np.abs(frames), axis=-1).astype(np.float32)
    seq_features["crest_factor"] = (peak / (rms_energy + 1e-10)).astype(np.float32)
    
    # 8. Temporal Energy Variance (rolling variance)
    for i in range(num_frames):
        start_idx = max(0, i - variance_window_frames + 1)
        win_rms = rms_energy[start_idx : i + 1]
        seq_features["temporal_energy_variance"][i] = float(np.var(win_rms)) if len(win_rms) > 1 else 0.0
        
    return seq_features
