"""
audio_io.py - Audio I/O, Normalization, Resampling, and Framing Utilities for PS52_DANC

Provides robust audio handling using NumPy float32 representation at 16,000 Hz target sample rate.

Audio Representation Rationale:
--------------------------------
- Sample Rate: 16,000 Hz (Standard wideband speech sampling rate, covering up to 8 kHz audio frequency bandwidth).
- Format: 1D NumPy float32 array scaled in range [-1.0, 1.0].
- Channel: Mono (Single channel) for direct compatibility with tactical headset microphone signals and signal processing algorithms.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import scipy.signal
import soundfile as sf

TARGET_SAMPLE_RATE = 16000

def validate_audio(data: np.ndarray, expected_sr: Optional[int] = None, sr: Optional[int] = None) -> bool:
    """
    Validates that audio data is a non-empty, 1D NumPy float array with finite numerical values.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(data)}")
    if data.size == 0:
        raise ValueError("Audio data array is empty.")
    if data.ndim != 1:
        raise ValueError(f"Expected 1D mono audio array, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Audio data contains non-finite values (NaN or Inf).")
    if expected_sr is not None and sr is not None and sr != expected_sr:
        raise ValueError(f"Sample rate mismatch: expected {expected_sr} Hz, got {sr} Hz")
    return True

def normalize_audio(data: np.ndarray, max_peak: float = 0.99) -> np.ndarray:
    """
    Safely normalizes 1D audio signal peak amplitude to max_peak.
    """
    validate_audio(data)
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        return ((data / peak) * max_peak).astype(np.float32)
    return data.astype(np.float32)

def load_audio(
    filepath: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    normalize: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Loads audio file using soundfile, converts to mono float32, resamples to target_sr if required.
    Does NOT modify or overwrite the original audio file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path.resolve()}")
    
    data, orig_sr = sf.read(str(path), dtype="float32")
    
    # Convert stereo/multichannel to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Resample if sample rate differs from target_sr
    if orig_sr != target_sr:
        gcd = np.gcd(int(target_sr), int(orig_sr))
        up = int(target_sr) // gcd
        down = int(orig_sr) // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
    
    if normalize:
        data = normalize_audio(data)
        
    validate_audio(data, expected_sr=target_sr, sr=target_sr)
    return data.astype(np.float32), target_sr

def save_audio(
    filepath: Union[str, Path],
    data: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    allow_clip_normalize: bool = False
) -> Path:
    """
    Saves float32 1D mono audio data to a WAV file.
    Checks for clipping (|x| > 1.0) and handles/reports appropriately.
    """
    validate_audio(data, expected_sr=TARGET_SAMPLE_RATE, sr=sample_rate)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    max_val = float(np.max(np.abs(data)))
    if max_val > 1.0:
        if allow_clip_normalize:
            data = normalize_audio(data, max_peak=0.99)
        else:
            data = np.clip(data, -1.0, 1.0)
            
    sf.write(str(path), data.astype(np.float32), sample_rate, subtype="PCM_16")
    return path

def frame_audio(data: np.ndarray, frame_length: int = 320, hop_length: int = 160) -> np.ndarray:
    """
    Slices 1D mono audio signal into 2D overlapping frames [num_frames, frame_length].
    Default: 320 samples = 20ms at 16kHz, hop 160 = 10ms at 16kHz.
    """
    validate_audio(data)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive integers.")
        
    n_samples = len(data)
    if n_samples < frame_length:
        padded = np.pad(data, (0, frame_length - n_samples), mode="constant")
        return padded.reshape(1, frame_length).astype(np.float32)
        
    num_frames = 1 + (n_samples - frame_length) // hop_length
    shape = (num_frames, frame_length)
    strides = (data.strides[0] * hop_length, data.strides[0])
    frames = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return np.copy(frames).astype(np.float32)

def unframe_audio(frames: np.ndarray, hop_length: int = 160, original_length: Optional[int] = None) -> np.ndarray:
    """
    Reconstructs 1D audio signal from 2D overlapping frames [num_frames, frame_length]
    using overlap-add (OLA) with a Hann synthesis window.
    """
    if frames.ndim != 2:
        raise ValueError(f"Expected 2D frames array [num_frames, frame_length], got shape {frames.shape}")
        
    num_frames, frame_length = frames.shape
    window = np.hanning(frame_length).astype(np.float32)
    output_length = (num_frames - 1) * hop_length + frame_length
    out = np.zeros(output_length, dtype=np.float32)
    norm = np.zeros(output_length, dtype=np.float32)
    
    for i in range(num_frames):
        start = i * hop_length
        out[start : start + frame_length] += frames[i] * window
        norm[start : start + frame_length] += window**2
        
    nonzero = norm > 1e-8
    out[nonzero] /= norm[nonzero]
    
    if original_length is not None and original_length <= output_length:
        out = out[:original_length]
        
    return out.astype(np.float32)
