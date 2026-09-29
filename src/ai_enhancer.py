r"""
ai_enhancer.py — Classical Wiener Spectral Speech Enhancement Baseline for PS52_DANC
======================================================================================

Implements a classical, explainable speech enhancement baseline using short-time
Fourier transform (STFT), noise power spectral density (PSD) tracking, and
decision-directed Wiener spectral gain masking.

DISCLAIMER & SCOPE:
--------------------
- This module implements a CLASSICAL DSP baseline (Wiener filtering & PSD tracking).
- It is NOT a neural network, AI, or machine learning model.
- It establishes a benchmark against which future neural enhancement (Stage 4D)
  and the hybrid controller (Stage 5) can be compared.

MATHEMATICAL FOUNDATION:
------------------------
1. Signal Model:
   y(t) = s(t) + n(t)  -->  Y(k, m) = S(k, m) + N(k, m)
   where k is the frequency bin index and m is the frame index.

2. Noise PSD Estimation:
   P_n(k, m) is initialized from early non-speech frames and updated via recursive
   smoothing during estimated non-speech periods:
   P_n(k, m) = alpha_n * P_n(k, m-1) + (1 - alpha_n) * |Y(k, m)|^2

3. Decision-Directed A Priori SNR Estimation (Ephraim & Malah):
   A posteriori SNR: gamma(k, m) = |Y(k, m)|^2 / (P_n(k, m) + eps)
   A priori SNR:     xi(k, m)    = alpha_dd * |G(k, m-1) * Y(k, m-1)|^2 / (P_n(k, m) + eps)
                                    + (1 - alpha_dd) * max(gamma(k, m) - 1, 0)

4. Wiener Gain Mask:
   G(k, m) = max(G_floor, xi(k, m) / (xi(k, m) + 1))
   Phase is strictly preserved: S_hat(k, m) = G(k, m) * Y(k, m)

5. Speech Presence Confidence Score (C_speech in [0, 1]):
   Computed from observed noisy signal evidence:
   - Spectral SNR margin across speech-dominant frequencies (300 - 3400 Hz)
   - Frame energy relative to noise baseline
   - Smoothed over time via exponential moving average (EMA)
   - NEVER uses ground-truth clean speech reference.
"""

from typing import Tuple, Optional, Dict, Union
import numpy as np
import scipy.signal

TARGET_SAMPLE_RATE = 16000


class SpectralEnhancer:
    """
    Classical Wiener-style spectral speech enhancer.
    
    Parameters:
      sample_rate: Sampling frequency in Hz (default 16000).
      frame_length: Window size in samples (320 = 20ms at 16kHz).
      hop_length: Hop size in samples (160 = 10ms at 16kHz).
      n_fft: FFT length (512 bins).
      gain_floor: Minimum spectral gain multiplier (default 0.05 = -26 dB).
      alpha_dd: Decision-directed a priori SNR smoothing factor (default 0.95).
      alpha_noise: Recursive noise PSD smoothing factor (default 0.92).
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        frame_length: int = 320,
        hop_length: int = 160,
        n_fft: int = 512,
        gain_floor: float = 0.05,
        alpha_dd: float = 0.95,
        alpha_noise: float = 0.92,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.n_freqs = self.n_fft // 2 + 1
        self.gain_floor = float(gain_floor)
        self.alpha_dd = float(alpha_dd)
        self.alpha_noise = float(alpha_noise)

        # Analysis and synthesis Hann windows
        self.window = np.hanning(self.frame_length).astype(np.float32)

        # Internal state
        self.noise_psd = None
        self.prev_gain = None
        self.prev_mag = None
        self.smoothed_confidence = 0.0
        self.frame_count = 0
        self.reset()

    def reset(self) -> None:
        """Resets all internal filter memory and PSD estimates."""
        self.noise_psd = np.ones(self.n_freqs, dtype=np.float32) * 1e-4
        self.prev_gain = np.ones(self.n_freqs, dtype=np.float32) * 0.5
        self.prev_mag = np.zeros(self.n_freqs, dtype=np.float32)
        self.smoothed_confidence = 0.0
        self.frame_count = 0

    def _estimate_frame_confidence(
        self,
        mag_spec: np.ndarray,
        noise_psd: np.ndarray,
        prior_snr: np.ndarray
    ) -> float:
        """
        Estimates speech presence confidence C_speech in [0.0, 1.0] from observed evidence:
          1. Average a priori SNR in telephonic speech band (300 Hz - 3400 Hz).
          2. Spectral energy concentration ratio.
          3. Total spectral power vs estimated noise floor.
        """
        freq_bins = np.linspace(0, self.sample_rate / 2, self.n_freqs)
        speech_band_idx = (freq_bins >= 300.0) & (freq_bins <= 3400.0)

        # 1. Mean speech-band SNR in dB
        speech_snr_lin = np.mean(prior_snr[speech_band_idx]) if np.any(speech_band_idx) else 1e-4
        speech_snr_db = 10.0 * np.log10(max(float(speech_snr_lin), 1e-6))

        # 2. Total power ratio vs noise
        tot_sig_pwr = float(np.sum(mag_spec ** 2))
        tot_noise_pwr = float(np.sum(noise_psd)) + 1e-10
        pwr_ratio_db = 10.0 * np.log10(max(tot_sig_pwr / tot_noise_pwr, 1e-6))

        # Sigmoid mapping centered at 3 dB SNR
        snr_factor = 1.0 / (1.0 + np.exp(-0.25 * (speech_snr_db - 3.0)))
        pwr_factor = 1.0 / (1.0 + np.exp(-0.25 * (pwr_ratio_db - 2.0)))

        raw_conf = 0.6 * snr_factor + 0.4 * pwr_factor
        return float(np.clip(raw_conf, 0.0, 1.0))

    def enhance_frame(
        self, frame: np.ndarray, update_noise: bool = True
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Processes a single 1D frame of audio.
        
        Returns:
          (enhanced_frame, frame_confidence, gain_vector)
        """
        if len(frame) != self.frame_length:
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))
            else:
                frame = frame[:self.frame_length]

        windowed = frame * self.window
        fft_complex = np.fft.rfft(windowed, n=self.n_fft)
        mag = np.abs(fft_complex).astype(np.float32)
        power = (mag ** 2).astype(np.float32)

        # Initialize noise PSD on first few frames
        if self.frame_count < 5:
            if self.frame_count == 0:
                self.noise_psd = power.copy()
            else:
                self.noise_psd = 0.7 * self.noise_psd + 0.3 * power
            self.frame_count += 1

        eps = 1e-10
        # A posteriori SNR: gamma = |Y|^2 / P_n
        gamma = power / (self.noise_psd + eps)

        # Decision-directed a priori SNR: xi = alpha * (G_prev * |Y_prev|)^2 / P_n + (1 - alpha) * max(gamma - 1, 0)
        prev_enhanced_pwr = (self.prev_gain * self.prev_mag) ** 2
        xi_pred = prev_enhanced_pwr / (self.noise_psd + eps)
        xi_inst = np.maximum(gamma - 1.0, 0.0)
        xi = self.alpha_dd * xi_pred + (1.0 - self.alpha_dd) * xi_inst

        # Wiener Gain Mask: G = max(gain_floor, xi / (xi + 1))
        gain = xi / (xi + 1.0 + eps)
        gain = np.clip(gain, self.gain_floor, 1.0).astype(np.float32)

        # Speech Presence Confidence
        frame_conf = self._estimate_frame_confidence(mag, self.noise_psd, xi)
        self.smoothed_confidence = (
            0.8 * self.smoothed_confidence + 0.2 * frame_conf
            if self.frame_count > 1 else frame_conf
        )

        # Adaptive Noise PSD Update (during low speech probability / speech pauses)
        if update_noise and self.frame_count >= 5:
            # Slower update when speech is likely, faster update during pauses
            noise_update_rate = self.alpha_noise if self.smoothed_confidence < 0.4 else 0.98
            # Only update noise with bins that show low instant SNR
            noise_mask = gamma < 2.0
            self.noise_psd[noise_mask] = (
                noise_update_rate * self.noise_psd[noise_mask] +
                (1.0 - noise_update_rate) * power[noise_mask]
            )

        # Apply gain to complex spectrum (strictly preserving phase)
        enhanced_fft = gain * fft_complex
        enhanced_time = np.fft.irfft(enhanced_fft, n=self.n_fft)[:self.frame_length]
        enhanced_frame = (enhanced_time * self.window).astype(np.float32)

        # Update state
        self.prev_gain = gain.copy()
        self.prev_mag = mag.copy()
        self.frame_count += 1

        return enhanced_frame, float(self.smoothed_confidence), gain

    def enhance(
        self, signal: np.ndarray, return_diagnostics: bool = False
    ) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict]]:
        """
        Enhances a complete 1D mono audio signal array using STFT and overlap-add synthesis.
        
        Returns:
          (enhanced_signal, mean_confidence) or
          (enhanced_signal, mean_confidence, diagnostics_dict)
        """
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=-1)
        if len(signal) == 0:
            empty = np.zeros(0, dtype=np.float32)
            diag = {"confidences": np.zeros(0), "mean_gain": 0.0, "processing_time_sec": 0.0, "real_time_factor": 0.0}
            return (empty, 0.0, diag) if return_diagnostics else (empty, 0.0)

        import time
        t_start = time.perf_counter()

        orig_len = len(signal)
        self.reset()

        # Handle very short signals
        if orig_len < self.frame_length:
            padded = np.pad(signal, (0, self.frame_length - orig_len))
            enh_frame, conf, _ = self.enhance_frame(padded)
            enhanced = enh_frame[:orig_len]
            t_proc = time.perf_counter() - t_start
            diag = {
                "confidences": np.array([conf], dtype=np.float32),
                "mean_gain": float(np.mean(self.prev_gain)),
                "processing_time_sec": t_proc,
                "real_time_factor": t_proc / (orig_len / self.sample_rate + 1e-8),
            }
            return (enhanced, conf, diag) if return_diagnostics else (enhanced, conf)

        # Slice into overlapping frames
        n_samples = orig_len
        num_frames = 1 + (n_samples - self.frame_length) // self.hop_length
        out_len = (num_frames - 1) * self.hop_length + self.frame_length

        out_signal = np.zeros(out_len, dtype=np.float32)
        norm_window = np.zeros(out_len, dtype=np.float32)
        confidences = []
        gains = []

        win_sq = self.window ** 2

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            raw_frame = signal[start:end]

            enh_frame, frame_conf, gain = self.enhance_frame(raw_frame)

            out_signal[start:end] += enh_frame
            norm_window[start:end] += win_sq
            confidences.append(frame_conf)
            gains.append(np.mean(gain))

        # Overlap-add window normalization
        nonzero = norm_window > 1e-6
        out_signal[nonzero] /= norm_window[nonzero]

        # Truncate to original length
        enhanced = out_signal[:orig_len]

        # Scale/clip check to guarantee [-1, 1] bounded finite output
        max_peak = float(np.max(np.abs(enhanced))) if len(enhanced) > 0 else 0.0
        if max_peak > 1.0:
            enhanced = enhanced / max_peak

        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.99, neginf=-0.99).astype(np.float32)
        t_proc = time.perf_counter() - t_start
        audio_dur = orig_len / self.sample_rate
        rtf = t_proc / (audio_dur + 1e-8)

        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        diagnostics = {
            "confidences": np.array(confidences, dtype=np.float32),
            "mean_confidence": mean_conf,
            "mean_gain": float(np.mean(gains)) if gains else 1.0,
            "processing_time_sec": t_proc,
            "audio_duration_sec": audio_dur,
            "real_time_factor": rtf,
            "num_frames": num_frames,
        }

        if return_diagnostics:
            return enhanced, mean_conf, diagnostics
        return enhanced, mean_conf


# Backward compatibility alias for health_check and architecture references
AIEnhancer = SpectralEnhancer
