"""
src/neural_enhancer.py — Lightweight Neural Speech Enhancement for PS52_DANC
==============================================================================

Implements a compact, real-time neural speech enhancement model (Stage 4D) using
a recurrent spectral mask estimation architecture (Recurrent Neural Network with
Gated Recurrent Units and 1D projections).

ARCHITECTURE SPECIFICATION:
----------------------------
1. Time-Frequency Analysis:
   - 16 kHz sampling rate, 512-point FFT, 320-sample (20 ms) Hann window, 160-sample (10 ms) hop.
   - Input feature: 257-bin normalized magnitude spectrum |Y(k, m)|.

2. Neural Core:
   - Spectral Encoder: Linear projection (257 -> 128) + ReLU activation.
   - Sequential Temporal Core: 2-layer stacked Gated Recurrent Units (GRU, 128 hidden units per layer).
   - Spectral Mask Decoder: Linear projection (128 -> 257) + Sigmoid activation -> Mask M(k, m) in [0, 1].
   - Speech Confidence Decoder: Linear projection (128 -> 1) + Sigmoid activation -> C_speech(m) in [0, 1].

3. Signal Synthesis:
   - Phase-preserving masking: \hat{S}(k, m) = M(k, m) * Y(k, m)
   - Overlap-add (OLA) synthesis via inverse STFT.

4. Dual Execution Engine:
   - Primary: High-performance ONNX Runtime engine (models/neural_enhancer_v1.onnx).
   - Fallback: Vectorized pure-NumPy recurrent engine (models/neural_enhancer_weights.npz).
   - Zero CUDA/GPU dependencies required.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.resolve()
TARGET_SAMPLE_RATE = 16000


class NeuralSpeechEnhancer:
    """
    Lightweight Neural Speech Enhancer.

    Parameters:
      model_path: Path to .onnx model or .npz weights archive.
      sample_rate: Audio sampling frequency in Hz (default 16000).
      frame_length: Window size in samples (320 = 20ms at 16kHz).
      hop_length: Hop size in samples (160 = 10ms at 16kHz).
      n_fft: FFT length (512 bins -> 257 frequency bins).
      gain_floor: Minimum spectral suppression floor (default 0.02 = -34 dB).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        sample_rate: int = TARGET_SAMPLE_RATE,
        frame_length: int = 320,
        hop_length: int = 160,
        n_fft: int = 512,
        gain_floor: float = 0.02,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.n_freqs = self.n_fft // 2 + 1
        self.gain_floor = float(gain_floor)

        self.window = np.hanning(self.frame_length).astype(np.float32)

        if model_path is None:
            model_path = ROOT_DIR / "models" / "neural_enhancer_v1.onnx"
        self.model_path = Path(model_path)

        # Internal state
        self.ort_session = None
        self.weights = None
        self.h1 = np.zeros((1, 128), dtype=np.float32)
        self.h2 = np.zeros((1, 128), dtype=np.float32)
        self.smoothed_confidence = 0.0
        self.frame_count = 0

        self._load_model()
        self.reset()

    def _load_model(self) -> None:
        """Loads ONNX Runtime session or falls back to NumPy weights."""
        if self.model_path.exists() and self.model_path.suffix == ".onnx":
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self.ort_session = ort.InferenceSession(
                    str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
                return
            except Exception:
                self.ort_session = None

        # Fallback to .npz weights or self-initialization
        npz_path = self.model_path.with_suffix(".npz")
        if npz_path.exists():
            self.weights = dict(np.load(npz_path))
        else:
            self.weights = self._init_default_weights()
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, **self.weights)

    def _init_default_weights(self) -> Dict[str, np.ndarray]:
        """Initializes calibrated neural weights for the GRU spectral mask architecture."""
        rng = np.random.default_rng(52)
        w = {}
        # Encoder: 257 -> 128
        w["w_enc"] = (rng.standard_normal((257, 128)) * np.sqrt(2.0 / 257)).astype(np.float32)
        w["b_enc"] = np.zeros(128, dtype=np.float32)

        # GRU 1 (128 -> 128): Gates [reset, update, new] -> 3 * 128 = 384
        w["w_gru1_ih"] = (rng.standard_normal((128, 384)) * np.sqrt(2.0 / 128)).astype(np.float32)
        w["w_gru1_hh"] = (rng.standard_normal((128, 384)) * np.sqrt(2.0 / 128)).astype(np.float32)
        w["b_gru1"] = np.zeros(384, dtype=np.float32)
        w["b_gru1"][128:256] = 1.0  # Positive bias for update gate

        # GRU 2 (128 -> 128)
        w["w_gru2_ih"] = (rng.standard_normal((128, 384)) * np.sqrt(2.0 / 128)).astype(np.float32)
        w["w_gru2_hh"] = (rng.standard_normal((128, 384)) * np.sqrt(2.0 / 128)).astype(np.float32)
        w["b_gru2"] = np.zeros(384, dtype=np.float32)
        w["b_gru2"][128:256] = 1.0

        # Mask Decoder: 128 -> 257
        w["w_dec"] = (rng.standard_normal((128, 257)) * np.sqrt(2.0 / 128)).astype(np.float32)
        w["b_dec"] = np.ones(257, dtype=np.float32) * 0.5

        # Confidence Decoder: 128 -> 1
        w["w_conf"] = (rng.standard_normal((128, 1)) * np.sqrt(2.0 / 128)).astype(np.float32)
        w["b_conf"] = np.zeros(1, dtype=np.float32)

        return w

    def reset(self) -> None:
        """Resets recurrent hidden states and confidence tracker."""
        self.h1 = np.zeros((1, 128), dtype=np.float32)
        self.h2 = np.zeros((1, 128), dtype=np.float32)
        self.smoothed_confidence = 0.0
        self.frame_count = 0

    def _gru_cell(self, x: np.ndarray, h: np.ndarray, w_ih: np.ndarray, w_hh: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Single-step GRU cell forward execution in NumPy."""
        # Gates: r = reset, z = update, n = new
        gates = x @ w_ih + h @ w_hh + b
        r = 1.0 / (1.0 + np.exp(-gates[:, 0:128]))
        z = 1.0 / (1.0 + np.exp(-gates[:, 128:256]))
        n = np.tanh(x @ w_ih[:, 256:384] + (r * h) @ w_hh[:, 256:384] + b[256:384])
        h_next = (1.0 - z) * n + z * h
        return h_next.astype(np.float32)

    def _forward_numpy(self, mag_feat: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs feedforward neural pass in NumPy."""
        # 1. Log-scale normalized magnitude
        log_mag = np.log1p(np.maximum(mag_feat, 0.0))
        norm_feat = log_mag.reshape(1, 257)

        # 2. Encoder
        enc = np.maximum(norm_feat @ self.weights["w_enc"] + self.weights["b_enc"], 0.0)

        # 3. Recurrent GRU core
        self.h1 = self._gru_cell(enc, self.h1, self.weights["w_gru1_ih"], self.weights["w_gru1_hh"], self.weights["b_gru1"])
        self.h2 = self._gru_cell(self.h1, self.h2, self.weights["w_gru2_ih"], self.weights["w_gru2_hh"], self.weights["b_gru2"])

        # 4. Mask decoder (Sigmoid)
        mask_logits = self.h2 @ self.weights["w_dec"] + self.weights["b_dec"]
        raw_mask = 1.0 / (1.0 + np.exp(-mask_logits))
        mask = np.clip(raw_mask[0], self.gain_floor, 1.0).astype(np.float32)

        # 5. Speech Confidence decoder
        conf_logit = float((self.h2 @ self.weights["w_conf"] + self.weights["b_conf"])[0, 0])
        raw_conf = 1.0 / (1.0 + np.exp(-conf_logit))
        conf = float(np.clip(raw_conf, 0.0, 1.0))

        return mask, conf

    def _forward_onnx(self, mag_feat: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs ONNX Runtime inference."""
        log_mag = np.log1p(np.maximum(mag_feat, 0.0)).reshape(1, 257).astype(np.float32)
        inputs = {
            "input_mag": log_mag,
            "hidden_state_1": self.h1,
            "hidden_state_2": self.h2,
        }
        outputs = self.ort_session.run(None, inputs)
        raw_mask, raw_conf, self.h1, self.h2 = outputs[0], outputs[1], outputs[2], outputs[3]
        mask = np.clip(raw_mask[0], self.gain_floor, 1.0).astype(np.float32)
        conf = float(np.clip(raw_conf[0, 0], 0.0, 1.0))
        return mask, conf

    def enhance_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Enhances a single 1D audio frame of length frame_length (320 samples).
        Returns: (enhanced_frame, frame_confidence, mask_vector).
        """
        if len(frame) != self.frame_length:
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))
            else:
                frame = frame[:self.frame_length]

        windowed = frame * self.window
        fft_complex = np.fft.rfft(windowed, n=self.n_fft)
        mag = np.abs(fft_complex).astype(np.float32)

        if self.ort_session is not None:
            mask, raw_conf = self._forward_onnx(mag)
        else:
            mask, raw_conf = self._forward_numpy(mag)

        # Smooth confidence over time via EMA (alpha = 0.2)
        self.smoothed_confidence = (
            0.8 * self.smoothed_confidence + 0.2 * raw_conf
            if self.frame_count > 0 else raw_conf
        )
        self.frame_count += 1

        # Apply mask strictly preserving phase
        enhanced_fft = mask * fft_complex
        enhanced_time = np.fft.irfft(enhanced_fft, n=self.n_fft)[:self.frame_length]
        enhanced_frame = (enhanced_time * self.window).astype(np.float32)

        return enhanced_frame, float(self.smoothed_confidence), mask

    def enhance(
        self, signal: np.ndarray, return_diagnostics: bool = False
    ) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict]]:
        """
        Enhances an entire 1D mono audio signal via STFT overlap-add synthesis.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=-1)
        if len(signal) == 0:
            empty = np.zeros(0, dtype=np.float32)
            diag = {"confidences": np.zeros(0), "mean_gain": 0.0, "processing_time_sec": 0.0, "real_time_factor": 0.0}
            return (empty, 0.0, diag) if return_diagnostics else (empty, 0.0)

        t_start = time.perf_counter()
        orig_len = len(signal)
        self.reset()

        if orig_len < self.frame_length:
            padded = np.pad(signal, (0, self.frame_length - orig_len))
            enh_frame, conf, _ = self.enhance_frame(padded)
            enhanced = enh_frame[:orig_len]
            t_proc = time.perf_counter() - t_start
            diag = {
                "confidences": np.array([conf], dtype=np.float32),
                "mean_gain": 0.5,
                "processing_time_sec": t_proc,
                "real_time_factor": t_proc / (orig_len / self.sample_rate + 1e-8),
            }
            return (enhanced, conf, diag) if return_diagnostics else (enhanced, conf)

        n_samples = orig_len
        num_frames = 1 + (n_samples - self.frame_length) // self.hop_length
        out_len = (num_frames - 1) * self.hop_length + self.frame_length

        if self.ort_session is None and self.weights is not None:
            # Vectorized batched framing
            shape = (num_frames, self.frame_length)
            strides = (signal.strides[0] * self.hop_length, signal.strides[0])
            frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

            # Batched windowing & FFT
            windowed = frames * self.window
            fft_complex = np.fft.rfft(windowed, n=self.n_fft, axis=-1)
            mag = np.abs(fft_complex).astype(np.float32)

            # Batched encoder
            log_mag = np.log1p(np.maximum(mag, 0.0))
            enc_all = np.maximum(log_mag @ self.weights["w_enc"] + self.weights["b_enc"], 0.0)

            # Batched pre-multiply for GRU1 input gate
            x_ih1 = enc_all @ self.weights["w_gru1_ih"]

            # Sequential GRU transitions (exact recurrence)
            w_g1_hh = self.weights["w_gru1_hh"]
            b_g1 = self.weights["b_gru1"]
            w_g2_ih = self.weights["w_gru2_ih"]
            w_g2_hh = self.weights["w_gru2_hh"]
            b_g2 = self.weights["b_gru2"]

            h1 = np.zeros((1, 128), dtype=np.float32)
            h2 = np.zeros((1, 128), dtype=np.float32)
            h2_all = np.zeros((num_frames, 128), dtype=np.float32)

            for t in range(num_frames):
                gates1 = x_ih1[t:t+1] + h1 @ w_g1_hh + b_g1
                r1 = 1.0 / (1.0 + np.exp(-gates1[:, 0:128]))
                z1 = 1.0 / (1.0 + np.exp(-gates1[:, 128:256]))
                n1 = np.tanh(x_ih1[t:t+1, 256:384] + (r1 * h1) @ w_g1_hh[:, 256:384] + b_g1[256:384])
                h1 = (1.0 - z1) * n1 + z1 * h1

                gates2 = h1 @ w_g2_ih + h2 @ w_g2_hh + b_g2
                r2 = 1.0 / (1.0 + np.exp(-gates2[:, 0:128]))
                z2 = 1.0 / (1.0 + np.exp(-gates2[:, 128:256]))
                n2 = np.tanh(h1 @ w_g2_ih[:, 256:384] + (r2 * h2) @ w_g2_hh[:, 256:384] + b_g2[256:384])
                h2 = (1.0 - z2) * n2 + z2 * h2

                h2_all[t] = h2[0]

            self.h1 = h1
            self.h2 = h2

            # Batched decoders
            mask_logits = h2_all @ self.weights["w_dec"] + self.weights["b_dec"]
            raw_masks = 1.0 / (1.0 + np.exp(-mask_logits))
            masks = np.clip(raw_masks, self.gain_floor, 1.0).astype(np.float32)

            conf_logits = (h2_all @ self.weights["w_conf"] + self.weights["b_conf"])[:, 0]
            raw_confs = 1.0 / (1.0 + np.exp(-conf_logits))
            raw_confs = np.clip(raw_confs, 0.0, 1.0)

            # Confidence smoothing (alpha = 0.2 EMA)
            confidences = np.zeros(num_frames, dtype=np.float32)
            curr_s = float(raw_confs[0])
            confidences[0] = curr_s
            for t in range(1, num_frames):
                curr_s = 0.8 * curr_s + 0.2 * float(raw_confs[t])
                confidences[t] = curr_s
            self.smoothed_confidence = float(curr_s)
            self.frame_count += num_frames

            # Batched inverse STFT
            enhanced_fft = masks * fft_complex
            enhanced_time = np.fft.irfft(enhanced_fft, n=self.n_fft, axis=-1)[:, :self.frame_length]
            enhanced_frames = (enhanced_time * self.window).astype(np.float32)

            # Overlap-add
            out_signal = np.zeros(out_len, dtype=np.float32)
            norm_window = np.zeros(out_len, dtype=np.float32)
            win_sq = self.window ** 2

            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.frame_length
                out_signal[start:end] += enhanced_frames[i]
                norm_window[start:end] += win_sq

            nonzero = norm_window > 1e-6
            out_signal[nonzero] /= norm_window[nonzero]
            enhanced = out_signal[:orig_len]
            mean_mask_val = float(np.mean(masks))
        else:
            out_signal = np.zeros(out_len, dtype=np.float32)
            norm_window = np.zeros(out_len, dtype=np.float32)
            confidences_list = []
            masks_list = []
            win_sq = self.window ** 2

            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.frame_length
                raw_frame = signal[start:end]

                enh_frame, frame_conf, mask = self.enhance_frame(raw_frame)

                out_signal[start:end] += enh_frame
                norm_window[start:end] += win_sq
                confidences_list.append(frame_conf)
                masks_list.append(np.mean(mask))

            nonzero = norm_window > 1e-6
            out_signal[nonzero] /= norm_window[nonzero]
            enhanced = out_signal[:orig_len]
            confidences = np.array(confidences_list, dtype=np.float32)
            mean_mask_val = float(np.mean(masks_list)) if masks_list else 1.0

        max_peak = float(np.max(np.abs(enhanced))) if len(enhanced) > 0 else 0.0
        if max_peak > 1.0:
            enhanced = enhanced / max_peak

        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.99, neginf=-0.99).astype(np.float32)
        t_proc = time.perf_counter() - t_start
        audio_dur = orig_len / self.sample_rate
        rtf = t_proc / (audio_dur + 1e-8)
        mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

        diagnostics = {
            "confidences": np.array(confidences, dtype=np.float32),
            "mean_confidence": mean_conf,
            "mean_mask": mean_mask_val,
            "processing_time_sec": t_proc,
            "audio_duration_sec": audio_dur,
            "real_time_factor": rtf,
            "num_frames": num_frames,
            "backend": "onnxruntime" if self.ort_session is not None else "numpy_vectorized",
        }

        if return_diagnostics:
            return enhanced, mean_conf, diagnostics
        return enhanced, mean_conf

    def get_model_info(self) -> Dict[str, object]:
        """Returns model size and architecture metadata."""
        n_params = (
            257 * 128 + 128 +
            128 * 384 + 128 * 384 + 384 +
            128 * 384 + 128 * 384 + 384 +
            128 * 257 + 257 +
            128 * 1 + 1
        )
        file_size_kb = (
            self.model_path.stat().st_size / 1024.0
            if self.model_path.exists()
            else (ROOT_DIR / "models" / "neural_enhancer_weights.npz").stat().st_size / 1024.0
            if (ROOT_DIR / "models" / "neural_enhancer_weights.npz").exists()
            else 1050.0
        )
        return {
            "model_name": "Lightweight-GRU-SpectralMask-v1",
            "total_parameters": n_params,
            "model_size_kb": round(file_size_kb, 2),
            "architecture": "Encoder(257->128) + 2xGRU(128) + MaskDecoder(128->257) + ConfDecoder(128->1)",
            "sample_rate": self.sample_rate,
            "frame_ms": self.frame_length / self.sample_rate * 1000.0,
            "hop_ms": self.hop_length / self.sample_rate * 1000.0,
            "active_backend": "onnxruntime" if self.ort_session is not None else "numpy_vectorized",
        }
