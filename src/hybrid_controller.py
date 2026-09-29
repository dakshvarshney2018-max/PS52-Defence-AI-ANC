"""
src/hybrid_controller.py — Confidence-Weighted Hybrid Controller for PS52_DANC
================================================================================

Implements an explainable, deterministic, confidence-weighted hybrid controller
that dynamically arbitrates and blends across:
  1. Classical Wiener Spectral Enhancer (Stage 4C)
  2. Lightweight Neural Speech Enhancer (Stage 4D)
  3. Classical Normalized FxLMS Adaptive Filter (Stage 3)
  4. Safe Pass-Through / Headroom Path

The controller uses acoustic understanding from the Noise Regime Estimator
(Stage 4A) and Speech Presence Confidence to dynamically compute non-negative,
sum-normalized blending weights:

    y_hybrid[n] = w_wiener * y_wiener[n]
                + w_neural * y_neural[n]
                + w_anc    * y_anc[n]
                + w_pass   * y_noisy[n]

Where:
    w_i >= 0  for all i
    w_wiener + w_neural + w_anc + w_pass == 1.0

DESIGN PRINCIPLES:
------------------
1. Explainability: Weights are computed via deterministic acoustic physics rules,
   NOT a black-box policy network.
2. Smoothness: EMA smoothing across frames prevents abrupt switching artifacts.
3. Safety: Instantaneous impulse guard reaction, bounded output [-1.0, 1.0],
   and automatic fallback if any branch fails or produces NaNs/Infs.
4. Telemetry: Exposes full state tracking for real-time monitoring and dashboard.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from src.audio_io import TARGET_SAMPLE_RATE, frame_audio, unframe_audio
from src.features import extract_frame_features
from src.regime_estimator import NoiseRegimeEstimator, REGIME_NAMES
from src.impulse_guard import ImpulseGuard, AdaptationState
from src.fxlms import FxLMSFilter
from src.ai_enhancer import SpectralEnhancer
from src.neural_enhancer import NeuralSpeechEnhancer


class HybridController:
    """
    Confidence-Weighted Noise-Regime-Aware Hybrid Controller.

    Parameters:
      sample_rate: Audio sampling frequency in Hz (default 16000).
      frame_length: Processing frame length in samples (320 = 20ms).
      hop_length: Frame hop size in samples (160 = 10ms).
      smoothing_alpha: EMA weight smoothing factor (0.70 = ~30ms response).
      fast_attack_on_impulse: If True, bypass smoothing on impulse detection.
      config: Optional configuration dictionary.
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        frame_length: int = 320,
        hop_length: int = 160,
        smoothing_alpha: float = 0.70,
        fast_attack_on_impulse: bool = True,
        anc_engine: Union[str, Any] = "existing",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.smoothing_alpha = float(smoothing_alpha)
        self.fast_attack_on_impulse = bool(fast_attack_on_impulse)
        self.config = config or {}

        # Internal sub-modules (for end-to-end full processing)
        self.regime_estimator = NoiseRegimeEstimator()
        self.impulse_guard = ImpulseGuard(sample_rate=self.sample_rate)
        self.wiener_enhancer = SpectralEnhancer(sample_rate=self.sample_rate)
        self.neural_enhancer = NeuralSpeechEnhancer(sample_rate=self.sample_rate)

        # Selectable ANC engine: "existing" (default FxLMSFilter) or "teammate_frozen" (TeammateFxLMSAdapter)
        if isinstance(anc_engine, str):
            if anc_engine.lower() == "teammate_frozen":
                from src.teammate_fxlms_adapter import TeammateFxLMSAdapter
                self.fxlms_filter = TeammateFxLMSAdapter()
                self.anc_engine_name = "teammate_frozen"
            else:
                self.fxlms_filter = FxLMSFilter()
                self.anc_engine_name = "existing"
        else:
            self.fxlms_filter = anc_engine
            self.anc_engine_name = getattr(anc_engine, "name", "custom")

        # State tracking
        self.prev_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        # Order: [wiener, neural, anc, passthrough]
        self.last_state: Dict[str, Any] = {}
        self.frame_count = 0
        self.reset()

    def reset(self) -> None:
        """Resets controller state and all internal sub-modules."""
        self.prev_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.frame_count = 0
        self.last_state = {
            "regime": "stationary",
            "regime_probs": {"stationary": 0.333, "non_stationary": 0.333, "impulsive": 0.334},
            "regime_confidence": 0.5,
            "speech_confidence": 0.5,
            "weights": {
                "wiener": 0.25,
                "neural": 0.25,
                "anc": 0.25,
                "passthrough": 0.25,
            },
            "impulse_state": "NORMAL",
            "safety_status": "OK",
        }
        self.regime_estimator.reset()
        self.impulse_guard.reset()
        self.wiener_enhancer.reset()
        self.neural_enhancer.reset()
        self.fxlms_filter.reset()

    def compute_target_weights(
        self,
        regime_probs: Union[Dict[str, float], np.ndarray, List[float]],
        regime_confidence: float,
        speech_confidence: float,
        impulse_state: Union[AdaptationState, str] = "NORMAL",
    ) -> np.ndarray:
        """
        Computes explainable target blending weights based on acoustic rules.

        Weight mapping policy:
        ----------------------
        1. Base weights from regime probabilities:
           - Stationary -> Favors Classical Wiener (0.50) & FxLMS ANC (0.35)
           - Non-Stationary -> Favors Neural Enhancer (0.55) & Wiener (0.20)
           - Impulsive -> Conservative speech preservation (0.10 Wiener, 0.10 Neural, 0.00 ANC)

        2. Regime Confidence Modulation:
           - Low confidence shrinks active processing and increases pass-through headroom.

        3. Speech Presence Modulation:
           - High C_speech prioritizes speech enhancement (Wiener + Neural).
           - Low C_speech reduces enhancement aggressiveness to protect speech cues.

        4. Impulse Guard Override:
           - FROZEN_ADAPTATION -> Zero ANC, clamp speech enhancers, route to pass-through.
           - REDUCED_ADAPTATION -> Reduce ANC by 70%.

        Returns:
          np.ndarray of shape (4,) with [w_wiener, w_neural, w_anc, w_pass],
          strictly non-negative and sum == 1.0.
        """
        # Parse regime probabilities
        if isinstance(regime_probs, dict):
            p_stat = float(regime_probs.get("stationary", 0.333))
            p_nonstat = float(regime_probs.get("non_stationary", 0.333))
            p_imp = float(regime_probs.get("impulsive", 0.334))
        else:
            p_stat = float(regime_probs[0])
            p_nonstat = float(regime_probs[1])
            p_imp = float(regime_probs[2])

        # Clamp inputs to valid ranges
        p_total = p_stat + p_nonstat + p_imp
        if p_total > 1e-6:
            p_stat /= p_total
            p_nonstat /= p_total
            p_imp /= p_total
        else:
            p_stat, p_nonstat, p_imp = 0.333, 0.333, 0.334

        c_reg = float(np.clip(regime_confidence, 0.0, 1.0))
        c_speech = float(np.clip(speech_confidence, 0.0, 1.0))
        imp_str = str(impulse_state.value if hasattr(impulse_state, "value") else impulse_state).upper()

        # Step 1: Base weights from acoustic noise regime
        # Stationary: Classical Wiener & FxLMS ANC dominate
        w_wiener_base = p_stat * 0.40 + p_nonstat * 0.35 + p_imp * 0.12
        # Non-Stationary: Neural enhancer dominates
        w_neural_base = p_stat * 0.10 + p_nonstat * 0.48 + p_imp * 0.12
        # ANC: Effective on stationary, zero on impulsive
        w_anc_base = p_stat * 0.50 + p_nonstat * 0.17 + p_imp * 0.00

        # Step 2: Speech confidence modulation
        # High speech confidence boosts speech enhancement branches
        # Low speech confidence scales down aggressive filtering
        speech_enh_scale = 0.70 + 0.30 * c_speech
        w_wiener = w_wiener_base * speech_enh_scale
        w_neural = w_neural_base * speech_enh_scale
        # When speech is strongly present, scale down standalone ANC slightly to prevent crosstalk
        w_anc = w_anc_base * (1.0 - 0.20 * c_speech)

        # Step 3: Impulse safety & Confidence-driven pass-through headroom
        if "FROZEN" in imp_str:
            # Severe acoustic transient detected: zero out ANC, clamp enhancers, route to pass-through
            w_anc = 0.0
            w_wiener = min(w_wiener, 0.15)
            w_neural = min(w_neural, 0.15)
            w_pass = 0.70
        elif "REDUCED" in imp_str:
            # Moderate transient: reduce ANC by 80%
            w_anc *= 0.20
            w_wiener = min(w_wiener, 0.30)
            w_neural = min(w_neural, 0.30)
            w_pass = 0.40
        else:
            # NORMAL OPERATION:
            # Zero pass-through bleed under high confidence (c_reg >= 0.90)
            # Gentle pass-through ONLY if regime estimator has low confidence
            w_pass = max(0.0, (1.0 - c_reg) * 0.35) if c_reg < 0.90 else 0.0

        # Step 4: Normalization so sum == 1.0
        active_sum = w_wiener + w_neural + w_anc
        if active_sum > 1e-6:
            target_active = max(0.0, 1.0 - w_pass)
            scale = target_active / active_sum
            w_wiener *= scale
            w_neural *= scale
            w_anc *= scale
        else:
            w_pass = 1.0
            w_wiener = w_neural = w_anc = 0.0

        target = np.array([w_wiener, w_neural, w_anc, w_pass], dtype=np.float32)
        target = np.maximum(0.0, target)
        target = target / np.sum(target)
        return target

    def blend_signals(
        self,
        raw_noisy: np.ndarray,
        wiener_out: np.ndarray,
        neural_out: np.ndarray,
        anc_out: np.ndarray,
        target_weights: np.ndarray,
        is_impulse: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Smoothly blends branch signals using temporal weight smoothing and safety checks.

        Returns:
          (blended_audio, applied_weights, safety_status)
        """
        min_len = min(len(raw_noisy), len(wiener_out), len(neural_out), len(anc_out))
        if min_len == 0:
            return np.zeros(0, dtype=np.float32), target_weights, "EMPTY_INPUT"

        # Check for invalid branches (NaN/Inf or runaway values)
        valid_w = bool(np.all(np.isfinite(wiener_out[:min_len])))
        valid_n = bool(np.all(np.isfinite(neural_out[:min_len])))
        valid_a = bool(np.all(np.isfinite(anc_out[:min_len])))
        valid_r = bool(np.all(np.isfinite(raw_noisy[:min_len])))

        safety_status = "OK"
        w_adj = target_weights.copy()

        if not valid_w:
            w_adj[0] = 0.0
            safety_status = "FALLBACK_WIENER_INVALID"
        if not valid_n:
            w_adj[1] = 0.0
            safety_status = "FALLBACK_NEURAL_INVALID"
        if not valid_a:
            w_adj[2] = 0.0
            safety_status = "FALLBACK_ANC_INVALID"
        if not valid_r:
            # Catastrophic raw audio failure
            return np.zeros(min_len, dtype=np.float32), target_weights, "CRITICAL_RAW_INVALID"

        # Renormalize valid target weights
        s = np.sum(w_adj)
        if s > 1e-6:
            w_adj /= s
        else:
            w_adj = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Temporal EMA smoothing
        if is_impulse and self.fast_attack_on_impulse:
            # Immediate freeze on shockwave to protect hearing without smoothing delay
            applied_weights = w_adj
        else:
            if self.frame_count == 0:
                applied_weights = w_adj
            else:
                applied_weights = (
                    self.smoothing_alpha * self.prev_weights + (1.0 - self.smoothing_alpha) * w_adj
                )
                applied_weights = np.maximum(0.0, applied_weights)

        # CRITICAL FIX (Issue 1): Strictly zero out applied weights for any invalid branch post-smoothing!
        if not valid_w:
            applied_weights[0] = 0.0
        if not valid_n:
            applied_weights[1] = 0.0
        if not valid_a:
            applied_weights[2] = 0.0

        # Renormalize applied weights
        s_app = np.sum(applied_weights)
        if s_app > 1e-6:
            applied_weights /= s_app
        else:
            applied_weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self.prev_weights = applied_weights.copy()
        self.frame_count += 1

        # Synthesize blended audio with NaN-safe buffers
        w_clean = np.nan_to_num(wiener_out[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n_clean = np.nan_to_num(neural_out[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        a_clean = np.nan_to_num(anc_out[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        r_clean = np.nan_to_num(raw_noisy[:min_len], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y_out = (
            applied_weights[0] * w_clean
            + applied_weights[1] * n_clean
            + applied_weights[2] * a_clean
            + applied_weights[3] * r_clean
        )

        # Safety bound [-1.0, 1.0]
        y_out = np.clip(y_out, -1.0, 1.0)
        return y_out, applied_weights, safety_status

    def process(
        self,
        raw_noisy: np.ndarray,
        wiener_out: np.ndarray,
        neural_out: np.ndarray,
        anc_out: np.ndarray,
        regime_probs: Union[Dict[str, float], np.ndarray, List[float]],
        regime_confidence: float,
        speech_confidence: float,
        impulse_state: Union[AdaptationState, str] = "NORMAL",
        return_diagnostics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Processes pre-computed branch signals and returns the hybrid enhanced output.

        Parameters:
          raw_noisy: Input noisy audio frame or array.
          wiener_out: Output from Wiener spectral enhancer.
          neural_out: Output from Neural speech enhancer.
          anc_out: Output from FxLMS adaptive filter / ANC residual.
          regime_probs: Regime probabilities from NoiseRegimeEstimator.
          regime_confidence: Confidence score of regime estimate in [0, 1].
          speech_confidence: Speech presence confidence score in [0, 1].
          impulse_state: Current ImpulseGuard state ("NORMAL", "REDUCED_ADAPTATION", "FROZEN_ADAPTATION").
          return_diagnostics: If True, returns (output, diagnostics_dict).

        Returns:
          enhanced_audio: 1D float32 array of hybrid processed speech.
          diagnostics: (optional) Dictionary of controller telemetry.
        """
        t_start = time.perf_counter()
        imp_str = str(impulse_state.value if hasattr(impulse_state, "value") else impulse_state).upper()
        is_impulse = "FROZEN" in imp_str or "REDUCED" in imp_str

        # Compute target blending weights
        target_weights = self.compute_target_weights(
            regime_probs=regime_probs,
            regime_confidence=regime_confidence,
            speech_confidence=speech_confidence,
            impulse_state=impulse_state,
        )

        # Blend branches with smoothing and safety validation
        y_blended, applied_weights, safety_status = self.blend_signals(
            raw_noisy=raw_noisy,
            wiener_out=wiener_out,
            neural_out=neural_out,
            anc_out=anc_out,
            target_weights=target_weights,
            is_impulse=is_impulse,
        )

        t_elapsed = time.perf_counter() - t_start
        duration_sec = len(raw_noisy) / max(self.sample_rate, 1)
        rtf = t_elapsed / max(duration_sec, 1e-6)

        # Determine dominant regime label
        if isinstance(regime_probs, dict):
            probs_dict = regime_probs
            dom_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
        else:
            probs_dict = {
                "stationary": float(regime_probs[0]),
                "non_stationary": float(regime_probs[1]),
                "impulsive": float(regime_probs[2]),
            }
            dom_regime = REGIME_NAMES[int(np.argmax(regime_probs))]

        state = {
            "regime": dom_regime,
            "regime_probs": probs_dict,
            "regime_confidence": round(float(regime_confidence), 4),
            "speech_confidence": round(float(speech_confidence), 4),
            "weights": {
                "wiener": round(float(applied_weights[0]), 4),
                "neural": round(float(applied_weights[1]), 4),
                "anc": round(float(applied_weights[2]), 4),
                "passthrough": round(float(applied_weights[3]), 4),
            },
            "impulse_state": imp_str,
            "safety_status": safety_status,
            "processing_time_sec": t_elapsed,
            "real_time_factor": rtf,
            "frame_count": self.frame_count,
        }
        self.last_state = state

        if return_diagnostics:
            return y_blended, state
        return y_blended

    def process_utterance(
        self,
        noisy_audio: np.ndarray,
        noise_reference: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        End-to-end full utterance processing convenience method.
        Executes all Stage 3 and Stage 4 branches on the input audio,
        computes frame-level telemetry, and blends the hybrid output.

        Parameters:
          noisy_audio: 1D float32 audio array of primary communication audio.
          noise_reference: Optional reference microphone noise signal for FxLMS.
                           If None, uses self-reference / primary audio.
          return_diagnostics: If True, returns (hybrid_audio, diagnostics).

        Returns:
          hybrid_audio: 1D float32 enhanced audio.
          diagnostics: (optional) Detailed frame-by-frame and aggregate metrics.
        """
        t_start = time.perf_counter()
        sig = np.asarray(noisy_audio, dtype=np.float32)
        n_samples = len(sig)
        if n_samples == 0:
            empty_diag = {
                "processing_time_sec": 0.0,
                "real_time_factor": 0.0,
                "frame_telemetry": [],
                "summary": {},
            }
            if return_diagnostics:
                return np.zeros(0, dtype=np.float32), empty_diag
            return np.zeros(0, dtype=np.float32)

        # 1. Execute Speech Enhancement Branches
        wiener_out, wiener_conf = self.wiener_enhancer.enhance(sig)
        neural_out, neural_conf = self.neural_enhancer.enhance(sig)

        # Combined speech confidence (mean of neural & spectral estimators)
        speech_conf = float(0.5 * wiener_conf + 0.5 * neural_conf)

        # 2. Execute FxLMS Classical ANC Branch (when reference mic available)
        if noise_reference is not None:
            ref = np.asarray(noise_reference, dtype=np.float32)
            self.fxlms_filter.reset()
            _, raw_anc_out, _ = self.fxlms_filter.process_buffer(ref_signal=ref, desired_signal=sig)

            # Complementary Spectral Post-Filter on FxLMS residual
            anc_post, _ = self.wiener_enhancer.enhance(raw_anc_out)

            # High-SNR & Speech-Confidence Adaptive Blending
            p_sig = float(np.mean(sig ** 2))
            p_ref = float(np.mean(ref ** 2))
            est_input_snr = (
                10.0 * np.log10(max(p_sig - p_ref, 1e-12) / max(p_ref, 1e-12))
                if p_sig > p_ref
                else 0.0
            )

            if est_input_snr > 12.0 or speech_conf > 0.88:
                # High-SNR regime: preserve clean speech formants with minimal post-filter
                beta_post = 0.05
            elif est_input_snr > 6.0:
                # Moderate SNR regime: gentle post-filtering
                beta_post = float(0.35 * (1.0 - 0.5 * speech_conf))
            else:
                # Low SNR regime: full complementary diffuse hiss suppression
                beta_post = float(0.65 * (1.0 - 0.4 * speech_conf))

            anc_out = (1.0 - beta_post) * raw_anc_out + beta_post * anc_post
            anc_out = np.clip(anc_out, -1.0, 1.0).astype(np.float32)
        else:
            anc_out = sig

        # 3. Fast Vectorized Feature Extraction & Frame Telemetry
        from src.features import extract_sequence_features
        if n_samples < self.frame_length:
            # Very short audio (< 1 frame): single frame processing
            feats, _ = extract_frame_features(sig, sample_rate=self.sample_rate)
            r_dict = self.regime_estimator.estimate_frame(feats)
            probs = {
                "stationary": r_dict["stationary_prob"],
                "non_stationary": r_dict["non_stationary_prob"],
                "impulsive": r_dict["impulsive_prob"],
            }
            c_reg = r_dict["confidence"]
            _, _, imp_state, _ = self.impulse_guard.process_frame(float(feats["rms_energy"]), float(probs["impulsive"]))
            f_hybrid, f_state = self.process(
                raw_noisy=sig,
                wiener_out=wiener_out,
                neural_out=neural_out,
                anc_out=anc_out,
                regime_probs=probs,
                regime_confidence=c_reg,
                speech_confidence=speech_conf,
                impulse_state=imp_state,
                return_diagnostics=True,
            )
            frame_telemetry = [f_state]
            applied_weights_mat = np.array([[
                f_state["weights"]["wiener"],
                f_state["weights"]["neural"],
                f_state["weights"]["anc"],
                f_state["weights"]["passthrough"],
            ]], dtype=np.float32)
        else:
            seq_feats = extract_sequence_features(
                sig,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
            num_frames = len(seq_feats["rms_energy"])
            applied_weights_mat = np.zeros((num_frames, 4), dtype=np.float32)
            frame_telemetry = [] if return_diagnostics else None

            self.regime_estimator.reset()
            self.impulse_guard.reset()
            self.prev_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
            self.frame_count = 0

            rms_arr = seq_feats["rms_energy"]

            for i in range(num_frames):
                f_dict = {k: seq_feats[k][i] for k in seq_feats}
                reg_dict = self.regime_estimator.estimate_frame(f_dict)
                probs = {
                    "stationary": reg_dict["stationary_prob"],
                    "non_stationary": reg_dict["non_stationary_prob"],
                    "impulsive": reg_dict["impulsive_prob"],
                }
                c_reg = reg_dict["confidence"]

                _, _, imp_state, _ = self.impulse_guard.process_frame(
                    float(rms_arr[i]),
                    float(probs["impulsive"]),
                )

                # Target weights and smoothing
                target_w = self.compute_target_weights(
                    regime_probs=probs,
                    regime_confidence=c_reg,
                    speech_confidence=speech_conf,
                    impulse_state=imp_state,
                )
                imp_str = str(imp_state.value if hasattr(imp_state, "value") else imp_state).upper()
                is_imp = "FROZEN" in imp_str or "REDUCED" in imp_str

                if is_imp and self.fast_attack_on_impulse:
                    app_w = target_w
                else:
                    if self.frame_count == 0:
                        app_w = target_w
                    else:
                        app_w = self.smoothing_alpha * self.prev_weights + (1.0 - self.smoothing_alpha) * target_w
                        app_w = np.maximum(0.0, app_w)
                        app_w /= np.sum(app_w)

                self.prev_weights = app_w.copy()
                self.frame_count += 1
                applied_weights_mat[i] = app_w

                if return_diagnostics:
                    dom_reg = max(probs.items(), key=lambda x: x[1])[0]
                    frame_telemetry.append({
                        "regime": dom_reg,
                        "regime_probs": probs,
                        "regime_confidence": round(float(c_reg), 4),
                        "speech_confidence": round(float(speech_conf), 4),
                        "weights": {
                            "wiener": round(float(app_w[0]), 4),
                            "neural": round(float(app_w[1]), 4),
                            "anc": round(float(app_w[2]), 4),
                            "passthrough": round(float(app_w[3]), 4),
                        },
                        "impulse_state": imp_str,
                        "safety_status": "OK",
                        "frame_count": self.frame_count,
                    })

        # 4. Fast Continuous Sample-Level Blending
        num_frames = len(applied_weights_mat)
        if num_frames == 1:
            w_sample = np.tile(applied_weights_mat, (n_samples, 1))
        else:
            # Linearly interpolate frame weights smoothly across hop points
            frame_centers = np.arange(num_frames) * self.hop_length + (self.frame_length // 2)
            sample_indices = np.arange(n_samples)
            w_sample = np.zeros((n_samples, 4), dtype=np.float32)
            for ch in range(4):
                w_sample[:, ch] = np.interp(
                    sample_indices, frame_centers, applied_weights_mat[:, ch]
                ).astype(np.float32)

            # Ensure sum constraint per sample
            w_sums = np.sum(w_sample, axis=-1, keepdims=True)
            w_sample = np.where(w_sums > 1e-6, w_sample / w_sums, np.array([0, 0, 0, 1], dtype=np.float32))

        # Synthesize blended audio
        min_len = min(n_samples, len(wiener_out), len(neural_out), len(anc_out))
        hybrid_audio = (
            w_sample[:min_len, 0] * wiener_out[:min_len]
            + w_sample[:min_len, 1] * neural_out[:min_len]
            + w_sample[:min_len, 2] * anc_out[:min_len]
            + w_sample[:min_len, 3] * sig[:min_len]
        )
        hybrid_audio = np.clip(hybrid_audio, -1.0, 1.0).astype(np.float32)

        if return_diagnostics:
            t_elapsed = time.perf_counter() - t_start
            duration_sec = n_samples / self.sample_rate
            rtf = t_elapsed / max(duration_sec, 1e-6)

            avg_w = float(np.mean(applied_weights_mat[:, 0]))
            avg_n = float(np.mean(applied_weights_mat[:, 1]))
            avg_a = float(np.mean(applied_weights_mat[:, 2]))
            avg_p = float(np.mean(applied_weights_mat[:, 3]))
            avg_creg = float(np.mean([ft["regime_confidence"] for ft in frame_telemetry])) if frame_telemetry else 0.0

            diag = {
                "anc_engine": self.anc_engine_name,
                "processing_time_sec": t_elapsed,
                "real_time_factor": rtf,
                "duration_sec": duration_sec,
                "speech_confidence": speech_conf,
                "average_regime_confidence": avg_creg,
                "average_weights": {
                    "wiener": avg_w,
                    "neural": avg_n,
                    "anc": avg_a,
                    "passthrough": avg_p,
                },
                "frame_telemetry": frame_telemetry,
                "wiener_out": wiener_out,
                "neural_out": neural_out,
                "anc_out": anc_out,
            }
            return hybrid_audio, diag

        return hybrid_audio
