"""
regime_estimator.py - Acoustic Noise Regime Estimator for PS52_DANC

An explainable, deterministic evidence-based rule engine (NOT a neural network)
that estimates probabilities for three noise regimes:
1. Stationary (e.g. steady engine hum, pink/white background noise)
2. Non-Stationary (e.g. wind gusts, background speech chatter)
3. Impulsive (e.g. acoustic shockwaves, gunshot transients)

Evidence Rules:
----------------
- High Crest Factor (> 3.5) AND Energy Ratio (> 2.0 vs prior baseline) -> High Impulsive Evidence
- Low Crest Factor (< 3.0) & Low Energy Variance (< 0.003) -> High Stationary Evidence
- Moderate Energy Variance (>= 0.003) & Moderate Crest Factor (<= 3.5) -> High Non-Stationary Evidence

Includes exponential temporal hysteresis smoothing to prevent frame-flickering.
"""

from typing import Dict, Tuple, Any, List, Optional
import numpy as np

REGIME_NAMES = ["stationary", "non_stationary", "impulsive"]

class NoiseRegimeEstimator:
    """
    Explainable Noise Regime Estimator & Confidence Engine.
    """
    def __init__(
        self,
        smoothing_alpha: float = 0.25,
        rolling_window_len: int = 20,
        temperature: float = 1.0
    ):
        self.smoothing_alpha = float(smoothing_alpha)
        self.rolling_window_len = int(rolling_window_len)
        self.temperature = float(temperature)
        
        # State tracking for sequence processing & hysteresis
        self.prev_probs = np.array([0.333, 0.333, 0.334], dtype=np.float32)
        self.feature_history: List[Dict[str, float]] = []

    def reset(self) -> None:
        """Resets estimator history and state."""
        self.prev_probs = np.array([0.333, 0.333, 0.334], dtype=np.float32)
        self.feature_history.clear()

    def _compute_evidence_scores(self, features: Dict[str, float], window_features: List[Dict[str, float]]) -> np.ndarray:
        """
        Computes evidence scores [S_stat, S_nonstat, S_imp] based on physical acoustic principles.
        """
        crest_factor = features.get("crest_factor", 1.0)
        rms_energy = features.get("rms_energy", 0.0)
        spectral_flux = features.get("spectral_flux", 0.0)
        
        past_window = window_features[:-1] if len(window_features) > 1 else window_features
        if len(past_window) > 0:
            flux_arr = [f.get("spectral_flux", 0.0) for f in past_window]
            rms_arr = [f.get("rms_energy", 0.0) for f in past_window]
            mean_flux = float(np.mean(flux_arr))
            var_energy = float(np.var(rms_arr))
            mean_rms = float(np.mean(rms_arr)) + 1e-8
            energy_ratio = rms_energy / mean_rms
        else:
            mean_flux = spectral_flux
            var_energy = 0.0
            energy_ratio = 1.0
            
        # 1. Impulsive Evidence Score
        s_imp = 0.0
        if crest_factor > 3.5:
            s_imp += 8.0 * (crest_factor - 3.5)
        if energy_ratio > 2.0:
            s_imp += 6.0 * min(energy_ratio - 2.0, 5.0)
        if var_energy > 0.05:
            s_imp += 4.0
            
        # 2. Stationary Evidence Score
        s_stat = 0.0
        if var_energy < 0.003 and energy_ratio < 1.6:
            s_stat += 5.0 * (0.003 - var_energy) / 0.003
        if crest_factor < 3.0:
            s_stat += 3.0 * (3.0 - crest_factor)
            
        # 3. Non-Stationary Evidence Score
        s_nonstat = 0.0
        if var_energy >= 0.003 and crest_factor <= 3.5:
            s_nonstat += 5.0 * min(var_energy / 0.02, 2.0)
        if mean_flux >= 0.55 and crest_factor <= 3.5:
            s_nonstat += 3.0
            
        # Near-silence handling (RMS < 1e-4) -> Default to stationary
        if rms_energy < 1e-4:
            s_stat += 5.0
            s_nonstat = 0.0
            s_imp = 0.0
            
        return np.array([s_stat, s_nonstat, s_imp], dtype=np.float32)

    def estimate_frame(
        self,
        features: Dict[str, float],
        apply_smoothing: bool = True
    ) -> Dict[str, Any]:
        """
        Estimates probabilities and regime for a single frame feature set.
        """
        self.feature_history.append(features)
        if len(self.feature_history) > self.rolling_window_len:
            self.feature_history.pop(0)
            
        scores = self._compute_evidence_scores(features, self.feature_history)
        
        # Numerically stable softmax probability mapping
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted / max(0.1, self.temperature))
        raw_probs = exp_scores / np.sum(exp_scores)
        
        if apply_smoothing:
            smoothed_probs = (1.0 - self.smoothing_alpha) * self.prev_probs + self.smoothing_alpha * raw_probs
            smoothed_probs = smoothed_probs / np.sum(smoothed_probs)
            self.prev_probs = smoothed_probs
            probs = smoothed_probs
        else:
            probs = raw_probs
            
        p_stat, p_nonstat, p_imp = float(probs[0]), float(probs[1]), float(probs[2])
        
        # Confidence Score C in [0.0, 1.0]
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2])
        confidence = float(np.clip(margin / (1.0 - 0.3333), 0.0, 1.0))
        
        best_idx = int(np.argmax(probs))
        predicted_regime = REGIME_NAMES[best_idx]
        
        return {
            "stationary_prob": p_stat,
            "non_stationary_prob": p_nonstat,
            "impulsive_prob": p_imp,
            "confidence": confidence,
            "predicted_regime": predicted_regime
        }

    def estimate_sequence(
        self,
        seq_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Estimates regime probabilities across an entire feature sequence.
        """
        self.reset()
        num_frames = len(seq_features["rms_energy"])
        
        p_stat_arr = np.zeros(num_frames, dtype=np.float32)
        p_nonstat_arr = np.zeros(num_frames, dtype=np.float32)
        p_imp_arr = np.zeros(num_frames, dtype=np.float32)
        conf_arr = np.zeros(num_frames, dtype=np.float32)
        regimes_list = []
        
        for i in range(num_frames):
            frame_feats = {name: float(seq_features[name][i]) for name in seq_features.keys()}
            res = self.estimate_frame(frame_feats, apply_smoothing=True)
            
            p_stat_arr[i] = res["stationary_prob"]
            p_nonstat_arr[i] = res["non_stationary_prob"]
            p_imp_arr[i] = res["impulsive_prob"]
            conf_arr[i] = res["confidence"]
            regimes_list.append(res["predicted_regime"])
            
        avg_probs = [np.mean(p_stat_arr), np.mean(p_nonstat_arr), np.mean(p_imp_arr)]
        
        if np.mean(p_imp_arr > 0.5) > 0.015:
            overall_regime = "impulsive"
        elif avg_probs[1] > 0.20 or np.mean(p_nonstat_arr > 0.3) > 0.15:
            overall_regime = "non_stationary"
        else:
            overall_regime = "stationary"
            
        mean_conf = float(np.mean(conf_arr))
        
        return {
            "stationary_prob": p_stat_arr,
            "non_stationary_prob": p_nonstat_arr,
            "impulsive_prob": p_imp_arr,
            "confidence": conf_arr,
            "predicted_regimes": regimes_list,
            "overall_regime": overall_regime,
            "mean_confidence": mean_conf
        }
