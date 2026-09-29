"""
impulse_guard.py - Transient & Impulsive Noise Guard for PS52_DANC

Detects sudden high-energy transient events (e.g. acoustic shockwaves, blasts)
and controls FxLMS step-size adaptation to prevent filter divergence while
protecting human hearing.

Adaptation States:
------------------
1. NORMAL            : Step size multiplier = 1.0 (Normal convergence)
2. REDUCED_ADAPTATION: Step size multiplier = 0.2 (Reduced step size during moderate transient)
3. FROZEN_ADAPTATION : Step size multiplier = 0.0 (Weights held constant during severe shockwave)

Recovery Behavior:
------------------
When an impulse subsides, state is held for release_time_ms (hangover window),
then step_size_multiplier smoothly ramps back to 1.0 over recovery window.
"""

from enum import Enum
from typing import Tuple, Dict, Any, List
import numpy as np

class AdaptationState(str, Enum):
    NORMAL = "NORMAL"
    REDUCED_ADAPTATION = "REDUCED_ADAPTATION"
    FROZEN_ADAPTATION = "FROZEN_ADAPTATION"

class ImpulseGuard:
    """
    Lightweight Transient Impulse Energy Detector & Adaptation Safety Guard.
    """
    def __init__(
        self,
        energy_threshold_multiplier: float = 5.0,
        sample_rate: int = 16000,
        win_size_ms: float = 5.0,
        attack_time_ms: float = 1.0,
        release_time_ms: float = 50.0,
        alpha_baseline: float = 0.005,
        min_baseline_energy: float = 1e-4,
        eps: float = 1e-8
    ):
        self.threshold_multiplier = float(energy_threshold_multiplier)
        self.sample_rate = int(sample_rate)
        self.win_size_samples = max(1, int((win_size_ms / 1000.0) * sample_rate))
        self.release_samples = max(1, int((release_time_ms / 1000.0) * sample_rate))
        self.alpha_baseline = float(alpha_baseline)
        self.min_baseline_energy = float(min_baseline_energy)
        self.eps = float(eps)
        
        # State variables
        self.window_buf = np.zeros(self.win_size_samples, dtype=np.float32)
        self.baseline_energy = float(min_baseline_energy)
        self.release_counter = 0
        self.current_state = AdaptationState.NORMAL
        self.current_multiplier = 1.0
        self.frame_energy_history: List[float] = []
        self.frame_history_max = 20
        self.frame_release_frames = max(
            1, int(round(self.release_samples / max(1, int(0.01 * self.sample_rate))))
        )

    def reset(self) -> None:
        """Resets guard internal state."""
        self.window_buf.fill(0.0)
        self.baseline_energy = self.min_baseline_energy
        self.release_counter = 0
        self.current_state = AdaptationState.NORMAL
        self.current_multiplier = 1.0
        self.frame_energy_history = []

    def process_sample(self, sample: float) -> Tuple[bool, float, AdaptationState, float]:
        """
        Processes a single sample and returns:
          (impulse_detected, energy_ratio, adaptation_state, step_size_multiplier)
        """
        # Shift sample into rolling window
        self.window_buf = np.roll(self.window_buf, 1)
        self.window_buf[0] = sample
        
        # Compute short-time RMS energy
        short_energy = float(np.mean(self.window_buf**2))
        
        # Energy ratio relative to baseline
        ratio = short_energy / (self.baseline_energy + self.eps)
        
        impulse_detected = False
        
        if ratio >= 2.0 * self.threshold_multiplier:
            # Severe Impulse Shockwave
            impulse_detected = True
            self.current_state = AdaptationState.FROZEN_ADAPTATION
            self.current_multiplier = 0.0
            self.release_counter = self.release_samples
            
        elif ratio >= self.threshold_multiplier:
            # Moderate Transient Impulse
            impulse_detected = True
            self.current_state = AdaptationState.REDUCED_ADAPTATION
            self.current_multiplier = 0.2
            self.release_counter = self.release_samples
            
        else:
            # Below threshold: check hangover / release phase
            if self.release_counter > 0:
                impulse_detected = True
                self.release_counter -= 1
                ramp_progress = 1.0 - (self.release_counter / self.release_samples)
                self.current_multiplier = min(1.0, 0.2 + 0.8 * ramp_progress)
                self.current_state = AdaptationState.REDUCED_ADAPTATION if self.current_multiplier < 0.9 else AdaptationState.NORMAL
            else:
                self.current_state = AdaptationState.NORMAL
                self.current_multiplier = 1.0
                # Update baseline energy slowly during normal conditions (bounded below by min_baseline_energy)
                new_baseline = (1.0 - self.alpha_baseline) * self.baseline_energy + self.alpha_baseline * short_energy
                self.baseline_energy = max(self.min_baseline_energy, new_baseline)
                
        return impulse_detected, ratio, self.current_state, self.current_multiplier

    def process_frame(
        self,
        rms_energy: float,
        impulsive_confidence: float = 0.0,
    ) -> Tuple[bool, float, AdaptationState, float]:
        '''Robust frame-level transient detector using recent median/MAD energy.'''
        energy = max(float(rms_energy) ** 2, self.eps)
        c_imp = float(np.clip(impulsive_confidence, 0.0, 1.0))

        if not hasattr(self, "frame_energy_history"):
            self.frame_energy_history = []
        if not hasattr(self, "frame_history_max"):
            self.frame_history_max = 20
        if not hasattr(self, "frame_release_frames"):
            self.frame_release_frames = max(
                1, int(round(self.release_samples / max(1, int(0.01 * self.sample_rate))))
            )

        hist = self.frame_energy_history

        if len(hist) < 4:
            hist.append(energy)
            self.current_state = AdaptationState.NORMAL
            self.current_multiplier = 1.0
            return False, 1.0, self.current_state, self.current_multiplier

        baseline = float(np.median(hist))
        mad = float(np.median(np.abs(np.asarray(hist, dtype=np.float64) - baseline)))
        robust_sigma = max(1.4826 * mad, baseline * 0.05, self.eps)

        ratio = energy / max(baseline, self.min_baseline_energy, self.eps)
        robust_z = (energy - baseline) / robust_sigma

        severe = (
            (ratio >= 6.0 and robust_z >= 6.0 and c_imp >= 0.55)
            or (ratio >= 12.0 and robust_z >= 8.0)
        )
        moderate = (
            (ratio >= 4.0 and robust_z >= 4.0 and c_imp >= 0.35)
            or (ratio >= 8.0 and robust_z >= 6.0)
        )

        if severe:
            self.current_state = AdaptationState.FROZEN_ADAPTATION
            self.current_multiplier = 0.0
            self.release_counter = self.frame_release_frames
            detected = True
        elif moderate:
            self.current_state = AdaptationState.REDUCED_ADAPTATION
            self.current_multiplier = 0.2
            self.release_counter = self.frame_release_frames
            detected = True
        elif self.release_counter > 0:
            self.release_counter -= 1
            progress = 1.0 - (
                self.release_counter / max(1, self.frame_release_frames)
            )
            self.current_multiplier = min(1.0, 0.2 + 0.8 * progress)
            self.current_state = (
                AdaptationState.REDUCED_ADAPTATION
                if self.current_multiplier < 0.9
                else AdaptationState.NORMAL
            )
            detected = True
        else:
            self.current_state = AdaptationState.NORMAL
            self.current_multiplier = 1.0
            detected = False

        if not detected:
            hist.append(energy)
            if len(hist) > self.frame_history_max:
                del hist[:-self.frame_history_max]

        return detected, ratio, self.current_state, self.current_multiplier

    def process_buffer(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Processes 1D signal array and returns diagnostic state arrays:
          - impulse_detected: bool array
          - energy_ratio: float array
          - adaptation_state: list of state strings
          - step_size_multipliers: float array
        """
        n_samples = len(signal)
        detected_arr = np.zeros(n_samples, dtype=bool)
        ratio_arr = np.zeros(n_samples, dtype=np.float32)
        state_list = []
        mult_arr = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            det, r, st, mult = self.process_sample(signal[i])
            detected_arr[i] = det
            ratio_arr[i] = r
            state_list.append(st.value)
            mult_arr[i] = mult
            
        return {
            "impulse_detected": detected_arr,
            "energy_ratio": ratio_arr,
            "adaptation_state": state_list,
            "step_size_multipliers": mult_arr
        }
