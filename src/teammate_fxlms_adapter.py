"""
src/teammate_fxlms_adapter.py — Adapter for Teammate's Frozen Classical FxLMS Baseline
========================================================================================

Wraps the teammate's exact frozen Filtered-X Least Mean Squares (FxLMS) baseline
from `PS52_FxLMS_AI_HANDOFF/PS52_FxLMS_AI_HANDOFF/01_FxLMS_SOURCE/dataset_fxlms_single_test.py`
and exposes an interface fully compatible with the PS52_DANC ANC branch and HybridController.

SEMANTIC BOUNDARY & PROVENANCE NOTICE:
--------------------------------------
- Algorithm Label: "Offline Simulated-Reference Acoustic ANC"
- This adapter executes the teammate's FROZEN classical baseline.
- Primary and secondary acoustic paths are SIMULATED mathematical models, NOT measured
  headset/microphone/speaker responses.
- The reference signal represents an idealized isolated reference microphone.
- FxLMS parameters are strictly frozen:
    * Fs = 16000 Hz
    * filter_length = 64 taps
    * mu = 0.01
    * epsilon = 1e-8
    * leakage = 0.0 (none)
    * impulse_guard inside baseline = OFF (False)
- Impulse protection is NOT applied inside the frozen FxLMS algorithm; instead,
  it is supervised at the HybridController / safety arbitration layer.
"""

from pathlib import Path
import importlib.util
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.signal import lfilter

# Dynamically load teammate's exact frozen source without modifying the handoff folder
_HANDOFF_SOURCE_PATH = (
    Path(__file__).parent.parent
    / "PS52_FxLMS_AI_HANDOFF"
    / "PS52_FxLMS_AI_HANDOFF"
    / "01_FxLMS_SOURCE"
    / "dataset_fxlms_single_test.py"
).resolve()

if not _HANDOFF_SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Teammate frozen FxLMS source not found at: {_HANDOFF_SOURCE_PATH}"
    )

_spec = importlib.util.spec_from_file_location("teammate_fxlms_source", _HANDOFF_SOURCE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Expose exact frozen functions and constants from teammate source
teammate_run_fxlms = _mod.run_fxlms
teammate_create_primary_path = _mod.create_primary_path
teammate_create_secondary_path = _mod.create_secondary_path

FROZEN_SAMPLE_RATE = int(_mod.SAMPLE_RATE)
FROZEN_FILTER_LENGTH = int(_mod.FILTER_LENGTH)
FROZEN_MU = float(_mod.MU)
FROZEN_EPSILON = float(_mod.EPSILON)
FROZEN_LEAKAGE = 0.0
FROZEN_IMPULSE_GUARD = False


class TeammateFxLMSAdapter:
    """
    Adapter encapsulating the teammate's frozen classical FxLMS baseline.

    Compatible with FxLMSFilter interface used by HybridController:
      - process_buffer(ref_signal, desired_signal, step_size_multipliers) -> (y_out, e_out, x_prime)
      - reset()
      - get_weights()
    """

    def __init__(
        self,
        filter_length: int = FROZEN_FILTER_LENGTH,
        mu: float = FROZEN_MU,
        epsilon: float = FROZEN_EPSILON,
        sample_rate: int = FROZEN_SAMPLE_RATE,
    ):
        # Enforce frozen parameters
        self.sample_rate = int(sample_rate)
        self.filter_length = int(filter_length)
        self.step_size_mu = float(mu)
        self.epsilon = float(epsilon)
        self.leakage = FROZEN_LEAKAGE
        self.impulse_guard_enabled = FROZEN_IMPULSE_GUARD

        # Pre-load exact frozen acoustic paths (float64)
        self.primary_path = teammate_create_primary_path()
        self.secondary_path = teammate_create_secondary_path()

        # State storage:
        # Note: The underlying teammate `run_fxlms` function is an offline batch procedure
        # that allocates local state per invocation. This adapter stores the resulting
        # final weights, outputs, and telemetry from the most recent execution.
        self.weights = np.zeros(self.filter_length, dtype=np.float32)
        self.last_weights_float64 = np.zeros(self.filter_length, dtype=np.float64)
        self.last_anti_noise: Optional[np.ndarray] = None
        self.last_residual_error: Optional[np.ndarray] = None
        self.last_error_power: Optional[np.ndarray] = None

        self.name = "teammate_frozen"
        self.semantic_label = "Offline Simulated-Reference Acoustic ANC"

    def reset(self) -> None:
        """
        Resets adapter state and clears cached weights and buffers.
        (Underlying teammate FxLMS is stateless per execution; reset ensures clean state).
        """
        self.weights.fill(0.0)
        self.last_weights_float64.fill(0.0)
        self.last_anti_noise = None
        self.last_residual_error = None
        self.last_error_power = None

    def get_weights(self) -> np.ndarray:
        """Returns a copy of current adaptive filter weights in float32."""
        return np.copy(self.weights)

    def simulate_disturbance(self, reference: np.ndarray) -> np.ndarray:
        """
        Convenience method: Passes reference through the teammate's exact 8-tap
        simulated primary path P(z) and normalizes RMS to match the reference RMS.
        Matches teammate's dataset_fxlms_single_test.py disturbance generation.
        """
        ref_64 = np.asarray(reference, dtype=np.float64)
        dist = lfilter(self.primary_path, [1.0], ref_64)

        ref_rms = float(np.sqrt(np.mean(ref_64 ** 2) + self.epsilon))
        dist_rms = float(np.sqrt(np.mean(dist ** 2) + self.epsilon))
        if dist_rms > self.epsilon:
            dist = dist * (ref_rms / dist_rms)

        return dist.astype(np.float32)

    def process_buffer(
        self,
        ref_signal: np.ndarray,
        desired_signal: np.ndarray,
        step_size_multipliers: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes the exact frozen FxLMS algorithm on input buffers.

        Parameters:
          ref_signal: 1D array of reference noise signal x(n) (float32).
          desired_signal: 1D array of disturbance signal d(n) (float32).
          step_size_multipliers: Ignored for the frozen baseline to maintain
                                 exact fidelity with teammate characterization.
                                 (Impulse safety is enforced via HybridController).

        Returns:
          anti_noise_y: Anti-noise signal generated by controller (float32).
          residual_e  : Residual error signal d(n) - y_s(n) (float32).
          filtered_ref: Filtered reference signal x'(n) through secondary path (float32).
        """
        # 1. Validation
        ref_arr = np.asarray(ref_signal)
        des_arr = np.asarray(desired_signal)

        if ref_arr.ndim != 1 or des_arr.ndim != 1:
            raise ValueError("Reference and desired signals must be 1D arrays.")
        if len(ref_arr) != len(des_arr):
            raise ValueError(
                f"Length mismatch: ref ({len(ref_arr)}) vs desired ({len(des_arr)})."
            )
        if len(ref_arr) == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty

        if not np.all(np.isfinite(ref_arr)) or not np.all(np.isfinite(des_arr)):
            raise ValueError("Inputs to FxLMS must contain only finite values (no NaN/Inf).")

        # 2. Precision conversion: float32 -> float64 for exact teammate math
        ref_64 = ref_arr.astype(np.float64)
        des_64 = des_arr.astype(np.float64)

        # 3. Filter reference through secondary path estimate (float64)
        # Matches teammate: filtered_reference_signal = lfilter(secondary_path, [1.0], reference)
        filtered_ref_64 = lfilter(self.secondary_path, [1.0], ref_64)

        # 4. Execute exact teammate frozen FxLMS
        anti_noise_64, error_64, weights_64, error_power_64 = teammate_run_fxlms(
            reference=ref_64,
            disturbance=des_64,
            secondary_path=self.secondary_path,
            filter_length=self.filter_length,
            mu=self.step_size_mu,
        )

        # 5. Store state & telemetry
        self.last_weights_float64 = np.copy(weights_64)
        self.weights = weights_64.astype(np.float32)
        self.last_anti_noise = anti_noise_64.astype(np.float32)
        self.last_residual_error = error_64.astype(np.float32)
        self.last_error_power = error_power_64.astype(np.float32)

        # 6. Return pipeline-compatible float32 arrays
        return (
            self.last_anti_noise,
            self.last_residual_error,
            filtered_ref_64.astype(np.float32),
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry from the most recent execution."""
        max_weight = float(np.max(np.abs(self.last_weights_float64)))
        weight_norm = float(np.linalg.norm(self.last_weights_float64))
        max_error_pwr = (
            float(np.max(self.last_error_power))
            if self.last_error_power is not None
            else 0.0
        )
        return {
            "name": self.name,
            "semantic_label": self.semantic_label,
            "filter_length": self.filter_length,
            "mu": self.step_size_mu,
            "epsilon": self.epsilon,
            "leakage": self.leakage,
            "max_abs_weight": max_weight,
            "weight_l2_norm": weight_norm,
            "max_error_power": max_error_pwr,
            "impulse_guard_inside_baseline": self.impulse_guard_enabled,
        }
