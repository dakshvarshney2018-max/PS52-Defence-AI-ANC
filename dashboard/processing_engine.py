"""
dashboard/processing_engine.py — Genuine Multi-Method Execution & Audio Caching Engine
========================================================================================

Executes the actual selected DSP / AI / ANC / Hybrid processing methods on genuine
noisy scenario recordings and clean speech references:
  - Raw: Original noisy input
  - Wiener: Classical Wiener spectral enhancer (src/ai_enhancer.py)
  - Neural: Lightweight 2-layer GRU speech enhancer (src/neural_enhancer.py)
  - FxLMS: Normalized FxLMS adaptive filter (src/teammate_fxlms_adapter.py)
  - Hybrid: Confidence-weighted hybrid controller (src/hybrid_controller.py)

Features:
  1. Real-time deterministic execution — never synthesizes or fabricates audio.
  2. Multi-tier caching:
     - Tier 1: Persistent disk cache at results/scenario_lab_cache/<method>/<scenario_id>_<method>.wav
     - Tier 2: Streamlit @st.cache_data memory caching for zero-latency UI re-renders.
  3. Live objective metric computation (SNR, ΔSNR, STOI, PESQ, RTF) strictly derived from
     the actual audio signals.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from pystoi import stoi
from pesq import pesq, PesqError

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ai_enhancer import SpectralEnhancer
from src.neural_enhancer import NeuralSpeechEnhancer
from src.teammate_fxlms_adapter import TeammateFxLMSAdapter
from src.hybrid_controller import HybridController


TARGET_SAMPLE_RATE = 16000

METHODS = ["raw", "wiener", "neural", "fxlms", "hybrid"]
METHOD_LABELS = {
    "raw": "Raw / Noisy",
    "wiener": "Wiener",
    "neural": "Neural",
    "fxlms": "FxLMS",
    "hybrid": "Hybrid",
}


def get_cache_dir(repo_root: Union[str, Path]) -> Path:
    p = Path(repo_root) / "results" / "scenario_lab_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_audio_snr(clean: np.ndarray, processed: np.ndarray) -> float:
    min_len = min(len(clean), len(processed))
    if min_len == 0:
        return 0.0
    c = clean[:min_len].astype(np.float64)
    p = processed[:min_len].astype(np.float64)
    err = p - c
    p_clean = float(np.mean(c ** 2))
    p_err = float(np.mean(err ** 2))
    if p_err < 1e-12:
        return 50.0
    if p_clean < 1e-12:
        return -50.0
    return float(10.0 * np.log10(p_clean / p_err))


def safe_stoi(clean: np.ndarray, processed: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> float:
    try:
        min_len = min(len(clean), len(processed))
        if min_len < 512:
            return 0.0
        score = float(
            stoi(
                clean[:min_len].astype(np.float64),
                processed[:min_len].astype(np.float64),
                sr,
                extended=False,
            )
        )
        return float(score) if np.isfinite(score) else 0.0
    except Exception:
        return 0.0


def safe_pesq(clean: np.ndarray, degraded: np.ndarray, sr: int = TARGET_SAMPLE_RATE, mode: str = "wb") -> Tuple[Optional[float], str]:
    try:
        min_len = min(len(clean), len(degraded))
        if min_len < 4000:
            return None, "AUDIO_TOO_SHORT_LT_0.25S"
        if not np.all(np.isfinite(clean[:min_len])) or not np.all(np.isfinite(degraded[:min_len])):
            return None, "NON_FINITE_VALUES"
        p_clean = float(np.mean(clean[:min_len] ** 2))
        p_deg = float(np.mean(degraded[:min_len] ** 2))
        if p_clean < 1e-12 or p_deg < 1e-12:
            return None, "SILENCE_OR_ZERO_ENERGY"
        c_sig = np.clip(clean[:min_len].astype(np.float32), -1.0, 1.0)
        d_sig = np.clip(degraded[:min_len].astype(np.float32), -1.0, 1.0)
        score = float(pesq(sr, c_sig, d_sig, mode))
        return round(score, 4), "SUCCESS"
    except PesqError as pe:
        return None, f"PESQ_ERROR_{pe}"
    except Exception as exc:
        return None, f"EXCEPTION_{type(exc).__name__}: {exc}"


def run_method_pipeline(
    method: str,
    noisy_audio: np.ndarray,
    clean_audio: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Executes the genuine processing method on input audio.
    Returns: (processed_audio, exec_time_ms, telemetry_dict).
    """
    sig = np.asarray(noisy_audio, dtype=np.float32)
    min_len = len(sig)
    if clean_audio is not None:
        min_len = min(min_len, len(clean_audio))
        sig = sig[:min_len]
        clean_ref = np.asarray(clean_audio[:min_len], dtype=np.float32)
        noise_ref = (sig - clean_ref).astype(np.float32)
    else:
        noise_ref = None

    method = method.lower().strip()

    if method == "raw":
        return sig, 0.0, {"method": "raw", "status": "ORIGINAL_NOISY"}

    elif method == "wiener":
        enh = SpectralEnhancer(sample_rate=sample_rate)
        t0 = time.perf_counter()
        out, conf, diag = enh.enhance(sig, return_diagnostics=True)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return out[:min_len], t_ms, {"speech_confidence": float(conf), "mean_gain": float(diag.get("mean_gain", 0.0))}

    elif method == "neural":
        enh = NeuralSpeechEnhancer(sample_rate=sample_rate)
        t0 = time.perf_counter()
        out, conf, diag = enh.enhance(sig, return_diagnostics=True)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return out[:min_len], t_ms, {"speech_confidence": float(conf), "mean_gain": float(diag.get("mean_gain", 0.0))}

    elif method == "fxlms":
        if noise_ref is None:
            # Standalone fallback: no reference microphone
            return sig, 0.0, {"warning": "NO_REFERENCE_MIC", "anc_active": False}
        adapter = TeammateFxLMSAdapter(sample_rate=sample_rate)
        t0 = time.perf_counter()
        _, out, _ = adapter.process_buffer(ref_signal=noise_ref, desired_signal=sig)
        t_ms = (time.perf_counter() - t0) * 1000.0
        diag = adapter.get_diagnostics()
        diag["anc_active"] = True
        return out[:min_len], t_ms, diag

    elif method == "hybrid":
        ctrl = HybridController(sample_rate=sample_rate, anc_engine="teammate_frozen")
        t0 = time.perf_counter()
        out, diag = ctrl.process_utterance(sig, noise_reference=noise_ref, return_diagnostics=True)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return out[:min_len], t_ms, diag

    else:
        raise ValueError(f"Unknown processing method: '{method}'. Expected one of: raw, wiener, neural, fxlms, hybrid.")


def get_or_process_scenario_audio(
    repo_root: Union[str, Path],
    scenario_id: str,
    category: str,
    method: str,
    noisy_path: Union[str, Path],
    clean_path: Optional[Union[str, Path]] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Loads or dynamically processes audio for a specific scenario and method.
    Uses persistent disk caching at results/scenario_lab_cache/<method>/<scenario_id>_<method>.wav.
    """
    repo_root = Path(repo_root)
    cache_dir = get_cache_dir(repo_root) / method
    cache_dir.mkdir(parents=True, exist_ok=True)

    wav_cache_path = cache_dir / f"{scenario_id}_{method}.wav"
    meta_cache_path = cache_dir / f"{scenario_id}_{method}_meta.json"

    # Tier 1 Check: Disk Cache Hit
    if not force_recompute and wav_cache_path.exists() and meta_cache_path.exists():
        try:
            audio, sr = sf.read(str(wav_cache_path))
            with open(meta_cache_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "audio": audio.astype(np.float32),
                "sr": sr,
                "wav_path": wav_cache_path,
                "metrics": meta.get("metrics", {}),
                "telemetry": meta.get("telemetry", {}),
                "cached": True,
            }
        except Exception:
            # If reading cache fails, proceed to recompute
            pass

    # Cache Miss: Load audio and execute pipeline
    noisy_file = Path(noisy_path)
    if not noisy_file.exists():
        raise FileNotFoundError(f"Noisy input file not found: {noisy_file}")
    noisy_audio, sr = sf.read(str(noisy_file))
    if noisy_audio.ndim > 1:
        noisy_audio = np.mean(noisy_audio, axis=-1)
    noisy_audio = noisy_audio.astype(np.float32)

    clean_audio = None
    if clean_path is not None:
        clean_file = Path(clean_path)
        if clean_file.exists():
            clean_audio, _ = sf.read(str(clean_file))
            if clean_audio.ndim > 1:
                clean_audio = np.mean(clean_audio, axis=-1)
            clean_audio = clean_audio.astype(np.float32)

    min_len = len(noisy_audio)
    if clean_audio is not None:
        min_len = min(min_len, len(clean_audio))
        noisy_audio = noisy_audio[:min_len]
        clean_audio = clean_audio[:min_len]

    # Execute actual method
    processed_audio, exec_time_ms, telemetry = run_method_pipeline(
        method=method,
        noisy_audio=noisy_audio,
        clean_audio=clean_audio,
        sample_rate=sr,
    )
    processed_audio = np.clip(processed_audio[:min_len], -1.0, 1.0).astype(np.float32)

    # Compute objective metrics directly from the genuine audio arrays
    duration_sec = min_len / float(sr)
    rtf = (exec_time_ms / 1000.0) / max(duration_sec, 1e-6)

    metrics = {
        "scenario_id": scenario_id,
        "method": method,
        "exec_time_ms": round(exec_time_ms, 2),
        "rtf": round(rtf, 6),
    }

    if clean_audio is not None:
        in_snr = compute_audio_snr(clean_audio, noisy_audio)
        out_snr = compute_audio_snr(clean_audio, processed_audio)
        gain_db = out_snr - in_snr
        stoi_score = safe_stoi(clean_audio, processed_audio, sr=sr)
        pesq_score, _ = safe_pesq(clean_audio, processed_audio, sr=sr)

        metrics.update({
            "input_snr_db": round(in_snr, 4),
            "output_snr_db": round(out_snr, 4),
            "gain_db": round(gain_db, 4),
            "stoi": round(stoi_score, 4),
            "pesq": round(pesq_score, 4) if pesq_score is not None else None,
        })
    else:
        metrics.update({
            "input_snr_db": None,
            "output_snr_db": None,
            "gain_db": None,
            "stoi": None,
            "pesq": None,
        })

    # Save to disk cache
    try:
        sf.write(str(wav_cache_path), processed_audio, sr)
        serializable_telemetry = {}
        for k, v in telemetry.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                serializable_telemetry[k] = v
            elif isinstance(v, np.ndarray):
                serializable_telemetry[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serializable_telemetry[k] = float(v)

        with open(meta_cache_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "telemetry": serializable_telemetry}, f, indent=2)
    except Exception as e:
        # Cache write error should not prevent returning audio
        pass

    return {
        "audio": processed_audio,
        "sr": sr,
        "wav_path": wav_cache_path,
        "metrics": metrics,
        "telemetry": telemetry,
        "cached": False,
    }
