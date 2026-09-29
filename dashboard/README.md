# PS52 — Defence Acoustic Intelligence Lab Dashboard

Visualization-only Streamlit dashboard for the PS52 adaptive noise cancellation system. It does not modify, rerun, or regenerate any engineering artifact — it reads dynamically from the authoritative candidate evaluation artifacts under `results/candidate_evaluation/`, frozen baseline evidence under `results/final_evaluation/`, and audio files under `data/processed/defence_scenarios/` and `results/defence_scenarios/audio_samples/`.

---

## 1. Launch Command

From the repository root (`C:\Users\kapil\.gemini\antigravity\scratch\PS52_DANC`):

```bash
python -m streamlit run dashboard/app.py
```

The sidebar provides a **Repository root** input that defaults to the repository root directory. The application validates that all required evidence files exist upon startup.

---

## 2. Dashboard Architecture & Sections

The dashboard is structured around five core views:

1. **Mission Control**: First-screen operational overview reporting controlled benchmark KPIs (300 mixtures), defence scenario KPIs (110 threats), target status pills (STOI > 0.85 MET, PESQ > 2.5 NOT MET, RTF < 0.10x NOT MET for complete pipeline), total PESQ evaluation count (2,050 evaluations), and audited candidate vs baseline improvements.
2. **Scenario Lab**: Interactive tactical scenario exploration across all 7 defence categories and 5 processing methods (`Raw`, `Wiener`, `Neural`, `FxLMS`, `Candidate Hybrid`). Features BEFORE vs AFTER waveform and spectrogram transformations, audio playback for genuine recorded noisy input, representative hybrid output, and clean speech references, acoustic regime intelligence, and branch weight telemetry.
3. **Performance Evidence**: Comprehensive multi-metric benchmark evaluations: 5-way SNR gain comparison, SNR gain vs input SNR, STOI vs input SNR (with 0.85 target line), PESQ vs input SNR (with 2.5 target line), method comparisons for Controlled (300) and Defence (110), 7-category threat breakdowns, noise-regime gains, and desktop CPU runtime distribution.
4. **Impulse Safety**: Dedicated ballistic shockwave protection analysis comparing unshielded FxLMS against ImpulseGuard-shielded Hybrid across all 10 impulse threat events (`artillery` and `simulated_gunshot`). Features the 5-stage timeline (`NORMAL` -> `IMPULSE DETECTED` -> `FROZEN_ADAPTATION` -> `ANC = 0` -> `SAFE RECOVERY`) and full supervisory disclosure.
5. **Architecture & Limitations**: Scientific signal flow diagram, continuous hybrid weighting equation, component specifications, and explicit claim boundaries.

---

## 3. Authoritative Evidence Sources

All metrics and comparisons are strictly data-driven from disk:

| Section | Authoritative Source Artifacts |
|---|---|
| **Mission Control** | `results/candidate_evaluation/candidate_summary.json`<br>`results/final_evaluation/pesq/pesq_summary.json`<br>`results/final_evaluation/final_summary.json` |
| **Scenario Lab** | `results/candidate_evaluation/candidate_defence_results.csv`<br>`results/defence_scenarios/audio_samples/`<br>`data/processed/defence_scenarios/` |
| **Performance Evidence** | `results/candidate_evaluation/candidate_summary.json`<br>`results/candidate_evaluation/candidate_300_results.csv`<br>`results/candidate_evaluation/candidate_defence_results.csv`<br>`results/final_evaluation/final_summary.json` |
| **Impulse Safety** | `results/candidate_evaluation/candidate_defence_results.csv`<br>`results/final_evaluation/final_defence_results.csv` (impulse event telemetry) |
| **Architecture & Limitations** | `results/candidate_evaluation/candidate_summary.json` (`tested_system`) + `CANDIDATE_EVALUATION_REPORT.md` |

---

## 4. Audio & DSP Evidence Protocol

- **Genuine Recordings**: Authentic noisy input WAVs and clean reference WAVs exist under `data/processed/defence_scenarios/`.
- **Representative Outputs**: Verified representative hybrid output WAVs for every threat category exist under `results/defence_scenarios/audio_samples/` (`artillery_hybrid.wav`, `drone_uav_like_hybrid.wav`, `helicopter_hybrid.wav`, `simulated_gunshot_hybrid.wav`, `siren_hybrid.wav`, `vehicle_engine_hybrid.wav`, `wind_hybrid.wav`).
- **No Signal Fabrication**: If individual per-case audio was not exported during benchmarking, a clean "Audio evidence unavailable" state is presented. No synthetic or reconstructed waveforms are generated.
- **Residual Noise Quantification**: Residual suppression is evidenced objectively through measured ΔSNR (+3.37 dB defence mean), STOI, and wideband PESQ metrics.

---

## 5. Runtime & Performance Profile

- **Isolated Neural Enhancement Component**: Operates at **0.0121x RTF** (a 3.18x speedup over the 0.0330x baseline), well within the 0.10x low-power embedded reference target.
- **End-to-End Hybrid Pipeline**: Operates at **0.1601x RTF** (controlled) and **0.1542x RTF** (defence), comfortably maintaining real-time execution (< 1.0x) on desktop CPU. The complete pipeline remains above the 0.10x target due to the sample-by-sample teammate FxLMS filter loop.

---

## 6. Scientific Claim Boundaries

1. **ANC**: Offline simulated-reference acoustic ANC — not a physical headset ANC product.
2. **Data**: 110 curated demonstration-oriented tactical scenarios — not a comprehensive battlefield dataset.
3. **UAV / Drone**: Generic / UAV-like synthetic tones — not a recorded military UAV.
4. **Gunshot**: Simulated gunshot (Friedlander shockwave synthesis) — not a live firearm recording.
5. **Artillery**: Historic Sexton 25-pdr field recording.
6. **Neural Model**: Lightweight GRU trained on the 300-mixture dataset for integration proof-of-concept — no clean held-out generalization claim.
7. **Runtime**: Complete Hybrid RTF is 0.1601x / 0.1542x (<1.0x real-time maintained; <0.10x target achieved by neural component alone).
8. **PESQ**: Dataset-wide >2.5 target was not achieved (Controlled: 1.7394, Defence: 1.6632); compliance appears at high input SNR (+20 dB reaches 2.5738) only.
9. **Supervisory Protection**: ImpulseGuard is a supervisory protection mechanism; the frozen teammate FxLMS algorithm itself is not modified.
10. **Method Ranking**: Hybrid is not universally the best method on every metric in every condition — see Performance Evidence for direct comparisons.
