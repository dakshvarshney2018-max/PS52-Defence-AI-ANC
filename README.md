# PS52: Defence AI + Adaptive Noise Cancellation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ps52-defence-ai-anc.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

> **Smart India Hackathon (SIH) — Problem Statement 52 (PS52)**
> **Confidence-Weighted, Noise-Regime-Aware Hybrid ANC for Defence Speech Communication**

---

## 🎯 Project Overview

This repository hosts the final lightweight Streamlit deployment for **PS52 — Defence AI + Adaptive Noise Cancellation**. 

The system implements a deterministic, regime-aware hybrid controller combining:
1. **Classical Wiener Spectral Enhancer** (Stationary diffuse hiss suppression)
2. **Lightweight 2-Layer GRU Neural Masker** (263k parameters, RTF = 0.0121x on CPU)
3. **Normalized Filtered-X Least Mean Squares (FxLMS)** (Acoustic anti-noise cancellation)
4. **ImpulseGuard Supervisory Layer** (Instantaneous protection against gunshots and artillery shockwaves)

```
SENSE  ──>  UNDERSTAND  ──>  CANCEL  ──>  MEASURE  ──>  PROVE
```

---

## 🚀 Live Streamlit Cloud Deployment

This repository is optimized for one-click deployment on **Streamlit Community Cloud**:
1. Fork or clone this repository: `https://github.com/dakshvarshney2018-max/PS52-Defence-AI-ANC`
2. Connect to [share.streamlit.io](https://share.streamlit.io).
3. Set **Main file path** to `dashboard/app.py` (or `app.py`).
4. Launch!

---

## 💻 Local Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### Installation
```bash
git clone https://github.com/dakshvarshney2018-max/PS52-Defence-AI-ANC.git
cd PS52-Defence-AI-ANC
pip install -r requirements.txt
```

### Launch Dashboard
```bash
python -m streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Repository Structure

```text
PS52-Defence-AI-ANC/
├── .streamlit/
│   └── config.toml                  # Dark theme & headless server configuration
├── configs/
│   ├── default_config.yaml          # Pipeline hyperparameters
│   └── defence_scenarios.yaml       # Curated defence threat scenarios
├── dashboard/
│   ├── app.py                       # Mission Control executive view
│   ├── architecture.py              # System topology & hardware roadmap
│   ├── audio_utils.py               # Waveform & spectrogram figure renderers
│   ├── data_loader.py               # Evidence and scenario loader
│   ├── impulse_safety.py            # ImpulseGuard transient protection dashboard
│   ├── performance.py               # Benchmark metric visualizations
│   ├── processing_engine.py         # Multi-method DSP & live audio processing engine
│   ├── scenario_lab.py              # 5-method interactive audio listening lab
│   └── theme.py                     # Custom dark theme and CSS components
├── data/
│   └── processed/
│       └── defence_scenarios/       # Curated recordings (noisy + clean references)
│           ├── artillery/
│           ├── drone_uav_like/
│           ├── helicopter/
│           ├── simulated_gunshot/
│           ├── siren/
│           ├── vehicle_engine/
│           ├── wind/
│           └── clean_references/
├── models/
│   ├── neural_enhancer_v1.npz       # ONNX / NumPy weight checkpoint
│   └── neural_enhancer_weights.npz  # Dual-engine recurrent GRU weights
├── PS52_FxLMS_AI_HANDOFF/
│   └── PS52_FxLMS_AI_HANDOFF/
│       └── 01_FxLMS_SOURCE/
│           └── dataset_fxlms_single_test.py  # Frozen teammate FxLMS source
├── results/
│   ├── candidate_evaluation/        # Audited candidate acceptance evidence
│   ├── final_evaluation/            # Authoritative frozen baseline evidence
│   ├── defence_scenarios/           # Representative audio samples
│   └── scenario_lab_cache/          # Pre-cached representative demo outputs
├── src/
│   ├── ai_enhancer.py               # Classical Wiener spectral enhancer
│   ├── audio_io.py                  # 16 kHz mono framing & overlap-add
│   ├── features.py                  # 12 short-time acoustic features
│   ├── fxlms.py                     # Normalized FxLMS adaptive filter
│   ├── hybrid_controller.py         # Phase 2 regime-aware hybrid controller
│   ├── impulse_guard.py             # Transient detection & adaptation freeze
│   ├── metrics.py                   # SNR, STOI, PESQ, RTF evaluation functions
│   ├── neural_enhancer.py           # 2-layer GRU neural enhancer
│   ├── regime_estimator.py          # Evidence-based noise regime classifier
│   └── teammate_fxlms_adapter.py    # Bridge to frozen teammate FxLMS
├── .gitignore                       # Standard Python ignore rules
├── app.py                           # Root deployment entrypoint
├── packages.txt                     # Debian dependencies (libsndfile1, ffmpeg)
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

---

## 📊 Audited Evidence & Performance

| Benchmark | ΔSNR Gain | STOI Intelligibility | PESQ Quality | Pipeline RTF |
| :--- | :---: | :---: | :---: | :---: |
| **Controlled Benchmark (300)** | **+8.45 dB** | **0.9575** | **1.7161** | **0.1554x** |
| **Curated Defence Threats (110)** | **+3.37 dB** | **0.8870** | **1.6632** | **0.1542x** |
| **Isolated Neural Enhancer** | — | — | — | **0.0121x** *(MET)* |

- **Real-Time Factor (RTF)**: < 0.16x desktop CPU end-to-end; neural component 0.0121x (meets < 0.10x embedded target).
- **Transient Safety**: Instantaneous adaptation freeze on shockwaves prevents destructive filter divergence.

---

## 📜 License
Developed for Smart India Hackathon (SIH) 2024.
