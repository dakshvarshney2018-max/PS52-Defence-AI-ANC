"""
PS52 - Defence Acoustic Intelligence Lab
Visualization-only Streamlit dashboard for the PS52 adaptive noise
cancellation system. Reads exclusively from results/final_evaluation/
and data/defence_scenarios/. Does not modify, rerun, or regenerate
any engineering artifact.

Launch from the repository root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DASHBOARD_DIR.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (
    inject_css, kpi_card, claim_card,
    PLOTLY_TEMPLATE, METHOD_COLORS, REGIME_COLORS,
    BG_PRIMARY, BORDER, ACCENT_CYAN, STATUS_GOOD, STATUS_WARN, STATUS_BAD,
)
from data_loader import get_repo_root, check_data_available, get_store
import audio_utils as au

try:
    st.set_page_config(
        page_title="PS52 - Defence Acoustic Intelligence Lab",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass
inject_css()

METHODS = ["raw", "wiener", "neural", "fxlms", "hybrid"]
METHOD_LABELS = {"raw": "Raw / Noisy", "wiener": "Wiener", "neural": "Neural",
                  "fxlms": "FxLMS", "hybrid": "Hybrid"}


# =========================================================================
# Sidebar: data source + navigation
# =========================================================================
def sidebar_data_source() -> Path:
    st.sidebar.markdown("**DATA SOURCE**")
    default_root = str(get_repo_root())
    root_input = st.sidebar.text_input(
        "Repository root", value=default_root,
        help="Folder that contains results/ and data/. Defaults to the "
             "parent of this dashboard/ folder.",
    )
    st.session_state["repo_root_override"] = root_input
    repo_root = Path(root_input)

    ok, missing = check_data_available(repo_root)
    if ok:
        st.sidebar.markdown(
            '<span class="badge good">● EVIDENCE FILES FOUND</span>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="badge warn">● EVIDENCE FILES MISSING</span>',
            unsafe_allow_html=True,
        )
        with st.sidebar.expander("Missing files"):
            for m in missing:
                st.code(m, language=None)
    return repo_root


def header():
    inject_css()
    st.markdown(
        """
        <div class="lab-header">
            <div class="lab-title">PS52 &bull; DEFENCE AI + ADAPTIVE NOISE CANCELLATION</div>
            <div class="lab-subtitle">Confidence-Weighted, Noise-Regime-Aware Hybrid ANC</div>
            <div class="lab-pipeline">SENSE &rarr; UNDERSTAND &rarr; CANCEL &rarr; MEASURE &rarr; PROVE</div>
            <div class="badge-row">
                <span class="badge good">&#9679; SYSTEM READY</span>
                <span class="badge cyan">&#9679; EVIDENCE FROZEN</span>
                <span class="badge warn">&#9679; ANC MODE: OFFLINE / SIMULATED REFERENCE</span>
                <span class="badge cyan">&#9679; 2,050 PESQ EVALUATIONS (WB ITU-T P.862.2)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# Section 1 -- Mission Control
# =========================================================================
def section_mission_control(store):
    st.markdown('<div class="section-tag">01 &mdash; Mission Control</div>', unsafe_allow_html=True)

    cs = store.candidate_summary
    ps = store.pesq_summary
    c300 = cs["overall_300_mixtures"]
    c110 = cs.get("overall_110_defence", cs.get("overall_110_defence_scenarios"))
    neural_rtf = store.neural_component_rtf

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**CONTROLLED BENCHMARK &mdash; 300 mixtures**")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("SNR Gain (Hybrid)", f"+{c300['mean_hybrid_gain_db']:.2f} dB",
                              f"input {c300['mean_input_snr_db']:.2f} dB"), unsafe_allow_html=True)
        c2.markdown(kpi_card("STOI (Hybrid)", f"{c300['mean_hybrid_stoi']:.4f}",
                              f"raw {c300['mean_raw_stoi']:.4f}"), unsafe_allow_html=True)
        c3.markdown(kpi_card("PESQ (Hybrid)", f"{c300['mean_hybrid_pesq']:.4f}",
                              "target &gt; 2.5"), unsafe_allow_html=True)
        c4.markdown(kpi_card("RTF (Hybrid)", f"{c300['mean_hybrid_rtf']:.4f}x",
                              "desktop CPU mean (target &lt; 0.10x)"), unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8b98ac;font-size:0.8rem;margin-top:6px;'>"
                    f"n = {c300['total_count']} controlled mixtures &bull; Neural Component RTF: <b>{neural_rtf:.4f}x</b></div>",
                    unsafe_allow_html=True)

    with colB:
        st.markdown("**DEFENCE SCENARIO BENCHMARK &mdash; 110 curated scenarios**")
        d1, d2, d3, d4 = st.columns(4)
        d1.markdown(kpi_card("SNR Gain (Hybrid)", f"+{c110['mean_hybrid_gain_db']:.2f} dB",
                              f"input {c110['mean_input_snr_db']:.2f} dB"), unsafe_allow_html=True)
        d2.markdown(kpi_card("STOI (Hybrid)", f"{c110['mean_hybrid_stoi']:.4f}",
                              f"raw {c110['mean_raw_stoi']:.4f}"), unsafe_allow_html=True)
        d3.markdown(kpi_card("PESQ (Hybrid)", f"{c110['mean_hybrid_pesq']:.4f}",
                              "target &gt; 2.5"), unsafe_allow_html=True)
        d4.markdown(kpi_card("RTF (Hybrid)", f"{c110['mean_hybrid_rtf']:.4f}x",
                              "defence mean, desktop CPU"), unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8b98ac;font-size:0.8rem;margin-top:6px;'>"
                    f"n = {c110['total_count']} defence scenarios &bull; Curated Threat Scenarios</div>",
                    unsafe_allow_html=True)

    with st.expander("Candidate vs Frozen Baseline Audited Improvements"):
        base_300 = store.final_summary["overall_300_mixtures"]
        base_110 = store.final_summary.get("overall_110_defence_scenarios", store.final_summary.get("overall_110_defence"))
        base_pesq_300 = ps["overall_300_mixtures"]["hybrid"]["mean"]
        base_pesq_110 = ps["overall_110_defence"]["hybrid"]["mean"]
        base_rtf_300 = store.final_summary["runtime_diagnostics"]["mean_rtf"]
        base_rtf_110 = store.df_defence_baseline["hybrid_rtf"].mean()

        comp_df = pd.DataFrame([
            {
                "Benchmark": "Controlled (300)",
                "Baseline ΔSNR": f"+{base_300['mean_hybrid_gain_db']:.2f} dB",
                "Candidate ΔSNR": f"+{c300['mean_hybrid_gain_db']:.2f} dB",
                "ΔSNR Gain": f"+{c300['mean_hybrid_gain_db'] - base_300['mean_hybrid_gain_db']:+.2f} dB",
                "Baseline STOI": f"{base_300['mean_hybrid_stoi']:.4f}",
                "Candidate STOI": f"{c300['mean_hybrid_stoi']:.4f}",
                "STOI Delta": f"{c300['mean_hybrid_stoi'] - base_300['mean_hybrid_stoi']:+.4f}",
                "Baseline PESQ": f"{base_pesq_300:.4f}",
                "Candidate PESQ": f"{c300['mean_hybrid_pesq']:.4f}",
                "PESQ Delta": f"{c300['mean_hybrid_pesq'] - base_pesq_300:+.4f}",
                "Baseline RTF": f"{base_rtf_300:.4f}x",
                "Candidate RTF": f"{c300['mean_hybrid_rtf']:.4f}x",
                "Speedup": "-5.1%",
            },
            {
                "Benchmark": "Defence (110)",
                "Baseline ΔSNR": f"+{base_110['mean_hybrid_gain_db']:.2f} dB",
                "Candidate ΔSNR": f"+{c110['mean_hybrid_gain_db']:.2f} dB",
                "ΔSNR Gain": f"+{c110['mean_hybrid_gain_db'] - base_110['mean_hybrid_gain_db']:+.2f} dB",
                "Baseline STOI": f"{base_110['mean_hybrid_stoi']:.4f}",
                "Candidate STOI": f"{c110['mean_hybrid_stoi']:.4f}",
                "STOI Delta": f"{c110['mean_hybrid_stoi'] - base_110['mean_hybrid_stoi']:+.4f}",
                "Baseline PESQ": f"{base_pesq_110:.4f}",
                "Candidate PESQ": f"{c110['mean_hybrid_pesq']:.4f}",
                "PESQ Delta": f"{c110['mean_hybrid_pesq'] - base_pesq_110:+.4f}",
                "Baseline RTF": f"{base_rtf_110:.4f}x",
                "Candidate RTF": f"{c110['mean_hybrid_rtf']:.4f}x",
                "Speedup": "-7.3%",
            },
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "This system runs **offline, simulated-reference acoustic ANC** on a "
        "**desktop CPU**, evaluated against curated demonstration-oriented "
        "tactical scenarios. It is not a physical headset ANC product, and "
        "the 110-scenario defence set is not a comprehensive battlefield "
        "dataset. See Architecture & Limitations for full claim boundaries.",
        icon="ℹ️",
    )


# =========================================================================
# Placeholder registrations for the remaining sections (implemented in
# sections_*.py and attached below via exec-free imports to keep this
# file readable). See scenario_lab.py, performance.py, impulse_safety.py,
# architecture.py.
# =========================================================================
from scenario_lab import section_scenario_lab
from performance import section_performance_evidence
from impulse_safety import section_impulse_safety
from architecture import section_architecture_limitations


def main():
    inject_css()
    repo_root = sidebar_data_source()
    ok, missing = check_data_available(repo_root)

    header()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**NAVIGATION**")
    section = st.sidebar.radio(
        "Section",
        [
            "1. Mission Control",
            "2. Scenario Lab",
            "3. Performance Evidence",
            "4. Impulse Safety",
            "5. Architecture & Limitations",
        ],
        label_visibility="collapsed",
    )

    if not ok:
        st.error(
            "Evaluation artifacts not found at the configured repository "
            "root. Set the correct **Repository root** in the sidebar "
            "(the folder containing `results/` and `data/`)."
        )
        with st.expander("Missing files"):
            for m in missing:
                st.code(m, language=None)
        st.stop()

    store = get_store(str(repo_root))

    if section.startswith("1"):
        section_mission_control(store)
    elif section.startswith("2"):
        section_scenario_lab(store)
    elif section.startswith("3"):
        section_performance_evidence(store)
    elif section.startswith("4"):
        section_impulse_safety(store)
    elif section.startswith("5"):
        section_architecture_limitations(store)


if __name__ == "__main__":
    main()
