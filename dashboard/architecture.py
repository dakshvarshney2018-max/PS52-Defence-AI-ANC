from __future__ import annotations

import streamlit as st

from theme import claim_card, BORDER, BG_PANEL, ACCENT_CYAN, TEXT_PRIMARY, TEXT_MUTED, FONT_MONO


def _flow_box(label: str, sub: str = "", min_width: str = "170px") -> str:
    sub_html = f'<div style="font-size:0.68rem;color:{TEXT_MUTED};margin-top:3px;letter-spacing:0.02em;">{sub}</div>' if sub else ""
    return (
        f'<div style="background:{BG_PANEL};border:1px solid {BORDER};border-radius:6px;'
        f'padding:9px 16px;text-align:center;font-family:{FONT_MONO};font-size:0.80rem;'
        f'font-weight:600;color:{TEXT_PRIMARY};min-width:{min_width};'
        f'box-shadow:0 2px 4px rgba(0,0,0,0.3);">'
        f'{label}{sub_html}</div>'
    )


def _arrow_down() -> str:
    return (
        f'<div style="text-align:center;color:{ACCENT_CYAN};font-size:1.15rem;'
        f'line-height:1.2;margin:2px 0;user-select:none;">&#8595;</div>'
    )


def _architecture_diagram() -> str:
    branches = (
        f'<div style="display:flex;gap:8px;justify-content:center;width:100%;flex-wrap:wrap;">'
        f'{_flow_box("WIENER", "spectral mask", min_width="105px")}'
        f'{_flow_box("NEURAL", "vectorized GRU", min_width="105px")}'
        f'{_flow_box("FxLMS", "adaptive ANC", min_width="105px")}'
        f'{_flow_box("PASS-THROUGH", "safe headroom", min_width="105px")}'
        f'</div>'
    )
    parts = [
        '<div style="display:flex;flex-direction:column;align-items:center;width:100%;max-width:520px;margin:0 auto;">',
        _flow_box("REFERENCE MIC / AUDIO INPUT"),
        _arrow_down(),
        _flow_box("ACOUSTIC FEATURE EXTRACTION", "12 acoustic features"),
        _arrow_down(),
        _flow_box("REGIME & CONFIDENCE ESTIMATOR", "rule-based softmax"),
        _arrow_down(),
        branches,
        _arrow_down(),
        _flow_box("CONFIDENCE & SAFETY CONTROLLER", "continuous dynamic weighting"),
        _arrow_down(),
        _flow_box("IMPULSEGUARD SUPERVISORY LAYER", "transient detection + ANC freeze"),
        _arrow_down(),
        _flow_box("WEIGHTED HYBRID OUTPUT", "sum of active branch signals"),
        _arrow_down(),
        _flow_box("MEASURE / PROVE", "SNR &bull; STOI &bull; PESQ &bull; RTF"),
        '</div>'
    ]
    return "".join(parts)


def section_architecture_limitations(store):
    st.markdown('<div class="section-tag">05 &mdash; Architecture & Limitations</div>', unsafe_allow_html=True)

    col_diag, col_eq = st.columns([1.3, 1])
    with col_diag:
        st.markdown("**SIGNAL PATH**")
        st.markdown(_architecture_diagram(), unsafe_allow_html=True)
    with col_eq:
        st.markdown("**HYBRID BLEND**")
        st.latex(r"y_{hybrid}[n] = w_w y_{wiener}[n] + w_n y_{neural}[n] + w_a y_{anc}[n] + w_p y_{noisy}[n]")
        st.caption("Weights are estimated per-frame by the regime/confidence "
                   "estimator and always sum to 1.0.")
        ts = store.candidate_summary.get("tested_system", store.final_summary["tested_system"])
        st.markdown("**COMPONENT SUMMARY**")
        st.markdown(
            f"- **System version**: `{ts.get('version', 'candidate_optimized_v1')}`\n"
            f"- **Regime estimator**: 12 acoustic features + rule-based softmax\n"
            f"- **Wiener branch**: Decision-directed spectral mask\n"
            f"- **Neural branch**: {ts.get('neural_enhancer', 'Vectorized Lightweight GRU (263k params)')}\n"
            f"- **FxLMS filter**: L=64, μ=0.01, leakage=0.0 (frozen teammate engine)\n"
            f"- **ImpulseGuard**: Active in Hybrid (clamps w_anc to 0.000 on ballistic transients)"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**CLAIM BOUNDARIES**")
    st.caption("These boundaries are load-bearing for how this system's results should be read.")

    cand_300 = store.candidate_summary["overall_300_mixtures"]
    cand_110 = store.candidate_summary.get("overall_110_defence", store.candidate_summary.get("overall_110_defence_scenarios"))

    cards = [
        ("ANC", "Offline Simulated-Reference Acoustic ANC — not physical headset ANC."),
        ("DATA", "110 curated demonstration-oriented tactical scenarios — not a comprehensive battlefield dataset."),
        ("UAV / DRONE", "Generic / UAV-like synthetic tones — not a recorded military UAV."),
        ("GUNSHOT", "Simulated gunshot (Friedlander shockwave synthesis) — not a live firearm recording."),
        ("ARTILLERY", "Historic Sexton 25-pdr field recording."),
        ("NEURAL MODEL", "Lightweight GRU trained on the 300-mixture dataset for integration "
                          "proof-of-concept — no clean held-out generalization claim."),
        ("RUNTIME", f"Desktop CPU. Isolated neural component operates at {store.neural_component_rtf:.4f}x RTF "
                    f"(well within the 0.10x budget). End-to-end Hybrid achieves {cand_300['mean_hybrid_rtf']:.4f}x (controlled) "
                    f"and {cand_110['mean_hybrid_rtf']:.4f}x (defence), maintaining real-time (< 1.0x) while remaining above the 0.10x target."),
        ("PESQ", f"Dataset-wide >2.5 target not achieved (Controlled: {cand_300['mean_hybrid_pesq']:.4f}, "
                 f"Defence: {cand_110['mean_hybrid_pesq']:.4f}); compliance appears at high input SNR only."),
        ("METHOD RANKING", "Hybrid is not universally the best method on every metric in every "
                            "condition — see Performance Evidence for direct comparisons."),
    ]
    cols = st.columns(3)
    for i, (k, v) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(claim_card(k, v), unsafe_allow_html=True)

