"""
Audio loading + Plotly figure builders for the Scenario Lab.

Only reads real WAV files already on disk (noisy input, clean
reference). Never synthesizes or fabricates a processed/output
signal -- see the in-app note in the Scenario Lab about why
per-method processed audio is not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram

from theme import PLOTLY_TEMPLATE, ACCENT_CYAN, METHOD_COLORS, BG_PRIMARY, BORDER


@st.cache_data(show_spinner=False)
def read_wav(path_str: str):
    """Returns (sample_rate, float32 samples in [-1, 1]) or None if missing."""
    path = Path(path_str)
    if not path.exists():
        return None
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


def _base_layout(fig: go.Figure, title: str, y_title: str, height: int = 260):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=13, family="IBM Plex Mono")),
        height=height,
        margin=dict(l=50, r=20, t=40, b=35),
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        xaxis=dict(title="Time (seconds)", gridcolor=BORDER),
        yaxis=dict(title=y_title, gridcolor=BORDER),
        legend=dict(orientation="h", y=1.15, font=dict(size=10)),
    )
    return fig


def waveform_figure(sr_a, samples_a, label_a, sr_b=None, samples_b=None, label_b=None,
                     title="Waveform", color_a=ACCENT_CYAN, color_b="#5b6478"):
    fig = go.Figure()
    if samples_a is not None:
        t_a = np.arange(len(samples_a)) / sr_a
        fig.add_trace(go.Scatter(x=t_a, y=samples_a, mode="lines", name=label_a,
                                  line=dict(color=color_a, width=1)))
    if samples_b is not None:
        t_b = np.arange(len(samples_b)) / sr_b
        fig.add_trace(go.Scatter(x=t_b, y=samples_b, mode="lines", name=label_b,
                                  line=dict(color=color_b, width=1)))
    return _base_layout(fig, title, "Amplitude")


def spectrogram_figure(sr, samples, title="Spectrogram"):
    fig = go.Figure()
    if samples is None or len(samples) < 256:
        fig.add_annotation(text="Audio not available", showarrow=False,
                            font=dict(color="#8b98ac"))
        return _base_layout(fig, title, "Frequency (Hz)")
    f, t, Sxx = scipy_spectrogram(samples, fs=sr, nperseg=256, noverlap=192)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    fig.add_trace(go.Heatmap(
        x=t, y=f, z=Sxx_db, colorscale="Viridis",
        colorbar=dict(title="dB", tickfont=dict(size=9)),
    ))
    return _base_layout(fig, title, "Frequency (Hz)")
