"""
PS52 DATASET-DRIVEN FxLMS - SINGLE MIXTURE TEST

Purpose
-------
Run a scientifically controlled FxLMS experiment using one actual
PS52 dataset entry.

This version:
1. Reads mixture_metadata.csv
2. Finds the corresponding clean speech and noise WAV files
3. Reconstructs the requested SNR
4. Treats the isolated noise as the simulated reference signal
5. Passes the noise through a simulated primary acoustic path
6. Runs normalized FxLMS
7. Uses a simulated secondary acoustic path
8. Produces residual noise
9. Adds residual noise to clean speech
10. Saves metrics and diagnostic plots

IMPORTANT
---------
The primary and secondary acoustic paths are SIMULATED.
They are NOT measured headset/microphone/speaker responses.

This is an offline software experiment, not a physical headset test.
"""

import os
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

# Project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "PS52_DATA_FOR_TEAM")

METADATA_FILE = os.path.join(
    DATA_DIR,
    "metadata",
    "mixture_metadata.csv"
)

SPEECH_DIR = os.path.join(
    DATA_DIR,
    "speech",
    "speech"
)

NOISE_DIR = os.path.join(
    DATA_DIR,
    "noise",
    "noise"
)

RESULTS_DIR = os.path.join(
    SCRIPT_DIR,
    "results_dataset_fxlms"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TARGET_MIXTURE_ID = "train_00000"

FILTER_LENGTH = 64

# Normalized FxLMS step size.
# We deliberately use a conservative value for stability.
MU = 0.01

# Small value preventing division by zero.
EPSILON = 1e-8

# Plot only this many seconds of audio.
PLOT_SECONDS = 3.0


# ============================================================
# AUDIO UTILITIES
# ============================================================

def load_wav(path):
    """
    Load a WAV file and convert it to float32 in approximately
    the range [-1, 1].
    """

    sample_rate, audio = wavfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        max_value = max(abs(info.min), info.max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    return sample_rate, audio


def rms(signal):
    """Root-mean-square amplitude."""

    return float(np.sqrt(np.mean(signal ** 2) + EPSILON))


def power(signal):
    """Mean-square signal power."""

    return float(np.mean(signal ** 2))


def snr_db(signal, noise):
    """
    Calculate SNR:

        SNR = 10 log10(P_signal / P_noise)
    """

    p_signal = power(signal)
    p_noise = power(noise)

    return 10.0 * np.log10(
        (p_signal + EPSILON) /
        (p_noise + EPSILON)
    )


def normalize_rms(signal, target_rms):
    """
    Scale signal so that its RMS becomes target_rms.
    """

    current = rms(signal)

    if current < EPSILON:
        return signal.copy()

    return signal * (target_rms / current)


# ============================================================
# METADATA
# ============================================================

def load_metadata_entry(mixture_id):
    """
    Find one mixture entry inside mixture_metadata.csv.
    """

    with open(METADATA_FILE, "r", newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["mixture_id"] == mixture_id:
                return row

    raise ValueError(
        f"Could not find mixture ID: {mixture_id}"
    )


# ============================================================
# PATH RECONSTRUCTION
# ============================================================

def find_speech_file(metadata_path):
    """
    The metadata contains Windows paths from the teammate's machine.

    We therefore ignore the old absolute path and use only the
    filename to locate the corresponding WAV inside our local
    PS52_DATA_FOR_TEAM directory.
    """

    filename = os.path.basename(
        metadata_path.replace("\\", "/")
    )

    local_path = os.path.join(
        SPEECH_DIR,
        filename
    )

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Speech file not found:\n{local_path}"
        )

    return local_path


def find_noise_file(metadata_path, noise_type):
    """
    Reconstruct local noise path from the filename and noise regime.
    """

    filename = os.path.basename(
        metadata_path.replace("\\", "/")
    )

    local_path = os.path.join(
        NOISE_DIR,
        noise_type,
        filename
    )

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Noise file not found:\n{local_path}"
        )

    return local_path


# ============================================================
# SIGNAL LENGTH HANDLING
# ============================================================

def repeat_to_length(signal, target_length):
    """
    Repeat a signal cyclically until target_length is reached.
    """

    if len(signal) == 0:
        raise ValueError("Cannot repeat an empty signal.")

    repetitions = int(
        np.ceil(target_length / len(signal))
    )

    extended = np.tile(signal, repetitions)

    return extended[:target_length]


# ============================================================
# SNR CONTROL
# ============================================================

def scale_noise_to_snr(clean, noise, target_snr_db):
    """
    Scale noise to achieve the requested SNR relative to clean speech.

        SNR_dB = 10 log10(P_clean / P_noise)

    Therefore:

        P_noise_target =
            P_clean / 10^(SNR/10)
    """

    clean_power = power(clean)
    noise_power = power(noise)

    if clean_power < EPSILON:
        raise ValueError("Clean speech power is too small.")

    if noise_power < EPSILON:
        raise ValueError("Noise power is too small.")

    target_ratio = 10.0 ** (
        target_snr_db / 10.0
    )

    target_noise_power = (
        clean_power / target_ratio
    )

    scale = np.sqrt(
        target_noise_power /
        noise_power
    )

    scaled_noise = noise * scale

    return scaled_noise


# ============================================================
# SIMULATED ACOUSTIC PATHS
# ============================================================

def create_primary_path():
    """
    Simulated primary acoustic path.

    Represents:

        noise source -> listener's ear

    This is NOT a measured physical headset response.

    The coefficients introduce delay, attenuation and
    mild acoustic coloration.
    """

    path = np.array(
        [
            0.70,
            0.25,
            -0.10,
            0.06,
            0.03,
            -0.02,
            0.015,
            0.01
        ],
        dtype=np.float64
    )

    # Normalize energy so that the path does not arbitrarily
    # amplify or attenuate the entire experiment.
    path /= np.sqrt(np.sum(path ** 2))

    return path


def create_secondary_path():
    """
    Simulated secondary acoustic path.

    Represents:

        controller -> speaker -> air -> error microphone

    Again, this is a mathematical model, NOT a measured response.
    """

    path = np.array(
        [
            0.80,
            0.18,
            -0.08,
            0.04,
            0.025,
            -0.015
        ],
        dtype=np.float64
    )

    path /= np.sqrt(np.sum(path ** 2))

    return path


# ============================================================
# NORMALIZED FxLMS
# ============================================================

def run_fxlms(
    reference,
    disturbance,
    secondary_path,
    filter_length=64,
    mu=0.01
):
    """
    Normalized Filtered-x LMS.

    reference:
        x(n)

    disturbance:
        d(n)

    secondary_path:
        S(z)

    Returns:
        anti_noise
        residual/error
        weights
        error_power
    """

    n_samples = len(reference)

    # Adaptive controller weights.
    weights = np.zeros(
        filter_length,
        dtype=np.float64
    )

    # Reference history.
    x_buffer = np.zeros(
        filter_length,
        dtype=np.float64
    )

    # Filtered reference history.
    x_filtered = np.zeros(
        filter_length,
        dtype=np.float64
    )

    # Controller output.
    anti_noise = np.zeros(
        n_samples,
        dtype=np.float64
    )

    # Residual error.
    error = np.zeros(
        n_samples,
        dtype=np.float64
    )

    # Power tracking.
    error_power = np.zeros(
        n_samples,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Filter reference through secondary-path estimate.
    # --------------------------------------------------------

    filtered_reference_signal = lfilter(
        secondary_path,
        [1.0],
        reference
    )

    # --------------------------------------------------------
    # Main FxLMS loop
    # --------------------------------------------------------

    for n in range(n_samples):

        # Shift reference history.
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = reference[n]

        # Shift filtered-reference history.
        x_filtered[1:] = x_filtered[:-1]
        x_filtered[0] = filtered_reference_signal[n]

        # Controller output:
        #
        # y(n) = w^T(n) x(n)

        anti_noise[n] = np.dot(
            weights,
            x_buffer
        )

        # The anti-noise travels through the secondary path.
        #
        # For a causal sample-by-sample simulation, calculate
        # the current contribution of previous controller outputs.

        start = max(
            0,
            n - len(secondary_path) + 1
        )

        secondary_segment = anti_noise[
            start:n + 1
        ]

        secondary_coefficients = secondary_path[
            :len(secondary_segment)
        ][::-1]

        anti_noise_at_error_mic = np.dot(
            secondary_segment,
            secondary_coefficients
        )

        # ----------------------------------------------------
        # Residual
        #
        # e(n) = d(n) - y_s(n)
        # ----------------------------------------------------

        error[n] = (
            disturbance[n]
            - anti_noise_at_error_mic
        )

        # ----------------------------------------------------
        # Normalized FxLMS update
        #
        # w(n+1) =
        # w(n) +
        # mu * e(n) * x_filtered(n)
        # ----------------------------------------------
        #
        # Normalization prevents very large updates when
        # the reference signal has high instantaneous energy.
        # ----------------------------------------------------

        norm = (
            EPSILON +
            np.dot(
                x_filtered,
                x_filtered
            )
        )

        step = (
            mu *
            error[n] /
            norm
        )

        weights += (
            step *
            x_filtered
        )

        error_power[n] = error[n] ** 2

    return (
        anti_noise,
        error,
        weights,
        error_power
    )


# ============================================================
# SAVE AUDIO
# ============================================================

def save_wav(path, signal, sample_rate):
    """
    Save float signal as 32-bit WAV.
    """

    signal = signal.astype(np.float32)

    # Prevent numerical overflow.
    peak = np.max(np.abs(signal))

    if peak > 0.999:
        signal = signal / peak * 0.999

    wavfile.write(
        path,
        sample_rate,
        signal
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 60)
    print("PS52 DATASET-DRIVEN FxLMS - SINGLE MIXTURE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Read metadata
    # --------------------------------------------------------

    metadata = load_metadata_entry(
        TARGET_MIXTURE_ID
    )

    speech_metadata_path = metadata["speech_file"]
    noise_metadata_path = metadata["noise_file"]

    noise_type = metadata["noise_type"]

    target_snr = float(
        metadata["target_snr_db"]
    )

    print()
    print("Mixture ID:")
    print(TARGET_MIXTURE_ID)

    print()
    print("Noise regime:")
    print(noise_type)

    print()
    print("Target SNR:")
    print(f"{target_snr:.2f} dB")

    # --------------------------------------------------------
    # 2. Locate local files
    # --------------------------------------------------------

    speech_path = find_speech_file(
        speech_metadata_path
    )

    noise_path = find_noise_file(
        noise_metadata_path,
        noise_type
    )

    print()
    print("Speech:")
    print(speech_path)

    print()
    print("Noise:")
    print(noise_path)

    # --------------------------------------------------------
    # 3. Load audio
    # --------------------------------------------------------

    speech_sr, speech = load_wav(
        speech_path
    )

    noise_sr, noise = load_wav(
        noise_path
    )

    if speech_sr != SAMPLE_RATE:
        raise ValueError(
            f"Speech sample rate is {speech_sr}, "
            f"expected {SAMPLE_RATE}."
        )

    if noise_sr != SAMPLE_RATE:
        raise ValueError(
            f"Noise sample rate is {noise_sr}, "
            f"expected {SAMPLE_RATE}."
        )

    # --------------------------------------------------------
    # 4. Match noise duration to speech
    # --------------------------------------------------------

    noise = repeat_to_length(
        noise,
        len(speech)
    )

    # --------------------------------------------------------
    # 5. Scale noise to metadata SNR
    # --------------------------------------------------------

    scaled_noise = scale_noise_to_snr(
        speech,
        noise,
        target_snr
    )

    reconstructed_input = (
        speech +
        scaled_noise
    )

    reconstructed_snr = snr_db(
        speech,
        scaled_noise
    )

    # --------------------------------------------------------
    # 6. Reference signal
    #
    # In this controlled experiment:
    #
    # x(n) = isolated noise
    #
    # This represents what an idealized reference microphone
    # would observe.
    # --------------------------------------------------------

    reference = scaled_noise.copy()

    # --------------------------------------------------------
    # 7. Primary acoustic path
    #
    # noise source -> ear
    # --------------------------------------------------------

    primary_path = create_primary_path()

    disturbance = lfilter(
        primary_path,
        [1.0],
        reference
    )

    # Normalize disturbance RMS to preserve the intended
    # noise level while retaining acoustic coloration.
    disturbance = normalize_rms(
        disturbance,
        rms(reference)
    )

    # --------------------------------------------------------
    # 8. Secondary acoustic path
    #
    # speaker -> air -> error microphone
    # --------------------------------------------------------

    secondary_path = create_secondary_path()

    # --------------------------------------------------------
    # 9. Run FxLMS
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("Running FxLMS...")
    print("-" * 60)

    anti_noise, residual, weights, error_power = run_fxlms(
        reference=reference,
        disturbance=disturbance,
        secondary_path=secondary_path,
        filter_length=FILTER_LENGTH,
        mu=MU
    )

    # --------------------------------------------------------
    # 10. Metrics
    # --------------------------------------------------------

    input_noise_power = power(
        disturbance
    )

    residual_noise_power = power(
        residual
    )

    input_snr = snr_db(
        speech,
        disturbance
    )

    output_snr = snr_db(
        speech,
        residual
    )

    snr_improvement = (
        output_snr -
        input_snr
    )

    attenuation_db = 10.0 * np.log10(
        (input_noise_power + EPSILON) /
        (residual_noise_power + EPSILON)
    )

    # Final communication signal.
    processed_speech = (
        speech +
        residual
    )

    # --------------------------------------------------------
    # 11. Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print()
    print(f"Mixture ID:              {TARGET_MIXTURE_ID}")
    print(f"Noise regime:            {noise_type}")
    print(f"Target SNR:              {target_snr:.2f} dB")
    print(f"Reconstructed SNR:       {reconstructed_snr:.2f} dB")
    print()
    print(f"Input SNR at ear:        {input_snr:.2f} dB")
    print(f"Output SNR:              {output_snr:.2f} dB")
    print(f"SNR improvement:         {snr_improvement:.2f} dB")
    print()
    print(
        f"Input noise power:       "
        f"{input_noise_power:.8f}"
    )

    print(
        f"Residual noise power:    "
        f"{residual_noise_power:.8f}"
    )

    print(
        f"Noise attenuation:       "
        f"{attenuation_db:.2f} dB"
    )

    print()
    print(f"Filter length:            {FILTER_LENGTH} taps")
    print(f"Step size (mu):           {MU}")
    print()
    print("Primary path:             SIMULATED")
    print("Secondary path:           SIMULATED")

    # --------------------------------------------------------
    # 12. Save audio
    # --------------------------------------------------------

    save_wav(
        os.path.join(
            RESULTS_DIR,
            "reference_noise.wav"
        ),
        reference,
        SAMPLE_RATE
    )

    save_wav(
        os.path.join(
            RESULTS_DIR,
            "disturbance_before_anc.wav"
        ),
        disturbance,
        SAMPLE_RATE
    )

    save_wav(
        os.path.join(
            RESULTS_DIR,
            "residual_after_fxlms.wav"
        ),
        residual,
        SAMPLE_RATE
    )

    save_wav(
        os.path.join(
            RESULTS_DIR,
            "processed_speech.wav"
        ),
        processed_speech,
        SAMPLE_RATE
    )

    # --------------------------------------------------------
    # 13. Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        RESULTS_DIR,
        "metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "PS52 DATASET-DRIVEN FxLMS\n"
        )

        f.write(
            "====================================\n"
        )

        f.write(
            f"Mixture ID: {TARGET_MIXTURE_ID}\n"
        )

        f.write(
            f"Noise regime: {noise_type}\n"
        )

        f.write(
            f"Target SNR: {target_snr:.4f} dB\n"
        )

        f.write(
            f"Reconstructed SNR: "
            f"{reconstructed_snr:.4f} dB\n"
        )

        f.write(
            f"Input SNR at ear: "
            f"{input_snr:.4f} dB\n"
        )

        f.write(
            f"Output SNR: "
            f"{output_snr:.4f} dB\n"
        )

        f.write(
            f"SNR improvement: "
            f"{snr_improvement:.4f} dB\n"
        )

        f.write(
            f"Noise attenuation: "
            f"{attenuation_db:.4f} dB\n"
        )

        f.write(
            f"Filter length: "
            f"{FILTER_LENGTH}\n"
        )

        f.write(
            f"Mu: {MU}\n"
        )

        f.write(
            "Primary path: SIMULATED\n"
        )

        f.write(
            "Secondary path: SIMULATED\n"
        )

    # --------------------------------------------------------
    # 14. Generate plots
    # --------------------------------------------------------

    plot_samples = min(
        len(speech),
        int(PLOT_SECONDS * SAMPLE_RATE)
    )

    time = (
        np.arange(plot_samples) /
        SAMPLE_RATE
    )

    # ========================================================
    # Plot 1 - Reference vs disturbance
    # ========================================================

    plt.figure(figsize=(14, 5))

    plt.plot(
        time,
        reference[:plot_samples],
        label="Reference noise x(n)"
    )

    plt.plot(
        time,
        disturbance[:plot_samples],
        label="Disturbance at ear d(n)",
        alpha=0.8
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(
        "Reference Noise vs Simulated Disturbance"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "01_reference_vs_disturbance.png"
        ),
        dpi=150
    )

    plt.close()

    # ========================================================
    # Plot 2 - Noise before vs residual
    # ========================================================

    plt.figure(figsize=(14, 5))

    plt.plot(
        time,
        disturbance[:plot_samples],
        label="Noise before ANC"
    )

    plt.plot(
        time,
        residual[:plot_samples],
        label="Residual noise after FxLMS"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(
        "Noise Before vs Residual Noise After FxLMS"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "02_noise_before_after.png"
        ),
        dpi=150
    )

    plt.close()

    # ========================================================
    # Plot 3 - Speech before vs after
    # ========================================================

    noisy_signal = (
        speech +
        disturbance
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        time,
        noisy_signal[:plot_samples],
        label="No ANC"
    )

    plt.plot(
        time,
        processed_speech[:plot_samples],
        label="FxLMS"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(
        "Speech + Noise: Before vs After FxLMS"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "03_speech_before_after.png"
        ),
        dpi=150
    )

    plt.close()

    # ========================================================
    # Plot 4 - Error power / convergence
    # ========================================================

    # Moving average to make convergence easier to see.
    window = 1600

    if len(error_power) >= window:

        kernel = np.ones(window) / window

        smoothed_power = np.convolve(
            error_power,
            kernel,
            mode="valid"
        )

        power_time = (
            np.arange(len(smoothed_power)) /
            SAMPLE_RATE
        )

    else:

        smoothed_power = error_power

        power_time = (
            np.arange(len(error_power)) /
            SAMPLE_RATE
        )

    plt.figure(figsize=(14, 5))

    plt.plot(
        power_time,
        10.0 * np.log10(
            smoothed_power + EPSILON
        )
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Residual power (dB)")

    plt.title(
        "FxLMS Residual Error Power / Convergence"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "04_fxlms_convergence.png"
        ),
        dpi=150
    )

    plt.close()

    # ========================================================
    # Plot 5 - Controller weights
    # ========================================================

    plt.figure(figsize=(12, 5))

    plt.stem(
        np.arange(FILTER_LENGTH),
        weights
    )

    plt.xlabel("Filter tap")
    plt.ylabel("Weight")

    plt.title(
        "Final FxLMS Adaptive Filter Weights"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "05_final_filter_weights.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print()
    print("Results saved in:")

    print(
        RESULTS_DIR
    )

    print()
    print("Generated files:")

    print("  reference_noise.wav")
    print("  disturbance_before_anc.wav")
    print("  residual_after_fxlms.wav")
    print("  processed_speech.wav")
    print("  metrics.txt")
    print("  01_reference_vs_disturbance.png")
    print("  02_noise_before_after.png")
    print("  03_speech_before_after.png")
    print("  04_fxlms_convergence.png")
    print("  05_final_filter_weights.png")

    print()
    print("IMPORTANT:")
    print(
        "Primary and secondary acoustic paths are simulated."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()