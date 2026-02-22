"""
Barn Audio Denoising Pipeline
=============================

Python implementation approximating the iZotope RX 11 workflow:

1. DC offset removal and gain normalization
2. Band-pass filtering (50–1800 Hz)
3. Spectral denoising (broadband noise reduction)
4. Transient / artifact attenuation (RX Spectral Repair approximation)
5. Optional de-clip approximation
6. Optional EQ matching (spectral envelope matching to reference)
7. Optional loudness normalization (LUFS)

Dependencies (install via pip):
    numpy
    scipy
    librosa
    soundfile
    noisereduce
    pyloudnorm
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
import noisereduce as nr
import pyloudnorm as pyln


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from file and convert to mono if needed.
    Returns:
        y : float32 mono signal in [-1, 1]
        sr: sample rate
    """
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    # convert to float32 and clip just in case
    y = np.asarray(y, dtype=np.float32)
    max_abs = np.max(np.abs(y)) + 1e-9
    if max_abs > 1.0:
        y = y / max_abs
    return y, sr


def save_audio(path: str, y: np.ndarray, sr: int) -> None:
    """Save mono float signal to WAV, clipping to [-1, 1]."""
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr)


# ----------------------------------------------------------------------
# Stage 1: DC removal & gain normalization
# ----------------------------------------------------------------------

def remove_dc_and_normalize_peak(
    y: np.ndarray,
    peak_target: float = 0.9
) -> np.ndarray:
    """
    Remove DC offset (mean) and peak-normalize.

    Args:
        y: audio signal
        peak_target: desired absolute peak after normalization (<= 1.0)

    Returns:
        y_out: processed signal
    """
    # Remove DC offset
    y_out = y - np.mean(y)

    # Peak normalize
    peak = np.max(np.abs(y_out)) + 1e-9
    y_out = y_out / peak * peak_target

    return y_out


# ----------------------------------------------------------------------
# Stage 2: Band-pass filter 50–1800 Hz (Butterworth)
# ----------------------------------------------------------------------

def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(
    y: np.ndarray,
    fs: int,
    lowcut: float = 50.0,
    highcut: float = 1800.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth band-pass filter (50–1800 Hz by default).

    Args:
        y: audio signal
        fs: sample rate
        lowcut, highcut: band edges in Hz
        order: filter order

    Returns:
        y_bp: band-passed signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y_bp = signal.filtfilt(b, a, y)
    return y_bp.astype(np.float32)


# ----------------------------------------------------------------------
# Stage 3: Spectral denoising (noisereduce)
# ----------------------------------------------------------------------

def extract_noise_clip(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    duration_sec: float,
) -> Optional[np.ndarray]:
    """
    Extract a noise-only segment from y for use as noise profile.

    Args:
        y: audio signal
        sr: sample rate
        start_sec: start time in seconds
        duration_sec: duration in seconds

    Returns:
        noise_clip or None if invalid region
    """
    if duration_sec <= 0:
        return None

    start = int(start_sec * sr)
    end = start + int(duration_sec * sr)
    if start >= len(y):
        return None
    end = min(end, len(y))
    if end <= start:
        return None
    return y[start:end]


def spectral_denoise(
    y: np.ndarray,
    sr: int,
    noise_clip: Optional[np.ndarray] = None,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    """
    Apply spectral gating denoising using noisereduce.

    Args:
        y: audio signal (band-passed)
        sr: sample rate
        noise_clip: optional noise-only segment
        prop_decrease: reduction strength (0–1)

    Returns:
        y_denoised
    """
    if noise_clip is not None:
        y_out = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_clip,
            stationary=False,
            prop_decrease=prop_decrease,
        )
    else:
        y_out = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=False,
            prop_decrease=prop_decrease,
        )
    return y_out.astype(np.float32)


# ----------------------------------------------------------------------
# Stage 4: Transient / artifact attenuation (Spectral Repair approximation)
# ----------------------------------------------------------------------

def attenuate_transients(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    energy_thresh_db: float = 3.0,
    context_frames: int = 2,
) -> np.ndarray:
    """
    Approximate RX Spectral Repair by attenuating frames with unusually high energy.

    Detection:
        - Compute STFT
        - Compute frame-wise energy
        - Mark frames where log energy exceeds
          (median_energy + energy_thresh_db)

    Repair:
        - For marked frames, replace magnitude with median of neighboring frames.

    Args:
        y: input audio
        sr: sample rate
        frame_length, hop_length: STFT parameters
        energy_thresh_db: threshold above median energy (dB)
        context_frames: how many neighbor frames to use for median

    Returns:
        y_out: repaired audio signal
    """
    # STFT
    S = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)

    # Frame-wise energy
    frame_energy = np.sum(mag**2, axis=0)
    # Avoid log of zero
    frame_energy = np.maximum(frame_energy, 1e-12)
    log_energy = librosa.power_to_db(frame_energy, ref=np.max)

    median_energy = np.median(log_energy)
    transient_mask = log_energy > (median_energy + energy_thresh_db)

    mag_repaired = mag.copy()
    n_frames = mag.shape[1]

    for idx, is_transient in enumerate(transient_mask):
        if not is_transient:
            continue
        left = max(0, idx - context_frames)
        right = min(n_frames, idx + context_frames + 1)
        # Median over neighboring non-transient frames if possible
        neighbor_indices = [
            i for i in range(left, right) if i != idx and not transient_mask[i]
        ]
        if not neighbor_indices:
            # Fallback: median over full context region
            neighbor_indices = list(range(left, right))
        mag_repaired[:, idx] = np.median(mag[:, neighbor_indices], axis=1)

    S_repaired = mag_repaired * np.exp(1j * phase)
    y_out = librosa.istft(S_repaired, hop_length=hop_length, length=len(y))

    return y_out.astype(np.float32)


# ----------------------------------------------------------------------
# Stage 5: De-clip approximation
# ----------------------------------------------------------------------

def simple_declip(y: np.ndarray, clip_ratio: float = 0.99) -> np.ndarray:
    """
    Simple de-clip approximation: detect samples near absolute peak and
    interpolate over them.

    Args:
        y: audio signal
        clip_ratio: fraction of max amplitude to treat as clipping

    Returns:
        y_out: de-clipped signal
    """
    y_out = y.copy()
    peak = np.max(np.abs(y_out)) + 1e-9
    threshold = clip_ratio * peak

    clipped = np.abs(y_out) >= threshold
    if not np.any(clipped):
        return y_out

    idx = np.arange(len(y_out))
    # Interpolate over clipped samples using non-clipped neighbors
    y_out[clipped] = np.interp(idx[clipped], idx[~clipped], y_out[~clipped])
    return y_out.astype(np.float32)


# ----------------------------------------------------------------------
# Stage 6: EQ match approximation
# ----------------------------------------------------------------------

def compute_avg_spectrum(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute average magnitude spectrum over time.

    Args:
        y: audio signal
        sr: sample rate

    Returns:
        avg_spec: 1D array of average magnitude per frequency bin
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    avg_spec = np.mean(S, axis=1)
    return avg_spec


def apply_eq_match(
    y: np.ndarray,
    sr: int,
    ref_spectrum: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Apply spectral envelope matching to approximate RX EQ Match.

    Args:
        y: audio signal
        sr: sample rate
        ref_spectrum: average spectrum of reference signal (1D)
        n_fft, hop_length: STFT parameters

    Returns:
        y_eq: EQ-matched signal
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)

    target_spec = np.mean(mag, axis=1)
    # Gain curve: reference / target
    gain = ref_spectrum / (target_spec + eps)
    gain = gain[:, None]  # broadcast across frames

    mag_matched = mag * gain
    S_matched = mag_matched * np.exp(1j * phase)
    y_eq = librosa.istft(S_matched, hop_length=hop_length, length=len(y))
    return y_eq.astype(np.float32)


# ----------------------------------------------------------------------
# Optional loudness normalization (LUFS)
# ----------------------------------------------------------------------

def loudness_normalize(
    y: np.ndarray,
    sr: int,
    target_lufs: float = -24.0,
) -> np.ndarray:
    """
    Loudness normalization using pyloudnorm.

    Args:
        y: audio signal
        sr: sample rate
        target_lufs: target integrated loudness

    Returns:
        y_out: loudness-normalized signal
    """
    meter = pyln.Meter(sr)  # mono meter
    loudness = meter.integrated_loudness(y)
    y_out = pyln.normalize.loudness(y, loudness, target_lufs)
    return y_out.astype(np.float32)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def preprocess_clip(
    y: np.ndarray,
    sr: int,
    noise_clip: Optional[np.ndarray] = None,
    ref_spectrum: Optional[np.ndarray] = None,
    target_lufs: Optional[float] = None,
    band_low: float = 50.0,
    band_high: float = 1800.0,
    denoise_strength: float = 0.8,
    transient_energy_thresh_db: float = 3.0,
    apply_declipping: bool = True,
) -> np.ndarray:
    """
    Run the full denoising pipeline on a single mono clip.

    Args:
        y: input audio (raw)
        sr: sample rate
        noise_clip: optional noise-only segment
        ref_spectrum: optional reference average spectrum for EQ match
        target_lufs: optional target loudness
        band_low, band_high: band-pass edges
        denoise_strength: spectral denoise prop_decrease (0–1)
        transient_energy_thresh_db: threshold for transient detection
        apply_declipping: whether to run simple_declip

    Returns:
        y_proc: processed audio
    """
    # Stage 1: DC removal + peak normalization
    y_proc = remove_dc_and_normalize_peak(y, peak_target=0.9)

    # Stage 2: Band-pass 50–1800 Hz
    y_proc = apply_bandpass(y_proc, sr, lowcut=band_low, highcut=band_high)

    # Stage 3: Spectral denoise
    y_proc = spectral_denoise(
        y_proc,
        sr,
        noise_clip=noise_clip,
        prop_decrease=denoise_strength,
    )

    # Stage 4: Transient attenuation
    y_proc = attenuate_transients(
        y_proc,
        sr,
        frame_length=2048,
        hop_length=512,
        energy_thresh_db=transient_energy_thresh_db,
    )

    # Stage 5: De-clip (optional)
    if apply_declipping:
        y_proc = simple_declip(y_proc, clip_ratio=0.99)

    # Stage 6: EQ match (optional)
    if ref_spectrum is not None:
        y_proc = apply_eq_match(y_proc, sr, ref_spectrum)

    # Stage 7: Loudness normalization (optional)
    if target_lufs is not None:
        y_proc = loudness_normalize(y_proc, sr, target_lufs=target_lufs)

    # Final safety clip
    y_proc = np.clip(y_proc, -1.0, 1.0)

    return y_proc.astype(np.float32)


# ----------------------------------------------------------------------
# CLI wrapper
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Barn audio denoising pipeline (RX-11-inspired, Python implementation)."
    )
    parser.add_argument(
        "input",
        help="Input WAV file OR directory containing WAV files.",
    )
    parser.add_argument(
        "output",
        help="Output WAV file OR directory (if input is directory).",
    )
    parser.add_argument(
        "--noise-start",
        type=float,
        default=0.0,
        help="Start time (s) of noise-only segment within each file.",
    )
    parser.add_argument(
        "--noise-duration",
        type=float,
        default=0.0,
        help="Duration (s) of noise-only segment. 0 disables explicit noise profile.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional reference WAV file for EQ match (should be band-pass 50–1800 Hz).",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=None,
        help="Optional target LUFS for loudness normalization (e.g., -24.0).",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=0.8,
        help="Spectral denoise strength (prop_decrease, 0–1, default 0.8).",
    )
    parser.add_argument(
        "--transient-thresh-db",
        type=float,
        default=3.0,
        help="Energy threshold (dB above median) to mark transient frames.",
    )
    parser.add_argument(
        "--no-declip",
        action="store_true",
        help="Disable simple de-clip stage.",
    )
    return parser


def process_file(
    in_path: str,
    out_path: str,
    args: argparse.Namespace,
    ref_spectrum: Optional[np.ndarray],
) -> None:
    """Run pipeline on a single file and write result to out_path."""
    print(f"Processing: {in_path}")
    y, sr = load_audio(in_path)

    # Extract noise clip if requested
    noise_clip = None
    if args.noise_duration > 0.0:
        noise_clip = extract_noise_clip(
            y, sr, start_sec=args.noise_start, duration_sec=args.noise_duration
        )

    y_proc = preprocess_clip(
        y,
        sr,
        noise_clip=noise_clip,
        ref_spectrum=ref_spectrum,
        target_lufs=args.target_lufs,
        denoise_strength=args.denoise_strength,
        transient_energy_thresh_db=args.transient_thresh_db,
        apply_declipping=not args.no_declip,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_audio(out_path, y_proc, sr)
    print(f"Saved:      {out_path}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Prepare reference spectrum if provided
    ref_spectrum = None
    if args.reference is not None:
        print(f"Loading reference for EQ match: {args.reference}")
        y_ref, sr_ref = load_audio(args.reference)
        y_ref = apply_bandpass(y_ref, sr_ref, lowcut=50.0, highcut=1800.0)
        ref_spectrum = compute_avg_spectrum(y_ref, sr_ref)

    # Single file vs directory mode
    if os.path.isfile(args.input):
        # Single file
        out_path = args.output
        process_file(args.input, out_path, args, ref_spectrum)
    elif os.path.isdir(args.input):
        # Directory mode
        in_dir = args.input
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)

        for root, _, files in os.walk(in_dir):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue
                rel_dir = os.path.relpath(root, in_dir)
                in_path = os.path.join(root, fname)
                out_subdir = os.path.join(out_dir, rel_dir)
                out_path = os.path.join(out_subdir, fname)
                process_file(in_path, out_path, args, ref_spectrum)
    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()
