# ============================================================
# Extract 24 acoustic features from .mp3/.wav files
# ============================================================
#
# 1. Lets you upload a ZIP file containing folders / subfolders with audio files.
# 2. Unzips into a "data" directory.
# 3. Recursively finds all .wav and .mp3 files.
# 4. For each file, computes 24 acoustic features:
#       - duration_s
#       - snr_db
#       - f0_mean_hz, f0_min_hz, f0_max_hz
#       - intensity_min_db, intensity_max_db
#       - f1_mean_hz, f2_mean_hz
#       - rms_energy_mean, rms_energy_std
#       - spectral_centroid_mean
#       - spectral_bandwidth_mean
#       - spectral_rolloff_95
#       - zcr_mean, zcr_std
#       - time_to_peak_s
#       - mfcc1_mean, mfcc1_std
#       - mfcc2_mean, mfcc2_std
#       - mfcc3_mean, mfcc3_std
#       - voiced_ratio
# 5. Saves results to an Excel file with columns:
#       main_folder, file_name, relative_path, + 24 feature columns.
#
# ============================================================
# 0. Install dependencies
# ============================================================

#pip install -q librosa==0.10.2.post1 soundfile praat-parselmouth openpyxl

# ============================================================
# 1. Imports
# ============================================================

import os
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import noisereduce as nr
import librosa
import parselmouth  # praat-parselmouth
from scipy.signal import butter, filtfilt, stft
import types
import sys
if 'imp' not in sys.modules:
    imp = types.ModuleType('imp')
    # Add common functions that older versions of flatbuffers/tensorlow look for
    imp.find_module = lambda *args: None
    imp.load_module = lambda *args: None
    sys.modules['imp'] = imp
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================
# 2. Upload ZIP and extract it into ./data
# ============================================================
#data_root = input("Enter the full path to your audio directory: ").strip()
data_root = "data"  

# ============================================================
# 3. Helper functions: feature computation
# ============================================================

SR_TARGET = 16000        # Target sampling rate for feature extraction
FRAME_LENGTH = 1024      # STFT frame length
HOP_LENGTH = 512         # STFT hop length
F0_MIN_HZ = 50.0
F0_MAX_HZ = 1250.0
EPS = 1e-10

LOWCUT = 50.0
HIGHCUT = 1800.0

def butter_bandpass_filter(data, sr, lowcut, highcut, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def clean_audio(y, sr):
    """
    Approximates the iZotope RX workflow:
    1. DC Offset removal
    2. Gain Normalization
    3. Band-pass filtering (50-1800Hz)
    4. Spectral Noise Reduction
    """
    # 1. DC Offset & Normalization
    y = y - np.mean(y)
    y = librosa.util.normalize(y)
    
    # 2. Band-pass filter to remove wind/low-rumble and high-freq static
    y_filtered = butter_bandpass_filter(y, sr, LOWCUT, HIGHCUT)
    
    # 3. Spectral Denoising (Broadband noise reduction)
    # Using non-stationary reduce_noise for better wildlife background handling
    y_denoised = nr.reduce_noise(y=y_filtered, sr=sr, prop_decrease=0.8)
    
    return y_denoised

def load_audio(path, sr=SR_TARGET):
    """Loads and immediately applies the denoising pipeline."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    
    if len(y) > 0:
        y = clean_audio(y, sr)
    return y, sr

def compute_duration(y, sr):
    """
    Compute clip duration in seconds.
    """
    return len(y) / float(sr) if sr > 0 else np.nan

def compute_snr_db(y):
    """
    Approximate signal-to-noise ratio (SNR) in dB.
    We use frame-wise RMS. The 95th percentile RMS is treated as "signal",
    and the 5th percentile as "noise".
    """
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    if len(rms) == 0:
        return np.nan
    signal_rms = np.percentile(rms, 95)
    noise_rms = np.percentile(rms, 5)
    # Avoid log of zero
    if noise_rms < EPS or signal_rms < EPS:
        return np.nan
    snr_db = 20.0 * np.log10(signal_rms / noise_rms)
    return float(snr_db)

def compute_f0_and_voiced_ratio(y, sr):
    """
    Estimate F0 contour using librosa.pyin and compute:
      - f0_mean_hz, f0_min_hz, f0_max_hz
      - voiced_ratio (fraction of frames marked as voiced)
    """
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=F0_MIN_HZ,
            fmax=F0_MAX_HZ,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan

    if f0 is None or len(f0) == 0:
        return np.nan, np.nan, np.nan, np.nan

    f0_valid = f0[~np.isnan(f0)]
    n_frames = len(f0)
    n_voiced = np.count_nonzero(~np.isnan(f0))
    voiced_ratio = (n_voiced / n_frames) if n_frames > 0 else np.nan

    if len(f0_valid) == 0:
        return np.nan, np.nan, np.nan, voiced_ratio

    f0_mean = float(np.mean(f0_valid))
    f0_min = float(np.min(f0_valid))
    f0_max = float(np.max(f0_valid))
    return f0_mean, f0_min, f0_max, voiced_ratio

def compute_intensity_stats(y):
    """
    Compute approximate intensity (in dB) using frame-wise RMS.
    """
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    if len(rms) == 0:
        return np.nan, np.nan

    # Convert RMS to dB (full scale)
    rms_db = librosa.amplitude_to_db(rms + EPS, ref=1.0)
    intensity_min = float(np.min(rms_db))
    intensity_max = float(np.max(rms_db))
    return intensity_min, intensity_max

def compute_formants(y, sr, n_formants=2):
    """
    Estimate formant frequencies (F1, F2) using praat-parselmouth.
    We take the average value over time for each formant.
    """
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        formant = snd.to_formant_burg()
        times = np.linspace(0, snd.duration, num=50)
        formants = {i: [] for i in range(1, n_formants + 1)}

        for t in times:
            for i in range(1, n_formants + 1):
                value = formant.get_value_at_time(i, t)
                if not math.isnan(value):
                    formants[i].append(value)

        f_means = []
        for i in range(1, n_formants + 1):
            if len(formants[i]) == 0:
                f_means.append(np.nan)
            else:
                f_means.append(float(np.mean(formants[i])))

        # Pad with NaN if fewer than requested formants found
        while len(f_means) < n_formants:
            f_means.append(np.nan)
        return f_means[0], f_means[1]

    except Exception as e:
        return np.nan, np.nan

def compute_band_level_metrics(y, sr):
    """
    Compute band-level descriptors:
      - rms_energy_mean, rms_energy_std
      - spectral_centroid_mean
      - spectral_bandwidth_mean
      - spectral_rolloff_95
      - zcr_mean, zcr_std
      - time_to_peak_s  (time of maximum RMS)
    """
    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    if len(rms) == 0:
        rms_mean = rms_std = np.nan
    else:
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

    # Spectral centroid & bandwidth
    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    if S.size == 0:
        centroid_mean = np.nan
        bandwidth_mean = np.nan
    else:
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        centroid_mean = float(np.mean(centroid)) if len(centroid) > 0 else np.nan
        bandwidth_mean = float(np.mean(bandwidth)) if len(bandwidth) > 0 else np.nan

    # Spectral rolloff (95%)
    if S.size == 0:
        rolloff_95 = np.nan
    else:
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
        rolloff_95 = float(np.mean(rolloff)) if len(rolloff) > 0 else np.nan

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    if len(zcr) == 0:
        zcr_mean = zcr_std = np.nan
    else:
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

    # Time to peak energy (using RMS)
    if len(rms) == 0:
        time_to_peak_s = np.nan
    else:
        peak_frame = int(np.argmax(rms))
        time_to_peak_s = float(peak_frame * HOP_LENGTH / sr)

    return (
        rms_mean,
        rms_std,
        centroid_mean,
        bandwidth_mean,
        rolloff_95,
        zcr_mean,
        zcr_std,
        time_to_peak_s,
    )

def compute_mfcc_features(y, sr, n_mfcc=3):
    """
    Compute MFCC-based features:
      - For the first n_mfcc coefficients, compute mean and std.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # mfcc shape: (n_mfcc, n_frames)
    feats = []
    for i in range(n_mfcc):
        coef = mfcc[i]
        if len(coef) == 0:
            feats.append(np.nan)
            feats.append(np.nan)
        else:
            feats.append(float(np.mean(coef)))
            feats.append(float(np.std(coef)))
    return feats

# ============================================================
# 4. Traverse the extracted directory and collect audio files
# ============================================================

def find_audio_files(root_dir, exts=(".wav", ".mp3", ".flac", ".ogg", ".m4a")):
    """
    Recursively find audio files under root_dir with given extensions.
    Returns a list of absolute file paths.
    """
    audio_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(exts):
                full_path = os.path.join(dirpath, fname)
                audio_paths.append(full_path)
    return audio_paths

audio_files = find_audio_files(data_root)
print(f"Found {len(audio_files)} audio files.")

if len(audio_files) == 0:
    print("WARNING: No audio files found. Please check your ZIP structure and extensions.")

# ============================================================
# 5. Compute 24 features for each file
# ============================================================

rows = []

for idx, path in enumerate(sorted(audio_files)):
    rel_path = os.path.relpath(path, data_root)
    parts = rel_path.split(os.sep)

    # main_folder = top-level folder under data_root (if any)
    if len(parts) >= 2:
        main_folder = parts[0]
        sub_folder = os.path.join(*parts[1:-1]) if len(parts) > 2 else ""
    else:
        main_folder = ""
        sub_folder = ""

    # --- Class label based on top-level folder ---
    if "HFC" in sub_folder.upper():
        label = 1   # Unhealthy
    elif "LFC" in sub_folder.upper():
        label = 0   # Healthy
    else:
        print(f"  WARNING: Could not determine label for {rel_path}. Marking as NaN.")
        label = np.nan  # Unknown / other folder


    file_name = parts[-1]

    print(f"[{idx+1}/{len(audio_files)}] Processing: {rel_path}")

    try:
        y, sr = load_audio(path, sr=SR_TARGET)
        if len(y) == 0 or sr <= 0:
            raise ValueError("Empty audio or invalid sample rate")

        # --- Temporal ---
        duration_s = compute_duration(y, sr)

        # --- Signal-to-noise ratio ---
        snr_db = compute_snr_db(y)

        # --- F0 stats + voiced ratio ---
        f0_mean_hz, f0_min_hz, f0_max_hz, voiced_ratio = compute_f0_and_voiced_ratio(y, sr)

        # --- Intensity stats ---
        intensity_min_db, intensity_max_db = compute_intensity_stats(y)

        # --- Formants (F1, F2) ---
        f1_mean_hz, f2_mean_hz = compute_formants(y, sr, n_formants=2)

        # --- Band-level metrics ---
        (
            rms_energy_mean,
            rms_energy_std,
            spectral_centroid_mean,
            spectral_bandwidth_mean,
            spectral_rolloff_95,
            zcr_mean,
            zcr_std,
            time_to_peak_s,
        ) = compute_band_level_metrics(y, sr)

        # --- MFCCs (first 3, mean + std) ---
        (
            mfcc1_mean,
            mfcc1_std,
            mfcc2_mean,
            mfcc2_std,
            mfcc3_mean,
            mfcc3_std,
        ) = compute_mfcc_features(y, sr, n_mfcc=3)

        row = {
            "label": label,
            # 24 acoustic features:
            "duration_s": duration_s,
            "snr_db": snr_db,
            "f0_mean_hz": f0_mean_hz,
            "f0_min_hz": f0_min_hz,
            "f0_max_hz": f0_max_hz,
            "intensity_min_db": intensity_min_db,
            "intensity_max_db": intensity_max_db,
            "f1_mean_hz": f1_mean_hz,
            "f2_mean_hz": f2_mean_hz,
            "rms_energy_mean": rms_energy_mean,
            "rms_energy_std": rms_energy_std,
            "spectral_centroid_mean": spectral_centroid_mean,
            "spectral_bandwidth_mean": spectral_bandwidth_mean,
            "spectral_rolloff_95": spectral_rolloff_95,
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std,
            "time_to_peak_s": time_to_peak_s,
            "mfcc1_mean": mfcc1_mean,
            "mfcc1_std": mfcc1_std,
            "mfcc2_mean": mfcc2_mean,
            "mfcc2_std": mfcc2_std,
            "mfcc3_mean": mfcc3_mean,
            "mfcc3_std": mfcc3_std,
            "voiced_ratio": voiced_ratio,
        }

    except Exception as e:
        print(f"  ERROR processing {rel_path}: {e}")
        row = {
            "main_folder": main_folder,
            "sub_folder": sub_folder,
            "file_name": file_name,
            "relative_path": rel_path,
            "duration_s": np.nan,
            "snr_db": np.nan,
            "f0_mean_hz": np.nan,
            "f0_min_hz": np.nan,
            "f0_max_hz": np.nan,
            "intensity_min_db": np.nan,
            "intensity_max_db": np.nan,
            "f1_mean_hz": np.nan,
            "f2_mean_hz": np.nan,
            "rms_energy_mean": np.nan,
            "rms_energy_std": np.nan,
            "spectral_centroid_mean": np.nan,
            "spectral_bandwidth_mean": np.nan,
            "spectral_rolloff_95": np.nan,
            "zcr_mean": np.nan,
            "zcr_std": np.nan,
            "time_to_peak_s": np.nan,
            "mfcc1_mean": np.nan,
            "mfcc1_std": np.nan,
            "mfcc2_mean": np.nan,
            "mfcc2_std": np.nan,
            "mfcc3_mean": np.nan,
            "mfcc3_std": np.nan,
            "voiced_ratio": np.nan,
        }

    rows.append(row)

# ============================================================
# 6. Save to CSV and make it downloadable
# ============================================================
df = pd.DataFrame(rows)

df.drop('label', axis=1, inplace=True)  # Remove label column for prediction
df.to_csv("feature.csv", index=False)

scaler = StandardScaler()
scaler = joblib.load("scaler.pkl")
df = scaler.transform(df)
model = load_model("lightweight_cnn_model.h5")

# 1. 'df' is already a numpy array from scaler.transform
# 2. Reshape the whole block at once: (Total_Samples, 24, 1)
input_data = np.expand_dims(df, axis=-1)

# 3. Predict everything in one go
predictions = model.predict(input_data, verbose=0)

# 4. Process results
for i, pred in enumerate(predictions):
    score = pred[0]
    binary_score = 1 if score >= 0.5 else 0
    confidence = score if score > 0.5 else 1 - score
    print(f"Index {i} | Label: {binary_score} | Confidence: {confidence:.2f}")
    
