# ============================
# SECTION 1: Setup Dropbox Links & Install Dependencies
# ============================
import os
import urllib.parse
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt, spectrogram

# Add your Dropbox links here
dropbox_links = [

"file_1_link",
"file_2_link",

# include all dropbox files link for processing it simultaneously.

]

print("Dropbox links loaded")

# ============================
# SECTION 2.1: For files not in actual .wav format, Read It as Raw Audio
# ============================

import numpy as np

def read_raw_audio(file_path, sr=44100, dtype='int16'):
    with open(file_path, 'rb') as f:
        raw = f.read()
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= np.iinfo(np.int16).max  # normalize to [-1, 1] for float
    return audio, sr

# ============================
# SECTION 2: Define Bandpass Filter and Full-file Processing
# ============================
# Function to apply band-pass filter using fourth order Butterworth design
# Removes noise outside 50–1800 Hz (cow vocalization range)
def butter_bandpass_filter(data, sr, lowcut=50.0, highcut=1800.0, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Process full audio and export spectrogram power in dB to CSV (only 50–1800 Hz bins)
# Also save bandpass-filtered audio as a new WAV file

def process_and_export_csv_and_wav(file_path, output_dir='output'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        try:
          data, sr = sf.read(file_path)
        except:
          print("Standard read failed. Trying raw import...")
          data, sr = read_raw_audio(file_path, sr=44100)

        if data.ndim > 1:
            data = data.mean(axis=1)  # Convert stereo to mono

        # Apply full audio band-pass filter
        filtered = butter_bandpass_filter(data, sr)

        # Spectrogram calculation
        f, t, Sxx = spectrogram(filtered, sr, nperseg=1024, noverlap=512)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Only keep frequency bins within 50–1800 Hz
        valid_idx = np.where((f >= 50) & (f <= 1800))[0]
        f_filtered = f[valid_idx]
        Sxx_db = Sxx_db[valid_idx, :]

        # Collect rows for CSV
        rows = []
        for i, time_val in enumerate(t):
            row = {"timestamp": time_val}
            for j, freq in enumerate(f_filtered):
                row[f"{int(freq)}Hz"] = Sxx_db[j, i]
            rows.append(row)

        # Save CSV with _pass_filter.csv suffix
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        csv_path = os.path.join(output_dir, f"{base_filename}_pass_filter.csv")
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Save Filtered WAV with _pass_filter.wav suffix
        filtered_path = os.path.join(output_dir, f"{base_filename}_pass_filter.wav")
        sf.write(filtered_path, filtered, sr)

        print(f"Filtered audio saved: {filtered_path}")
        print(f"Spectral CSV saved: {csv_path}")

        return csv_path, filtered_path
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# ============================
# SECTION 3: Execute Download → Filter → CSV + WAV Export
# ============================
# Downloads WAVs from Dropbox, applies bandpass, saves CSV and filtered WAV
for url in dropbox_links:
    try:
        # Parse the file name directly from the URL
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        local_wav = filename.split('?')[0] if '?' in filename else filename

        print(f"\nDownloading {local_wav}...")
        os.system(f'wget -q -O "{local_wav}" "{url}"')

        if not os.path.exists(local_wav):
            print(f"Failed to download {local_wav}")
            continue

        print(f"Processing {local_wav} with band-pass filter, saving filtered WAV and exporting CSV...")
        csv_path, wav_path = process_and_export_csv_and_wav(local_wav)
        if csv_path and wav_path:
            print(f"Done: CSV saved at {csv_path}")
            print(f"Done: Filtered WAV saved at {wav_path}")
        else:
            print("Skipped due to error.")

    except Exception as e:
        print(f"Unexpected error with {url}: {e}")

# ============================
# SECTION 4: Download the Output folder on local system
# ============================
#Zip the folder
!zip -r output.zip output

#download the zip file
from google.colab import files
files.download('output.zip')

# ============================
# SECTION 5: Delete the files in Output folder
# ============================
#shell command
!rm -rf output/*

#delete downloaded files in root folder
!find . -maxdepth 1 -type f -exec rm -f {} \;

pip install librosa matplotlib numpy

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature

def plot_waveform_and_spectrum(file_path, label):
    y, sr = librosa.load(file_path, sr=None)

    plt.figure(figsize=(12, 4))
    plt.title(f"Waveform - {label}")
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # Spectrogram
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    plt.figure(figsize=(12, 5))
    plt.title(f"Spectrogram - {label}")
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# Compare noise and moo
plot_waveform_and_spectrum("noise_clip.wav", "Noise Segment")
plot_waveform_and_spectrum("moo_clip.wav", "Moo Call")

import numpy as np
import librosa.feature

def extract_basic_features(y, sr):
    return {
        'Duration (s)': librosa.get_duration(y=y, sr=sr),
        'RMS Energy': np.mean(librosa.feature.rms(y=y)),
        'Spectral Centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'Zero Crossing Rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        'Harmonic-to-Noise': np.mean(librosa.effects.harmonic(y))
    }

# Load both
y_noise, sr_noise = librosa.load("noise_clip.wav", sr=None)
y_moo, sr_moo = librosa.load("moo_clip.wav", sr=None)

features_noise = extract_basic_features(y_noise, sr_noise)
features_moo = extract_basic_features(y_moo, sr_moo)

print("Noise Features:", features_noise)
print("Moo Features:", features_moo)