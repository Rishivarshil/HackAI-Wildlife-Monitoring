import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import soundfile as sf
from pydub import AudioSegment

# ============================================================
# App Title
# ============================================================

st.title("Audio Spectrogram & Feature Extractor")
st.write("Upload a .wav or .mp4 file to generate spectrograms and extract acoustic features.")

# ============================================================
# File Upload
# ============================================================

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp4"])

if uploaded_file is not None:

    file_ext = uploaded_file.name.split(".")[-1].lower()

    try:
        # ------------------------------------------------------------
        # Convert MP4 → WAV if necessary
        # ------------------------------------------------------------
        if file_ext == "mp4":
            st.info("Converting MP4 to WAV...")

            audio_bytes = uploaded_file.read()
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp4")

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            y, sr = librosa.load(wav_buffer, sr=16000, mono=True)

        elif file_ext == "wav":
            y, sr = librosa.load(uploaded_file, sr=16000, mono=True)

        else:
            st.error("Unsupported file type.")
            st.stop()

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.stop()

    st.success(f"Audio loaded! Duration: {len(y)/sr:.2f} seconds")

    # ============================================================
    # 1️⃣ Linear Spectrogram
    # ============================================================

    st.subheader("Linear Spectrogram")

    fig1, ax1 = plt.subplots()
    D = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    D_db = librosa.amplitude_to_db(D, ref=np.max)

    img1 = librosa.display.specshow(
        D_db,
        sr=sr,
        hop_length=512,
        x_axis='time (s)',
        y_axis='linear',
        ax=ax1
    )
    fig1.colorbar(img1, ax=ax1, format="%+2.0f dB")
    ax1.set_title("Linear Frequency Spectrogram")

    st.pyplot(fig1)

    # ============================================================
    # 2️⃣ Mel Spectrogram
    # ============================================================

    st.subheader("Mel Spectrogram")

    fig2, ax2 = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)

    img2 = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        ax=ax2
    )
    fig2.colorbar(img2, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Mel Spectrogram")

    st.pyplot(fig2)

    # ============================================================
    # 3️⃣ Feature Extraction
    # ============================================================

    st.subheader("Extracted Features")

    duration = len(y) / sr

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    mfcc1_mean = np.mean(mfcc[0])
    mfcc2_mean = np.mean(mfcc[1])
    mfcc3_mean = np.mean(mfcc[2])

    data = {
        "duration_s": [duration],
        "rms_mean": [rms_mean],
        "rms_std": [rms_std],
        "spectral_centroid_mean": [centroid_mean],
        "zcr_mean": [zcr_mean],
        "mfcc1_mean": [mfcc1_mean],
        "mfcc2_mean": [mfcc2_mean],
        "mfcc3_mean": [mfcc3_mean],
    }

    df = pd.DataFrame(data)

    st.dataframe(df)

    # ============================================================
    # 4️⃣ CSV Download
    # ============================================================

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Features as CSV",
        data=csv_buffer.getvalue(),
        file_name="audio_features.csv",
        mime="text/csv",
    )