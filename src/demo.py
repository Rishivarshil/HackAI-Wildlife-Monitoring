import io

import librosa

import streamlit as st
import subprocess
import os
import tempfile
import sys
from PIL import Image
from pydub import AudioSegment
from moviepy import VideoFileClip


# ============================================================
# App Boot Check (Prevents Blank Page Mystery)
# ============================================================

st.title("Audio Processing Dashboard")
st.write("Upload a file. The backend script (main.py) will process it and generate 3 PNG outputs.")

# ============================================================
# File Upload
# ============================================================
#Send to Image Script
def convert_mp4_to_wav(input_path, output_path=None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".wav"

    with VideoFileClip(input_path) as video:
        audio = video.audio
        audio.write_audiofile(
            output_path,
            fps=16000,
            nbytes=2,
            codec="pcm_s16le"
        )

    return output_path

uploaded_file = st.file_uploader("Upload File", type=["wav", "mp4"])

if uploaded_file is not None:

    st.success("File uploaded successfully.")

    try:
        # ------------------------------------------------------------
        # Save uploaded file to disk FIRST
        # ------------------------------------------------------------
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_input_path = tmp_file.name

        file_ext = suffix.lower()[1:]

        if file_ext == "mp4":
            with st.spinner("Processing... please wait."):
                boundingbox = subprocess.run(
                    ["py", "-3.12", "animalClassifier.py", temp_input_path],
                    capture_output=True,
                    text=True
                )
                
                st.write("Return code:", boundingbox.returncode)
                
                if boundingbox.returncode != 0:
                    st.error("Backend processing failed.")
                    st.text_area("Error Log:", boundingbox.stderr) # Helpful for debugging
                else:
                    st.success("Processing complete.")
                    
                    # Define the path where animalClassifier.py saves the file
                    output_video_path = "output.mp4"
                    
                    # Check if the file actually exists before trying to display it
                    if os.path.exists(output_video_path):
                        st.write("### Resulting Detection Video")
                        
                        # Use st.video to render the mp4 file
                        with open(output_video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                            
                        # Optional: Provide a download button for the processed video
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name="processed_wildlife.mp4",
                            mime="video/mp4"
                        )
                    else:
                        st.warning("Processing finished, but output.mp4 was not found.")

            
        # ------------------------------------------------------------
        # Convert MP4 → WAV if needed
        # ------------------------------------------------------------
            st.info("Converting MP4 to WAV...")
            input_path = convert_mp4_to_wav(temp_input_path)
        else:
            input_path = temp_input_path

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.stop()

    st.info("Running backend processing...")
    print(input_path)

    # ------------------------------------------------------------
    # Run main.py
    # ------------------------------------------------------------

    with st.spinner("Processing... please wait."):
        result = subprocess.run(
            ["py", "-3.12", "main.py", input_path],
            capture_output=True,
            text=True
        )

    if result.returncode != 0:
        st.error("Backend processing failed.")
    else:
        st.success("Processing complete.")

    # ------------------------------------------------------------
    # Show backend errors if any
    # ------------------------------------------------------------
    if result.returncode != 0:
        st.error("❌ Backend processing failed.")
        st.text(result.stderr)
    else:
        st.success("✅ Processing complete!")

        # ------------------------------------------------------------
        # Display generated PNGs
        # ------------------------------------------------------------
        png_files = ["unnormalized.png", "normalized.png"]

        for img_file in png_files:
            if os.path.exists(img_file):
                image = Image.open(img_file)
                st.image(image, caption=img_file)
            else:
                st.warning(f"{img_file} not found.")
    if "Predicted label: 0" in result.stdout:
        st.success("✅ Prediction: No Anomaly Detected!")
    else:
        st.warning("❌ Prediction: Anomaly Detected.")