import streamlit as st
import subprocess
import os
import tempfile
import sys
from PIL import Image

# ============================================================
# App Boot Check (Prevents Blank Page Mystery)
# ============================================================

st.title("Audio Processing Dashboard")
st.write("Upload a file. The backend script (main.py) will process it and generate 3 PNG outputs.")

# ============================================================
# File Upload
# ============================================================

uploaded_file = st.file_uploader("Upload File", type=["wav", "mp4"])

if uploaded_file is not None:

    st.success("File uploaded successfully.")

    # ------------------------------------------------------------
    # Save uploaded file temporarily
    # ------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name

    st.info("Running backend processing...")

    # ------------------------------------------------------------
    # Run main.py safely using same interpreter
    # ------------------------------------------------------------
    with st.spinner("Processing... please wait."):
        result = subprocess.run(
            [sys.executable, "main.py", input_path],
            capture_output=True,
            text=True
        )

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
                st.image(image, caption=img_file, use_column_width=True)
            else:
                st.warning(f"{img_file} not found.")