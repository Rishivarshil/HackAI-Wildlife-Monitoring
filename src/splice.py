import os
import pandas as pd
import librosa
import soundfile as sf

# --- Configuration ---
WAV_FOLDER = "audios"
TXT_FOLDER = "labels"
OUTPUT_ROOT = "LHC"

# Ensure output directory exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def segment_bites():
    # Loop through wav files
    for wav_file in os.listdir(WAV_FOLDER):
        if not wav_file.endswith(".wav"):
            continue
            
        file_id = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(WAV_FOLDER, wav_file)
        txt_path = os.path.join(TXT_FOLDER, file_id + ".txt")
        
        # Check if matching label file exists
        if not os.path.exists(txt_path):
            print(f"Skipping {wav_file}: No matching .txt found.")
            continue

        print(f"Processing: {wav_file}...")
        
        # 1. Load the full audio file
        y, sr = librosa.load(wav_path, sr=None)

        # 2. Read the label file (assuming Tab-separated: Start, End, Label)
        # Change sep='\t' to sep='\s+' if it's space-separated
        try:
            df = pd.read_csv(txt_path, sep='\t', header=None, names=['start', 'end', 'label'])
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            continue

        # 3. Slice and Export
        for i, row in df.iterrows():
            if row['label'].strip().lower() == 'bite':
                start_sample = int(row['start'] * sr)
                end_sample = int(row['end'] * sr)
                
                # Extract the bite clip
                bite_audio = y[start_sample:end_sample]
                
                # Construct output name: originalName_bite_01.wav
                output_filename = f"{file_id}_bite_{i:03d}.wav"
                output_path = os.path.join(OUTPUT_ROOT, output_filename)
                
                # Save the new clip
                sf.write(output_path, bite_audio, sr)

    print("Segmentation complete!")

if __name__ == "__main__":
    segment_bites()