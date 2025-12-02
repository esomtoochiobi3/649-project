import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
GENRE_DIR = 'data/genres/'         # path to GTZAN genre folders 
OUTPUT_CSV = 'features/features.csv'
SAMPLE_RATE = 22050
DURATION = 30  # seconds

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # aggregate statistics
    features = {
        'mfcc_mean': np.mean(mfcc),
        'mfcc_std': np.std(mfcc),
        'chroma_mean': np.mean(chroma),
        'contrast_mean': np.mean(contrast),
        'zcr_mean': np.mean(zcr),
        'tempo': tempo
    }

    # flatten to single row
    return features

def main():
    rows = []

    genres = sorted(os.listdir(GENRE_DIR))

    print("Extracting features from GTZAN dataset...\n")

    for genre in genres:
        genre_path = os.path.join(GENRE_DIR, genre)
        if not os.path.isdir(genre_path):
            continue

        for filename in tqdm(os.listdir(genre_path), desc=f"[{genre}]"):
            if not filename.endswith(".wav"):
                continue

            file_path = os.path.join(genre_path, filename)
            try:
                feats = extract_features(file_path)
                feats['filename'] = filename
                feats['label'] = genre
                rows.append(feats)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Features saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
