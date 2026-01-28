# Program Name: cnn_baseline_study.py
# Version: 1.0
# Description: Implements a standard 2D-CNN Baseline for PhysioNet 2016.
#              1. Loads Audio & Labels.
#              2. Computes 2D Mel-Spectrograms (fixed 5s duration).
#              3. Trains a 2D-CNN (Conv2D -> MaxPool -> Dense).
#              4. Reports Accuracy & F1-Score for Table I.

import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import glob
from google.colab import drive

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONF = {
    'SR': 2000,           # Same as proposed method
    'DURATION': 5.0,      # Fixed 5s segments
    'N_MELS': 64,         # Standard for Audio CNNs
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'BATCH_SIZE': 32,
    'EPOCHS': 15          # Enough for convergence on this size
}

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
def load_references(base_path):
    """Parses REFERENCE.csv files."""
    ref_files = glob.glob(os.path.join(base_path, '**', 'REFERENCE.csv'), recursive=True)
    labels = {}
    print(f"Found {len(ref_files)} reference files.")
    for ref_file in ref_files:
        folder = os.path.dirname(ref_file)
        try:
            df = pd.read_csv(ref_file, names=['file', 'label'], header=None)
            for _, row in df.iterrows():
                full_path = os.path.join(folder, f"{row['file']}.wav")
                # Label -1 = Normal (0), 1 = Abnormal (1)
                labels[full_path] = 1 if row['label'] == 1 else 0
        except: pass
    return labels

def extract_2d_features(file_list, labels_dict):
    """Extracts 2D Mel-Spectrograms for CNN Input."""
    X = []
    y = []
    
    # Calculate expected width for 5 seconds
    expected_frames = int(np.ceil((CONF['DURATION'] * CONF['SR']) / CONF['HOP_LEN']))
    
    print(f"Extracting 2D Spectrograms (Target Shape: {CONF['N_MELS']}x{expected_frames})...")
    
    for f in tqdm(file_list):
        if f not in labels_dict: continue
        try:
            # Load and fix duration to exactly 5s
            y_sig, sr = librosa.load(f, sr=CONF['SR'], duration=CONF['DURATION'])
            
            # Pad if too short, truncate if too long
            target_len = int(CONF['DURATION'] * CONF['SR'])
            if len(y_sig) < target_len:
                y_sig = np.pad(y_sig, (0, target_len - len(y_sig)))
            else:
                y_sig = y_sig[:target_len]
                
            # Compute Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=y_sig, sr=sr, n_mels=CONF['N_MELS'], 
                n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN']
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalization [-1, 1] usually works best for CNNs
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
            
            # Ensure exact shape (sometimes off by 1 frame due to rounding)
            if mel_norm.shape[1] > expected_frames:
                mel_norm = mel_norm[:, :expected_frames]
            elif mel_norm.shape[1] < expected_frames:
                mel_norm = np.pad(mel_norm, ((0,0), (0, expected_frames - mel_norm.shape[1])))
                
            X.append(mel_norm)
            y.append(labels_dict[f])
            
        except Exception as e:
            continue
            
    # Reshape for CNN (Samples, Freq, Time, Channels)
    X = np.array(X)
    X = X[..., np.newaxis] 
    return X, np.array(y)

# ==========================================
# 3. CNN MODEL ARCHITECTURE
# ==========================================
def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Conv Block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten & Dense
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)
    
    # 1. Load Data
    labels_map = load_references(CONF['BASE_PATH'])
    files = list(labels_map.keys())
    
    if not files:
        print("Error: No data found.")
    else:
        # 2. Extract Features
        X, y = extract_2d_features(files, labels_map)
        print(f"\nFeature Shape: {X.shape}")
        
        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 4. Train CNN
        print("\nTraining CNN Baseline...")
        model = build_cnn(input_shape=X_train.shape[1:])
        history = model.fit(X_train, y_train, 
                            epochs=CONF['EPOCHS'], 
                            batch_size=CONF['BATCH_SIZE'], 
                            validation_data=(X_test, y_test),
                            verbose=1)
        
        # 5. Evaluate
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n\n=== CNN BASELINE RESULTS (For Table I) ===")
        print(f"{'Method':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
        print("-" * 50)
        print(f"{'CNN (Black-Box)':<25} | {acc:.4f}     | {f1:.4f}")
        print("-" * 50)