# Program Name: hpss_baseline_study.py
# Version: 1.0
# Description: Implements Harmonic-Percussive Source Separation (HPSS) Baseline.
#              1. Separates Audio into Harmonic (Murmurs) and Percussive (Beats).
#              2. Stacks them as a 2-Channel Image (Channel 1: H, Channel 2: P).
#              3. Trains a model to see if this "Fixed" separation is better than NMF.

import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONF = {
    'SR': 2000,
    'DURATION': 5.0,
    'N_MELS': 64,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'BATCH_SIZE': 32,
    'EPOCHS': 15 
}

# ==========================================
# 2. DATA LOADING & HPSS EXTRACTION
# ==========================================
def load_references(base_path):
    ref_files = glob.glob(os.path.join(base_path, '**', 'REFERENCE.csv'), recursive=True)
    labels = {}
    for ref_file in ref_files:
        folder = os.path.dirname(ref_file)
        try:
            df = pd.read_csv(ref_file, names=['file', 'label'], header=None)
            for _, row in df.iterrows():
                labels[os.path.join(folder, f"{row['file']}.wav")] = 1 if row['label'] == 1 else 0
        except: pass
    return labels

def extract_hpss_features(file_list, labels_dict):
    X = []
    y = []
    expected_frames = int(np.ceil((CONF['DURATION'] * CONF['SR']) / CONF['HOP_LEN']))
    
    print(f"Extracting HPSS Features (Harmonic + Percussive Channels)...")
    
    for f in tqdm(file_list):
        if f not in labels_dict: continue
        try:
            y_sig, sr = librosa.load(f, sr=CONF['SR'], duration=CONF['DURATION'])
            
            # Pad/Truncate
            target_len = int(CONF['DURATION'] * CONF['SR'])
            if len(y_sig) < target_len: y_sig = np.pad(y_sig, (0, target_len - len(y_sig)))
            else: y_sig = y_sig[:target_len]
            
            # --- CORE HPSS LOGIC ---
            # Separate the signal components
            y_harmonic, y_percussive = librosa.effects.hpss(y_sig)
            
            # Compute Mel Spec for EACH component
            mel_h = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=CONF['N_MELS'], n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN'])
            mel_p = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=CONF['N_MELS'], n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN'])
            
            # Normalize
            mel_h = librosa.power_to_db(mel_h, ref=np.max)
            mel_p = librosa.power_to_db(mel_p, ref=np.max)
            mel_h = (mel_h - mel_h.min()) / (mel_h.max() - mel_h.min() + 1e-9)
            mel_p = (mel_p - mel_p.min()) / (mel_p.max() - mel_p.min() + 1e-9)
            
            # Fix shapes
            if mel_h.shape[1] > expected_frames: 
                mel_h = mel_h[:, :expected_frames]
                mel_p = mel_p[:, :expected_frames]
            elif mel_h.shape[1] < expected_frames:
                pad_width = expected_frames - mel_h.shape[1]
                mel_h = np.pad(mel_h, ((0,0), (0, pad_width)))
                mel_p = np.pad(mel_p, ((0,0), (0, pad_width)))
            
            # Stack into 2 Channels: [Freq, Time, 2] -> (Ch1=Harmonic, Ch2=Percussive)
            combined = np.stack([mel_h, mel_p], axis=-1)
            
            X.append(combined)
            y.append(labels_dict[f])
            
        except Exception as e: continue
            
    return np.array(X), np.array(y)

# ==========================================
# 3. MODEL (Simple 2-Channel CNN)
# ==========================================
def build_hpss_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Process the 2 channels (Physics-Informed Inputs)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load Data
    labels_map = load_references(CONF['BASE_PATH'])
    files = list(labels_map.keys())
    
    if files:
        # Extract
        X, y = extract_hpss_features(files, labels_map)
        print(f"\nHPSS Feature Shape: {X.shape} (Note: Last dim is 2 for H/P channels)")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train
        print("\nTraining HPSS Baseline...")
        model = build_hpss_model(X_train.shape[1:])
        model.fit(X_train, y_train, epochs=CONF['EPOCHS'], batch_size=CONF['BATCH_SIZE'], verbose=1)
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n\n=== HPSS BASELINE RESULTS (For Table I) ===")
        print(f"{'Method':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
        print("-" * 50)
        print(f"{'Baseline (HPSS)':<25} | {acc:.4f}     | {f1:.4f}")
        print("-" * 50)