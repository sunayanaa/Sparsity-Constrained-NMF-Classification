# Program Name: cnn_rebuild_and_gradcam.py
# Version: 2.1 (Keras 3 Compatibility Fix)
# Description: 
#   1. Re-trains CNN using Functional API .
#   2. Generates the 'Blobby' Grad-CAM visualization.

import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
CONF = {
    'SR': 2000,
    'DURATION': 5.0,
    'N_MELS': 64,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'BATCH_SIZE': 32,
    'EPOCHS': 5  # Quick training for demo
}

def get_data():
    # 1. Load Labels
    ref_files = glob.glob(os.path.join(CONF['BASE_PATH'], '**', 'REFERENCE.csv'), recursive=True)
    labels = {}
    for ref_file in ref_files:
        folder = os.path.dirname(ref_file)
        try:
            df = pd.read_csv(ref_file, names=['file', 'label'], header=None)
            for _, row in df.iterrows():
                labels[os.path.join(folder, f"{row['file']}.wav")] = 1 if row['label'] == 1 else 0
        except: pass
    
    # 2. Extract Features (First 500 for speed)
    files = list(labels.keys())[:500] 
    print(f"Loading {len(files)} files for visualization demo...")
    
    X, y = [], []
    expected_frames = int(np.ceil((CONF['DURATION'] * CONF['SR']) / CONF['HOP_LEN']))
    
    for f in tqdm(files):
        try:
            y_sig, sr = librosa.load(f, sr=CONF['SR'], duration=CONF['DURATION'])
            target_len = int(CONF['DURATION'] * CONF['SR'])
            if len(y_sig) < target_len: y_sig = np.pad(y_sig, (0, target_len - len(y_sig)))
            else: y_sig = y_sig[:target_len]
            
            mel = librosa.feature.melspectrogram(y=y_sig, sr=sr, n_mels=CONF['N_MELS'], 
                                                n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN'])
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
            
            if mel_norm.shape[1] > expected_frames: mel_norm = mel_norm[:, :expected_frames]
            elif mel_norm.shape[1] < expected_frames: 
                mel_norm = np.pad(mel_norm, ((0,0), (0, expected_frames - mel_norm.shape[1])))
                
            X.append(mel_norm)
            y.append(labels[f])
        except: continue
        
    X = np.array(X)[..., np.newaxis]
    return X, np.array(y)

# ==========================================
# 2. BUILD CNN (FUNCTIONAL API FIX)
# ==========================================
def build_and_train_cnn(X, y):
    input_shape = X.shape[1:]
    
    # --- FIX: Use Functional API Explicitly ---
    inputs = Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_1')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Block 2 (Target for Grad-CAM)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_last')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Dense Head
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training CNN (Functional API)...")
    model.fit(X, y, epochs=CONF['EPOCHS'], batch_size=CONF['BATCH_SIZE'], verbose=1)
    return model

# ==========================================
# 3. GRAD-CAM GENERATOR
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Now that model is Functional, this works safely in Keras 3
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()

# ==========================================
# 4. MAIN RUN
# ==========================================
if __name__ == "__main__":
    # 1. Prepare Data & Model
    X, y = get_data()
    model = build_and_train_cnn(X, y)
    
    # 2. Find Abnormal Sample
    pos_indices = np.where(y == 1)[0]
    idx = pos_indices[0] if len(pos_indices) > 0 else 0

    # 3. Generate Heatmap
    print("\nGenerating Grad-CAM Visualization...")
    img = X[idx]
    heatmap = make_gradcam_heatmap(np.expand_dims(img, axis=0), model, 'conv2d_last')
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 4. Plot Comparison
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze(), aspect='auto', origin='lower', cmap='magma')
    plt.title('Input Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img.squeeze(), aspect='auto', origin='lower', cmap='gray', alpha=0.3)
    plt.imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.6)
    plt.title('CNN Grad-CAM Focus')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('gradcam_comparison.png') # Auto-save for download
    plt.show()
    print("Success! 'gradcam_comparison.png' saved.")