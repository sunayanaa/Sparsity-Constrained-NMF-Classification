# Program Name: vit_baseline_study.py
# Version: 1.0
# Description: Implements a Standard "Patch-Based" Vision Transformer (ViT).
#              1. Loads Audio & Labels.
#              2. Computes 2D Mel-Spectrograms (resized to 64x64).
#              3. Implements ViT (Patches -> Projection -> Transformer -> MLP).
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
    'SR': 2000,
    'DURATION': 5.0,
    'N_MELS': 64,         # Height
    'TARGET_WIDTH': 64,   # Resize width to 64 for square patches
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'PATCH_SIZE': 8,      # Standard ViT patch size
    'PROJECTION_DIM': 64,
    'NUM_HEADS': 4,
    'TRANSFORMER_LAYERS': 4,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'BATCH_SIZE': 32,
    'EPOCHS': 20          # ViTs need slightly longer training
}

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
def load_references(base_path):
    ref_files = glob.glob(os.path.join(base_path, '**', 'REFERENCE.csv'), recursive=True)
    labels = {}
    print(f"Found {len(ref_files)} reference files.")
    for ref_file in ref_files:
        folder = os.path.dirname(ref_file)
        try:
            df = pd.read_csv(ref_file, names=['file', 'label'], header=None)
            for _, row in df.iterrows():
                full_path = os.path.join(folder, f"{row['file']}.wav")
                labels[full_path] = 1 if row['label'] == 1 else 0
        except: pass
    return labels

def extract_vit_features(file_list, labels_dict):
    X = []
    y = []
    
    print(f"Extracting features for ViT (Target: 64x64, Patch: {CONF['PATCH_SIZE']})...")
    
    for f in tqdm(file_list):
        if f not in labels_dict: continue
        try:
            y_sig, sr = librosa.load(f, sr=CONF['SR'], duration=CONF['DURATION'])
            
            # Standard Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=y_sig, sr=sr, n_mels=CONF['N_MELS'], 
                n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN']
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize [-1, 1]
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
            
            # Resize to exactly 64x64 for clean patching
            # (Mel Bands, Time) -> (64, Time) -> Resize to (64, 64)
            mel_resized = tf.image.resize(mel_norm[..., np.newaxis], [64, 64]).numpy()
            
            X.append(mel_resized)
            y.append(labels_dict[f])
            
        except Exception as e:
            continue
            
    return np.array(X), np.array(y)

# ==========================================
# 3. ViT MODEL ARCHITECTURE
# ==========================================
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def build_vit(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # 1. Create Patches
    num_patches = (input_shape[0] // CONF['PATCH_SIZE']) * (input_shape[1] // CONF['PATCH_SIZE'])
    patches = Patches(CONF['PATCH_SIZE'])(inputs)
    
    # 2. Encode Patches (Linear Projection + Position Emb)
    encoded_patches = PatchEncoder(num_patches, CONF['PROJECTION_DIM'])(patches)
    
    # 3. Transformer Blocks
    for _ in range(CONF['TRANSFORMER_LAYERS']):
        # Layer Norm 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=CONF['NUM_HEADS'], key_dim=CONF['PROJECTION_DIM'], dropout=0.1
        )(x1, x1)
        # Skip Connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer Norm 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(units=CONF['PROJECTION_DIM'] * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(units=CONF['PROJECTION_DIM'], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        # Skip Connection 2
        encoded_patches = layers.Add()([x3, x2])

    # 4. Classification Head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    
    logits = layers.Dense(1, activation='sigmoid')(representation)
    
    model = models.Model(inputs=inputs, outputs=logits)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)
    
    labels_map = load_references(CONF['BASE_PATH'])
    files = list(labels_map.keys())
    
    if not files:
        print("Error: No data found.")
    else:
        # Extract
        X, y = extract_vit_features(files, labels_map)
        print(f"\nFeature Shape: {X.shape}") # Should be (N, 64, 64, 1)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train
        print("\nTraining Standard ViT Baseline...")
        model = build_vit(input_shape=(64, 64, 1))
        history = model.fit(X_train, y_train, 
                            epochs=CONF['EPOCHS'], 
                            batch_size=CONF['BATCH_SIZE'], 
                            validation_data=(X_test, y_test),
                            verbose=1)
        
        # Evaluate
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n\n=== STANDARD ViT BASELINE RESULTS (For Table I) ===")
        print(f"{'Method':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
        print("-" * 50)
        print(f"{'Standard ViT (Patch)':<25} | {acc:.4f}     | {f1:.4f}")
        print("-" * 50)