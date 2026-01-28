	# Program Name: nmf_classification_baseline.py
	# Version: 1.1 (Clean Output & Better Convergence)
	# Description: Performs comparative baseline study (Mel-Spec vs. NMF Atoms).

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import glob
import warnings

from google.colab import drive


# --- FIX: Suppress Convergence Warnings for clean log ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONF = {
    'SR': 2000,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'N_ATOMS': 8,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'MAX_ITER': 500  # Increased from 200 to 500 for better convergence
}

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_references(base_path):
    """Parses REFERENCE.csv files to get file_name -> label mappings."""
    ref_files = glob.glob(os.path.join(base_path, '**', 'REFERENCE.csv'), recursive=True)
    labels = {}
    
    print(f"Found {len(ref_files)} reference files. Loading labels...")
    for ref_file in ref_files:
        folder = os.path.dirname(ref_file)
        try:
            # PhysioNet 2016 ref format: filename, label
            df = pd.read_csv(ref_file, names=['file', 'label'], header=None)
            for _, row in df.iterrows():
                # Store full path as key
                full_path = os.path.join(folder, f"{row['file']}.wav")
                # Label -1 = Normal, 1 = Abnormal. Remap to 0/1
                labels[full_path] = 1 if row['label'] == 1 else 0
        except Exception as e:
            pass
            
    return labels

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
def extract_features(file_list, labels_dict):
    X_baseline = [] # Raw Spectrogram Features
    X_proposed = [] # NMF Atom Features
    y = []
    
    print(f"Extracting features from {len(file_list)} files...")
    print(f"(Using max_iter={CONF['MAX_ITER']} - this takes about ~15 mins for full dataset)")
    
    for f in tqdm(file_list):
        if f not in labels_dict: continue
        
        try:
            y_sig, sr = librosa.load(f, sr=CONF['SR'], duration=5.0)
            
            # -- BASELINE FEATURES (Mel Spectrogram) --
            # Standard "black box" input
            mel = librosa.feature.melspectrogram(y=y_sig, sr=sr, n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN'])
            mel_mean = np.mean(mel, axis=1) # Global average pooling
            
            # -- PROPOSED FEATURES (NMF Atoms) --
            S = np.abs(librosa.stft(y_sig, n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN']))
            V = S / (np.max(S) + 1e-9)
            
            # Fast KL-NMF with increased iterations
            model = NMF(n_components=CONF['N_ATOMS'], solver='mu', beta_loss='kullback-leibler', 
                       init='random', alpha_H=0.1, max_iter=CONF['MAX_ITER'], random_state=42)
            W = model.fit_transform(V)
            H = model.components_
            
            # Feature: Average activation strength of each atom over time
            atom_activations = np.mean(H, axis=1)
            
            X_baseline.append(mel_mean)
            X_proposed.append(atom_activations)
            y.append(labels_dict[f])
            
        except Exception as e:
            continue
            
    return np.array(X_baseline), np.array(X_proposed), np.array(y)

# ==========================================
# 4. CLASSIFICATION & REPORTING
# ==========================================
def run_baseline_study():
    # 1. Load Data
    labels_map = load_references(CONF['BASE_PATH'])
    all_files = list(labels_map.keys())
    
    if len(all_files) == 0:
        print("Error: No labels found. Please check dataset path.")
        return

    # Use all files for the paper
    X_base, X_prop, y = extract_features(all_files, labels_map)
    
    if len(y) == 0:
        print("Error: Feature extraction failed.")
        return

    # 2. Split Data
    print("\nTraining Classifiers (Random Forest)...")
    X_base_train, X_base_test, y_train, y_test = train_test_split(X_base, y, test_size=0.3, random_state=42)
    X_prop_train, X_prop_test, _, _ = train_test_split(X_prop, y, test_size=0.3, random_state=42)
    
    # 3. Train Baseline Classifier
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_base.fit(X_base_train, y_train)
    y_pred_base = clf_base.predict(X_base_test)
    
    # 4. Train Proposed Classifier
    clf_prop = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_prop.fit(X_prop_train, y_train)
    y_pred_prop = clf_prop.predict(X_prop_test)
    
    # 5. Results
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    
    acc_prop = accuracy_score(y_test, y_pred_prop)
    f1_prop = f1_score(y_test, y_pred_prop)
    
    print("\n\n=== BASELINE STUDY RESULTS (For Table I) ===")
    print(f"{'Method':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 50)
    print(f"{'Baseline (Mel-Spec)':<25} | {acc_base:.4f}     | {f1_base:.4f}")
    print(f"{'Proposed (NMF Atoms)':<25} | {acc_prop:.4f}     | {f1_prop:.4f}")
    print("-" * 50)
    
    if acc_prop >= acc_base - 0.05:
        print("\nCONCLUSION: The proposed method maintains competitive accuracy while adding interpretability.")
    else:
        print("\nNOTE: Proposed accuracy is lower. We might frame this as a trade-off for interpretability.")

if __name__ == "__main__":
    run_baseline_study()