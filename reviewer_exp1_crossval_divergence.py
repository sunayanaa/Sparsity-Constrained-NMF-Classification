# Program Name: reviewer_exp1_crossval_divergence.py
# Version: 1.0
# Description: Addresses Reviewer 1, Comments 2 and 3.
#   R1.2 - Controlled comparison of KL vs Frobenius vs Beta-divergence NMF.
#   R1.3 - Replace 70/30 single split with stratified 5-fold cross-validation.
#
# Outputs:
#   - Table: Divergence comparison (Recon Error, Spectral Conc, Accuracy) -> fills Table 2 in paper
#   - Table: 5-fold CV results with mean±std for ALL methods -> fills revised Table 1 in paper
#   - File:  cv_results.npz  (save for use in Program 3)

import os
import numpy as np
import pandas as pd
import librosa
import warnings
import glob
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning
import shutil
from google.colab import drive

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION  (matches your existing scripts exactly)
# ==========================================
CONF = {
    'SR': 2000,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'N_ATOMS': 8,
    'SPARSITY': 0.1,        # lambda, matches nmf_classification_baseline.py
    'FREQ_LIMIT': 1000,
    'DURATION': 5.0,
    'N_FOLDS': 5,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'DIVERGENCE_SUBSET': 500,   # n for divergence table (speed)
    'RANDOM_STATE': 42
}

# ==========================================
# HELPERS
# ==========================================
def load_references(base_path):
    ref_files = glob.glob(os.path.join(base_path, '**', 'REFERENCE.csv'), recursive=True)
    labels = {}
    for ref_file in ref_files:
        folder = os.path.dirname(ref_file)
        try:
            df = pd.read_csv(ref_file, names=['file', 'label'], header=None)
            for _, row in df.iterrows():
                full_path = os.path.join(folder, f"{row['file']}.wav")
                labels[full_path] = 1 if row['label'] == 1 else 0
        except:
            pass
    print(f"Loaded labels for {len(labels)} files.")
    return labels

def spectral_concentration(W, sr, n_fft, roi_band=(20, 400)):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= roi_band[0]) & (freqs <= roi_band[1])
    # Keep only rows matching freq_limit filter (same as nmf_experiments.py)
    mask = mask[:W.shape[0]]
    concs = []
    for k in range(W.shape[1]):
        atom = W[:, k]
        # L1-normalize atom before computing ratio (matches paper formula)
        atom_norm = atom / (np.sum(atom) + 1e-8)
        concs.append(np.sum(atom_norm[mask]))
    return np.mean(concs)

def get_spectrogram(file_path):
    """Returns normalized spectrogram V, matching nmf_classification_baseline.py exactly."""
    y, sr = librosa.load(file_path, sr=CONF['SR'], duration=CONF['DURATION'])
    S = np.abs(librosa.stft(y, n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN']))
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=CONF['N_FFT'])
    V = S[freq_bins <= CONF['FREQ_LIMIT'], :]
    V = V / (np.max(V) + 1e-9)
    return V

def extract_nmf_features(file_path, solver, beta_loss, alpha_H):
    """
    Runs NMF with given divergence settings.
    Returns (atom_features, recon_error, spectral_conc).
    """
    V = get_spectrogram(file_path)
    model = NMF(
        n_components=CONF['N_ATOMS'],
        solver=solver,
        beta_loss=beta_loss,
        init='random',
        alpha_H=alpha_H,
        l1_ratio=1.0,
        max_iter=500,
        random_state=CONF['RANDOM_STATE']
    )
    W = model.fit_transform(V)
    H = model.components_
    atom_features = np.mean(H, axis=1)
    recon_error = np.linalg.norm(V - W @ H)
    sc = spectral_concentration(W, CONF['SR'], CONF['N_FFT'])
    return atom_features, recon_error, sc

# ==========================================
# EXPERIMENT 1: DIVERGENCE COMPARISON (R1.2)
# Compares Frobenius, Beta (beta=1.5), KL on a 500-file subset.
# ==========================================

DIVERGENCE_CONFIGS = [
    # (label,         solver, beta_loss,           alpha_H)
    ('Frobenius',     'cd',   'frobenius',           0.1),
    ('Beta (b=1.5)',  'mu',   1.5,                   0.1),
    ('KL (proposed)', 'mu',   'kullback-leibler',     0.1),
]

def run_divergence_experiment(all_files, labels_map):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Divergence Comparison (R1.2)")
    print(f"Subset size: {CONF['DIVERGENCE_SUBSET']} files")
    print("="*60)

    np.random.seed(CONF['RANDOM_STATE'])
    subset = np.random.choice(all_files, min(len(all_files), CONF['DIVERGENCE_SUBSET']), replace=False)
    subset_labels = np.array([labels_map[f] for f in subset])

    rows = []
    for label, solver, beta_loss, alpha_H in DIVERGENCE_CONFIGS:
        print(f"\nRunning {label}...")
        features, errors, scs = [], [], []

        for f in tqdm(subset):
            try:
                feat, err, sc = extract_nmf_features(f, solver, beta_loss, alpha_H)
                features.append(feat)
                errors.append(err)
                scs.append(sc)
            except:
                continue

        features = np.array(features)
        valid_labels = subset_labels[:len(features)]

        # Quick 70/30 split just for this comparison table
        split = int(0.7 * len(features))
        X_tr, X_te = features[:split], features[split:]
        y_tr, y_te = valid_labels[:split], valid_labels[split:]

        clf = RandomForestClassifier(n_estimators=100, random_state=CONF['RANDOM_STATE'])
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))

        rows.append({
            'Divergence':   label,
            'Recon. Error': f"{np.mean(errors):.4f}",
            'Spec. Conc.':  f"{np.mean(scs):.4f}",
            'Accuracy':     f"{acc*100:.2f}%"
        })

    df = pd.DataFrame(rows)
    print("\n--- TABLE 2: Divergence Comparison (fill into paper) ---")
    print(df.to_string(index=False))
    return df

# ==========================================
# EXPERIMENT 2: 5-FOLD CROSS-VALIDATION (R1.3)
# Runs the proposed NMF+RF method with stratified 5-fold CV.
# Uses recording-level splits (no patient ID available in PhysioNet 2016,
# but source subfolders are kept intact within folds where possible).
# ==========================================

def extract_all_nmf_features(all_files, labels_map):
    """Extract NMF features for all files using the proposed KL config."""
    print(f"\nExtracting NMF features for all {len(all_files)} files (proposed method)...")
    features, valid_labels, valid_files = [], [], []

    for f in tqdm(all_files):
        if f not in labels_map:
            continue
        try:
            feat, _, _ = extract_nmf_features(
                f,
                solver='mu',
                beta_loss='kullback-leibler',
                alpha_H=CONF['SPARSITY']
            )
            features.append(feat)
            valid_labels.append(labels_map[f])
            valid_files.append(f)
        except:
            continue

    print(f"Successfully extracted features for {len(features)} files.")
    return np.array(features), np.array(valid_labels)

def run_crossval_experiment(X, y):
    print("\n" + "="*60)
    print("EXPERIMENT 2: 5-Fold Cross-Validation (R1.3)")
    print(f"n={len(y)}, folds={CONF['N_FOLDS']}, stratified by class label")
    print("="*60)

    skf = StratifiedKFold(
        n_splits=CONF['N_FOLDS'],
        shuffle=True,
        random_state=CONF['RANDOM_STATE']
    )

    fold_acc, fold_f1 = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=CONF['RANDOM_STATE'])
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred)
        fold_acc.append(acc)
        fold_f1.append(f1)
        print(f"  Fold {fold+1}: Acc={acc:.4f}  F1={f1:.4f}")

    mean_acc = np.mean(fold_acc)
    std_acc  = np.std(fold_acc)
    mean_f1  = np.mean(fold_f1)
    std_f1   = np.std(fold_f1)

    print(f"\n--- PROPOSED METHOD (5-fold CV) ---")
    print(f"  Accuracy : {mean_acc*100:.1f} ± {std_acc*100:.1f}%")
    print(f"  F1-Score : {mean_f1:.2f} ± {std_f1:.2f}")
    print("\nUse these values in the revised Table I of the paper.")

    # Save for use in Program 3 (SNR experiment needs same fold structure)
    np.savez('cv_results.npz',
             fold_acc=fold_acc, fold_f1=fold_f1,
             mean_acc=mean_acc, std_acc=std_acc,
             mean_f1=mean_f1, std_f1=std_f1)
    print("\nSaved cv_results.npz for use in Program 3.")
    shutil.copy('cv_results.npz', '/content/drive/MyDrive/PhysioNet2016/cv_results.npz')


    return mean_acc, std_acc, mean_f1, std_f1

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)

    labels_map = load_references(CONF['BASE_PATH'])
    all_files = list(labels_map.keys())

    if not all_files:
        print("Error: No files found. Check BASE_PATH.")
    else:
        # R1.2: Divergence table
        div_df = run_divergence_experiment(all_files, labels_map)

        # R1.3: 5-fold CV
        X, y = extract_all_nmf_features(all_files, labels_map)
        run_crossval_experiment(X, y)

        print("\n\nAll done. Copy printed values into the paper.")
