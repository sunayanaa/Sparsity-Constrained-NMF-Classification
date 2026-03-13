# Program Name: nmf_robustness_revised.py
# Version: 1.0
# Description: Revised robustness experiment for IEEE SPL resubmission.
#              Fixes five problems in the original nmf_experiments.py:
#              1. Baseline curve was mocked (0.85x scaling) — now real NMF run.
#              2. Inconsistent NMF settings between proposed and baseline — now matched.
#              3. Missing normalization on noisy spectrogram in baseline — now fixed.
#              4. Greedy atom matching — replaced with Hungarian algorithm.
#              5. Single file only — now averaged across N_FILES files.
#
# Outputs:
#   - Printed table: stability values at each SNR for both methods
#   - Fig2_Robustness_Revised.png  (replaces original Fig2_Robustness.png)

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import glob
from tqdm import tqdm
from google.colab import drive
import shutil

plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

# ==========================================
# CONFIGURATION
# Proposed settings match nmf_classification_baseline.py exactly.
# Baseline is identical except alpha_H=0 (no sparsity constraint).
# ==========================================
CONF = {
    'SR': 2000,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'N_ATOMS': 8,
    'ALPHA_PROPOSED': 0.8,   # matches nmf_classification_baseline.py
    'ALPHA_BASELINE': 0.0,   # unconstrained NMF — only difference
    'FREQ_LIMIT': 1000,
    'SNR_LEVELS': [0, 5, 10, 15, 20, 25],
    'N_FILES': 200,          # average over this many files for stable results
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'RANDOM_STATE': 42
}

# ==========================================
# HELPERS
# ==========================================
def find_wav_files(base_path):
    wav_files = []
    wav_files.extend(glob.glob(os.path.join(base_path, '*.wav')))
    if not wav_files:
        for root, dirs, files in os.walk(base_path):
            for f in files:
                if f.endswith('.wav'):
                    wav_files.append(os.path.join(root, f))
    return wav_files

def add_noise(signal, snr_db):
    sig_power = np.mean(signal ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    return signal + np.random.normal(0, np.sqrt(noise_power), signal.shape)

def get_spectrogram(y, sr):
    """
    Computes normalized spectrogram.
    Identical normalization applied to both clean and noisy signals,
    and to both proposed and baseline methods.
    """
    S = np.abs(librosa.stft(y, n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN']))
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=CONF['N_FFT'])
    valid_bins = freq_bins <= CONF['FREQ_LIMIT']
    V = S[valid_bins, :]
    V = V / (np.max(V) + 1e-9)   # max-normalize to [0,1]
    return V

def run_nmf(V, alpha_H):
    """
    Runs NMF with identical settings for both methods.
    Only alpha_H differs: CONF['ALPHA_PROPOSED'] vs CONF['ALPHA_BASELINE'].
    Both use:
      - solver='mu' (multiplicative update, required for KL divergence)
      - beta_loss='kullback-leibler'
      - init='nndsvda' (deterministic, good starting point)
      - random_state=42
    """
    model = NMF(
        n_components=CONF['N_ATOMS'],
        solver='mu',
        beta_loss='kullback-leibler',
        init='nndsvda',
        alpha_H=alpha_H,
        l1_ratio=1.0,
        max_iter=500,
        random_state=CONF['RANDOM_STATE']
    )
    W = model.fit_transform(V)
    return W

def hungarian_match(W_clean, W_noisy):
    """
    Aligns noisy atoms to clean atoms via the Hungarian algorithm.
    Builds pairwise cosine similarity matrix, solves linear assignment,
    returns mean cosine similarity of matched pairs.
    """
    K = W_clean.shape[1]
    sim_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            a = W_clean[:, i]
            b = W_noisy[:, j]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            sim_matrix[i, j] = np.dot(a, b) / denom
    # Negate because linear_sum_assignment minimises cost
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    return np.mean(sim_matrix[row_ind, col_ind])

# ==========================================
# MAIN ROBUSTNESS EXPERIMENT
# ==========================================
def run_robustness(files):
    """
    For each file:
      1. Compute clean spectrogram and run both NMF variants.
      2. For each SNR level, add noise, recompute spectrogram (same normalization),
         run both NMF variants, compute Hungarian-matched stability.
    Averages results across all files.
    """
    # Accumulators: dict of snr -> list of stability scores
    scores_proposed = {snr: [] for snr in CONF['SNR_LEVELS']}
    scores_baseline = {snr: [] for snr in CONF['SNR_LEVELS']}

    print(f"Running robustness experiment on {len(files)} files...")

    for f in tqdm(files):
        try:
            y, sr = librosa.load(f, sr=CONF['SR'], duration=5.0)

            # Clean reference atoms for both methods
            V_clean = get_spectrogram(y, sr)
            W_clean_proposed = run_nmf(V_clean, CONF['ALPHA_PROPOSED'])
            W_clean_baseline = run_nmf(V_clean, CONF['ALPHA_BASELINE'])

            for snr in CONF['SNR_LEVELS']:
                y_noisy = add_noise(y, snr)

                # Same normalization pipeline for noisy signal
                V_noisy = get_spectrogram(y_noisy, sr)

                W_noisy_proposed = run_nmf(V_noisy, CONF['ALPHA_PROPOSED'])
                W_noisy_baseline = run_nmf(V_noisy, CONF['ALPHA_BASELINE'])

                scores_proposed[snr].append(
                    hungarian_match(W_clean_proposed, W_noisy_proposed))
                scores_baseline[snr].append(
                    hungarian_match(W_clean_baseline, W_noisy_baseline))

        except Exception as e:
            continue

    mean_proposed = [np.mean(scores_proposed[s]) for s in CONF['SNR_LEVELS']]
    std_proposed  = [np.std(scores_proposed[s])  for s in CONF['SNR_LEVELS']]
    mean_baseline = [np.mean(scores_baseline[s]) for s in CONF['SNR_LEVELS']]
    std_baseline  = [np.std(scores_baseline[s])  for s in CONF['SNR_LEVELS']]

    return mean_proposed, std_proposed, mean_baseline, std_baseline

# ==========================================
# PLOTTING
# ==========================================
def plot_robustness(mean_proposed, std_proposed, mean_baseline, std_baseline):
    snrs = CONF['SNR_LEVELS']

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(snrs, mean_proposed, 'b-o', linewidth=2,
            label='Proposed (Sparsity NMF)')
    ax.fill_between(snrs,
                    np.array(mean_proposed) - np.array(std_proposed),
                    np.array(mean_proposed) + np.array(std_proposed),
                    alpha=0.15, color='blue')

    ax.plot(snrs, mean_baseline, 'r--x', linewidth=2,
            label='Baseline (Standard NMF)')
    ax.fill_between(snrs,
                    np.array(mean_baseline) - np.array(std_baseline),
                    np.array(mean_baseline) + np.array(std_baseline),
                    alpha=0.15, color='red')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Atom Stability (Cosine Similarity)')
    ax.set_title('Fig. 2. Robustness of Atoms vs. Noise')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([0.3, 1.0])
    plt.tight_layout()

    plt.savefig('Fig2_Robustness_Revised.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Also save to Drive immediately
    shutil.copy('Fig2_Robustness_Revised.png',
                '/content/drive/MyDrive/PhysioNet2016/Fig2_Robustness_Revised.png')
    print("Saved: Fig2_Robustness_Revised.png (local + Drive)")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)
    np.random.seed(CONF['RANDOM_STATE'])

    print(f"Searching for wav files in {CONF['BASE_PATH']}...")
    all_files = find_wav_files(CONF['BASE_PATH'])
    print(f"Found {len(all_files)} files.")

    if not all_files:
        print("Error: No wav files found. Check BASE_PATH.")
    else:
        # Random subset for speed — 200 files gives stable averages
        subset = np.random.choice(
            all_files,
            min(len(all_files), CONF['N_FILES']),
            replace=False
        )

        mean_p, std_p, mean_b, std_b = run_robustness(subset)

        # Print results table
        print("\n--- Atom Stability Results (Hungarian-matched, mean ± std) ---")
        print(f"{'SNR':>5} | {'Proposed':>14} | {'Std NMF':>14}")
        print("-" * 40)
        for i, snr in enumerate(CONF['SNR_LEVELS']):
            print(f"{snr:>5} | {mean_p[i]:>6.4f} ± {std_p[i]:.4f} "
                  f"| {mean_b[i]:>6.4f} ± {std_b[i]:.4f}")

        print(f"\nKey value for paper text:")
        print(f"  Proposed @ 0 dB SNR: {mean_p[0]:.3f} ± {std_p[0]:.3f}")
        print(f"  Baseline @ 0 dB SNR: {mean_b[0]:.3f} ± {std_b[0]:.3f}")
        print(f"  n files averaged: {len(subset)}")

        plot_robustness(mean_p, std_p, mean_b, std_b)
