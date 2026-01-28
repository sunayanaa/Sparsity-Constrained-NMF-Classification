# Program Name: nmf_ablation_study.py
# Version: 1.0
# Description: Conducts an Ablation Study on NMF Rank (K).
#              - Varies K from 4 to 16.
#              - Measures Trade-off: Reconstruction Error vs. Sparsity.
#              - Generates the 'Elbow Plot' for the paper.

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.decomposition import NMF
import glob
from tqdm import tqdm

from google.colab import drive

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONF = {
    'SR': 2000,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'K_VALUES': [4, 6, 8, 10, 12, 16], # The Ranks to test
    'SPARSITY': 0.1,
    'FREQ_LIMIT': 1000,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'SUBSET_SIZE': 200 # Number of files to test per K (for speed)
}

# ==========================================
# 2. METRICS
# ==========================================
def hoyer_sparsity(x):
    n = len(x)
    norm2 = np.linalg.norm(x, 2)
    norm1 = np.linalg.norm(x, 1)
    if norm2 == 0: return 0.0
    num = np.sqrt(n) - (norm1 / (norm2 + 1e-9))
    den = np.sqrt(n) - 1
    return num / den

# ==========================================
# 3. ABLATION LOOP
# ==========================================
def run_ablation():
    # Find files
    files = glob.glob(os.path.join(CONF['BASE_PATH'], '**', '*.wav'), recursive=True)
    if not files:
        print("Error: No files found.")
        return
    
    # Use a fixed random subset for fair comparison
    np.random.seed(42)
    subset_files = np.random.choice(files, min(len(files), CONF['SUBSET_SIZE']), replace=False)
    
    results_mse = []
    results_sparsity = []

    print(f"--- Running Ablation on K={CONF['K_VALUES']} (n={len(subset_files)}) ---")

    for k in CONF['K_VALUES']:
        errors = []
        sparsities = []
        
        print(f"Testing Rank K={k}...")
        
        for f in tqdm(subset_files, leave=False):
            try:
                y, sr = librosa.load(f, sr=CONF['SR'], duration=5.0)
                S = np.abs(librosa.stft(y, n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN']))
                freq_bins = librosa.fft_frequencies(sr=sr, n_fft=CONF['N_FFT'])
                V = S[freq_bins <= CONF['FREQ_LIMIT'], :]
                
                # Normalize
                V = V / (np.max(V) + 1e-9)
                
                # NMF with current K
                model = NMF(n_components=k, solver='mu', beta_loss='kullback-leibler', 
                           init='random', alpha_H=CONF['SPARSITY'], max_iter=200, random_state=42)
                
                W = model.fit_transform(V)
                H = model.components_
                
                # Metric 1: Reconstruction Error
                err = np.linalg.norm(V - W @ H)
                errors.append(err)
                
                # Metric 2: Avg Atom Sparsity
                sp = np.mean([hoyer_sparsity(H[i, :]) for i in range(k)])
                sparsities.append(sp)
                
            except:
                continue
        
        results_mse.append(np.mean(errors))
        results_sparsity.append(np.mean(sparsities))

    # ==========================================
    # 4. PLOTTING THE ELBOW
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(7, 4))

    color = 'tab:red'
    ax1.set_xlabel('Number of Atoms (K)')
    ax1.set_ylabel('Reconstruction Error', color=color)
    ax1.plot(CONF['K_VALUES'], results_mse, 'o--', color=color, linewidth=2, label='Error (MSE)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Avg Sparsity Index', color=color)  # we already handled the x-label with ax1
    ax2.plot(CONF['K_VALUES'], results_sparsity, 's-', color=color, linewidth=2, label='Sparsity')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Fig. 4. Ablation Study: Selecting Optimal Rank K')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    print("\n--- ABLATION RESULTS ---")
    print(f"{'K':<5} | {'Error':<10} | {'Sparsity':<10}")
    for i, k in enumerate(CONF['K_VALUES']):
        print(f"{k:<5} | {results_mse[i]:.4f}     | {results_sparsity[i]:.4f}")
    
    print("\nInterpretation: We look for the 'Elbow' where Error drops but Sparsity is still high.")

if __name__ == "__main__":
    run_ablation()