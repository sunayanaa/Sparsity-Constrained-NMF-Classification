# Program Name: reviewer_exp3_sparsity_distribution.py
# Version: 1.0
# Description: Addresses Reviewer 1, Comment 6.
#   R1.6 - Ensure identical max-normalization for both NMF and Grad-CAM heatmaps
#           before computing Hoyer sparsity.
#         - Report distribution of Map Sparsity across all test recordings,
#           not just a single example.
#         - Produces Fig_Sparsity_Dist.png (box + violin plot) for the paper.
#
# Prerequisites:
#   - PhysioNet 2016 data at CONF['BASE_PATH']
#   - Trains a fresh CNN internally (matches cnn_rebuild_and_gradcam.py architecture)

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import warnings
import glob
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from google.colab import drive

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

# ==========================================
# CONFIGURATION
# ==========================================
CONF = {
    'SR': 2000,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'N_ATOMS': 8,
    'SPARSITY': 0.1,
    'FREQ_LIMIT': 1000,
    'DURATION': 5.0,
    'N_MELS': 64,
    'SUBSET': 600,          # Files to use for distribution (balance speed vs stats)
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
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
                labels[os.path.join(folder, f"{row['file']}.wav")] = 1 if row['label'] == 1 else 0
        except:
            pass
    return labels

def hoyer_sparsity(x):
    """
    Hoyer sparsity on a NORMALISED map.
    Input x should already be in [0,1] (max-normalised).
    Flattens to 1D before computing.
    """
    x = x.flatten().astype(float)
    n = len(x)
    norm2 = np.linalg.norm(x, 2)
    if norm2 == 0:
        return 0.0
    norm1 = np.linalg.norm(x, 1)
    return (np.sqrt(n) - (norm1 / (norm2 + 1e-9))) / (np.sqrt(n) - 1 + 1e-9)

def normalise_map(M):
    """
    Identical max-normalisation applied to BOTH NMF and Grad-CAM maps.
    Clips negatives first (Grad-CAM uses ReLU so should be non-negative already).
    """
    M = np.maximum(M, 0)
    m = np.max(M)
    if m < 1e-9:
        return M
    return M / m

# ==========================================
# NMF HEATMAP SPARSITY
# ==========================================
def nmf_map_sparsity(y):
    """
    Builds the diagnostic heatmap M = sum_k w_k (W[:,k] * H[k,:])
    using the same formula as the paper (Eq. 4), then max-normalises
    and computes Hoyer sparsity.
    For the sparsity comparison we weight all atoms equally (w_k = 1/K)
    since we don't have a trained RF available here.
    The paper result used RF importance weights; for the distribution
    plot we use uniform weights which is slightly more conservative.
    """
    S = np.abs(librosa.stft(y, n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN']))
    freq_bins = librosa.fft_frequencies(sr=CONF['SR'], n_fft=CONF['N_FFT'])
    V = S[freq_bins <= CONF['FREQ_LIMIT'], :]
    V = V / (np.max(V) + 1e-9)

    model = NMF(
        n_components=CONF['N_ATOMS'], solver='mu',
        beta_loss='kullback-leibler', init='random',
        alpha_H=CONF['SPARSITY'], l1_ratio=1.0,
        max_iter=500, random_state=CONF['RANDOM_STATE']
    )
    W = model.fit_transform(V)
    H = model.components_

    # Uniform atom weights (conservative; paper uses RF importance)
    w = np.ones(CONF['N_ATOMS']) / CONF['N_ATOMS']
    M = sum(w[k] * np.outer(W[:, k], H[k, :]) for k in range(CONF['N_ATOMS']))

    M_norm = normalise_map(M)
    return hoyer_sparsity(M_norm)

# ==========================================
# CNN + GRAD-CAM HEATMAP SPARSITY
# ==========================================
def get_mel(y):
    target_len = int(CONF['DURATION'] * CONF['SR'])
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(
        y=y, sr=CONF['SR'], n_mels=CONF['N_MELS'],
        n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN'])
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    expected = int(np.ceil((CONF['DURATION'] * CONF['SR']) / CONF['HOP_LEN']))
    if mel_norm.shape[1] > expected:
        mel_norm = mel_norm[:, :expected]
    elif mel_norm.shape[1] < expected:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, expected - mel_norm.shape[1])))
    return mel_norm

def build_cnn(input_shape):
    """Matches cnn_rebuild_and_gradcam.py Functional API architecture."""
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_1')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_last')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def make_gradcam(img_array, model, last_conv_layer_name='conv2d_last'):
    """
    Generates Grad-CAM heatmap, upsampled to input size.
    Applies ReLU and max-normalisation (identical to NMF path).
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # Upsample to input spectrogram size
    h_in, w_in = img_array.shape[1], img_array.shape[2]
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], [h_in, w_in]
    ).numpy().squeeze()

    # *** Identical normalisation as NMF path ***
    heatmap_norm = normalise_map(heatmap_resized)
    return hoyer_sparsity(heatmap_norm)

# ==========================================
# MAIN EXPERIMENT
# ==========================================
def run_sparsity_distribution(subset_files, labels_map):
    print("\n" + "="*60)
    print("EXPERIMENT: Map Sparsity Distribution (R1.6)")
    print(f"n={len(subset_files)} files")
    print("="*60)

    # --- Step 1: Train CNN on the subset ---
    print("\nExtracting mel spectrograms for CNN training...")
    X_mel, y_all = [], []
    for f in tqdm(subset_files):
        if f not in labels_map:
            continue
        try:
            y, _ = librosa.load(f, sr=CONF['SR'], duration=CONF['DURATION'])
            X_mel.append(get_mel(y)[..., np.newaxis])
            y_all.append(labels_map[f])
        except:
            continue

    X_mel = np.array(X_mel)
    y_all = np.array(y_all)

    X_tr, X_te, y_tr, y_te, tr_files, te_files = train_test_split(
        X_mel, y_all, subset_files[:len(y_all)],
        test_size=0.3, random_state=CONF['RANDOM_STATE'], stratify=y_all
    )

    print(f"Training CNN on {len(X_tr)} samples...")
    cnn_model = build_cnn(input_shape=X_tr.shape[1:])
    cnn_model.fit(X_tr, y_tr, epochs=10, batch_size=32,
                  validation_split=0.1, verbose=0)
    print("CNN trained.")

    # --- Step 2: Compute sparsity for each test recording ---
    print(f"\nComputing Map Sparsity for {len(te_files)} test recordings...")
    nmf_sparsities = []
    gradcam_sparsities = []

    for i, f in enumerate(tqdm(te_files)):
        try:
            y, _ = librosa.load(f, sr=CONF['SR'], duration=CONF['DURATION'])

            # NMF sparsity
            sp_nmf = nmf_map_sparsity(y)
            nmf_sparsities.append(sp_nmf)

            # Grad-CAM sparsity (same normalisation)
            img = X_te[i][np.newaxis]   # shape (1, H, W, 1)
            sp_gc = make_gradcam(img, cnn_model)
            gradcam_sparsities.append(sp_gc)

        except:
            continue

    nmf_sparsities = np.array(nmf_sparsities)
    gradcam_sparsities = np.array(gradcam_sparsities)

    # --- Summary statistics ---
    print("\n--- Map Sparsity Distribution ---")
    print(f"{'Metric':<20} | {'NMF (Proposed)':>16} | {'Grad-CAM (CNN)':>16}")
    print("-" * 58)
    for stat_name, stat_fn in [('Median', np.median), ('Mean', np.mean),
                                ('Std Dev', np.std), ('Min', np.min), ('Max', np.max)]:
        print(f"{stat_name:<20} | {stat_fn(nmf_sparsities):>16.4f} | "
              f"{stat_fn(gradcam_sparsities):>16.4f}")
    print(f"\nTest set size: n={len(nmf_sparsities)}")
    print("Use median values as the representative figures in Table I.")

    return nmf_sparsities, gradcam_sparsities

def plot_sparsity_distribution(nmf_sparsities, gradcam_sparsities):
    fig, ax = plt.subplots(figsize=(5, 4))

    data = [gradcam_sparsities, nmf_sparsities]
    labels = ['CNN\n(Grad-CAM)', 'Proposed\n(NMF)']
    colors = ['#FF8C00', '#1E90FF']   # Orange, Blue — matches paper description

    parts = ax.violinplot(data, positions=[1, 2], widths=0.6,
                          showmedians=True, showextrema=True)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Overlay box plot for quartiles
    ax.boxplot(data, positions=[1, 2], widths=0.2,
               patch_artist=False, manage_ticks=False,
               medianprops=dict(color='black', linewidth=2),
               whiskerprops=dict(linestyle='--'))

    # Annotate medians
    for pos, vals in zip([1, 2], data):
        ax.text(pos, np.median(vals) + 0.02, f'{np.median(vals):.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Map Sparsity (Hoyer)')
    ax.set_title('Distribution of Map Sparsity\nacross Test Recordings')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig('Fig_Sparsity_Dist.png', dpi=300, bbox_inches='tight')
    plt.savefig('/content/drive/MyDrive/PhysioNet2016/Fig_Sparsity_Dist.png', dpi=300, bbox_inches='tight')

    plt.show()
    print("\nSaved: Fig_Sparsity_Dist.png")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)
    np.random.seed(CONF['RANDOM_STATE'])

    labels_map = load_references(CONF['BASE_PATH'])
    all_files = list(labels_map.keys())

    if not all_files:
        print("Error: No files found. Check BASE_PATH.")
    else:
        subset_files = np.random.choice(
            all_files, min(len(all_files), CONF['SUBSET']), replace=False
        ).tolist()

        nmf_sp, gc_sp = run_sparsity_distribution(subset_files, labels_map)
        plot_sparsity_distribution(nmf_sp, gc_sp)

        print("\n\nKey values for paper text (Section IV.D):")
        print(f"  NMF median Map Sparsity    : {np.median(nmf_sp):.3f}")
        print(f"  Grad-CAM median Map Sparsity: {np.median(gc_sp):.3f}")
        print(f"  Test set size n            : {len(nmf_sp)}")
