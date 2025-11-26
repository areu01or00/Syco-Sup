"""
Train Sycophancy Probes
=======================
Train linear probes for each extraction method (last_prompt, last_response, mean_response).

Requirements:
    pip install numpy scikit-learn

Inputs:
    - sycophancy_hidden_states.npz (from sycophancy_probe.py)
    - sycophancy_judged.csv (from judge_sycophancy.py)

Outputs:
    - sycophancy_probes.pkl (all probes for all extraction methods)
"""

import numpy as np
import csv
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from model_utils import get_model_key, get_output_dir

MODEL_KEY = get_model_key()
OUTPUT_DIR = get_output_dir(MODEL_KEY)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading hidden states and LLM judge labels...")

# Load hidden states
data = np.load(OUTPUT_DIR / "sycophancy_hidden_states.npz", allow_pickle=True)

# Load all extraction methods
EXTRACTION_METHODS = ['last_prompt', 'last_response', 'mean_response']

hidden_states = {}
for method in EXTRACTION_METHODS:
    hidden_states[f'neutral_{method}'] = data[f'neutral_{method}']
    hidden_states[f'biased_{method}'] = data[f'biased_{method}']

questions = data['questions']
num_layers = int(data['num_layers'])
hidden_dim = int(data['hidden_dim'])

print(f"Hidden states shape: {hidden_states['biased_last_prompt'].shape}")
print(f"Model: {num_layers} layers, hidden_dim={hidden_dim}")
print(f"Extraction methods: {EXTRACTION_METHODS}")

# Load LLM judge labels
with open(OUTPUT_DIR / "sycophancy_judged.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    judged = list(reader)

# Extract labels from LLM judge
y = np.array([1 if row['is_sycophantic'] == 'True' else 0 for row in judged])

print(f"\nTotal examples: {len(y)}")
print(f"Sycophantic (LLM judge): {sum(y)} ({100*sum(y)/len(y):.1f}%)")
print(f"Not sycophantic: {len(y) - sum(y)}")

# ============================================================
# 2. TRAIN PROBES FOR EACH EXTRACTION METHOD
# ============================================================

N_COMPONENTS = 64  # PCA dimensions

all_results = {}

for method in EXTRACTION_METHODS:
    print("\n" + "="*60)
    print(f"TRAINING PROBES: {method.upper()}")
    print("="*60)

    X_all = hidden_states[f'biased_{method}']  # [n_examples, num_layers+1, hidden_dim]

    layer_results = []

    for layer_idx in range(num_layers + 1):
        X = X_all[:, layer_idx, :]  # [n_examples, hidden_dim]

        # Normalize then PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train probe
        probe = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        probe.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, probe.predict(X_train))
        test_acc = accuracy_score(y_test, probe.predict(X_test))

        layer_results.append({
            'layer': layer_idx,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'probe': probe,
            'scaler': scaler,
            'pca': pca,
        })

        layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"
        print(f"  {layer_name}: train={train_acc:.2%}, test={test_acc:.2%}")

    # Find best layer for this method
    best_layer = max(layer_results, key=lambda x: x['test_acc'])
    print(f"\n*** BEST LAYER for {method}: {best_layer['layer']} (test acc: {best_layer['test_acc']:.2%}) ***")

    all_results[method] = {
        'layer_results': layer_results,
        'best_layer': best_layer['layer'],
        'best_test_acc': best_layer['test_acc'],
        'best_probe': best_layer['probe'],
        'best_scaler': best_layer['scaler'],
        'best_pca': best_layer['pca'],
    }

# ============================================================
# 3. COMPARE EXTRACTION METHODS
# ============================================================

print("\n" + "="*60)
print("COMPARISON OF EXTRACTION METHODS")
print("="*60)

print(f"\n{'Method':<20} {'Best Layer':<12} {'Test Acc':<12}")
print("-"*44)

best_method = None
best_acc = 0

for method in EXTRACTION_METHODS:
    r = all_results[method]
    print(f"{method:<20} {r['best_layer']:<12} {r['best_test_acc']:<12.2%}")
    if r['best_test_acc'] > best_acc:
        best_acc = r['best_test_acc']
        best_method = method

print("-"*44)
print(f"\n*** BEST OVERALL: {best_method} (layer {all_results[best_method]['best_layer']}, acc: {best_acc:.2%}) ***")

# ============================================================
# 4. SPARSE PROBE ON BEST METHOD
# ============================================================

print("\n" + "="*60)
print(f"SPARSE PROBE ANALYSIS ({best_method}, Layer {all_results[best_method]['best_layer']})")
print("="*60)

best_layer_idx = all_results[best_method]['best_layer']
X_best = hidden_states[f'biased_{best_method}'][:, best_layer_idx, :]
scaler_best = StandardScaler()
X_best_scaled = scaler_best.fit_transform(X_best)

# Train sparse probe directly on original features (no PCA) to find neurons
sparse_probe = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=2000, random_state=42)
sparse_probe.fit(X_best_scaled, y)

weights = sparse_probe.coef_[0]
n_nonzero = np.sum(np.abs(weights) > 1e-6)
print(f"Non-zero weights: {n_nonzero}/{hidden_dim} ({100*n_nonzero/hidden_dim:.1f}%)")

# Top neurons for sycophancy
top_positive = np.argsort(weights)[::-1][:10]
top_negative = np.argsort(weights)[:10]

print(f"\nTop 10 neurons PREDICTING sycophancy:")
for i, idx in enumerate(top_positive):
    if weights[idx] > 0:
        print(f"  {i+1}. Neuron {idx}: weight = {weights[idx]:.4f}")

print(f"\nTop 10 neurons AGAINST sycophancy:")
for i, idx in enumerate(top_negative):
    if weights[idx] < 0:
        print(f"  {i+1}. Neuron {idx}: weight = {weights[idx]:.4f}")

# ============================================================
# 5. SAVE ALL PROBES
# ============================================================

print("\n" + "="*60)
print("SAVING PROBES")
print("="*60)

probe_path = OUTPUT_DIR / "sycophancy_probes.pkl"
with open(probe_path, 'wb') as f:
    pickle.dump({
        'all_results': all_results,
        'best_method': best_method,
        'best_layer': all_results[best_method]['best_layer'],
        'best_test_acc': best_acc,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'extraction_methods': EXTRACTION_METHODS,
        'sparse_probe': sparse_probe,
        'sparse_scaler': scaler_best,
        'top_neurons_positive': top_positive.tolist(),
        'top_neurons_negative': top_negative.tolist(),
    }, f)
print(f"All probes saved to {probe_path}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
print("\nNext step: python test_probe_on_eval.py")
