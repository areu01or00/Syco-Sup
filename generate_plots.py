"""
Generate Plots for Blog Post
============================
Clean, publication-ready visualizations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os
from PIL import Image
from scipy.ndimage import gaussian_filter

from model_utils import get_model_key, get_output_dir

MODEL_KEY = get_model_key()
OUTPUT_DIR = get_output_dir(MODEL_KEY)
PLOTS_DIR = OUTPUT_DIR / "plots"

# Standard output size
STANDARD_WIDTH = 1200
STANDARD_HEIGHT = 800

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

with open(OUTPUT_DIR / "sycophancy_probes.pkl", "rb") as f:
    probe_data = pickle.load(f)

steering_results = {}
with open(OUTPUT_DIR / "intervention_results.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        alpha = int(float(row['alpha']))
        if alpha not in steering_results:
            steering_results[alpha] = {'correct': 0, 'total': 0}
        steering_results[alpha]['total'] += 1
        if row['is_correct'] == 'True':
            steering_results[alpha]['correct'] += 1

num_layers = probe_data['num_layers']
methods = ['last_prompt', 'last_response', 'mean_response']
method_labels = ['Last Prompt', 'Last Response', 'Mean Response']

def save_fig(fig, filename):
    """Save figure at standard dimensions."""
    temp_file = filename + '.tmp.png'
    fig.savefig(temp_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='white', pad_inches=0.1)
    plt.close(fig)

    img = Image.open(temp_file)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    canvas = Image.new('RGB', (STANDARD_WIDTH, STANDARD_HEIGHT), (255, 255, 255))

    img_ratio = img.width / img.height
    canvas_ratio = STANDARD_WIDTH / STANDARD_HEIGHT

    if img_ratio > canvas_ratio:
        new_width = STANDARD_WIDTH - 40
        new_height = int(new_width / img_ratio)
    else:
        new_height = STANDARD_HEIGHT - 40
        new_width = int(new_height * img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)
    x = (STANDARD_WIDTH - new_width) // 2
    y = (STANDARD_HEIGHT - new_height) // 2
    canvas.paste(img, (x, y))

    canvas.save(filename, 'PNG')
    os.remove(temp_file)
    print(f"  Saved: {filename}")

# ============================================================
# PLOT 1: SMOOTH LAYER HEATMAP (like the reference image)
# ============================================================

print("1. Smooth layer heatmap...")

fig, ax = plt.subplots(figsize=(12, 6))

# Build 2D data: rows = methods (0,1,2), cols = layers (0-28)
data = np.zeros((len(methods), num_layers + 1))
for i, method in enumerate(methods):
    for r in probe_data['all_results'][method]['layer_results']:
        data[i, r['layer']] = r['test_acc'] * 100

# Use scipy.ndimage.zoom for smooth upscaling
from scipy.ndimage import zoom

# Upsample: 3 methods -> 60 rows, 29 layers -> 290 cols
data_smooth = zoom(data, (20, 10), order=3)

# Apply slight gaussian blur for extra smoothness
data_smooth = gaussian_filter(data_smooth, sigma=2)

# Plot with jet colormap like reference
# Dynamic color range based on data
data_min = max(45, data.min() - 5)
data_max = min(100, data.max() + 5)
im = ax.imshow(data_smooth, aspect='auto', cmap='jet',
               extent=[0, num_layers, len(methods) - 0.5, -0.5],
               vmin=data_min, vmax=data_max)

ax.set_xlabel('Layer', fontsize=14)
ax.set_ylabel('Extraction Method', fontsize=14)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(method_labels, fontsize=12)
ax.set_xticks(np.arange(0, num_layers + 1, 4))

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Test Accuracy (%)', fontsize=12)

ax.set_title('Probe Accuracy by Layer and Extraction Method', fontsize=14, fontweight='bold')

save_fig(fig, str(PLOTS_DIR / '01_layer_heatmap.png'))

# ============================================================
# PLOT 2: LINE PLOT - Clean and simple
# ============================================================

print("2. Layer accuracy lines...")

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#d62728', '#1f77b4', '#2ca02c']
markers = ['o', 's', '^']

for method, label, color, marker in zip(methods, method_labels, colors, markers):
    layer_results = probe_data['all_results'][method]['layer_results']
    layers = [r['layer'] for r in layer_results]
    test_accs = [r['test_acc'] * 100 for r in layer_results]

    ax.plot(layers, test_accs, '-', color=color, linewidth=2.5,
            marker=marker, markersize=6, label=label)

    # Mark best point
    best_idx = np.argmax(test_accs)
    ax.scatter([layers[best_idx]], [test_accs[best_idx]],
               color=color, s=150, zorder=5, edgecolors='black', linewidth=2)
    ax.annotate(f'{test_accs[best_idx]:.1f}%',
                xy=(layers[best_idx], test_accs[best_idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
ax.set_xlabel('Layer', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Probe Accuracy Across Layers', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(-0.5, num_layers + 0.5)
# Dynamic y-axis based on actual data
all_test_accs = []
for method in methods:
    all_test_accs.extend([r['test_acc'] * 100 for r in probe_data['all_results'][method]['layer_results']])
y_min = max(45, min(all_test_accs) - 5)
y_max = min(100, max(all_test_accs) + 5)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)

save_fig(fig, str(PLOTS_DIR / '02_layer_lines.png'))

# ============================================================
# PLOT 3: STEERING CURVE
# ============================================================

print("3. Steering curve...")

fig, ax = plt.subplots(figsize=(10, 6))

alphas = sorted(steering_results.keys())
accuracies = [steering_results[a]['correct'] / steering_results[a]['total'] * 100 for a in alphas]

ax.plot(alphas, accuracies, '-o', color='#8e44ad', linewidth=3, markersize=12,
        markerfacecolor='white', markeredgewidth=2.5)

# Annotate each point (skip if too crowded)
for i, (a, acc) in enumerate(zip(alphas, accuracies)):
    # Only annotate if not overlapping with previous
    if i == 0 or abs(acc - accuracies[i-1]) > 3:
        offset = 3
        ax.text(a, acc + offset, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Mark optimal
best_idx = np.argmax(accuracies)
ax.scatter([alphas[best_idx]], [accuracies[best_idx]],
           color='#27ae60', s=300, zorder=5, edgecolors='black', linewidth=2, marker='*')

# Baseline
ax.axhline(y=accuracies[0], color='red', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Steering Strength (α)', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Intervention: Steering Curve', fontsize=14, fontweight='bold')
max_alpha = max(alphas)
ax.set_xlim(-5, max_alpha + 10)
ax.set_ylim(0, max(max(accuracies) + 10, 80))
ax.grid(True, alpha=0.3)

# Add baseline label inside plot area
ax.text(max_alpha * 0.7, accuracies[0] + 2, 'Baseline', fontsize=10, color='red')

save_fig(fig, str(PLOTS_DIR / '03_steering_curve.png'))

# ============================================================
# PLOT 4: BEFORE/AFTER BAR
# ============================================================

print("4. Before/after intervention...")

fig, ax = plt.subplots(figsize=(8, 6))

# Calculate actual values from data
alphas = sorted(steering_results.keys())
accuracies = [steering_results[a]['correct'] / steering_results[a]['total'] * 100 for a in alphas]
baseline_acc = accuracies[0]
best_idx = np.argmax(accuracies)
best_alpha = alphas[best_idx]
best_acc = accuracies[best_idx]
improvement = best_acc - baseline_acc

categories = [f'Baseline (α=0)', f'Optimal (α={int(best_alpha)})']
values = [baseline_acc, best_acc]
colors = ['#e74c3c', '#27ae60']

bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2, width=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=16, fontweight='bold')

# Arrow showing improvement
arrow_y_start = baseline_acc + 4
arrow_y_end = best_acc - 4
ax.annotate('', xy=(1, arrow_y_end), xytext=(0, arrow_y_start),
            arrowprops=dict(arrowstyle='->', color='black', lw=3))
ax.text(0.5, (baseline_acc + best_acc) / 2, f'+{improvement:.1f}%', ha='center', fontsize=20, fontweight='bold', color='#27ae60')

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Sycophancy Reduction via Steering', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(best_acc + 15, 70))

save_fig(fig, str(PLOTS_DIR / '04_intervention_bars.png'))

# ============================================================
# PLOT 5: DETECTION COMPARISON BAR
# ============================================================

print("5. Detection comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

best_accs = [probe_data['all_results'][m]['best_test_acc'] * 100 for m in methods]
best_layers = [probe_data['all_results'][m]['best_layer'] for m in methods]
colors = ['#d62728', '#1f77b4', '#2ca02c']

x = np.arange(len(methods))
bars = ax.bar(x, best_accs, color=colors, edgecolor='black', linewidth=2, width=0.6)

for bar, acc, layer in zip(bars, best_accs, best_layers):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%\n(Layer {layer})', ha='center', fontsize=12, fontweight='bold')

ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2.4, 51, 'Random', fontsize=11, color='gray')

ax.set_xticks(x)
ax.set_xticklabels(method_labels, fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Best Probe Accuracy by Extraction Method', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(best_accs) + 10)

save_fig(fig, str(PLOTS_DIR / '05_detection_bars.png'))

# ============================================================
# PLOT 6: TRAIN VS TEST GAP
# ============================================================

print("6. Train vs test...")

fig, ax = plt.subplots(figsize=(10, 6))

layer_results = probe_data['all_results']['mean_response']['layer_results']
layers = [r['layer'] for r in layer_results]
train_accs = [r['train_acc'] * 100 for r in layer_results]
test_accs = [r['test_acc'] * 100 for r in layer_results]

ax.plot(layers, train_accs, '-o', color='#1f77b4', linewidth=2.5, markersize=5, label='Train')
ax.plot(layers, test_accs, '-s', color='#d62728', linewidth=2.5, markersize=5, label='Test')
ax.fill_between(layers, train_accs, test_accs, alpha=0.2, color='orange')

ax.set_xlabel('Layer', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Train vs Test Accuracy (Mean Response)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(-0.5, num_layers + 0.5)
ax.grid(True, alpha=0.3)

save_fig(fig, str(PLOTS_DIR / '06_train_vs_test.png'))

# ============================================================
# PLOT 7: SPARSE PROBE NEURONS
# ============================================================

print("7. Top neurons...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

top_positive = probe_data['top_neurons_positive'][:10]
top_negative = probe_data['top_neurons_negative'][:10]
weights = probe_data['sparse_probe'].coef_[0]

# Pro-sycophancy (positive weights)
pos_weights = [weights[n] for n in top_positive]
ax1.barh(range(len(top_positive)), pos_weights, color='#d62728', edgecolor='black', height=0.7)
ax1.set_yticks(range(len(top_positive)))
ax1.set_yticklabels([f'Neuron {n}' for n in top_positive], fontsize=10)
ax1.set_xlabel('Weight', fontsize=12)
ax1.set_title('Pro-Sycophancy Neurons', fontsize=13, fontweight='bold')
ax1.invert_yaxis()

# Anti-sycophancy (negative weights)
neg_weights = [weights[n] for n in top_negative]
ax2.barh(range(len(top_negative)), neg_weights, color='#2ca02c', edgecolor='black', height=0.7)
ax2.set_yticks(range(len(top_negative)))
ax2.set_yticklabels([f'Neuron {n}' for n in top_negative], fontsize=10)
ax2.set_xlabel('Weight', fontsize=12)
ax2.set_title('Anti-Sycophancy Neurons', fontsize=13, fontweight='bold')
ax2.invert_yaxis()

fig.suptitle(f'Sparse Probe: Top 10 Neurons (Layer {probe_data["best_layer"]})',
             fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '07_neurons.png'))

# ============================================================
# PLOT 8: SEPARATE HEATMAPS FOR EACH METHOD
# ============================================================

print("8. Individual method heatmaps...")

fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True)

# Get overall min/max for consistent coloring
all_accs_heatmap = []
for method in methods:
    all_accs_heatmap.extend([r['test_acc'] * 100 for r in probe_data['all_results'][method]['layer_results']])
heatmap_vmin = max(45, min(all_accs_heatmap) - 5)
heatmap_vmax = min(100, max(all_accs_heatmap) + 5)

for idx, (method, label) in enumerate(zip(methods, method_labels)):
    ax = axes[idx]

    layer_results = probe_data['all_results'][method]['layer_results']
    accs = np.array([r['test_acc'] * 100 for r in layer_results]).reshape(-1, 1)

    im = ax.imshow(accs, cmap='RdYlGn', aspect='auto', vmin=heatmap_vmin, vmax=heatmap_vmax)

    # Annotate each cell
    mid_val = (heatmap_vmin + heatmap_vmax) / 2
    for i, acc in enumerate(accs.flatten()):
        color = 'white' if acc < mid_val - 5 or acc > mid_val + 5 else 'black'
        ax.text(0, i, f'{acc:.1f}', ha='center', va='center', fontsize=8, color=color)

    best_layer = probe_data['all_results'][method]['best_layer']
    ax.add_patch(plt.Rectangle((-0.5, best_layer-0.5), 1, 1,
                                fill=False, edgecolor='black', lw=3))

    ax.set_title(f'{label}\n(Best: Layer {best_layer})', fontsize=12, fontweight='bold')
    ax.set_xticks([])

    if idx == 0:
        ax.set_yticks(range(0, num_layers + 1, 4))
        ax.set_yticklabels([f'L{i}' for i in range(0, num_layers + 1, 4)])
        ax.set_ylabel('Layer', fontsize=12)

cbar = fig.colorbar(im, ax=axes, shrink=0.6, location='right')
cbar.set_label('Test Accuracy (%)', fontsize=11)

fig.suptitle('Layer-wise Probe Accuracy by Extraction Method', fontsize=14, fontweight='bold')

save_fig(fig, str(PLOTS_DIR / '08_method_heatmaps.png'))

# ============================================================
# PLOT 9: SUMMARY 4-PANEL
# ============================================================

print("9. Summary infographic...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# A: Detection bars
ax = axes[0, 0]
best_accs = [probe_data['all_results'][m]['best_test_acc'] * 100 for m in methods]
colors = ['#d62728', '#1f77b4', '#2ca02c']
ax.bar(['Last\nPrompt', 'Last\nResponse', 'Mean\nResponse'], best_accs,
       color=colors, edgecolor='black', linewidth=1.5)
for i, acc in enumerate(best_accs):
    ax.text(i, acc + 1, f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.6)
ax.set_ylabel('Test Accuracy (%)', fontsize=11)
ax.set_title('A) Detection Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(best_accs) + 10)

# B: Best layers
ax = axes[0, 1]
best_layers = [probe_data['all_results'][m]['best_layer'] for m in methods]
ax.bar(['Last\nPrompt', 'Last\nResponse', 'Mean\nResponse'], best_layers,
       color=colors, edgecolor='black', linewidth=1.5)
for i, layer in enumerate(best_layers):
    ax.text(i, layer + 0.5, f'L{layer}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Layer', fontsize=11)
ax.set_title('B) Best Layer per Method', fontsize=12, fontweight='bold')
ax.set_ylim(0, num_layers + 2)

# C: Steering curve
ax = axes[1, 0]
alphas = sorted(steering_results.keys())
accuracies = [steering_results[a]['correct'] / steering_results[a]['total'] * 100 for a in alphas]
ax.plot(alphas, accuracies, '-o', color='#8e44ad', linewidth=2.5, markersize=8)
best_idx = np.argmax(accuracies)
ax.scatter([alphas[best_idx]], [accuracies[best_idx]],
           color='#27ae60', s=150, zorder=5, edgecolors='black', linewidth=2, marker='*')
ax.axhline(y=accuracies[0], color='#e74c3c', linestyle='--', alpha=0.6)
ax.set_xlabel('Alpha (α)', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('C) Steering Curve', fontsize=12, fontweight='bold')
max_alpha_summary = max(alphas)
ax.set_xlim(-5, max_alpha_summary + 10)
ax.set_ylim(0, max(max(accuracies) + 10, 80))
ax.grid(True, alpha=0.3)

# D: Key results text
ax = axes[1, 1]
ax.axis('off')

# Calculate actual values
best_method_name = probe_data['best_method'].replace('_', ' ').title()
best_test_acc = probe_data['best_test_acc'] * 100
best_layer_num = probe_data['best_layer']
vs_random = best_test_acc - 60

sparse_weights = probe_data['sparse_probe'].coef_[0]
active_neurons = np.sum(np.abs(sparse_weights) > 1e-6)
total_neurons = len(sparse_weights)
sparsity_pct = 100 * active_neurons / total_neurons

text = f"""KEY RESULTS

DETECTION
• Best Method: {best_method_name}
• Best Accuracy: {best_test_acc:.1f}%
• Best Layer: {best_layer_num}
• vs Random: +{vs_random:.1f}%

INTERVENTION
• Optimal α: {int(best_alpha)}
• Improvement: +{improvement:.1f}%
  ({baseline_acc:.1f}% → {best_acc:.1f}%)

SPARSE PROBE
• Active neurons: {active_neurons}/{total_neurons} ({sparsity_pct:.1f}%)
• Interpretable direction found
"""
ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=12,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray', alpha=0.9))
ax.set_title('D) Summary', fontsize=12, fontweight='bold')

fig.suptitle('Sycophancy Probing: Complete Results', fontsize=15, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '09_summary.png'))

# ============================================================
# PLOT 10: PIPELINE DIAGRAM
# ============================================================

print("10. Pipeline diagram...")

fig, ax = plt.subplots(figsize=(16, 4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis('off')

# Get dataset size from judged file
with open(OUTPUT_DIR / "sycophancy_judged.csv", "r", encoding="utf-8") as f:
    n_dataset = sum(1 for _ in f) - 1  # subtract header

boxes = [
    (1.2, f'Dataset\n({n_dataset} examples)', '#3498db'),
    (3.5, 'Generate\nResponses', '#9b59b6'),
    (5.8, 'LLM\nJudge', '#e67e22'),
    (8.1, 'Extract\nHidden States', '#1abc9c'),
    (10.4, 'Train\nProbe', '#27ae60'),
    (12.7, 'Steering\nVector', '#e74c3c'),
    (15.0, 'Intervene', '#f39c12'),
]

for x, text, color in boxes:
    rect = plt.Rectangle((x-1, 1), 2, 2, facecolor=color,
                          edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, 2, text, ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')

# Arrows
for i in range(len(boxes)-1):
    ax.annotate('', xy=(boxes[i+1][0]-1.1, 2), xytext=(boxes[i][0]+1.1, 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

ax.set_title('Sycophancy Probing Pipeline', fontsize=14, fontweight='bold', y=0.9)

save_fig(fig, str(PLOTS_DIR / '10_pipeline.png'))

# ============================================================
# PLOT 11: NEURON ACTIVATION PATTERNS
# ============================================================

print("11. Neuron activation patterns...")

# Load hidden states
hs_data = np.load(OUTPUT_DIR / "sycophancy_hidden_states.npz", allow_pickle=True)

best_layer = probe_data['best_layer']
top_pos = probe_data['top_neurons_positive'][:5]
top_neg = probe_data['top_neurons_negative'][:5]

# biased = sycophantic, neutral = non-sycophantic
# Shape: (1000, 29, 1024) -> get best_layer
syco_activations = hs_data['biased_mean_response'][:, best_layer, :]  # (1000, 1024)
non_syco_activations = hs_data['neutral_mean_response'][:, best_layer, :]  # (1000, 1024)

fig, axes = plt.subplots(2, 5, figsize=(16, 8))

# Top row: Pro-sycophancy neurons
for i, neuron_idx in enumerate(top_pos):
    ax = axes[0, i]
    ax.hist(non_syco_activations[:, neuron_idx], bins=20, alpha=0.7,
            label='Non-syco', color='#2ca02c', density=True)
    ax.hist(syco_activations[:, neuron_idx], bins=20, alpha=0.7,
            label='Sycophantic', color='#d62728', density=True)
    ax.set_title(f'Neuron {neuron_idx}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Activation', fontsize=9)
    if i == 0:
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)

# Bottom row: Anti-sycophancy neurons
for i, neuron_idx in enumerate(top_neg):
    ax = axes[1, i]
    ax.hist(non_syco_activations[:, neuron_idx], bins=20, alpha=0.7,
            label='Non-syco', color='#2ca02c', density=True)
    ax.hist(syco_activations[:, neuron_idx], bins=20, alpha=0.7,
            label='Sycophantic', color='#d62728', density=True)
    ax.set_title(f'Neuron {neuron_idx}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Activation', fontsize=9)
    if i == 0:
        ax.set_ylabel('Density', fontsize=10)

axes[0, 2].text(0.5, 1.15, 'Pro-Sycophancy Neurons', transform=axes[0, 2].transAxes,
                ha='center', fontsize=12, fontweight='bold')
axes[1, 2].text(0.5, -0.25, 'Anti-Sycophancy Neurons', transform=axes[1, 2].transAxes,
                ha='center', fontsize=12, fontweight='bold')

fig.suptitle(f'Neuron Activation Distributions (Layer {best_layer})', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '11_neuron_activations.png'))

# ============================================================
# PLOT 12: WEIGHT DISTRIBUTION (SPARSITY)
# ============================================================

print("12. Weight distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

weights = probe_data['sparse_probe'].coef_[0]

# Histogram of all weights
ax1.hist(weights, bins=50, color='#3498db', edgecolor='black', alpha=0.8)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Weight Value', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Probe Weight Distribution', fontsize=13, fontweight='bold')

# Show sparsity
nonzero = np.sum(weights != 0)
total = len(weights)
ax1.text(0.95, 0.95, f'Non-zero: {nonzero}/{total}\n({100*nonzero/total:.1f}%)',
         transform=ax1.transAxes, ha='right', va='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Sorted absolute weights
sorted_weights = np.sort(np.abs(weights))[::-1]
ax2.plot(sorted_weights, color='#8e44ad', linewidth=2)
ax2.fill_between(range(len(sorted_weights)), sorted_weights, alpha=0.3, color='#8e44ad')
ax2.set_xlabel('Neuron Rank', fontsize=12)
ax2.set_ylabel('|Weight|', fontsize=12)
ax2.set_title('Sorted Weight Magnitudes', fontsize=13, fontweight='bold')
ax2.set_xlim(0, len(sorted_weights))

# Mark top neurons
top_k = 20
ax2.axvline(x=top_k, color='red', linestyle='--', linewidth=2)
ax2.text(top_k + 5, sorted_weights[0] * 0.9, f'Top {top_k}', fontsize=10, color='red')

fig.suptitle('Sparse Probe Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '12_weight_distribution.png'))

# ============================================================
# PLOT 13: LAYER-NEURON IMPORTANCE HEATMAP
# ============================================================

print("13. Layer-neuron heatmap...")

# We need to compute neuron importance at each layer
# Use correlation between each neuron and the label

fig, ax = plt.subplots(figsize=(14, 8))

# Compute importance scores for top neurons across all layers
top_neurons = list(top_pos[:10]) + list(top_neg[:10])  # Top 20 neurons

# Get all activations: biased (syco=1) vs neutral (syco=0)
biased_all = hs_data['biased_mean_response']  # (n_examples, num_layers+1, hidden_dim)
neutral_all = hs_data['neutral_mean_response']  # (n_examples, num_layers+1, hidden_dim)
n_examples = biased_all.shape[0]

importance_matrix = np.zeros((num_layers + 1, len(top_neurons)))

for layer in range(num_layers + 1):
    # Combine biased and neutral
    layer_biased = biased_all[:, layer, :]  # (n_examples, hidden_dim)
    layer_neutral = neutral_all[:, layer, :]  # (n_examples, hidden_dim)

    layer_activations = np.vstack([layer_biased, layer_neutral])  # (2*n_examples, hidden_dim)
    labels = np.array([1]*n_examples + [0]*n_examples)  # 1=syco, 0=non-syco

    for j, neuron_idx in enumerate(top_neurons):
        # Correlation between neuron activation and label
        corr = np.corrcoef(layer_activations[:, neuron_idx], labels)[0, 1]
        importance_matrix[layer, j] = corr

# Smooth it for visualization
importance_smooth = zoom(importance_matrix, (3, 3), order=3)
importance_smooth = gaussian_filter(importance_smooth, sigma=1)

im = ax.imshow(importance_smooth, cmap='RdBu_r', aspect='auto',
               extent=[-0.5, len(top_neurons)-0.5, num_layers, 0],
               vmin=-0.4, vmax=0.4)

ax.set_xlabel('Neuron', fontsize=12)
ax.set_ylabel('Layer', fontsize=12)
ax.set_xticks(range(len(top_neurons)))
ax.set_xticklabels([f'N{n}' for n in top_neurons], rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(0, num_layers + 1, 4))

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Correlation with Sycophancy', fontsize=11)

ax.axhline(y=best_layer, color='black', linestyle='--', linewidth=2)
ax.text(len(top_neurons) - 0.5, best_layer, f' Best Layer ({best_layer})',
        va='center', fontsize=10, fontweight='bold')

ax.set_title('Neuron-Sycophancy Correlation Across Layers', fontsize=14, fontweight='bold')

save_fig(fig, str(PLOTS_DIR / '13_layer_neuron_heatmap.png'))

# ============================================================
# PLOT 14: CIRCUIT DIAGRAM
# ============================================================

print("14. Circuit diagram...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Input layer (left)
ax.text(1, 5, f'Hidden\nState\n(L{best_layer})', ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', edgecolor='black', linewidth=2))

# Top pro-sycophancy neurons (middle-top)
pro_neurons = top_pos[:5]
for i, n in enumerate(pro_neurons):
    y = 8 - i * 1.2
    weight = weights[n]
    ax.add_patch(plt.Circle((6, y), 0.4, facecolor='#e74c3c', edgecolor='black', linewidth=2))
    ax.text(6, y, f'{n}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(7, y, f'w={weight:.3f}', ha='left', va='center', fontsize=8)
    # Connection from input
    ax.annotate('', xy=(5.6, y), xytext=(2, 5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, alpha=0.6))

# Top anti-sycophancy neurons (middle-bottom)
anti_neurons = top_neg[:5]
for i, n in enumerate(anti_neurons):
    y = 4 - i * 1.2
    weight = weights[n]
    ax.add_patch(plt.Circle((6, y), 0.4, facecolor='#27ae60', edgecolor='black', linewidth=2))
    ax.text(6, y, f'{n}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(7, y, f'w={weight:.3f}', ha='left', va='center', fontsize=8)
    # Connection from input
    ax.annotate('', xy=(5.6, y), xytext=(2, 5),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5, alpha=0.6))

# Output (right)
ax.text(12, 5, 'Sycophancy\nScore', ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#9b59b6', edgecolor='black', linewidth=2))

# Connections to output
for i in range(5):
    y_pro = 8 - i * 1.2
    y_anti = 4 - i * 1.2
    ax.annotate('', xy=(11, 5.3), xytext=(6.4, y_pro),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, alpha=0.6))
    ax.annotate('', xy=(11, 4.7), xytext=(6.4, y_anti),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5, alpha=0.6))

# Legend
ax.add_patch(plt.Circle((14, 8), 0.3, facecolor='#e74c3c', edgecolor='black'))
ax.text(14.5, 8, 'Pro-Sycophancy', va='center', fontsize=10)
ax.add_patch(plt.Circle((14, 7), 0.3, facecolor='#27ae60', edgecolor='black'))
ax.text(14.5, 7, 'Anti-Sycophancy', va='center', fontsize=10)

ax.set_title('Sycophancy Detection Circuit (Top 10 Neurons)', fontsize=14, fontweight='bold', y=0.98)

save_fig(fig, str(PLOTS_DIR / '14_circuit_diagram.png'))

# ============================================================
# DONE
# ============================================================

# ============================================================
# PLOT 15: PCA VISUALIZATION
# ============================================================

print("15. PCA visualization...")

from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Combine data
all_activations = np.vstack([syco_activations, non_syco_activations])
all_labels = np.array([1]*len(syco_activations) + [0]*len(non_syco_activations))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_activations)

# Full view
ax = axes[0]
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels,
                     cmap='RdYlGn_r', alpha=0.5, s=20)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax.set_title(f'PCA of Hidden States (Layer {best_layer})', fontsize=12, fontweight='bold')
# Custom legend
from matplotlib.lines import Line2D
leg_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                       markersize=8, label='Sycophantic'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                       markersize=8, label='Non-Sycophantic')]
ax.legend(handles=leg_elements, loc='upper right')

# Show probe direction in PCA space
ax = axes[1]
ax.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels,
           cmap='RdYlGn_r', alpha=0.3, s=15)
# Project probe direction to PCA space
probe_weights = probe_data['sparse_probe'].coef_[0]
probe_pca = pca.transform(probe_weights.reshape(1, -1))[0]
ax.arrow(0, 0, probe_pca[0]*5, probe_pca[1]*5, head_width=0.3, head_length=0.2,
         fc='black', ec='black', linewidth=2)
ax.set_xlabel(f'PC1', fontsize=11)
ax.set_ylabel(f'PC2', fontsize=11)
ax.set_title('Probe Direction in PCA Space', fontsize=12, fontweight='bold')

# Density plot
ax = axes[2]
from scipy.stats import gaussian_kde
syco_pca = pca_result[:len(syco_activations)]
non_syco_pca = pca_result[len(syco_activations):]

ax.hist(syco_pca[:, 0], bins=30, alpha=0.6, color='#d62728', label='Sycophantic', density=True)
ax.hist(non_syco_pca[:, 0], bins=30, alpha=0.6, color='#2ca02c', label='Non-Sycophantic', density=True)
ax.set_xlabel('PC1 Score', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution Along PC1', fontsize=12, fontweight='bold')
ax.legend()

fig.suptitle('Principal Component Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '15_pca_visualization.png'))

# ============================================================
# PLOT 16: t-SNE VISUALIZATION
# ============================================================

print("16. t-SNE visualization...")

from sklearn.manifold import TSNE

# Subsample for speed
n_samples = 500
idx_syco = np.random.choice(len(syco_activations), n_samples//2, replace=False)
idx_non = np.random.choice(len(non_syco_activations), n_samples//2, replace=False)

tsne_data = np.vstack([syco_activations[idx_syco], non_syco_activations[idx_non]])
tsne_labels = np.array([1]*(n_samples//2) + [0]*(n_samples//2))

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(tsne_data)

fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=tsne_labels,
                     cmap='RdYlGn_r', alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title(f't-SNE Visualization of Hidden States (Layer {best_layer})', fontsize=14, fontweight='bold')

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                          markersize=10, label='Sycophantic'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                          markersize=10, label='Non-Sycophantic')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

save_fig(fig, str(PLOTS_DIR / '16_tsne_visualization.png'))

# ============================================================
# PLOT 17: PROBE DIRECTION VISUALIZATION
# ============================================================

print("17. Probe direction visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Full weight vector
ax = axes[0]
ax.bar(range(len(probe_weights)), probe_weights, color='#3498db', alpha=0.7, width=1.0)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Neuron Index', fontsize=11)
ax.set_ylabel('Weight', fontsize=11)
ax.set_title('Full Probe Direction (1024 dims)', fontsize=12, fontweight='bold')
ax.set_xlim(0, len(probe_weights))

# Zoomed on non-zero
ax = axes[1]
nonzero_idx = np.where(probe_weights != 0)[0]
nonzero_weights = probe_weights[nonzero_idx]
colors = ['#d62728' if w > 0 else '#2ca02c' for w in nonzero_weights]
ax.bar(range(len(nonzero_weights)), nonzero_weights, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Non-zero Neuron Index', fontsize=11)
ax.set_ylabel('Weight', fontsize=11)
ax.set_title(f'Non-zero Weights Only ({len(nonzero_idx)} neurons)', fontsize=12, fontweight='bold')

fig.suptitle('Learned Steering Vector', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '17_probe_direction.png'))

# ============================================================
# PLOT 18: CONFIDENCE DISTRIBUTION
# ============================================================

print("18. Confidence distribution...")

# Get probe predictions
from sklearn.preprocessing import StandardScaler

scaler = probe_data['sparse_scaler']
probe = probe_data['sparse_probe']

# Prepare data
X = np.vstack([syco_activations, non_syco_activations])
y = np.array([1]*len(syco_activations) + [0]*len(non_syco_activations))

X_scaled = scaler.transform(X)
probs = probe.predict_proba(X_scaled)[:, 1]  # P(sycophantic)
preds = probe.predict(X_scaled)

correct_mask = (preds == y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confidence when correct vs incorrect
ax = axes[0]
confidence = np.abs(probs - 0.5) * 2  # Scale to 0-1
ax.hist(confidence[correct_mask], bins=20, alpha=0.7, color='#2ca02c',
        label=f'Correct (n={correct_mask.sum()})', density=True)
ax.hist(confidence[~correct_mask], bins=20, alpha=0.7, color='#d62728',
        label=f'Incorrect (n={(~correct_mask).sum()})', density=True)
ax.set_xlabel('Confidence', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Probe Confidence: Correct vs Incorrect', fontsize=12, fontweight='bold')
ax.legend()

# Probability distribution by class
ax = axes[1]
ax.hist(probs[y == 1], bins=20, alpha=0.7, color='#d62728',
        label='Sycophantic', density=True)
ax.hist(probs[y == 0], bins=20, alpha=0.7, color='#2ca02c',
        label='Non-Sycophantic', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('P(Sycophantic)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Predicted Probability Distribution', fontsize=12, fontweight='bold')
ax.legend()

fig.suptitle('Probe Confidence Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '18_confidence_distribution.png'))

# ============================================================
# PLOT 19: LAYER PROGRESSION
# ============================================================

print("19. Layer progression...")

fig, axes = plt.subplots(2, 4, figsize=(18, 10))

# Show PCA at different layers (dynamically based on model)
layer_step = max(1, num_layers // 7)
layers_to_show = [0] + list(range(layer_step, num_layers, layer_step))[:6] + [num_layers]
layers_to_show = layers_to_show[:8]  # Ensure max 8 layers

n_samples_hs = biased_all.shape[0]

for idx, layer in enumerate(layers_to_show):
    ax = axes[idx // 4, idx % 4]

    layer_syco = biased_all[:, layer, :]
    layer_non = neutral_all[:, layer, :]
    layer_all = np.vstack([layer_syco, layer_non])

    pca_layer = PCA(n_components=2)
    pca_result_layer = pca_layer.fit_transform(layer_all)

    ax.scatter(pca_result_layer[:n_samples_hs, 0], pca_result_layer[:n_samples_hs, 1],
               c='#d62728', alpha=0.3, s=10, label='Syco')
    ax.scatter(pca_result_layer[n_samples_hs:, 0], pca_result_layer[n_samples_hs:, 1],
               c='#2ca02c', alpha=0.3, s=10, label='Non-Syco')

    ax.set_title(f'Layer {layer}', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

fig.suptitle('Representation Evolution Across Layers (PCA)', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '19_layer_progression.png'))

# ============================================================
# PLOT 20: EXTRACTION METHOD COMPARISON SCATTER
# ============================================================

print("20. Extraction method comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

method_accs = {}
for method in methods:
    method_accs[method] = [r['test_acc'] * 100 for r in probe_data['all_results'][method]['layer_results']]

# Last Prompt vs Last Response
ax = axes[0]
ax.scatter(method_accs['last_prompt'], method_accs['last_response'],
           c=range(num_layers + 1), cmap='viridis', s=50, alpha=0.8)
ax.plot([55, 75], [55, 75], 'k--', alpha=0.5)
ax.set_xlabel('Last Prompt Accuracy (%)', fontsize=11)
ax.set_ylabel('Last Response Accuracy (%)', fontsize=11)
ax.set_title('Last Prompt vs Last Response', fontsize=12, fontweight='bold')

# Last Response vs Mean Response
ax = axes[1]
scatter = ax.scatter(method_accs['last_response'], method_accs['mean_response'],
                     c=range(num_layers + 1), cmap='viridis', s=50, alpha=0.8)
ax.plot([55, 75], [55, 75], 'k--', alpha=0.5)
ax.set_xlabel('Last Response Accuracy (%)', fontsize=11)
ax.set_ylabel('Mean Response Accuracy (%)', fontsize=11)
ax.set_title('Last Response vs Mean Response', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Layer', fontsize=10)

# All three methods
ax = axes[2]
layers = range(num_layers + 1)
ax.plot(layers, method_accs['last_prompt'], 'o-', label='Last Prompt', alpha=0.7)
ax.plot(layers, method_accs['last_response'], 's-', label='Last Response', alpha=0.7)
ax.plot(layers, method_accs['mean_response'], '^-', label='Mean Response', alpha=0.7)
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('All Methods by Layer', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Extraction Method Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '20_method_comparison.png'))

# ============================================================
# PLOT 21: ROC CURVE
# ============================================================

print("21. ROC curve...")

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
ax = axes[0]
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color='#8e44ad', lw=3, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
ax.fill_between(fpr, tpr, alpha=0.2, color='#8e44ad')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

# Precision-Recall Curve
ax = axes[1]
precision, recall, _ = precision_recall_curve(y, probs)
ap = average_precision_score(y, probs)

ax.plot(recall, precision, color='#e67e22', lw=3, label=f'PR (AP = {ap:.3f})')
ax.fill_between(recall, precision, alpha=0.2, color='#e67e22')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

fig.suptitle('Probe Performance Curves', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '21_roc_pr_curves.png'))

# ============================================================
# PLOT 22: CONFUSION MATRIX
# ============================================================

print("22. Confusion matrix...")

from sklearn.metrics import confusion_matrix, classification_report

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm = confusion_matrix(y, preds)

# Raw counts
ax = axes[0]
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Non-Syco', 'Sycophantic'], fontsize=11)
ax.set_yticklabels(['Non-Syco', 'Sycophantic'], fontsize=11)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if cm[i, j] > cm.max()/2 else 'black')

# Normalized
ax = axes[1]
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Non-Syco', 'Sycophantic'], fontsize=11)
ax.set_yticklabels(['Non-Syco', 'Sycophantic'], fontsize=11)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm_norm[i, j]:.2%}', ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm_norm[i, j] > 0.5 else 'black')

plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Classification Results', fontsize=14, fontweight='bold')
plt.tight_layout()

save_fig(fig, str(PLOTS_DIR / '22_confusion_matrix.png'))

# ============================================================
# DONE
# ============================================================

print("\n" + "="*40)
print("ALL 22 PLOTS GENERATED!")
print("="*40)
for f in sorted(os.listdir(PLOTS_DIR)):
    if f.endswith('.png'):
        print(f"  {f}")
