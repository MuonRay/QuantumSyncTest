# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 14:56:53 2025

@author: ektop
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# === Choose which signal to load: 'blue' or 'red' ===
signal_label = 'blue'  # or 'red'
filename = f"sliding_window_metrics_{signal_label}.txt"

# === Load Data from TXT File ===
window_starts = []
lyap_exps = []
samp_ents = []
corr_dims = []

with open(filename, 'r') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue  # skip malformed line
        ws, le, se, cd = parts
        window_starts.append(int(ws))
        lyap_exps.append(float(le))
        samp_ents.append(float(se))
        corr_dims.append(float(cd))

# Convert to arrays
window_starts = np.array(window_starts)
lyap_exps = np.array(lyap_exps)
samp_ents = np.array(samp_ents)
corr_dims = np.array(corr_dims)

# === Create Output Folder ===
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# === Plot 1: Largest Lyapunov Exponent ===
plt.figure(figsize=(10, 4))
plt.plot(window_starts, lyap_exps, label="Lyapunov Exponent", color='red')
plt.axhline(0, color='black', linestyle='--', lw=1)
plt.title(f"Largest Lyapunov Exponent – {signal_label.capitalize()}")
plt.xlabel("Start of Window (Index)")
plt.ylabel("Lyapunov Exponent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/lyapunov_exponent_plot_{signal_label}.png", dpi=300)
plt.show()

# === Plot 2: Sample Entropy with Color Map ===
fig, ax = plt.subplots(figsize=(10, 4))
sc = ax.scatter(window_starts, samp_ents, c=samp_ents, cmap='viridis', edgecolor='black')
cbar = plt.colorbar(sc)
cbar.set_label('Sample Entropy')
ax.set_title(f"Sample Entropy – {signal_label.capitalize()} (Higher = More Complex)")
ax.set_xlabel("Start of Window (Index)")
ax.set_ylabel("Sample Entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/sample_entropy_plot_{signal_label}.png", dpi=300)
plt.show()

# === Plot 3: Correlation Dimension ===
fig, ax = plt.subplots(figsize=(10, 4))
sc = ax.scatter(window_starts, corr_dims, c=corr_dims, cmap='plasma', edgecolor='black')
cbar = plt.colorbar(sc)
cbar.set_label('Correlation Dimension')
ax.set_title(f"Correlation Dimension – {signal_label.capitalize()} (Fractal = Strange Attractor)")
ax.set_xlabel("Start of Window (Index)")
ax.set_ylabel("Correlation Dimension")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_dimension_plot_{signal_label}.png", dpi=300)
plt.show()
