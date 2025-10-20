# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 15:31:53 2025

@author: ektop
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 15:24:48 2025

@author: ektop
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import warnings
import os
from mpl_toolkits.mplot3d import Axes3D  # for 3D embedding plot
import nolds

warnings.filterwarnings("ignore")

# Path to video file
video_path = 'video_test.mp4'
cap = cv2.VideoCapture(video_path)

green_vals = []
red_vals = []
blue_vals = []
times = []

start_time = time.time()

# HSV range for green (adjust if needed)
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# HSV ranges for red (handle wrap-around)
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# HSV range for blue
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

def takens_embedding(data, delay=5, dimension=3):
    """
    Construct a Takens embedding (delay‐embedding) for a 1D time series.
    """
    data = np.asarray(data)
    N = len(data)
    if N < (dimension - 1) * delay:
        raise ValueError("Time series too short for Takens embedding: N = {}, requires at least {}".format(N, (dimension-1)*delay + 1))
    embedded = np.array([data[i : i + delay*dimension : delay] for i in range(N - (dimension - 1)*delay)])
    return embedded

def plot_embedding_with_chimera(embedded_data, chimera_mask, title='Takens Embedding with Chimera', filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot normal points
    ax.plot(embedded_data[~chimera_mask, 0], embedded_data[~chimera_mask, 1], embedded_data[~chimera_mask, 2],
            lw=0.5, color='blue', label='Normal')

    # Plot chimera points
    ax.plot(embedded_data[chimera_mask, 0], embedded_data[chimera_mask, 1], embedded_data[chimera_mask, 2],
            lw=0.5, color='red', label='Chimera')

    ax.set_title(title)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t + τ)')
    ax.set_zlabel('x(t + 2τ)')
    ax.legend()
    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)

def handle_interrupt(sig, frame):
    # Save raw time series plot for green, red, blue
    plt.figure()
    plt.plot(times, green_vals, label='Green Values', color='green')
    plt.plot(times, red_vals, label='Red Values (Unsync)', color='red')
    plt.plot(times, blue_vals, label='Blue Values (Sync)', color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Intensity')
    plt.title('Color Mask Values vs Time')
    plt.legend()
    plt.tight_layout()
    time_series_fname = f"time_series_colors_{int(time.time())}.png"
    plt.savefig(time_series_fname, dpi=300)
    plt.show()
    plt.close()

    delay = 5
    dimension = 3

    # Takens embedding for green
    try:
        green_embed = takens_embedding(green_vals, delay=delay, dimension=dimension)
    except ValueError as e:
        print("Embedding failed:", e)
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

    # Sliding window nonlinear analysis for green
    window_size = 200
    step_size = 20

    results = []
    for start in range(0, len(green_vals) - window_size + 1, step_size):
        window = green_vals[start:start + window_size]
        try:
            le = nolds.lyap_r(window, emb_dim=dimension, tau=delay)
        except:
            le = np.nan
        try:
            se = nolds.sampen(window)
        except:
            se = np.nan
        try:
            cd = nolds.corr_dim(window, emb_dim=dimension, tau=delay)
        except:
            cd = np.nan
        results.append((start, le, se, cd))

    # Save sliding window metrics
    metrics_fname = f"sliding_window_metrics_green_{int(time.time())}.txt"
    with open(metrics_fname, "w") as f:
        f.write("WindowStart\tLyapunovExponent\tSampleEntropy\tCorrelationDimension\n")
        for r in results:
            f.write(f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}\n")
    print(f"[✓] Sliding window metrics saved: {metrics_fname}")

    # Extract metrics arrays
    starts, lyap_exps, samp_ents, corr_dims = zip(*results)
    starts = np.array(starts)
    lyap_exps = np.array(lyap_exps)
    samp_ents = np.array(samp_ents)
    corr_dims = np.array(corr_dims)

    red_arr = np.array(red_vals)
    blue_arr = np.array(blue_vals)

    # === Chimera detection mask with red/blue heuristic ===
    # Simple heuristic: chimera if LLE > 0 and 0.5 < SE < 1.5 (green signal)
    # and blue dominates red indicating sync flashing state
    
    epsilon = 1e-6
    chimera_windows = []
    for i, (le, se) in enumerate(zip(lyap_exps, samp_ents)):
        chimera_green = (le > 0) and (0.5 < se < 1.5)

        start = starts[i]
        end = start + window_size

        # Check array bounds to avoid NaN
        if end > len(red_arr) or end > len(blue_arr):
            chimera_windows.append(False)
            continue

        mean_red = np.mean(red_arr[start:end])
        mean_blue = np.mean(blue_arr[start:end])

        sync_ratio = mean_blue / (mean_red + epsilon)

        # Threshold for sync ratio can be tuned
        chimera_final = chimera_green and (sync_ratio > 1.5)
        chimera_windows.append(chimera_final)

    chimera_windows = np.array(chimera_windows)

    # Map chimera windows to embedding indices:
    chimera_mask = np.zeros(len(green_embed), dtype=bool)
    for i, is_chimera in enumerate(chimera_windows):
        if is_chimera:
            window_start = starts[i]
            window_end = window_start + window_size
            for emb_i in range(len(green_embed)):
                if window_start <= emb_i < window_end:
                    chimera_mask[emb_i] = True

    # Plot embedding with chimera highlighted
    embedding_fname = f"embedding_green_chimera_{int(time.time())}.png"
    plot_embedding_with_chimera(green_embed, chimera_mask,
                                title="Green Signal Takens Embedding with Chimera States (Red/Blue Heuristic)",
                                filename=embedding_fname)

    cap.release()
    cv2.destroyAllWindows()
    exit(0)

# Attach interrupt handler
signal.signal(signal.SIGINT, handle_interrupt)

print("Press Ctrl+C or 'q' to stop and perform analysis...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        handle_interrupt(None, None)
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green mask
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    res_green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Red mask (handle wrap-around hue for red)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Blue mask
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    src_h, src_w, _ = frame.shape
    max_value = src_h * src_w * 255.0

    greenVal = float(green_mask.sum()) / max_value
    redVal = float(red_mask.sum()) / max_value
    blueVal = float(blue_mask.sum()) / max_value

    green_vals.append(greenVal)
    red_vals.append(redVal)
    blue_vals.append(blueVal)

    # Optional: draw contours for green mask
    green_gray = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
    green_cnts, _ = cv2.findContours(green_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in green_cnts:
        cv2.drawContours(res_green, [cnt], -1, (0, 255, 0), 2)

    # Show results
    cv2.imshow('Green Detection', res_green)
    cv2.imshow('Red Detection (Unsync)', res_red)
    cv2.imshow('Blue Detection (Sync)', res_blue)

    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        handle_interrupt(None, None)
        break
