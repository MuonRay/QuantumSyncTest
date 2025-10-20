# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 14:46:35 2025

@author: ektop
"""

# -*- coding: utf-8 -*-
"""
Video colour‑mask → time series → Takens embedding → attractor + nonlinear metrics
Modified on Oct 19 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import warnings
import os
from mpl_toolkits.mplot3d import Axes3D  # for 3D embedding plot

# For nonlinear metrics
import nolds

warnings.filterwarnings("ignore")

# Path to video file
video_path = 'video_test.mp4'
cap = cv2.VideoCapture(video_path)

blue_vals = []
red_vals = []
times = []

start_time = time.time()

def takens_embedding(data, delay=5, dimension=3):
    """
    Construct a Takens embedding (delay‐embedding) for a 1D time series.

    Parameters:
      data: array‐like, the scalar time series
      delay: the time‐lag τ (in index units)
      dimension: embedding dimension m
    Returns:
      A 2D numpy array of shape (N ‑ (m‑1)*τ, m), each row is [x(t), x(t+τ), …, x(t+(m‑1)*τ)]
    """
    data = np.asarray(data)
    N = len(data)
    if N < (dimension - 1) * delay:
        raise ValueError("Time series too short for Takens embedding: N = {}, requires at least {}".format(N, (dimension-1)*delay + 1))
    # build the matrix
    embedded = np.array([ data[i : i + delay*dimension : delay] 
                          for i in range(N - (dimension - 1)*delay) ])
    return embedded

def plot_embedding(embedded_data, title='Takens Embedding', filename=None, colour='blue'):
    """
    3D plot of the embedded data.
    If filename provided, save the figure as a PNG.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(embedded_data[:,0], embedded_data[:,1], embedded_data[:,2], lw=0.5, color=colour)
    ax.set_title(title)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t + τ)')
    ax.set_zlabel('x(t + 2τ)')
    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)

def handle_interrupt(sig, frame):
    """Callback for Ctrl-C or keyboard interrupt: perform embedding and metrics then exit."""

    # --- Save & plot raw time series ---
    plt.figure()
    plt.plot(times, blue_vals, label='Blue Values')
    plt.plot(times, red_vals, label='Red Values')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Intensity')
    plt.title('Blue & Red Values vs Time')
    plt.legend()
    timeplot_fname = f"time_series_{int(time.time())}.png"
    plt.savefig(timeplot_fname, dpi=300)
    plt.show()
    plt.close()

    # --- Takens embedding ---
    delay = 5
    dimension = 3

    try:
        blue_embed = takens_embedding(blue_vals, delay=delay, dimension=dimension)
        red_embed  = takens_embedding(red_vals,  delay=delay, dimension=dimension)
    except ValueError as e:
        print("Embedding failed:", e)
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

    plot_embedding(blue_embed, title='Blue Signal Takens Embedding',
                   filename=f"embedding_blue_{int(time.time())}.png", colour='blue')
    plot_embedding(red_embed, title='Red Signal Takens Embedding',
                   filename=f"embedding_red_{int(time.time())}.png", colour='red')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(blue_embed[:,0], blue_embed[:,1], blue_embed[:,2], lw=0.5, color='blue', label='Blue')
    ax.plot(red_embed[:,0],  red_embed[:,1],  red_embed[:,2],  lw=0.5, color='red',  label='Red')
    ax.set_title('Combined Takens Embedding – Blue & Red')
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t + τ)')
    ax.set_zlabel('x(t + 2τ)')
    ax.legend()
    combined_fname = f"embedding_combined_{int(time.time())}.png"
    fig.savefig(combined_fname, dpi=300)
    plt.show()
    plt.close(fig)

    print("=== Sliding Window Nonlinear Analysis ===")

    # --- Sliding Window Analysis ---
    def sliding_analysis(data, label, window_size=200, step_size=20):
        results = []
        for start in range(0, len(data) - window_size + 1, step_size):
            window = data[start:start + window_size]
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
        # Write results to txt
        with open(f"sliding_window_metrics_{label}.txt", "w") as f:
            f.write("WindowStart\tLyapunovExponent\tSampleEntropy\tCorrelationDimension\n")
            for r in results:
                f.write(f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}\n")
        print(f"[✓] Sliding window metrics saved: sliding_window_metrics_{label}.txt")

    # Perform analysis on both signals
    sliding_analysis(blue_vals, 'blue')
    sliding_analysis(red_vals, 'red')

    cap.release()
    cv2.destroyAllWindows()
    exit(0)



# Attach interrupt handler
signal.signal(signal.SIGINT, handle_interrupt)

# Main video loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video read ended (no more frames).")
        # Trigger our handle_interrupt logic to do embedding & metrics
        handle_interrupt(None, None)
        break

    # Convert to HSV for colour thresholding
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define blue range
    lower_blue = np.array([100, 60, 60])
    upper_blue = np.array([150, 255, 255])

    # Define red ranges (two ranges to wrap-around HSV hue)
    lower_red1 = np.array([0,   70, 50])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255,255])

    # Generate masks
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

    # Apply masks to original frame for visualization
    res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    res_red  = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Compute normalized value of detected mask area
    src_h, src_w, _ = frame.shape
    max_value = src_h * src_w * 255.0
    blueVal = float(blue_mask.sum()) / max_value
    redVal  = float(red_mask.sum())  / max_value

    blue_vals.append(blueVal)
    red_vals.append(redVal)

    # Optional: draw contours (for visualization)
    blue_gray = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    blue_cnts, _ = cv2.findContours(blue_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in blue_cnts:
        cv2.drawContours(res_blue, [cnt], -1, (0,255,0), 2)

    red_gray = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
    red_cnts, _ = cv2.findContours(red_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in red_cnts:
        cv2.drawContours(res_red, [cnt], -1, (0,0,255), 2)

    # Show windows
    cv2.imshow('Blue Detection', res_blue)
    cv2.imshow('Red Detection',  res_red)

    # Append elapsed time
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Trigger interrupt behaviour
        handle_interrupt(None, None)
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
