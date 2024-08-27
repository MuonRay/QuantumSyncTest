# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:06:25 2023

@author: cosmi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import os

import warnings
warnings.filterwarnings("ignore")

cap = cv2.VideoCapture('video_test.mp4')

blue_vals = []
red_vals = []  # List to store red values
times = []

start_time = time.time()

def plot_and_save(blue_vals, red_vals, times):
    # Plot blue values
    plt.figure()
    plt.plot(times, blue_vals, color='blue', label='Blue Values')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Blue Values')
    plt.title('Blue Values vs Time')
    plt.legend()

    # Generate a unique filename for blue values plot based on current timestamp
    blue_filename = f"blue_plot_{time.strftime('%Y%m%d-%H%M%S')}.png"
    plt.savefig(blue_filename)
    plt.close()  # Close the figure after saving
    plt.show()   # Show the blue values plot

    # Plot red values
    plt.figure()
    plt.plot(times, red_vals, color='red', label='Red Values')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Red Values')
    plt.title('Red Values vs Time')
    plt.legend()
    
    # Generate a unique filename for red values plot based on current timestamp
    red_filename = f"red_plot_{time.strftime('%Y%m%d-%H%M%S')}.png"
    plt.savefig(red_filename)
    plt.close()  # Close the figure after saving
    plt.show()   # Show the red values plot


def handle_interrupt(signal, frame):
    plot_and_save(blue_vals, red_vals, times)
    exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

try:
    while True:
        ret, frame = cap.read()

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the blue color range
            lower_blue = np.array([100, 60, 60])
            upper_blue = np.array([150, 255, 255])

            # Define the red color range
            lower_red = np.array([0, 70, 50])
            upper_red = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            # Mask for blue and red
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            red_mask1 = cv2.inRange(hsv, lower_red, upper_red)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
            res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

            # Process blue mask
            src_height, src_width, _ = frame.shape
            max_value = src_height * src_width * 255
            blueVal = float(blue_mask.sum()) / float(max_value)
            blue_vals.append(blueVal)

            # Process red mask
            redVal = float(red_mask.sum()) / float(max_value)
            red_vals.append(redVal)

            # Process contours for blue
            blue_contours, _ = cv2.findContours(cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if blue_contours:
                for contour in blue_contours:
                    cv2.drawContours(res_blue, contour, -1, (0, 255, 0), 2)
            
            # Process contours for red
            red_contours, _ = cv2.findContours(cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if red_contours:
                for contour in red_contours:
                    cv2.drawContours(res_red, contour, -1, (0, 0, 255), 2)

            # Show the processed results
            cv2.imshow('Blue Detection', res_blue)
            cv2.imshow('Red Detection', res_red)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

        else:
            print("Read Failed")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    handle_interrupt(None, None)

# After exiting the main loop, create and save the key image.
if blue_vals and red_vals:  # Check if the lists are not empty
    # Normalize the lists to 0-255
    blue_vals_normalized = np.array(blue_vals[:256]).flatten() * 255  # Take the first 256 values and scale to 255
    red_vals_normalized = np.array(red_vals[:256]).flatten() * 255

    # Create 256x256 images for blue and red values
    blu_img = np.zeros((256, 256), dtype=np.uint8)
    red_img = np.zeros((256, 256), dtype=np.uint8)

    for i in range(256):
        blu_img[i, :] = blue_vals_normalized[i]  # Fill the column with blue values
        red_img[i, :] = red_vals_normalized[i]    # Fill the column with red values

    # Combine them into a 3-channel image
    key_image = cv2.merge((blu_img, np.zeros((256, 256), dtype=np.uint8), red_img))

    # Save the key image
    key_image_filename = 'key_image.png'
    cv2.imwrite(key_image_filename, key_image)
    print(f'Key image saved as {key_image_filename}')

# Finally, plot and display before exiting.
plot_and_save(blue_vals, red_vals, times)

cap.release()
cv2.destroyAllWindows()