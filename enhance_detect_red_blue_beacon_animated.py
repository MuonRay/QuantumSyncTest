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

import warnings
warnings.filterwarnings("ignore")

cap = cv2.VideoCapture('video_test.mp4')

blue_vals = []
red_vals = []  # List to store red values
times = []

start_time = time.time()

def handle_interrupt(signal, frame):
    plt.plot(times, blue_vals, label='Blue Values')
    plt.plot(times, red_vals, label='Red Values')  # Plot red values
    plt.xlabel('Time (seconds)')
    plt.ylabel('Values')
    plt.title('Blue and Red Values vs Time')
    plt.legend()

    # Generate a unique filename based on the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"plot_{timestamp}.png"
    
    plt.savefig(filename)  # Save the plot to a PNG file
    plt.show()
    exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

try:
    while True:
        ret, frame = cap.read()

        if ret == True:
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

cap.release()
cv2.destroyAllWindows()
