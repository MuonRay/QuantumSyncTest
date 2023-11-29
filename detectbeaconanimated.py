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

cap = cv2.VideoCapture('video_test2.mp4')

blue_vals = []
times = []

start_time = time.time()

def handle_interrupt(signal, frame):
    plt.plot(times, blue_vals)
    plt.xlabel('Time (seconds)')
    plt.ylabel('blueVal')
    plt.title('blueVal vs Time')
    
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

            lower_blue = np.array([100, 60, 60])
            upper_blue = np.array([150, 255, 255])

            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            res = cv2.bitwise_and(frame, frame, mask=mask)
            src_height, src_width, src_channels = frame.shape
            max_value = src_height * src_width * 255
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
            img = cv2.medianBlur(res, 5)
            ccimg = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            cimg = cv2.cvtColor(ccimg, cv2.COLOR_BGR2GRAY)

            ret, contours = cv2.findContours(cimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours is not None:
                print("beacon is found")
                contours = np.uint16(np.around(contours))
                blueVal = float(mask.sum()) / float(max_value)
                print(blueVal)

                for i in contours[0, :]:
                    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.circle(res, maxLoc, 50, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('flash', frame)
            cv2.imshow('detected circles', cimg)
            cv2.imshow('res', res)

            elapsed_time = time.time() - start_time
            blue_vals.append(blueVal)
            times.append(elapsed_time)

        else:
            print("Read Failed")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    plt.plot(times, blue_vals)
    plt.xlabel('Time (seconds)')
    plt.ylabel('blueVal')
    plt.title('blueVal vs Time')
    
    # Generate a unique filename based on the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"plot_{timestamp}.png"
    
    plt.savefig(filename)  # Save the plot to a PNG file
    plt.show()

cap.release()
cv2.destroyAllWindows()