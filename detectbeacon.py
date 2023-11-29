# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 04:01:57 2022

@author: cosmi

Load the video
Extract an image from the video in order to elect region of interest (ROI) 
by drawing a circle and locking around it
Go through the entire video to detect pixel changes in the ROI
Log/plot the time when the changes exceed a threshold

"""

import cv2
import numpy as np

import warnings

warnings.filterwarnings("ignore")

cap = cv2.VideoCapture('video_test2.mp4')

while True:
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #lower_yellow = np.array([20, 0, 0])
        #upper_yellow = np.array([40, 255, 255])
        #lower_red = np.array([160, 105, 84])
        #upper_red = np.array([179, 255, 255])
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
        #circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=30)
        

        # Find changes in the masked grayscale image
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
    else:
        print("Read Failed")

#wait for q key to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()