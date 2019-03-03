# your code goes here

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def prImg(img, frameName = 'img', time = 0):
    cv2.imshow(frameName, img)
    return cv2.waitKey(time)


def draw_circle(event,x,y,flags,param):
    global ix, iy, oldx, oldy, drawing, mode, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        oldx, oldy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (oldx, oldy), (0, 0, 0), 1)
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
                oldx, oldy = x, y

            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (oldx, oldy), (0, 0, 0), 1)
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
            oldx, oldy = x, y

        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)


def canny2(filename):
    imgOrig = cv2.imread(filename)
    imgOrig = cv2.resize(imgOrig, (500, 500))
    img = cv2.cvtColor(imgOrig, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)
    canny = cv2.Canny(img, 40, 500)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    catchmentArea = 0
    # print(contours)
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append([cX, cY])
        else:
            cX, cY = 0, 0
        catchmentArea += cv2.contourArea(c)
        cv2.circle(imgOrig, (cX, cY), 3, (0, 0, 0), -1)

    cv2.imwrite("static/imageWithClusters.png", imgOrig)
    return (points, catchmentArea)
