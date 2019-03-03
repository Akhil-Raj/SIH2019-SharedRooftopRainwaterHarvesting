from __future__ import division
import cv2
import numpy as np


def nothing(*arg):
    pass

def temp(filename):
	# Initial HSV GUI slider values to load on program start.
	icol = (36, 202, 59, 71, 255, 255)  # Green
	# icol = (18, 0, 196, 36, 255, 255)  # Yellow
	# icol = (89, 0, 0, 125, 255, 255)  # Blue
	# icol = (0, 100, 80, 10, 255, 255)   # Red
	# cv2.namedWindow('colorTest')
	# # Lower range colour sliders.
	# cv2.createTrackbar('lowR', 'colorTest', icol[0], 255, nothing)
	# cv2.createTrackbar('lowG', 'colorTest', icol[1], 255, nothing)
	# cv2.createTrackbar('lowB', 'colorTest', icol[2], 255, nothing)
	# # Higher range colour sliders.
	# cv2.createTrackbar('highR', 'colorTest', icol[3], 255, nothing)
	# cv2.createTrackbar('highG', 'colorTest', icol[4], 255, nothing)
	# cv2.createTrackbar('highB', 'colorTest', icol[5], 255, nothing)

	# Raspberry pi file path example.
	# frame = cv2.imread('/home/pi/python3/opencv/color-test/colour-circles-test.jpg')
	# Windows file path example.
	filename='static/'+filename
	frame = cv2.imread(filename)
	#frame = cv2.resize(frame, (500, 500))

	# Get HSV values from the GUI sliders.
	# lowR = cv2.getTrackbarPos('lowR', 'colorTest')
	# lowG = cv2.getTrackbarPos('lowG', 'colorTest')
	# lowB = cv2.getTrackbarPos('lowB', 'colorTest')
	# highR = cv2.getTrackbarPos('highR', 'colorTest')
	# highG = cv2.getTrackbarPos('highG', 'colorTest')
	# highB = cv2.getTrackbarPos('highB', 'colorTest')


	# Blur methods available, comment or uncomment to try different blur methods.
	frameBGR = frame
	# frameBGR = cv2.medianBlur(frameBGR, 7)
	# frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
	"""kernal = np.ones((15, 15), np.float32)/255
	frameBGR = cv2.filter2D(frameBGR, -1, kernal)"""

	# Show blurred image.
	#cv2.imshow('blurred', frameBGR)

	# HSV (Hue, Saturation, Value).
	# Convert the frame to HSV colour model.
	hsv = frameBGR

	# HSV values to define a colour range.
	colorLow = np.array([188, 185, 194])
	colorHigh = np.array([225, 206, 232])
	mask = cv2.inRange(hsv, colorLow, colorHigh)
	# Show the first mask
	#cv2.imshow('mask-plain', mask)

	#kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
	#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

	# Show morphological transformation mask
	#cv2.imshow('mask', mask)

	# Put mask over top of the original image.
	result = cv2.bitwise_and(frame, frame, mask=mask)


	# Show final output image
	cv2.imwrite("static/onlyBuilding.jpg", result)

	#k = cv2.waitKey(5) & 0xFF
	#if k == 27:
	#    break

	cv2.destroyAllWindows()