import cv2
imgOrig = cv2.imread('map.png')
imgOrig = cv2.resize(imgOrig, (500, 500))
cv2.imshow("img", imgOrig)
cv2.waitKey(0)