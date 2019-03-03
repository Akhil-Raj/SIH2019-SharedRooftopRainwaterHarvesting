# your code goes here

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def prImg(img, frameName = 'img', time = 0):
    cv2.imshow(frameName, img)
    return cv2.waitKey(time)

def drawOpenCVSymbol():
    side = 100
    len = 500
    breadth = 500
    gap = 5
    width = 30
    radius = (int)(side / 2 - gap / 2 - width / 2)
    thickness = 2

    PI = np.pi
    #print(PI)
    img = np.full((len, breadth, 3), 255, np.uint8)
    centerTop = (int(len / 2), int(len / 2 - (side * PI / 2 - (side / 2) / PI)))
    centerLeft = (int(len / 2 - side / 2), (int)(len / 2 + (side / 2) / PI))
    centerRight = (int(len / 2 + side / 2), (int)(len / 2 + (side / 2) / PI))
    #print(centerTop, centerLeft, centerRight)
    #print(np.array(centerLeft))
    cv2.ellipse(img, centerTop, (radius, radius), 0, 120, 420, (0, 0, 255), width, cv2.LINE_AA)
    cv2.ellipse(img, centerLeft, (radius, radius), 0, 0, 300, (0, 255, 0), width, cv2.LINE_AA)
    cv2.ellipse(img, centerRight, (radius, radius), 0, 300, 240 + 360, (255, 0, 0), width, cv2.LINE_AA)
    #cv2.polylines(img, [np.array([centerTop, centerRight, centerLeft], dtype=np.int32)], True, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([centerTop, centerRight, centerLeft], dtype=np.int32)], (255, 255, 255), lineType=cv2.LINE_AA)
    #print(np.array([centerTop, centerRight, centerLeft]))
    #print([np.array([[910, 641], [206, 632], [696, 488], [458, 485]])])
    #cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
    ims(img)

def ims(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

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

def nothing(x):
    pass

def mouseAsPaintBrush():
    global ix, iy, drawing, mode, img
    drawing = False  # true if mouse is pressed
    mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
    ix, iy = -1, -1
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break

    cv2.destroyAllWindows()

def trackbarAsColorPalette():
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

    cv2.destroyAllWindows()


def arithOps():
    # Load two images
    imgOld = cv2.imread('imgOld.JPG')
    imgOld = imgOld[0:3308, 0:5184]
    imgNew = cv2.imread('imgNew.JPG')
    imgNew = imgNew[0:3308, 0:5184]
    print(imgOld.shape, imgNew.shape)
    for i in np.arange(0, 1, 0.01):
        img = cv2.addWeighted(imgOld, i, imgNew, 1 - i, 0)
        #cv2.imwrite(os.path.join('/media/akhil/Code/sublimeFiles/transitioningImages', 'img' + str(int(i * 100)) + '.png'), img);
        img = cv2.resize(img, (800, 400))
        prImg(img, 1)

def giveHSV(event, x, y, flags, params):
    global drag
    if event == cv2.EVENT_LBUTTONDOWN:
        drag = True
        getHSV.append(hsv[y][x])

    elif event == cv2.EVENT_MOUSEMOVE and drag:
        #print('asdsad')
        getHSV.append(hsv[y][x])

    elif event == cv2.EVENT_LBUTTONUP:
        drag = False

def adapThresh():

    img = cv2.imread('chess.jpeg', 0)
    img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 71, 15)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def changingColorspaces():
    global drag
    drag = False
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame')
    cv2.namedWindow('draw')
    draw = np.full((500, 500), 255, np.uint8)
    cv2.setMouseCallback('frame', giveHSV)
    global hsv, getHSV
    getHSV = []
    while (1):

        # Take each frame
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        #cv2.fli
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #print(hsv)
        # define range of blue color in HSV
        mask = False
        for i in getHSV:
            lower_lim = np.array(i) - 1
            upper_lim = np.array(i) + 1

            # Threshold the HSV image to get only blue colors
            if mask is False:
                mask = cv2.inRange(hsv, lower_lim, upper_lim)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_lim, upper_lim))

        # Bitwise-AND mask and original image
        if mask is False:
            res = cv2.bitwise_and(frame, frame)
        else:
            (y, x) = (np.mean(np.where(mask == 255), 1))
            if not(math.isnan(x) or math.isnan(y)):
                x = int(x)
                y = int(y)
                print(x, y)
                cv2.putText(draw, '.', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            res = cv2.bitwise_and(frame, frame, mask=mask)
        #cv2.putText(frame, 'OpenCV', (10, 500), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        #cv2.imshow('hsv', hsv)
        cv2.imshow('mask', mask)
        cv2.imshow('draw', draw)
        #cv2.imshow('res', res)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('q'):
            draw = np.full((500, 500), 255, np.uint8)
    cv2.destroyAllWindows()

def getPoints(event, x, y, flags, param):
    global pts, img
    if event == cv2.EVENT_LBUTTONDOWN and getPoints.count < 3:
        pts.append([x, y])
        getPoints.count += 1
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

getPoints.count = 0

def geoTransfOfImgs():
    global pts, img
    pts = []
    img = cv2.imread('note.jpeg')
    prImg(img, time=1)
    cv2.setMouseCallback('img', getPoints)
    while(getPoints.count < 3):
        if(prImg(img, time = 1) == 27):
            return
    M = cv2.getAffineTransform(np.array(np.float32(pts)), np.array(np.float32([[0, 0], [500, 0], [500, 500]])))
    dst = cv2.warpAffine(img, M, (500, 500))
    prImg(dst, 'newImg', 0)

def imgThresh():
    img = cv2.imread('chess.jpeg', 0)

    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()

def smoothingImages():
    img = cv2.imread('messi.jpeg')
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.GaussianBlur(img, (5, 5), 0)
    dst2 = cv2.bilateralFilter(img, 9, 75, 75)

    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(dst2), plt.title('bil')
    plt.xticks([]), plt.yticks([])
    plt.show()

def morphTrans():
    img = cv2.imread('face.jpeg', 0)
    prImg(img)
    kernel = np.ones((5, 55), np.uint8) * 5
    erosion = cv2.erode(img, kernel, iterations=1)
    prImg(erosion, 'new')
    cv2.morphologyEx()

def imgGrads():
    img = cv2.imread('chess.jpeg', 0)

    # Output dtype = cv2.CV_8U
    sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap='gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap='gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

    plt.show()

def canny():
    cv2.namedWindow('img')
    cv2.namedWindow('origImg')
    img = cv2.imread('/home/akhil/Desktop/final1.png', 0)
    img = cv2.resize(img, (500, 500))
    prImg(img, 'origImg', time=1)
    edges = cv2.Canny(img, 100, 200)
    cv2.createTrackbar('Lower Threshold', 'img', 0, 1000, nothing)
    cv2.createTrackbar('Upper Threshold', 'img', 0, 1000, nothing)

    while(1):
        low = cv2.getTrackbarPos('Lower Threshold', 'img') # optimal 90
        high = cv2.getTrackbarPos('Upper Threshold', 'img') # optimal 380
        canny = cv2.Canny(img, 90, 380)
        if prImg(canny, time=1) == 27:
            break

def canny2():
    #cv2.namedWindow('img')
    cv2.namedWindow('origImg')
    imgOrig = cv2.imread('/media/akhil/Code/SIH/GS-PS/presentationThings/finalPresentation/onlyBuilding.jpg')
    imgOrig = cv2.resize(imgOrig, (500, 500))
    img = cv2.cvtColor(imgOrig, cv2.COLOR_RGB2GRAY)
    prImg(img, 'origImg', time=0)
    edges = cv2.Canny(img, 100, 200)
    canny = cv2.Canny(img, 40, 500)
    #prImg(canny)
    #print("SSSSSS")
    im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    catchmentArea = 0
    print(contours)
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
        #cv2.putText(imgOrig, "", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # display the image
    #print("SSSSSSSSSSSS")
    #print("catchmentArea : " + str(catchmentArea))
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)
    #mx, my = np.mean(points, 0)
    #cv2.circle(imgOrig, (int(mx), int(my)), 5, (0, 255, 0), -1)
    #cv2.putText(img, "Mean", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Image", imgOrig)
    cv2.imwrite("imageWithClusters.png", imgOrig)
    cv2.waitKey(0)
    return (points, catchmentArea)

def cannyAndContours():
    cv2.namedWindow('img')
    cv2.namedWindow('origImg')
    img = cv2.imread('/home/akhil/Desktop/satImg10.jpeg', 0)
    #img = cv2.resize(img, (250, 250))
    prImg(img, 'origImg', time=1)
    edges = cv2.Canny(img, 100, 200)
    cv2.createTrackbar('Lower Threshold', 'img', 0, 1000, nothing)
    cv2.createTrackbar('Upper Threshold', 'img', 0, 1000, nothing)

    while(1):
        low = cv2.getTrackbarPos('Lower Threshold', 'img') # optimal 90
        high = cv2.getTrackbarPos('Upper Threshold', 'img') # optimal 380
        canny = cv2.Canny(img, 90, 380)

        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img2 = np.zeros(img.shape)
        for i in range(len(contours)):
            cnt = contours[i]
            #area = cv2.contourArea(cnt)

            if cv2.contourArea(cnt) < 10 or cv2.contourArea(cnt) > 250:
               print(contours[i])
               contours[i] = np.array([[[0, 0]]])

        cv2.drawContours(img2, contours, -1, 255, 1)


        if prImg(canny, time=1) == 27 or prImg(img2, "iasdasdmg2", time=1) == 27:
            break

def imgPyrs():
    A = cv2.imread('apple.jpg')
    B = cv2.imread('orange.jpg')

    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
        #prImg(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
        #prImg(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
        LS.append(ls)
        #prImg(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, 6):
        #prImg(ls_)
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))

    cv2.imwrite('Pyramid_blending2.jpg', ls_)
    cv2.imwrite('Direct_blending.jpg', real)

def contours1():
    im = cv2.imread('/media/akhil/Code/SIH/GS-PS/presentationThings/final1.png')
    im = cv2.resize(im, (500, 500))
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    prImg(imgray, 'i1', time=1)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    #prImg(thresh, 'i2', time=1)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = np.zeros(img.shape)
    cv2.drawContours(img2, contours, -1, 255, 1)
    prImg(img2)

def contours3():
    cv2.namedWindow('img')
    cv2.namedWindow('origImg')
    im = cv2.imread('/media/akhil/Code/SIH/GS-PS/presentationThings/final1.png')
    im = cv2.resize(im, (500, 500))
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    prImg(imgray, 'origImg', time=1)

    cv2.createTrackbar('Lower Threshold', 'img', 0, 1000, nothing)
    cv2.createTrackbar('Upper Threshold', 'img', 0, 1000, nothing)
    areaThresh = 0

    while (1):
        low = cv2.getTrackbarPos('Lower Threshold', 'img')
        high = cv2.getTrackbarPos('Upper Threshold', 'img')

        ret, thresh = cv2.threshold(imgray, 67, 255, 0)
        # prImg(thresh, 'i2', time=1)
        img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

           # if area < 20:
            #    print(contours[i])
            #    contours[i] = np.array([[[0, 0]]])
            print(area)

        img2 = np.zeros(img.shape)
        cv2.drawContours(img2, contours, -1, 255, 1)
        if prImg(img2, time=1) == 27:
            break


def contours2():
    img = cv2.imread('messi.jpeg', 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    #prImg(thresh, time=0)
    img, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    idx = 3
    img2 = np.zeros(img.shape)
    cv2.drawContours(img2, contours, idx, 255, 1)
    prImg(img2, time=0)
    #print(contours)
    cnt = contours[idx]
    M = cv2.moments(cnt)
    print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print(perimeter, area, cx, cy)

if __name__ == '__main__':
    #drawOpenCVSymbol()
    #mouseAsPaintBrush()
    #trackbarAsColorPalette()
    #arithOps()
    #adapThresh()
    #changingColorspaces()
    #geoTransfOfImgs()
    #imgThresh()
    #smoothingImages()
    #morphTrans()
    #imgGrads()
    #canny()
    #imgPyrs()
    #contours1()
    #contours2()
    #contours3()
    canny2()
    #cannyAndContours()