import cv2
import numpy as np
from utils import *

### Camera options ###
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)

cap.set(3,frameWidth)
cap.set(4,frameHeight)

### Window options ###
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",55,255,empty)
cv2.createTrackbar("Threshold2","Parameters",50,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)


while True:
    success,img = cap.read()
    if success:
        imgContour = img.copy()

        imgBlur = cv2.GaussianBlur(img,(7,7),1)
        imgGray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)
        threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")
        minArea = cv2.getTrackbarPos("Area","Parameters")


        imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
        kernel = np.ones((5,5))
        imgDil = cv2.dilate(imgCanny,kernel,iterations=1)

        getContours(imgDil,imgContour,minArea)

        imgStack = stackImages(0.8,([img,imgGray,imgCanny],[img,imgDil,imgContour]))
        cv2.imshow("Result", imgStack)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break