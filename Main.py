import cv2
import numpy as np
import os
import picamera
from array import *

a=array('i',[0,0,0,0,0,0])
q=array('i',[0,0,0,0,0,0])
s,ma=0,0
flag=0

#cap = cv2.VideoCapture(0)
camera_feed = picamera.PiCamera()
camera_feed.resolution = (340,240)

while(1):
    camera_feed.capture('image.bmp')
    frame1=cv2.imread('image.bmp')
    #_, frame1 = cap.read()
    
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((5,5),np.float32)/25
    cv2.filter2D(frame1,-5,kernel)
    cv2.blur(frame1,(10,32))
    lower_red = np.array([90,100,100])
    upper_red = np.array([150,360,360])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame1,frame1, mask= mask)
    cv2.dilate(frame1,kernel,iterations = 1)
    cv2.morphologyEx(frame1, cv2.MORPH_OPEN, kernel)
    cv2.morphologyEx(frame1, cv2.MORPH_CLOSE, kernel)
    
    
    cv2.imshow('frame',frame1)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
##    frame=res
##    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
##    mask = cv2.erode(mask,element, iterations=2)
##    mask = cv2.dilate(mask,element,iterations=2)
##    #mask = cv2.dilate(mask,element,iterations=2)
##    mask = cv2.erode(mask,element)
##    
##    #Create Contours for all blue objects
##    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##    
##    
##    maximumArea = 0
##    bestContour = None
##    for contour in contours:
##        currentArea = cv2.contourArea(contour)
##        if currentArea > maximumArea:
##            bestContour = contour
##            maximumArea = currentArea
##     #Create a bounding box around the biggest blue object
##    if bestContour is not None:
##        x,y,w,h = cv2.boundingRect(bestContour)
##        cv2.rectangle(frame, (x,y),(x+w+30,y+h+30), (0,0,255), 3)
##        roi = frame[y:y+h+30,x:x+w+30]
##       # cv2.drawContours(roi,bestContour,-1,(0,255,0),-1)
        
         
        
    
    
    cv2.imwrite('capture1.jpg',res)
    MIN_MATCH_COUNT = 10
    #size = 2
    #for i in 2
    img1 = cv2.imread('capture1.jpg',0)# queryImage
    for i in range(1,7):
        img2 = cv2.imread(str(i)+'.jpg',0) # trainImage
        #img2=roi_gray
        # Initiate SIFT detector
        sift = cv2.SIFT()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

                # find the keypoints and descriptors with SIFT
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
       # print i,len(good)
        q[i-1]=len(good) 
        
    ma=max(q)
    #print q
    for i in range(1,7):
         if(ma==q[i-1]):
             s=i
             break
    #print ma,q       
    a[s-1]=a[s-1]+1
    for i in range(1,7):
         if(a[i-1]>2):
             flag=i
            # print a
             a=[0,0,0,0,0,0]
             break
    if (flag==1):
        os.system("mpg321 /home/pi/Desktop/Signlang/HOWMUCH.mp3")
        print 'first'
        flag=0
    if (flag==2):
        os.system("mpg321 /home/pi/Desktop/Signlang/GOODJOB.mp3")
        print 'second'
        flag=0
    if (flag==3):
        os.system("mpg321 /home/pi/Desktop/Signlang/WATER.mp3")
        print 'third'
        flag=0
    if (flag==4):
       os.system("mpg321 /home/pi/Desktop/Signlang/excuseme.mp3")
       print 'fourth'
       flag=0
    if (flag==5):
       os.system("mpg321 /home/pi/Desktop/Signlang/1.mp3")
       print 'fifth'
       flag=0
    if (flag==6):
       os.system("mpg321 /home/pi/Desktop/Signlang/haiall.mp3")
       print 'sixth'
       flag=0
    
    #Use this command to prevent freezes in the feed
    k = cv2.waitKey(5) & 0xFF
    #If escape is pressed close all windows
    if k == 27:
        break


cv2.destroyAllWindows() 
