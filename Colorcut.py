import cv2
#Import Numpy
import numpy as np
import os
from array import *
camera_feed = cv2.VideoCapture(0)
a=array('i',[0,0,0,0,0,0])
q=array('i',[0,0,0,0,0,0])
s,ma=0,0
flag=0
while(1):

    _,frame = camera_feed.read()
    #Convert the current frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Define the threshold for finding a blue object with hsv
    lower_blue = np.array([90,100,100])
    upper_blue = np.array([150,300,300])

    #Create a binary image, where anything blue appears white and everything else is black
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #Get rid of background noise using erosion and fill n the holes using dilation and erode the final image on last time
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.erode(mask,element, iterations=2)
    mask = cv2.dilate(mask,element,iterations=2)
    mask = cv2.erode(mask,element)
    
    #Create Contours for all blue objects
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximumArea = 0
    bestContour = None
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > maximumArea:
            bestContour = contour
            maximumArea = currentArea
     #Create a bounding box around the biggest blue object
    if bestContour is not None:
        x,y,w,h = cv2.boundingRect(bestContour)
        cv2.rectangle(frame, (x,y),(x+w+30,y+h+30), (0,0,255), 3)
        roi = frame[y:y+h+30,x:x+w+30]
        #if (w>70):
            
            #if (h>70):
            
        cv2.imshow('con',roi)
        cv2.imwrite('capture1.jpg',roi)
        MIN_MATCH_COUNT = 10
        #size = 2
        #for i in 2
        img1 = cv2.imread('capture1.jpg',0)# queryImage
        for i in range(1,7):
            img2 = cv2.imread('/home/ashaan/database1/'+str(i)+'.jpg',0) # trainImage
            #img2=roi_gray
            # Initiate SIFT detector
            sift = cv2.SIFT()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)

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
            #print i,len(good)
            q[i-1]=len(good)
        ma=max(q)
        for i in range(1,7):
             if(ma==q[i-1]):
                 s=i
                 break
        a[s-1]=a[s-1]+1
        for i in range(1,7):
             if(a[i-1]>3):
                 flag=i
                 print a
                 a=[0,0,0,0,0,0]
                 
                 break
        if (flag==1):
            os.system("mpg321 1.mp3")
            print 'first'
            
        if (flag==2):
            os.system("mpg321 2.mp3")
            print 'second'
        if (flag==3):
            os.system("mpg321 3.mp3")
            print 'third'
        if (flag==4):
           # os.system("mpg321 4.mp3")
           print 'fourth'
        if (flag==5):
           # os.system("mpg321 5.mp3")
           print 'fifth'
        if (flag==6):
           # os.system("mpg321 6.mp3")
           print 'sixth'

        
      
    k = cv2.waitKey(5) & 0xFF
    #If escape is pressed close all windows
    if k == 27:
        break


cv2.destroyAllWindows() 

        
   
