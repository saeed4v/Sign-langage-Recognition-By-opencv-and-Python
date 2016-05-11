import cv2
import numpy as np
#frame1 = cv2.imread('/home/pi/Desktop/Signlang/k.jpg',0)    
hsv = cv2.cvtColor('k.jpg', cv2.COLOR_BGR2HSV)

kernel = np.ones((5,5),np.float32)/25
cv2.filter2D('k.jpg',-5,kernel)
cv2.blur('k.jpg',(10,32))
lower_red = np.array([90,100,100])
upper_red = np.array([150,360,360])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and('k.jpg','k.jpg', mask= mask)
cv2.dilate(frame1,kernel,iterations = 1)
cv2.morphologyEx('k.jpg', cv2.MORPH_OPEN, kernel)
cv2.morphologyEx('k.jpg', cv2.MORPH_CLOSE, kernel)

cv2.imshow('res',res)

