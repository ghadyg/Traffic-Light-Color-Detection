import cv2
import numpy as np


img =cv2.resize(cv2.imread("C:/Users/Ghadi/Desktop/cv open tuto/ball.png"),(0,0),fx=2,fy=2)
img2 = img
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
upper=np.array([179,255,245])
lower=np.array([90,100,0])
mask=cv2.inRange(hsv,lower,upper)
B=np.ones((5,5),np.uint8)
mask1=cv2.dilate(mask,B,iterations=1)
result=cv2.medianBlur(mask,11)



contours,ret =cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    cv2.drawContours(img,cnt,-1,(255,0,0),5)
cv2.imshow('result',result)
cv2.imshow('img',img)

contours1,ret1 =cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for cnt in contours1:
    cv2.drawContours(img2,cnt,-1,(0,255,0),5)    


cv2.imshow('mask1',mask1)
cv2.imshow('img2',img2)


cv2.waitKey(0)



