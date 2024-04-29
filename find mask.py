import  numpy as np
import cv2

def empty(a):
    pass

cv2.namedWindow("trackbars")
cv2.resizeWindow("trackbars",640,240)

    
cv2.createTrackbar("Hm",'trackbars',0,179,empty)
cv2.createTrackbar("HM",'trackbars',179,179,empty)
cv2.createTrackbar("Sm",'trackbars',0,255,empty)
cv2.createTrackbar("SM",'trackbars',255,255,empty)
cv2.createTrackbar("Vm",'trackbars',0,255,empty)
cv2.createTrackbar("VM",'trackbars',255,255,empty)
while True:
    h_min=cv2.getTrackbarPos("Hm",'trackbars')
    h_max=cv2.getTrackbarPos("HM",'trackbars')
    s_min=cv2.getTrackbarPos("Sm",'trackbars')
    s_max=cv2.getTrackbarPos("SM",'trackbars')
    v_min=cv2.getTrackbarPos("Vm",'trackbars')
    v_max=cv2.getTrackbarPos("VM",'trackbars')
 

    img =cv2.imread('traffic-light-green.jpg')
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsi=cv2.cvtColor(img,cv2.COLOR_BGR2HSI)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])    
    
    mask=cv2.inRange(hsv,lower,upper)
    mask=cv2.medianBlur(mask, 5)
  
    result=cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('frame',hsi)
    #cv2.imshow('frame',mask)
    #cv2.imshow('frame1',hsv)
    #cv2.imshow('frame2',img)
    #cv2.imshow('frame3',result)
    if cv2.waitKey(1)==ord('q'):
        break
       
