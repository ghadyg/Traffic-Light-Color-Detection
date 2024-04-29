import cv2
image=cv2.imread("yellow.jpe")
ym=int(image.shape[0]/3)

ymi=int(image.shape[0]/3)
x=image.shape[1]
print (ym,ymi,x,image.shape[0],image.shape[1])
im1=image[ymi:,:image.shape[1]]
print(im1.shape)
print(image.shape)
cv2.imshow("frame",im1)

cv2.waitKey(0)
cv2.destroyAllWindows()