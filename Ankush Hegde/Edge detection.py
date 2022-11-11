import cv2 as cv;
import numpy as np

img = cv.imread(r'C:\Users\Ankush Hegde\Desktop\GetSetGo\Ankush Hegde\road-city-busy-motorbike.jpg')
"""cv.imshow('ig',img)
def rescale(frame,scale = 0.1):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_CUBIC)

# OR 

scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  

res = cv.resize(img, dim, interpolation = cv.INTER_CUBIC)"""

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('image',gray)
#----------------------------------------------------------
# laplacian method of edge detection 
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('laplacian',lap)


cv.waitKey(0)