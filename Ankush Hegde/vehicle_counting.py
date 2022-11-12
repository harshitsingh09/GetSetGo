import cv2 as cv
import glob
from vehicle_detector import VehicleDetector

# Load Veichle Detector
vd = VehicleDetector()
img = cv.imread(r'C:\Users\Ankush Hegde\Desktop\GetSetGo\Ankush Hegde\traffic-car-vehicle-way-wallpaper-preview.jpg')

box = vd.detect_vehicles(img)
count = len(box)
for i in box:
    x,y,w,h = i
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    cv.putText(img,"Vechiles:" + str(count),(20,50),1,1,(100,200,0),2)


cv.imshow('img',img)
cv.waitKey(0)
