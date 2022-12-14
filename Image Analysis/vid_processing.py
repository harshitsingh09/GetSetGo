import cv2
import numpy as np


kernel = np.ones((4,4),np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

# directory to save the ouput frames
pathIn = "contour_frames_3/"

vid=cv2.VideoCapture("tf.mp4")
i=0
while True:
    i+=1
    ret, frame=vid.read()
    ret, frame2=vid.read()
    frame=cv2.resize(frame,(240,320))
    frame2=cv2.resize(frame2,(240,320))
    new_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    new_img2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    diff_image=cv2.absdiff(new_img,new_img2)
    
    # image thresholding
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
    
    # image dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # shortlist contours appearing in the detection zone
    valid_cntrs = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
            if (y >= 90) & (cv2.contourArea(cntr) < 40):
                break
            valid_cntrs.append(cntr)
            
    # add contours to original frames
    dmy = frame
    cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
    cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), font, 0.6, (0, 180, 0), 2)
    cv2.line(dmy, (0, 80),(256,80),(100, 255, 255))
    dmy=cv2.resize(dmy,(240,320))
    cv2.imshow('test',dmy) 
