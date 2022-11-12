from random import randint

import cv2 as cv
import numpy as np

vid=cv.VideoCapture("C:\\Users\\Dell\\Documents\\GitHub\\GetSetGo\\Image Analysis\\tufu.mp4")
i=0
total=0
bg=cv.createBackgroundSubtractorMOG2(history=4,detectShadows=False)
while True:
    ret, frame=vid.read()
    frame=cv.resize(frame,(280,420))
    i+=1
    new_img=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    kernel=np.ones((3,3),dtype="uint8")
    ret,new_img=cv.threshold(new_img,30,255,cv.THRESH_BINARY)
    new_img=bg.apply(new_img)
    new_img=cv.erode(new_img,kernel,iterations=1)
    new_img=cv.dilate(new_img,kernel,iterations=2)
    cont,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    frame_copy=frame.copy()
    for c in cont:
        peri=cv.arcLength(c,True)
        approx=cv.approxPolyDP(c,peri*0.02,True)
        print(total)
        if cv.contourArea(c)>400 and len(approx):
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame_copy,(x,y),(x+w,y+h),(0,255,0),2)
            if(x<=200 and y>=80):   
                total+=1
    fore_grnd=cv.bitwise_and(frame,frame,mask=new_img)
    stacked=np.hstack((frame,fore_grnd,frame_copy))
    cv.imwrite("C:\\Users\\Dell\\Documents\\GitHub\\GetSetGo\\Image Analysis\\im_test\\"+str(i)+".jpeg",stacked)
    cv.imshow('test',stacked) 
    if cv.waitKey(1)==ord('q'):
      break

    
vid.release()
cv.destroyAllWindows()
