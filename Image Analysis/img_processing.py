from random import randint
import time
import cv2 as cv
import numpy as np

vid=cv.VideoCapture("C:\\Users\\Dell\\Documents\\GitHub\\GetSetGo\\Image Analysis\\car.mp4")
i=0
total=0
prev_frame_time=0
fps=500
bg=cv.createBackgroundSubtractorMOG2(history=500,detectShadows=False)
while True:
    ret, frame=vid.read()
    frame=cv.resize(frame,(280,420))
    i+=1
    new_img=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    kernel=np.ones((5,5),dtype="uint8")
    ret,new_img=cv.threshold(new_img,30,255,cv.THRESH_BINARY)
    new_img=bg.apply(new_img)
    new_img=cv.erode(new_img,kernel,iterations=1)
    new_img=cv.dilate(new_img,kernel,iterations=2)
    cont,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    frame_copy=frame.copy()
    for c in cont:
        if cv.contourArea(c)>400:
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame_copy,(x,y),(x+w,y+h),(0,255,0),2)  
            total+=1
    print(total//fps)
    fore_grnd=cv.bitwise_and(frame,frame,mask=new_img)
    stacked=np.hstack((frame,fore_grnd,frame_copy))
    cv.imwrite("C:\\Users\\Dell\\Documents\\GitHub\\GetSetGo\\Image Analysis\\im_test\\"+str(i)+".jpeg",stacked)
    cv.imshow('test',stacked) 
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    if cv.waitKey(1)==ord('q'):
      break

    
vid.release()
cv.destroyAllWindows()
