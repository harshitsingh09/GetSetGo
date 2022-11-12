from random import randint

import cv2 as cv
import numpy as np

vid=cv.VideoCapture("C:\\Users\\Dell\\Documents\\GitHub\\GetSetGo\\Image Analysis\\tf.mp4")
i=0
total=0
while True:
    ret, frame=vid.read()
    ret, frame2=vid.read()
    frame=cv.resize(frame,(240,320))
    frame2=cv.resize(frame2,(240,320))
    i+=1
    new_img=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    new_img2=cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    new_img=cv.absdiff(new_img,new_img2)
    ret,new_img=cv.threshold(new_img,30,250,cv.THRESH_BINARY)
    kernel=np.ones((5,5),int)
    new_img=cv.dilate(new_img,kernel,iterations=1)
    cont,_=cv.findContours(new_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for c in cont:
        peri=cv.arcLength(c,True)
        approx=cv.approxPolyDP(c,peri*0.035,True)
        #print(total)
        if cv.contourArea(c)>=90 and len(approx)==5:
            x,y,w,h = cv.boundingRect(c)
            print(x,y)
            if(x<=200 and y>=80):   
                total+=1
                cv.drawContours(frame,[approx],-1,(0,255,0),1)    
                cv.imwrite("C:\\Users\\Dell\\Documents\\GitHub\\GetSetGo\\Image Analysis\\im_test\\"+str(i)+".jpeg",frame)
        #cv.waitKey(2000)
          
    cv.imshow('test',frame) 
     
    if cv.waitKey(1)==ord('q'):
      break

    
vid.release()
cv.destroyAllWindows()
