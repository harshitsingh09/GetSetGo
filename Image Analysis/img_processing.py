from random import randint

import cv2 as cv
import numpy as np

vid=cv.VideoCapture("tf.mp4")
i=0
while True:
    ret, frame=vid.read()
    """frame[:,:,randint(0,2)]=randint(100,120)
    frame[:,:,randint(0,2)]=randint(120,150)
    frame[:,:,randint(0,2)]=randint(50,90)"""
    i+=1
    if i!=4 and i%4==0:
        new_img=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        new_img=cv.GaussianBlur(new_img,(3,3),cv.BORDER_DEFAULT)
        new_img=cv.Canny(new_img,20,240)
        kernel=np.ones((5,5),int)
        new_img=cv.morphologyEx(new_img,cv.MORPH_CLOSE,kernel)
        cont,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        total=0
        for c in cont:
            peri=cv.arcLength(c,True)
            approx=cv.approxPolyDP(c,peri*0.02,True)
            print(total)
            if len(approx)==4:
                cv.drawContours(frame,[approx],-1,(0,255,0),3)
                cv.imwrite("im_test/"+str(i)+".png",frame)
                total+=1
        cv.imshow('test',frame)
    if cv.waitKey(1)==ord('q'):
      break

    
vid.release()
cv.destroyAllWindows()
