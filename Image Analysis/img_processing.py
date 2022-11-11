from random import randint
import cv2 as cv
import numpy as np

vid=cv.VideoCapture("tf.mp4")
i=0
while True:
    ret, frame=vid.read()
    ret, frame2=vid.read()
    frame=cv.resize(frame,(240,320))
    frame2=cv.resize(frame2,(240,320))
    """frame[:,:,randint(0,2)]=randint(100,120)
    frame[:,:,randint(0,2)]=randint(120,150)
    frame[:,:,randint(0,2)]=randint(50,90)"""
    i+=1

    new_img=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    new_img2=cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    new_img=cv.absdiff(new_img,new_img2)
    cv.imshow('test',new_img)
    ret,new_img=cv.threshold(new_img,30,250,cv.THRESH_BINARY)
    kernel=np.ones((5,5),int)
    new_img=cv.dilate(new_img,kernel,iterations=1)
        
    cont,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    total=0
    for c in cont:
        #peri=cv.arcLength(c,True)
        #approx=cv.approxPolyDP(c,peri*0.02,True)
        print(total)
        if cv.contourArea(c)>=40:
            x,y,w,h = cv.boundingRect(c)
            if(x<=200 and y>=80):   
                total+=1
                cv.drawContours(frame,[approx],-1,(0,255,0),3)
            cv.imwrite("im_test/"+str(i)+".png",frame)
            
    
    
        #cv.waitKey(2000)
        
    if cv.waitKey(1)==ord('q'):
      break

    
vid.release()
cv.destroyAllWindows()
