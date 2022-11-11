import cv2 as cv
from random import randint
vid=cv.VideoCapture("an.mp4")
vid.set(3,1080)
vid.set(4,1920)
vid.set(6,1)
i=0
while True:
    ret, frame=vid.read()
    """frame[:,:,randint(0,2)]=randint(100,120)
    frame[:,:,randint(0,2)]=randint(120,150)
    frame[:,:,randint(0,2)]=randint(50,90)"""
    print(frame)
    cv.imshow('video',frame)
    cv.waitKey(3)
    i+=1
    if i!=4 and i%4==0:
        new_img=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        new_img=cv.GaussianBlur(new_img,(3,3),cv.BORDER_DEFAULT)
        new_img=cv.Canny(new_img,20,240)
        kernel=np.ones((5,5),uint8)
        new_img=cv.morphologyEx(new_img,cv.MORPH_CLOSE,kernel)
        cont,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        total=0
        for c in cont:
            peri=cv.arcLength(c,True)
            approx=cv.
            
                
        cv.imwrite("im_test/"+str(i)+".png",new_img)
    if cv.waitKey(1)==ord('q'):
      break
    
vid.release()
cv.destroyAllWindows()
