# TechVidvan Vehicle counting and Classification

# Import necessary packages

import time

import cv2
import numpy as np

from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap=cv2.VideoCapture(r'C:\Users\Dell\Documents\GitHub\GetSetGo\Image Analysis\model1.mp4')
input_size = 320

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 550   
up_line_position = middle_line_position - 30
down_line_position = middle_line_position + 30


# Store Coco Names in a list
classesFile = r"C:\Users\Dell\Documents\GitHub\GetSetGo\Image Analysis\vehicle-detection-classification-opencv\coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = r'C:\Users\Dell\Documents\GitHub\GetSetGo\Image Analysis\vehicle-detection-classification-opencv\yolov3-320.cfg'
modelWeigheights = r'C:\Users\Dell\Documents\GitHub\GetSetGo\Image Analysis\vehicle-detection-classification-opencv\yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend


# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id
    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1  

    print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)
        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        cv2.putText(img,f'{"vehicle "} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

def realTime():
    i=0
    prev_frame=0
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(1080,720))
        if(i%2==0):
            ih, iw, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            # Set the input of the network
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
            # Feed data to the network
            outputs = net.forward(outputNames)
            total=0
            # Find the objects from the network output
            postProcess(outputs,img)

            # Draw the crossing lines

            cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
            cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
            cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)
            # Draw counting texts in the frame
            cv2.putText(img, "vehicles:        "+str(sum(up_list)+sum(down_list))+"\n", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            test=(sum(up_list)+sum(down_list))
            new_frame=time.time()
            prev_frame=prev_frame+1/new_frame
            if(int(prev_frame)>50 and test*17*16<1080):
                test=0
            if(test*17*16>1080):
                 cv2.putText(img, "Condition:       "+"Heavy Traffic", (20, 60),cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            else:
                cv2.putText(img, "Condition:       "+"Light Traffic", (20, 60),cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

            # Show the frames
            cv2.imshow('Output', img)
            if cv2.waitKey(1) == ord('q'):
                break
        i+=1

    # Write the vehicle counting information in a file and save it
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realTime()