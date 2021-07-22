import cv2
import numpy as np
from time import sleep

def get_centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

min_width=35 #Minimum Rectangular Width
max_width=300 #Maximum Rectangular Width
min_height=25 #Minimum Rectangular Height
max_height=100 #Maximum Rectangular Height
offset=6 #Allowable error between pixel
line_position=550 #Count line position 
vfps= 60 #Vídeo FPS
detected = [] #detected
num_cars= 0 #Number of cars crossing the line
num_cars_frame=0

cap = cv2.VideoCapture('video2.mp4')
subtractor = cv2.createBackgroundSubtractorMOG2()
#Mixture of Gaussians
#subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

f=[]
while True:
    ret , frame1 = cap.read()
    tempo = float(1/vfps)
    sleep(tempo) 
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grayscale" , gray) 
    blur=cv2.medianBlur(gray,5)
    #blur = cv2.GaussianBlur(gray,(3,3),5) 
    #blur = cv2.blur(gray,(3,3),5)
    img_sub = subtractor.apply(blur)
    #cv2.imshow("Subtracted" , img_sub)  
    dilated = cv2.dilate(img_sub,np.ones((5,5))) #dilated
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _,dilated= cv2.threshold(dilated, 220, 255, cv2.THRESH_BINARY)    
    dilated = cv2.dilate(dilated,np.ones((5,5))) #dilated
    dilated2 = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel) #dilated
    dilated2 = cv2.morphologyEx (dilated2, cv2. MORPH_OPEN , kernel)
    contours,h=cv2.findContours(dilated2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    num_cars_frame=0
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_contour = (w >= min_width and w<=max_width) and (h >= min_height and h<=max_height) #validate_contour
        if not valid_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centre = get_centre(x, y, w, h)
        detected.append(centre)
        cv2.circle(frame1, centre, 4, (0, 0,255), -1)
        num_cars_frame+=1
       
    #cv2.putText(frame1, "TOTAL VEHICLE COUNT : "+str(num_cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.putText(frame1, "VEHICLE COUNT : "+str(num_cars_frame), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)    
    #cv2.imshow("Detected",dilated2)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
