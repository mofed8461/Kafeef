import cv2,os
import numpy as np
from PIL import Image 
import pickle
import time
from decimal import Decimal

from picamera.array import PiRGBArray
from picamera import PiCamera


#camera.resolution = (640, 480)
#camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(1.0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/Desktop/Code/trainer/trainer.yml')
cascadePath = "/home/pi/Desktop/Code/Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

#cv2.startWindowThread()
camera = PiCamera()

font = cv2.FONT_HERSHEY_SIMPLEX #cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) #Creates a font


def getConf(id):
    f = open('/home/pi/Desktop/Code/FacesConf/' + str(id), 'r')
    x = f.read()
    f.close()
    
    return Decimal(x)


ids = []
confs = []
newFace = 0

for tries in range(20):

    rawCapture = PiRGBArray(camera)
    cam = cv2.VideoCapture(0)

    camera.capture(rawCapture, format="bgr")
    im = rawCapture.array
    
    #ret, im =cam.read()
    #if ret == False:
    #    continue

    #print(ret)
    #gray=cv2.COLOR_BGR2GRAY

    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        
        
        
        if conf < getConf(nbr_predicted):
            print('new face')
            newFace = newFace + 1
            #exit(-1)
        else:
            ids.append(nbr_predicted)
            confs.append(conf)
        
        if (len(ids) > 5):
            print('exit ' + str(ids[0]))
            exit(ids[0])
        """
        if(nbr_predicted==7):
             nbr_predicted='Deaa' + str(conf)
        elif(nbr_predicted==3):
             nbr_predicted='mofed' + str(conf)
        elif(nbr_predicted==2):
             nbr_predicted='Anirban'
        else :
            nbr_predicted='Unoun'
        """
        ##cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 1,(0,0,255)) #Draw the text
        
        ##print([nbr_predicted, conf])
        #putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
        #print(nbr_predicted)
    #exit(nbr_predicted)
        ##cv2.imshow('im',im)
        ##cv2.waitKey(10)
        
        
        if newFace > 4:
            print('exit -1')
            exit(-1)
            
    if len(faces) == 0 :
        print('no faces')
        #exit(-2)

    #break

print('exit -2')
exit(-2)



