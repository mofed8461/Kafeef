import cv2,os
import numpy as np
from PIL import Image 
import pickle
import time

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
path = '/home/pi/Desktop/Code/dataSet'

#cv2.startWindowThread()
camera = PiCamera()

font = cv2.FONT_HERSHEY_SIMPLEX #cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) #Creates a font
while True:

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
        if nbr_predicted != -1:
            file = open('/home/pi/Desktop/Code/FacesConf/' + str(nbr_predicted), 'w')
            file.write(str(conf - 3))
            file.close()
            exit(0)
   
    if len(faces) == 0 :
        print('no faces')
        #exit(-2)

    #break






