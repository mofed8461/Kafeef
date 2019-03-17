import cv2
import sys

if len(sys.argv) != 2:
    exit(-1)

from picamera.array import PiRGBArray
from picamera import PiCamera
camera = PiCamera()

#cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('/home/pi/Desktop/Code/Classifiers/face.xml')
i=0
offset=50
#name=raw_input('enter your id')
name = sys.argv[1]
while True:
    #ret, im =cam.read()

    rawCapture = PiRGBArray(camera)
    cam = cv2.VideoCapture(0)

    camera.capture(rawCapture, format="bgr")
    im = rawCapture.array
    
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        #cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        #cv2.waitKey(100)
    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break

