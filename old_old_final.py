import speech_recognition as sr
from types import FunctionType

from subprocess import call
import RPi.GPIO as GPIO
import subprocess
from multiprocessing import Process
from multiprocessing import Queue
import pyaudio
import wave
import time
import os
import gc



import tflearn
import pyaudio
import wave
import tflearn as tf
import pyaudio
import numpy

import numpy
import numpy as np
import skimage.io  # scikit-image



from threading import Thread
GPIO.setwarnings(False)

outputQueue = Queue(maxsize=1)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 2
RECORD_SECONDS = 5

ana_jahez = "/home/pi/Downloads/readytoRec.wav"
i3lak_jehaaz = "/home/pi/Downloads/Voice-008.m4a"
bda_tasjeel = "/home/pi/Downloads/Voice-006.mp3"
intha_tasjeel = "/home/pi/Downloads/endRec.wav"
idafa_soot = "/home/pi/Downloads/Voice-005.m4a"
ultra_sound = "/home/pi/Downloads/Voice-010.mp3"
repeet_recor = "/home/pi/Downloads/Voice-011.m4a"
rec_new_person = "/home/pi/Downloads/newperson.wav"
doneTraining = "/home/pi/Downloads/done.wav"
rec_nobody = "/home/pi/Downloads/nobody.wav"
rec_ok = "/home/pi/Downloads/ok.wav"
rec_recording = "/home/pi/Downloads/recording.wav"
rec_finished = "/home/pi/Downloads/finished.wav"


rec_what = "/home/pi/Downloads/what.wav"
rec_saved = "/home/pi/Downloads/saved.wav"

rec_waitingcard = "/home/pi/Downloads/waitingcard.wav"
rec_tag_not_listed = "/home/pi/Downloads/tagNoListed.wav"


configFileName = '/home/pi/configFileName.txt'

fileText = ""
GPIO_TRIGGER = 18
GPIO_ECHO = 24

Button_1_Pin_5 = 5
Button_2_Pin_14 = 14












number_classes=10 # Digits

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)

model.load('SpeechRecognition/model')









tags = None
def initProject():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
    GPIO.setup(GPIO_ECHO, GPIO.IN)

    GPIO.setup(Button_1_Pin_5, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(Button_2_Pin_14, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    global fileText
    fileText = readConfigFile()
    print(fileText)
    global tags
    tags = fileText.split('\n')
    return

def readConfigFile():
    txt = readFile(configFileName)
    print(txt)
    print("$$$$")
    if(txt == "NAN" or txt == ""):
        writeToFile(configFileName)
        return readFile(configFileName)
    return txt

def writeToFile(fileName, strToWrite = "", typeWrite = 'a'):
    file = open(fileName, typeWrite) 
    file.write(strToWrite)
    file.close()
    return

def readFile(fileName):
    try:
        file = open(fileName, 'r') 
        faileAsStr =  file.read()
        file.close()
        return faileAsStr
    except IOError:
        print("No such file")
        return "NAN"

def record(waveFilePath):
    p = pyaudio.PyAudio()
    stream =  p.open(format = FORMAT,
                    rate = RATE,
                    channels = CHANNELS,
                    input_device_index = DEVICE_INDEX,
                    input = True,
                    output = False,
                    frames_per_buffer = CHUNK)

    print("* recording")

    frames = []
    while True:
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
        input_1_state =True
        input_1_state = GPIO.input(Button_1_Pin_5)
        if input_1_state == False:
            input_1_state =True
            break

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(waveFilePath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return

def playSound(filePath):
    os.system("omxplayer " + filePath)
    return

def distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    StartTime = time.time()
    StopTime = time.time()

    StaTime = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
        StoTime = time.time()
        delta = StoTime - StaTime
        if(delta > 0.2):
            return 2
        StartTime = time.time()

    StaTime = time.time()

    while GPIO.input(GPIO_ECHO) == 1:
        StoTime = time.time()
        delta = StoTime - StaTime
        if(delta > 0.2):
            return 2
        StopTime = time.time()

    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    return distance

def speechRec2():
    CHUNK = 4096
    f = wave.open('SpeechRecognition/rec.wav', "rb")
    # print("loading %s"%name)
    chunk = []
    data0 = f.readframes(CHUNK)
    while data0:  # f.getnframes()
        # data=numpy.fromstring(data0, dtype='float32')
        # data = numpy.fromstring(data0, dtype='uint16')
        data = numpy.fromstring(data0, dtype='uint8')
        data = (data + 128) / 255.  # 0-1 for Better convergence
        # chunks.append(data)
        chunk.extend(data)
        data0 = f.readframes(CHUNK)
        # finally trim:
        chunk = chunk[0:CHUNK * 2]  # should be enough for now -> cut
        chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
        # print("%s loaded"%name)

    demo=chunk
    result=model.predict([demo])
    result=numpy.argmax(result)
    print("predicted digit for %s : result = %d "%('rec.wav',result))

    return result

def methods(cls):
    return [x for x, y in cls.__dict__.items()]

def speechRec():
    text = ''
    r = sr.Recognizer()
    
    with sr.AudioFile('/home/pi/Desktop/Code/SpeechRecognition/rec.wav') as source:              # use "test.wav" as the audio source
        audio = r.record(source)                        # extract audio data from the file
    
    
    try:
        text = r.recognize_google(audio)
        print("recognized: " + text)   # recognize speech using Google Speech Recognition
    except sr.UnknownValueError:
	print("Could not understand audio")
    except sr.RequestError as e:
	print("Could not request results; {0}".format(e))
    except LookupError:                                 # speech is unintelligible
        print("Could not understand audio")

    result = 0
    
    if 'read' in text:
        result = 1

    if 'record' in text:
        result = 2

    if 'yes' in text:
        result = 3

    if 'no' in text:
        result = 4

    if 'happy' in text:
        result = 5

    if 'sad' in text:
        result = 6

    if 'search' in text or 'face' in text:
        result = 7

    return result


def postHappy():
    os.system('sudo python ~/Desktop/Code/fb/FBhappy.py')
    
def postSad():
    os.system('sudo python ~/Desktop/Code/fb/FBsad.py')
    

def readNFC(outputQueue):
    while True:
        if(outputQueue.empty()):
            pp = subprocess.Popen("nfc-poll", stdout=subprocess.PIPE, shell=True)
            (output, err) = pp.communicate()
            p_status = pp.wait()
            try:
                #print(output)
                tage = str(output).split('\n')[5].split("UID (NFCID1):")[1].strip()
                outputQueue.put(tage)
            except:
                print("split erorr")
            

p = Process(target=readNFC, args=(outputQueue,))
p.daemon = True
p.start()          


cc =0
initProject()


def set_tag(tag):
    playSound(rec_recording)

    tagee2 = tag.replace(" ", "_")
    record('/home/pi/Downloads/' + tagee2 + '.wav')
    playSound(rec_finished)

    playSound('/home/pi/Downloads/' + tagee2 + '.wav')
    
    tags.append(tag)
    writeToFile(configFileName, strToWrite = (tag + "\n"))




def read_tag(tag):
    tag2 = tag.replace(" ", "_")
    playSound("/home/pi/Downloads/" + tag2 + ".wav")
    playSound(rec_finished)

for i in range(30):
    print('_____________READY___________')

while True:
    gc.collect()

    input_2_state = GPIO.input(Button_2_Pin_14)
    input_1_state = GPIO.input(Button_1_Pin_5)
    if(input_1_state == False):
        print('what do you want?')
        playSound(rec_what)

        record('/home/pi/Desktop/Code/SpeechRecognition/rec.wav')
        # 0empty 1yes 2no 3x read 4c record
        #res = (os.system('python /home/pi/Desktop/Code/SpeechRecognition/test.py') == 256)
        res = speechRec()
            #ccchash = ['e':0empty, 'x':1read, 'c':2record, 'y':3yes, 'n':4no, 'h':5happy, 's':6sad, 'f':7search face]
        
        if input_2_state == False:
            res = 7
        
        if (res == 0 or res == 3 or res == 4):
            print('nothing.. :' + str(res))
            playSound(rec_ok)
            pass

        if (res == 1):
            print('read card.. waiting')
            playSound(rec_waitingcard)
            while True:
                print('..')
                if not outputQueue.empty():
                    tagee = outputQueue.get()
                    if(tagee not in tags):
                        print('tag not listed')
                        playSound(rec_tag_not_listed)
                        record('/home/pi/Desktop/Code/SpeechRecognition/rec.wav')
                        print('recording yes or no')
                        
                        #yes = (os.system('python /home/pi/Desktop/Code/SpeechRecognition/test.py') == 3)## 256 = 1..
                        yes = speechRec() == 3
                        if yes:
                            print('yes')
                            set_tag(tagee)
                            playSound(rec_ok)
                            print('done1')
                            break
                            
                        else:
                            print('no')
                            playSound(rec_ok)
                            break
                        
                    else:
                        read_tag(tagee)
                        playSound(rec_ok)
                        print('done2')
                        break


        if (res == 2):
            print('record card.. waiting')
            playSound(rec_waitingcard)
            while True:
                print('..')
                if not outputQueue.empty():
                    tagee = outputQueue.get()
                    set_tag(tagee)
                    playSound(rec_ok)
                    print('done1')
                    break
                         
        if (res == 5):
            postHappy()
            playSound(rec_ok)
            
        if (res == 6):
            postSad()
            playSound(rec_ok)
            
            
        if (res == 7):
            print('recognize face')
            
            faceid = os.system('python /home/pi/Desktop/Code/rec.py') / 256
            print(faceid)
            # unknown face
            if faceid == 255:
                print('new face')
                # unknown
                # who is this
                playSound(rec_new_person)
                
    
                
                record('/home/pi/Desktop/Code/SpeechRecognition/rec.wav')
                print('recording yes or no')

                #yes = (os.system('python /home/pi/Desktop/Code/SpeechRecognition/test.py') == 3)## 256 = 1..
                yes = speechRec() == 3
                if yes:
                    print('yes')
                    ids = int(readFile('/home/pi/Desktop/Code/FacesCount.txt'))
                    playSound(bda_tasjeel)
                    record('/home/pi/Desktop/Code/recordings/' + str(ids) + '.wav')
                    playSound(intha_tasjeel)
                    writeToFile('/home/pi/Desktop/Code/FacesCount.txt', strToWrite=str(ids + 1), typeWrite='w')
                    playSound(bda_tasjeel)
                    os.system('python /home/pi/Desktop/Code/DataSetGenerator.py ' + str(ids))
                    os.system('python /home/pi/Desktop/Code/trainer.py')
                    playSound(doneTraining)
                else:
                    print('no')
                    playSound(rec_ok)
                    
            elif faceid == 254:
                print('no faces found')
                playSound(rec_nobody)
            else:
                print('found ' + str(faceid))
                playSound('/home/pi/Desktop/Code/recordings/' + str(faceid) + '.wav')




















while True:
    
    input_2_state = GPIO.input(Button_2_Pin_14)

    
    #if not outputQueue.empty():
    #    tage = outputQueue.get()
    #    print(tage)
    input_1_state = GPIO.input(Button_1_Pin_5)
    if(input_1_state == False):
        print('entering face recognitnion mode')
        time.sleep(0.2)
        faceid = os.system('python /home/pi/Desktop/Code/rec.py') / 256
        print(faceid)
        # unknown face
        if faceid == 255:
            print('new face')
            # unknown
            # who is this
            playSound(rec_new_person)
            
            """
            while(input_1_state != False):
                input_1_state = GPIO.input(Button_1_Pin_5)
                pass
            """
            
            record('/home/pi/Desktop/Code/SpeechRecognition/rec.wav')
            print('recording yes or no')

            yes = (os.system('python /home/pi/Desktop/Code/SpeechRecognition/test.py') == 256)## 256 = 1..
            if yes:
                print('yes')
                ids = int(readFile('/home/pi/Desktop/Code/FacesCount.txt'))
                playSound(bda_tasjeel)
                record('/home/pi/Desktop/Code/recordings/' + str(ids) + '.wav')
                playSound(intha_tasjeel)
                writeToFile('/home/pi/Desktop/Code/FacesCount.txt', strToWrite=str(ids + 1), typeWrite='w')
                playSound(bda_tasjeel)
                os.system('python /home/pi/Desktop/Code/DataSetGenerator.py ' + str(ids))
                os.system('python /home/pi/Desktop/Code/trainer.py')
                playSound(doneTraining)
            else:
                print('no')
                playSound(ok)
                
        elif faceid == 254:
            print('no faces found')
            playSound(rec_nobody)
        else:
            print('found ' + str(faceid))
            playSound('/home/pi/Desktop/Code/recordings/' + str(faceid) + '.wav')















    if not outputQueue.empty():
        tagee = outputQueue.get()

        if(tagee not in tags):

            playSound(rec_tag_not_listed)
            print('tag not listed')
            """
            while(input_1_state != False):
                input_1_state = GPIO.input(Button_1_Pin_5)
                pass
            """
            
            time.sleep(0.2)
            print('record yes or no')
            
            record('/home/pi/Desktop/Code/SpeechRecognition/rec.wav')
            
            yes = (os.system('python /home/pi/Desktop/Code/SpeechRecognition/test.py') == 256)
            if yes:
                print('yes')
                tags.append(tagee)
                writeToFile(configFileName, strToWrite = (tagee + "\n"))

                playSound(ana_jahez)
                
                tagee2 = tagee.replace(" ", "_")
                record("/home/pi/Downloads/" + tagee2 + ".wav")
                playSound(intha_tasjeel)

                playSound("/home/pi/Downloads/" + tagee2 + ".wav")
                
            else:
                print('no')
                playSound(ok)
                
        else:
            input_2_state = GPIO.input(Button_2_Pin_14)

            if(input_2_state == False):
                print('recording existing tag')
                playSound(bda_tasjeel)## recording exsiting tag

                input_2_state = False

                while(input_2_state == False):
                    input_2_state = GPIO.input(Button_2_Pin_14)
                    pass
                
                
                tags.append(tagee)
                writeToFile(configFileName, strToWrite = (tagee + "\n"))
                playSound(intha_tasjeel)
                
                tagee2 = tagee.replace(" ", "_")
                try:
                    os.remove("/home/pi/Downloads/" + tagee2 + ".wav")
                except:
                    pass
                
                
                record("/home/pi/Downloads/" + tagee2 + ".wav")
                playSound("/home/pi/Downloads/" + tagee2 + ".wav")

                
                
            else:
                tagee2 = tagee.replace(" ", "_")
                name = "/home/pi/Downloads/" + tagee2 + ".wav"
                playSound(name)

        print(tagee)

        
        
        """
        
        while True:
            if not outputQueue.empty():
                tage2 = outputQueue.get()
                if(tage2 not in tags):
                    playSound(intha_tasjeel)
                    playSound(bda_tasjeel)
                    tagee2 = tage2.replace(" ", "_")
                    record("/home/pi/Downloads/" + tagee2 + ".wav")
                    playSound("/home/pi/Downloads/" + tagee2 + ".wav")
                else:
                    tags.append(tage2)
                    writeToFile(configFileName, strToWrite = (tage2 + "\n"))
                    playSound(intha_tasjeel)
                    playSound(bda_tasjeel)
                    tagee2 = tage2.replace(" ", "_")
                    record("/home/pi/Downloads/" + tagee2 + ".wav")
                    playSound("/home/pi/Downloads/" + tagee2 + ".wav")
                break  
                print(tage)
    input_1_state = GPIO.input(Button_1_Pin_5)
    if(input_1_state == False):
        time.sleep(0.2)
        playSound(ana_jahez)
        
        StartTim = time.time()
        StopTim = time.time()
        while True:
            
            #input_1_state = True
            input_1_state = GPIO.input(Button_1_Pin_5)
            if(input_1_state == False):
                playSound(i3lak_jehaaz)
                break

            StopTim = time.time()
            dist = distance()
            print(dist)
            
            if(dist > 8 and dist < 30):
                cc = cc + 1
                detta = StopTim - StartTim
                if(detta > 5 and cc > 4):
                    cc = 0
                    StartTim = time.time()
                    playSound(ultra_sound)
            if not outputQueue.empty():
                tagee = outputQueue.get()
                print (tagee)
                print (tags)
                if(tagee not in tags):
                    playSound(idafa_soot)
                    
                    press = 0
                    while True:
                        input_1_state = GPIO.input(Button_1_Pin_5)
                        if(input_1_state == False):
                            press = 1
                            break

                    
                    if(press == 1):
                        tags.append(tagee)
                        writeToFile(configFileName, strToWrite = (tagee + "\n"))
                        playSound(intha_tasjeel)

                        playSound(bda_tasjeel)
                        
                        tagee2 = tagee.replace(" ", "_")
                        record("/home/pi/Downloads/" + tagee2 + ".wav")
                        playSound("/home/pi/Downloads/" + tagee2 + ".wav")
                        

                else:
                    tagee2 = tagee.replace(" ", "_")
                    name = "/home/pi/Downloads/" + tagee2 + ".wav"
                    playSound(name)

                print(tagee)
        """
