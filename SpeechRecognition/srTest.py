import speech_recognition as sr

def speechRec():
    text = ''
    r = sr.Recognizer()
    # Code/SpeechRecognition/rec.wav'
    with sr.AudioFile('/home/pi/Desktop/harvard.wav') as source:              # use "test.wav" as the audio source
        audio = r.record(source)                        # extract audio data from the file
        
    print(type(audio))
    
    try:
        text = r.recognize_google(audio)
        print("recognized: " + text)   # recognize speech using Google Speech Recognition
    except sr.UnknownValueError:
	print("Could not understand audio1")
    except sr.RequestError as e:
	print("Could not request results; {0}".format(e))
    except LookupError:                                 # speech is unintelligible
        print("Could not understand audio2")

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


speechRec()