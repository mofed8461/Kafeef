#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import pyaudio
import wave
import tflearn as tf
import pyaudio
import speech_data
import numpy

import numpy
import numpy as np
import skimage.io  # scikit-image



number_classes=10 # Digits

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)

model.load('model')

demo_file = "~/Desktop/Code/SpeechRecognition/rec.wav"

CHUNK = 4096
f = wave.open(demo_file, "rb")
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
print("predicted digit for %s : result = %d "%(demo_file,result))

exit(result)
