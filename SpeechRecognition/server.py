#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyaudio
import time
import os

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
import socket
import pickle

from six.moves import xrange as range

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

from utils import maybe_download as maybe_download
from utils import sparse_tuple_from as sparse_tuple_from

############################ recording

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 2
RECORD_SECONDS = 5

def record(): # waveFilePath):
    p = pyaudio.PyAudio()
    stream =  p.open(format = FORMAT,
                    rate = RATE,
                    channels = CHANNELS,
                    input_device_index = DEVICE_INDEX,
                    input = True,
                    output = False,
                    frames_per_buffer = CHUNK)

    frames = []
    for i in range(0, 250):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)


    stream.stop_stream()
    stream.close()
    p.terminate()

    # wf = wave.open(waveFilePath, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    return frames

###################

def getAudioFromFile():
    audio = []
    fs, sub = wav.read('/Users/mamounlaptop/Desktop/test.wav')

    for s in sub:
        audio.append(np.short((int(s[0]) + int(s[1])) / 2))

    return audio

##################



# Constants
# SPACE_TOKEN = '<space>'
SPACE_TOKEN = ' '
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters


# Hyper-parameters
num_epochs = 1000
num_hidden = 100
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-5
momentum = 0.99

num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)











wordFiles = os.listdir('/Users/mamounlaptop/Desktop/aa/audios/')#''/Users/mamounlaptop/Desktop/dbwords/')
fileNames = []
tmp = []
for w in wordFiles:
    if w.endswith('.wav'):
        tmp.append('/Users/mamounlaptop/Desktop/aa/audios/' + w)#'/Users/mamounlaptop/Desktop/dbwords/' + w)
        fileNames.append(w)
wordFiles = tmp[:]






# Loading the data

# audio_filename = maybe_download('LDC93S1.wav', 93638)
# target_filename = maybe_download('LDC93S1.txt', 62)

# fs, audio = wav.read(audio_filename)

fs = 44100

HOST = ''  # Symbolic name meaning the local host
PORT = 50007  # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()


dataLen = conn.recv(1024)
print(dataLen)
time.sleep(5)
data = conn.recv(1024 * 1024 * 1000)

while True:
    if (len(data) == int(dataLen)):
        break
    data2 = conn.recv(4096)

    data = data + data2
    # print(len(data))


# print(len(data))
data = pickle.loads(data)

train_inputs = data[0]
train_seq_len = data[1]

# Readings targets
# with open(target_filename, 'r') as f:
#
#     #Only the last line is necessary
#     line = f.readlines()[-1]
#
#     # Get only the words between [a-z] and replace period for none
#     original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
#     targets = original.replace(' ', '  ')
#     targets = targets.split(' ')

# Adding blank label
targets = [''] #totalString[:]

# charSet = ['ge', 'ga', 'le', 'la', 'lo', 'to', '5e', 'te', '5a', 'ta', 'do', '<space>', 'de', 'da', '_so', 'sha', 'she', 'sho', 'em', 'el', 'eo', 'en', 'eh', 'ek', 'ee', 'ed', 'eg', 'ef', 'ea', 'za', 'eb', 'ez', 'et', 'es', 'er', 're', '3o', 'ra', 'o6', '3a', 'ro', '3e', 'esh', 'be', 'ba', 'wa', 'bo', 'e5', 'e7', 'ash', 'e3', 'on', 'om', 'ol', 'ok', 'oe', 'ob', 'oa', 'oz', 'ot', 'os', 'ag', 'ze', '6a', '6e', 'ho', 'ha', 'he', 'me', 'ma', 'mo', '3_a', '3_e', '3_o', '7e', 'ab', 'ae', 'ad', '7a', 'af', '7o', 'ah', 'ak', 'am', 'al', 'an', 'as', 'ar', 'at', 'no', 'na', 'ne', 'fa', 'a3', 'a5', 'fe', 'a7', 'a6', 'fo', 'ka', 'ke', 'ko', 'sa', 'se']
charSet = ['\x81', '\x83', '\x82', '\x85', '\x84', '\x87', '\x86', '\x89', '\x88', '\x8a', '\x94', '\xa1', ' ', '\xa7', '\xa9', '\xa8', '\xaa', '\xad', '\xac', '\xaf', '\xae', '\xb1', '\xb3', '\xb2', '\xb5', '\xb4', '\xb7', '\xb6', '\xb9', '\xba', '\xd9', '\xd8']

# list(set(targets))

# print(charSet)

num_classes = len(charSet) + 1 + 3
# targets = [charSet.index(x) for x in targets]

# Transform char into index
# targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

# Creating sparse representation to feed the placeholder
train_targets = [] #sparse_tuple_from([targets])

# We don't have a validation dataset :(
val_inputs, val_targets, val_seq_len = train_inputs, [], train_seq_len


# THE MAIN CODE!

graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    saver.restore(session, "data/model3.ckpt")



    feed = {inputs: train_inputs,
            targets: train_targets,
            seq_len: train_seq_len}


    # Decoding
    d = session.run(decoded[0], feed_dict=feed)
    str_decoded = ''.join([charSet[x] for x in np.asarray(d[1])])
    # Replacing blank label to none
    # str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    # str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

    # original = ''.join(totalString)
    # print('Original:\n%s' % original)

    print('Decoded:\n%s' % str_decoded)
    conn.send(str(len(str_decoded)))

    aaaa = os.listdir('/Users/mamounlaptop/Desktop/Screenshots/a/')  # ''/Users/mamounlaptop/Desktop/dbwords/')

    str_decoded = aaaa[0]
    print (str_decoded)
    time.sleep(10)
    conn.send(str_decoded)
    time.sleep(100)
    conn.close()




