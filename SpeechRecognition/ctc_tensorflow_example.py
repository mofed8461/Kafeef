#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import gc
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from random import shuffle
from random import randint

from six.moves import xrange as range

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

from utils import maybe_download as maybe_download
from utils import sparse_tuple_from as sparse_tuple_from


modelName = 'data/yn.ckpt'
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
num_hidden = 10
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-6
momentum = 0.9

num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)


# folder = '/Users/mamounlaptop/Desktop/dbwords/'
folder = 'traindata/'



wordFiles = os.listdir(folder)
fileNames = []
tmp = []
for w in wordFiles:
    if w.endswith('.wav'):
        tmp.append(folder + w)
        fileNames.append(w)
wordFiles = tmp[:]

# new, hello, read
cmds = ['no', 'yes']
fs = 44100
words = []
subs = []
for i in range(0, len(wordFiles)):
    fs, sub = wav.read(wordFiles[i])
    subs.append(sub)
    # totalString.extend(fileNames[i].replace('$.wav', '').replace('.wav', '').replace('x', '').split('$')[1:])
    words.append(str(fileNames[i].replace('.wav', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace('0', '')))


# Loading the data

# audio_filename = maybe_download('LDC93S1.wav', 93638)
# target_filename = maybe_download('LDC93S1.txt', 62)

# fs, audio = wav.read(audio_filename)


def getAudio(subs, strings):

    fs = 44100
    audio = []
    indexes = range(0, len(strings))
    indexes = list(indexes)

    shuffle(indexes)

    totalString = []

    for i in range(randint(1, 10)):
        totalString.extend(strings[indexes[i]])

        for s in subs[indexes[i]]:
            audio.append(np.short((int(s[0]) + int(s[1])) / 2))

        if (len(audio) % 2 == 1):
            audio.append(audio[len(audio) - 1])


    audio = np.asarray(audio)


    return audio, fs, totalString


# totalString = np.asarray(totalString)


def getInput(audio, fs, totalString):
    inputs = mfcc(audio, samplerate=fs)
    # Tranform in 3D array
    train_inputs = np.asarray(inputs[np.newaxis, :])
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
    train_seq_len = [train_inputs.shape[1]]

    targets = totalString[:]

    targets = [cmds.index(str(x)) for x in targets]

    train_targets = sparse_tuple_from([targets])

    gc.collect()

    return train_inputs, train_targets, train_seq_len



num_classes = len(cmds) + 2


print(cmds)



# Transform char into index
# targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

# Creating sparse representation to feed the placeholder

# We don't have a validation dataset :(


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

    # optimizer = tf.train.AdagradOptimizer(initial_learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate).minimize(cost)
    optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    # saver.restore(session, modelName)

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        gc.collect()
        if not os.path.exists('/Users/mamounlaptop/Desktop/delmetostop'):
            break

        val_inputs, val_targets, val_seq_len = [], [], []

        for batch in range(num_batches_per_epoch):

            audio, fs, totalString = getAudio(subs, words)

            train_inputs, train_targets, train_seq_len = getInput(audio, fs, totalString)

            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

            val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len

        train_cost /= num_examples
        train_ler /= num_examples

        val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    seq_len: val_seq_len}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        save_path = saver.save(session, modelName)

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))


    # Decoding
    d = session.run(decoded[0], feed_dict=feed)

    str_decoded = ''.join([cmds[x] for x in list(d[1])])

    original = ''.join(totalString)
    print('Original:\n%s' % original)
    print('Decoded:\n%s' % str_decoded)



