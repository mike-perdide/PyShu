import sys
import wave

import pyaudio
from pylab import *

from sklearn import mixture
from sklearn.externals.joblib import Memory

mem = Memory(cachedir='.')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FFT_STEP = 200


def snd_record(chunk=CHUNK, format=FORMAT,
           channels=CHANNELS, rate=RATE,
           record_seconds=RECORD_SECONDS):
    p = pyaudio.PyAudio()
    stream = p.open(format = format,
                    channels = channels,
                    rate = rate,
                    input = True,
                    frames_per_buffer = chunk)

    print "* recording"
    all = []
    all_samps = []
    for i in range(0, rate / chunk * record_seconds):
        print i, '/', rate/ chunk * record_seconds
        try:
            data = stream.read(chunk)
        except IOError, e:
            pass
        samps = np.fromstring(data, dtype=np.int16)
        for samp in samps:
            all_samps.append(samp)

    print "* done recording"

    stream.close()
    p.terminate()

    return all_samps


def process_fft(data):
    cached_fft = mem.cache(fft)

    x = array(xrange(FFT_STEP))
    all_fft = []
    for n in xrange(len(data)/FFT_STEP):
        data_set = data[FFT_STEP*n:FFT_STEP*(n+1)]

        data_fft = cached_fft(data_set)
        all_fft.append(data_fft)

    return array(all_fft)


def learn_fit(data):
    my_model = mixture.GMM(n_components=3)
    my_model.fit(data)

    all_results = []
    for one_fft in data:
        reversed_array = array(one_fft).reshape(1, one_fft.size)
        results = my_model.predict(reversed_array)
        all_results.append(results[0])

    return all_results

def graph_side_to_side(sound, predict):
    y = array(sound)
    plt.plot(y)

    x = array(xrange(len(predict))) * FFT_STEP
    plt.plot(x, array(predict) * 5000 - 40000)
    plt.show()

def graph_one(data):
    y = array(data)
    plt.plot(y)
    plt.show()
