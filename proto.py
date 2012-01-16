from pylab import *
from sklearn import mixture
import pickle

import pyaudio
import wave
import sys

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FFT_STEP = 1000


def record(chunk=CHUNK, format=FORMAT,
           channels=CHANNELS, rate=RATE,
           record_seconds=RECORD_SECONDS):
    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = chunk)

    print "* recording"
    all = []
    all_samps = []
    for i in range(0, RATE / chunk * RECORD_SECONDS):
        try:
            data = stream.read(chunk)
        except IOError, e:
            pass
            #pyaudio.paInputOverflowed:
#            data = '\x00' * chunk * format * channels
        samps = np.fromstring(data, dtype=np.int16)
        for samp in samps:
            all_samps.append(samp)

    print "* done recording"

    stream.close()
    p.terminate()

    return all_samps


def process_fft(data):
    x = array(xrange(FFT_STEP))
    all_fft = []
    for n in xrange(len(data)/FFT_STEP):
        data_set = data[FFT_STEP*n:FFT_STEP*(n+1)]

        data_fft = fft(data_set)
        all_fft.append(data_fft)

    return all_fft


def learn_fit(data):
    my_model = mixture.GMM(n_components=3)
    my_model.fit(data)

    all_results = []
    for one_fft in data:
        reversed_array = array(one_fft).reshape(1, one_fft.size)
        results = my_model.predict(reversed_array)
        all_results.append(results[0])

    return all_results

def graph(sound, predict):
    y = array(sound)
    plt.plot(y)

    x = array(xrange(len(predict))) * 1000
    plt.plot(x, array(predict) * 5000 - 20000)
    plt.show()


if __name__ == "__main__":
    recording = record()
    all_fft = process_fft(recording)
    prediction = learn_fit(all_fft)
    graph(recording, prediction)
