import pyaudio
from pylab import *
from pyshu.defaults import *

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
        print len(samps)
        for samp in samps:
            all_samps.append(samp)

    print "* done recording"

    stream.close()
    p.terminate()

    return all_samps


def play_array(snd_data):
    chunk = 1024
    print len(snd_data) % chunk

    p = pyaudio.PyAudio()

    # open stream
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    output = True)

    # play stream
    for data in [snd_data[n * chunk:(n+1) * chunk]
                 for n in xrange(len(snd_data) / chunk)]:
        stream.write(data)

    stream.close()
    p.terminate()
