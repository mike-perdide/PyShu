from pylab import *
from sklearn import mixture
from sklearn.externals.joblib import Memory
mem = Memory(".")

from pyshu.defaults import *

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
