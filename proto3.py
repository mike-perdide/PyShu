from pyshu.utils import all_recordings
from pylab import *
from scipy.signal import buttord, butter, lfilter, freqz
from scipy.signal.filter_design import lp2bp
from math import pi
from pyshu.defaults import RATE
from sklearn.externals.joblib import Memory
mem = Memory(".")

cached_buttord = mem.cache(buttord)
cached_butter = mem.cache(butter)
cached_lfilter = mem.cache(lfilter)


def high_pass(freq, raw_signal):
    corner_freq = freq * 2 * pi / RATE
    stop_freq = corner_freq * 0.5

    # Design the highpass Butterworth filter
    # See http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html
    (N, Wn) = cached_buttord(corner_freq, stop_freq, gpass=3, gstop=20)
    (b, a) = cached_butter(N, Wn, btype="high")

    # Apply the filter to the signal
    return cached_lfilter(b, a, raw_signal)


def low_pass(freq, raw_signal):
    corner_freq = freq * 2 * pi / RATE
    stop_freq = corner_freq * 1.5

    # Design the human frequencies filter
    # See http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html
    (N, Wn) = cached_buttord(corner_freq, stop_freq, gpass=3, gstop=20)
    (b, a) = cached_butter(N, Wn, btype="low")

    # Apply the filter to the signal
    return cached_lfilter(b, a, raw_signal)


def human_freq_filter(recording, sex):
    ## Adult male voice frequencies ranges from 85 to 180 Hz
    ## Adult female voice frequencies ranges from 165 to 255 Hz
    ## Source: wikipedia.org
    if sex == "M":
        freqs = (85, 180)
    elif sex == "F":
        freqs = (165, 255)

    # To perform a bandpass, do a lowpass filter on an highpass filter output
    return low_pass(freqs[1], high_pass(freqs[0], recording))


if __name__ == "__main__":
    # Record sound
    ambiant, r2_recordings, open_recordings = all_recordings()

    # Filter using human speech known frequencies
    r2_filtered_records = human_freq_filter(r2_recordings, "M")
    open_filtered_record = human_freq_filter(open_recordings, "M")

    # Filter-out the high frequencies
    # Detect the word edges
    # On each word, do a fft decomposition
    # Use unsupervised model to group fft decompositions
    # Use supervised model to learn each sequence of fft decompositions groups
