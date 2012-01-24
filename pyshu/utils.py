from pylab import *
from pyshu.defaults import *

def record_or_load(filename, message, length=5):
    try:
        with open(filename) as handle:
            recording = np.load(handle)
    except:
        raw_input(message + " Hit enter when ready. [Go!]")
        recording = snd_record(record_seconds=length)

        choice = raw_input("Display recording ? [Yn]")
        if not choice or choice[0] != 'n':
            plt.plot(recording)
            plt.show()

        choice = raw_input("Save recording ? [Yn]")
        if not choice or choice[0] != 'n':
            with open(filename, "w") as handle:
                np.save(handle, recording)

    return recording

def all_recordings():
    ambiant_noises = record_or_load("five_sec_ambiant.wav",
                                    "Recording 5 seconds of ambiant noises")

    ten_sec_r2 = record_or_load("ten_sec_r2.wav",
                                "Recording 10 seconds of you saying 'R2'"
                                "(6-10 times).", 10)

    ten_sec_open = record_or_load("ten_sec_open.wav",
                                  "Recording 10 seconds of you saying 'open'"
                                  "(6-10 times).", 10)

    return ambiant_noises, ten_sec_r2, ten_sec_open

def all_fft(records):
    ambiant_noises, ten_sec_r2, ten_sec_open = records

    fft_ambiant = process_fft(ambiant_noises)
    fft_r2 = process_fft(ten_sec_r2)
    fft_open = process_fft(ten_sec_open)

    return fft_ambiant, fft_r2, fft_open

def center_and_expand(_array, new_length):
    new_array = zeros(new_length)
    for i in xrange(len(_array)):
        new_array[int(double(i)/len(_array)*new_length)] = _array[i]

    return new_array

def detect_word_edges(ambiant_groups, fft_data, offset=0):
    # Other way to detect edges : we should try unsupervised learning with 2
    # groups.
    edges = []
    start = 0
#    print ambiant_groups

    len_data = len(fft_data)

    logfile = open("logfile", "w")
    for i in xrange(len_data - SIGNIFIANT_LEAP):
        group_count = bincount(fft_data[i:i+SIGNIFIANT_LEAP])
        ambiant_count = 0
        for group in ambiant_groups:
            if group < len(group_count):
                ambiant_count += group_count[group]


        is_ambiant = ambiant_count > .6 * SIGNIFIANT_LEAP

        logfile.write("[%d:%d], %s, count: %d\n" % (
            (offset + i) * FFT_STEP,
            (offset + i + SIGNIFIANT_LEAP) * FFT_STEP,
            repr(group_count),
            ambiant_count)
        )

        if not (start or is_ambiant):
            # This isn't ambiant noise
            # We haven't recorded the start of the word yet
            start = i + offset
        elif start and is_ambiant:
            edges.append((start, i + offset))
            start = 0

    logfile.close()

    return edges

def replace(_array, _replace_array):
    new_array = np.copy(_array)
    for key in _replace_array:
        new_array[_array==key] = 0

    return new_array

def find_most(data):
    count = bincount(data)
    return find(count > (.1 * sum(count)))
