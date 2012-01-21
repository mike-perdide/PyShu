import pickle
from pyshu import snd_record, process_fft, graph_side_to_side, graph_one
from pylab import *
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from IPython import embed

def fit_model(fft_data):
    model = KMeans(k=20)
    model.fit(fft_data)
    return model

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

def replace(_array, _replace_array):
    new_array = np.copy(_array)
    for key in _replace_array:
        new_array[_array==key] = 0

    return new_array

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


if __name__ == "__main__":

    try:
        with open("my_model.dump") as handle:
            fft_model, all_records, all_ffts  = np.load(handle)

        fft_ambiant, fft_r2, fft_open = all_ffts
        ambiant_noises, ten_sec_r2, ten_sec_open = all_records
    except:
        all_records = all_recordings()
        all_ffts = all_fft(all_records)
        fft_ambiant, fft_r2, fft_open = all_ffts

        # Fit model to recognize ambiant noise
        fft_model = fit_model(concatenate(all_ffts))

        with open("my_model.dump", 'w') as handle:
            np.save(handle, (fft_model, all_records, all_ffts))

#    embed()
    len_models_ambiant = len(fft_model.labels_) * len(fft_ambiant)/len(concatenate(all_ffts))
    values_to_zero_out = unique(fft_model.labels_[:len_models_ambiant])
    labels = replace(fft_model.labels_, values_to_zero_out)

#    graph_side_to_side(concatenate(all_records), labels)
#    graph_one(labels)
#    graph_one(ten_sec_r2)

    #graph_one(labels)
    limits = {"R2": [(1264, 1464), (1550, 1750), (1854, 1976), (2138, 2277),
                     (2456, 2655), (2773, 2940), (3055, 3215)],
              "open": [(3356, 3599), (3708, 3894), (4041, 4224), (4393, 4618),
                       (4723, 4947), (5066, 5284)]}

#    for 

#    while True:
#        word = "R2"
#        start = raw_input("Enter start (number of s) of 'R2' or enter 'X' to"
#                          "got to the next word.")
#
#        if start == "X" or start == "":
#            if word == "R2":
#                word = "open"
#                continue
#            else:
#                break
#        else:
#            start = int(start)
#
#        end = raw_input("Enter end (number of s) of 'R2'")
#        end = int(end)
#
#        limits[word] = (start, end)

    # Extract the fft of the words
    R2_fft_extracts = []
    for start, end in limits["R2"]:
        extract = labels[start:end]
        fitting_extract = center_and_expand(extract, 250)
        R2_fft_extracts.append(fitting_extract)

    open_fft_extracts = []
    for start, end in limits["open"]:
        extract = labels[start:end]
        fitting_extract = center_and_expand(extract, 250)
        open_fft_extracts.append(fitting_extract)

    # Build a array of the targets ("R2" is 0, "open" is 1)
    targets = r_[[0] * 7, [1] * 6]

    # Create new model and fit it with the 'R2' and 'open' occurences
    model = SVC(gamma=0.001)
    model.fit(concatenate((R2_fft_extracts, open_fft_extracts)), targets)
    print "This should be 0"
    print model.predict(R2_fft_extracts[3])
    print "This should be 1"
    print model.predict(open_fft_extracts[5])

    test = record_or_load("test_record.wav", "Say 'open' or 'R2'", length=3)
    test_fft = process_fft(test)

    predictions = [fft_model.predict(one_fft) for one_fft in test_fft]

    predictions = replace(predictions, values_to_zero_out)

    #graph_one(predictions)

    centered_test = center_and_expand(predictions[200:325], 250)
    #graph_one(centered_test)

    print model.predict(centered_test)
