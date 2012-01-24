import pickle
from pyshu.sound import snd_record, play_array
from pyshu.calculus import process_fft
from pyshu.graphs import graph_side_to_side, graph_one
from pyshu.defaults import FFT_STEP, RATE
from pyshu.utils import detect_word_edges, find_most, center_and_expand, record_or_load
from pylab import *
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from IPython import embed


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

    # Calculating lengths
    ratio = double(len(fft_model.labels_) / len(concatenate(all_ffts)))
    len_models_ambiant = int(len(fft_ambiant) * ratio)
    len_models_R2 = int(len(fft_r2) * ratio)
    len_models_open = int(len(fft_open) * ratio)

    # Calculating the values of the groups present in the ambiant noise recording
    labels = fft_model.labels_
    ambiant_groups = find_most(labels[:len_models_ambiant])

#    print unique(fft_model.labels_[:len_models_ambiant])
#    print bincount(fft_model.labels_[:len_models_ambiant])

#    labels = replace(fft_model.labels_, values_to_zero_out)
#    graph_side_to_side(concatenate(all_records), labels)
#    graph_one(ten_sec_r2)

#    HARDCODED EDGES
#    limits = {"R2": [(1264, 1464), (1550, 1750), (1854, 1976), (2138, 2277),
#                     (2456, 2655), (2773, 2940), (3055, 3215)],
#              "open": [(3356, 3599), (3708, 3894), (4041, 4224), (4393, 4618),
#                       (4723, 4947), (5066, 5284)]}

    # R2 edges detection
    r2_edges = detect_word_edges(
        ambiant_groups,
        labels[len_models_ambiant:len_models_ambiant + len_models_R2],
        offset=len_models_ambiant)
    open_edges = detect_word_edges(
        ambiant_groups,
        labels[len_models_R2 + len_models_ambiant:
               len_models_R2 + len_models_ambiant + len_models_open],
        offset=len_models_R2+len_models_ambiant)

    edges = {"R2": r2_edges, "open": open_edges}

    # Extract the fft of the words
    R2_fft_extracts = []
    for start, end in edges["R2"]:
        extract = labels[start:end]
        fitting_extract = center_and_expand(extract, 250)
        R2_fft_extracts.append(fitting_extract)

    open_fft_extracts = []
    for start, end in edges["open"]:
        extract = labels[start:end]
        fitting_extract = center_and_expand(extract, 250)
        open_fft_extracts.append(fitting_extract)

    # Build a array of the targets ("R2" is 0, "open" is 1)
    targets = r_[[0] * len(R2_fft_extracts), [1] * len(open_fft_extracts)]

    # Create new model and fit it with the 'R2' and 'open' occurences
    model = SVC(gamma=0.001)
    model.fit(concatenate((R2_fft_extracts, open_fft_extracts)), targets)
    print "This should be 0"
    print model.predict(R2_fft_extracts[3])
    print "This should be 1"
    print model.predict(open_fft_extracts[5])

    test = record_or_load("test_record.wav", "Say 'open' or 'R2'", length=3)
    test_fft = process_fft(test)

    predictions = array([fft_model.predict(one_fft)[0] for one_fft in test_fft])

    #graph_one(predictions)

    test_edges = detect_word_edges(
        ambiant_groups,
        predictions)
    print test_edges
    for start, end in test_edges:
        print start * FFT_STEP, end * FFT_STEP

    play_array(test)
    for start, end in test_edges:
    #    play_array(test[start * FFT_STEP:end * FFT_STEP])

        centered_test = center_and_expand(predictions[start:start], 250)
        print model.predict(centered_test)
    graph_one(test)
