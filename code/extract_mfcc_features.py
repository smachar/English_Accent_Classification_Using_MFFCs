import json
import os
import re
import math
import librosa
import time

SAMPLE_RATE = 22050
JSON_PATH = "json_wav_data"
DATASET_PATH = "wav_data"

#AUDIO_DURATION = 30 # measured in seconds min=15, max=45
#SIGNAL_LENGTH = SAMPLE_RATE * AUDIO_DURATION
#MFCC_FRAMES = SIGNAL_LENGTH/hop_length


def save_mfccs(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, segment_length=165375):
    """
        the dataset_path is like:
            afrikaans1.wav,
            afrikaans2.wav,
            afrikaans3.wav,
            afrikaans4.wav,
            afrikaans5.wav,
            agni1.wav,
            akan1.wav,
            ...

        Extracts the mfccs of each audio file in dataset_path, and saves them to a json file of the form
        {
            "name":["afrikaans", "albanian" ...],
            "mfcc":[array_of_afrikaans_1_mfccs(), array_of_afrikaans_2_mfccs(), ..., array_of_albanian_1_mfccs(), array_of_albanian_2_mfccs() ...]
            "label":[0, 0, ..., 1, 1, ...] #0 => afrikaans, 1 => albanian etc...
        }

        /// PARAMETERS ///
        dataset_path: the path to where the dataser is located
        json_path: the path to where to store the output

        n_mfcc=13, n_fft=2048, hop_length=512: are used by the librosa library to calculate the fourier transformation and the mfccs.
        segment_length: the number of samples to devide the signal into. set to 165375 which is a 2-samples for the min duration 15(secs)*22050(sample_rate)/2
    """
    data = {
        "name":[],
        "mfcc":[],
        "label":[]
    }

    prev_name = ""
    label_index = -1
    #loop through all files in dataset_path 
    for filename in sorted(os.listdir(dataset_path)):

        print("Processing the {} audio".format(filename))

        #extract the name from filename. ex: afrikaans1.wav => afrikaans and stor it once in data["name"]
        current_name = re.findall('([a-zA-Z]*)\d', filename)[0] #take the first match
    
        #store the name if it's not already stored
        if (current_name != prev_name):
            data["name"].append(current_name)
            prev_name = current_name
            label_index += 1
        
        #load the audio (filename)
        file_path = os.path.join(dataset_path, filename)
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        #calculating the expected number of segments to devide this signal into
        num_segs_to_devide_into = int(len(signal)/segment_length)

        #store the mfccs of the audio's segment in data["mfcc"]
        for seg in range(num_segs_to_devide_into):
            start = segment_length * seg
            end = start + segment_length

            #get mfcc for current segment
            mfcc = librosa.feature.mfcc(signal[start:end], sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T

            data["mfcc"].append(mfcc.tolist())
            #store the label of that mfcc in data["label"]
            data["label"].append(label_index)
            print("\t\tsegment {} : saved".format(seg+1))

        #break

        
    #save data to json file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__=="__main__":
    tstart = time.time()
    save_mfccs(DATASET_PATH, JSON_PATH)
    tend = time.time()
    print("Execution time :{}".format(tend-tstart))

#Execution time :1991.063691854477 secs => 33minutes & 11seconds
#Execution time :2033.1778016090393