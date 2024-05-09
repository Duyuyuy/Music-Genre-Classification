import json
import os
import librosa
import numpy as np


# DATASET_PATH = r"C:\Users\david.tran\OneDrive - PUMA\Desktop\Gztan\Data\genres_original"
# JSON_PATH = r"C:\Users\david.tran\PycharmProjects\pythonProject1\JsonData\gztan_640.json"
DATASET_PATH = r"demo file/audio"
JSON_PATH = r"demo file/json/demo.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
# Calculate the number of samples to remove (2 seconds)
samples_to_remove = int(10 * SAMPLE_RATE)

def save_melspec(dataset_path, json_path, n_fft=2048, hop_length=690):
    """Extracts mel_spec from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and features
    data = {
        "mapping": [],
        "labels": [],
        "mel_spec": []
    }

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                signal = signal[samples_to_remove:]

                mel_spec= librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft,
                                               hop_length=hop_length, n_mels= 128)

                # rounded_mel_spec
                rounded_mel_spec = [[float("{:0.7f}".format(value)) for value in sublist] for sublist in mel_spec]

                d=0
                for a in rounded_mel_spec:
                     if len(a)!=640:
                            print(len(a))
                            print(file_path)
                            d=1

                if d==0:
                    data["mel_spec"].append(rounded_mel_spec)
                    data["labels"].append(i - 1)
                    print("success")


        # save to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=2)

if __name__ == "__main__":
    save_melspec(DATASET_PATH, JSON_PATH)

