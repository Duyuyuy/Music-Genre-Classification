import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = r"D:\Data\process4"
JSON_PATH = r"D:/pythonProject/data/train.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "tempo": [],
        "spectral_contrast": [],
        "chroma_cqt": [],
        "beat_hist": [],
        "zero_cross_rate": []
    }


    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        fmin = 27.5
        fmax = 14000
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

                # Extract form and structure features
                tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sample_rate, start_bpm=100)
                beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)


                # Compute beat times using beat tracking algorithm
                # Compute histogram of beat times
                n_bins = 100  # Number of histogram bins
                hist, bin_edges = np.histogram(beat_times, bins=n_bins)

                # Normalize the histogram to have unit area
                beat_hist = hist / np.sum(hist)

                # Extract harmonic features
                chroma_cqt = librosa.feature.chroma_cqt(y=signal, sr=sample_rate)
                chroma_cqt = chroma_cqt.T

                # Calculate STFT and power spectrum
                stft = np.abs(librosa.stft(y=signal))
                power_spectrum = stft ** 2

                # Calculate mean of power spectrum
                mean_power = np.mean(power_spectrum, axis=1)

                # Calculate square root of mean to get RMSE
                rmse = np.sqrt(mean_power)

                # process all segments of audio file
                for d in range(num_segments):
                    try:
                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment


                        S = librosa.feature.melspectrogram(y=signal[start:finish], sr=sample_rate, n_fft=n_fft,
                                                           hop_length=hop_length,
                                                           fmin=fmin,
                                                           fmax=fmax,
                                                           n_mels=138)

                        # Zero-crossing rate
                        zero_cross = librosa.zero_crossings(y=signal[start:finish])
                        zero_cross_rate = len(zero_cross) / len(signal[start:finish])

                        # Spectral contrast
                        S = librosa.feature.melspectrogram(y=signal[start:finish], sr=sample_rate, n_fft=n_fft,
                                                           hop_length=hop_length)
                        spectral_contrast = librosa.feature.spectral_contrast(S=S)
                        spectral_contrast = spectral_contrast.T

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            # data["mfcc"].append(mfcc.tolist())
                            data["spectral_contrast"].append(spectral_contrast.tolist())
                            data["chroma_cqt"].append(chroma_cqt.tolist())
                            data["tempo"].append(tempo)
                            data["zero_cross_rate"].append(zero_cross_rate)
                            data["beat_hist"].append(beat_hist.tolist())

                            #                         data["feature"].append(x)
                            data["labels"].append(i - 1)
                            # print("{}, segment:{}".format(file_path, d + 1))
                    except EOFError as e:
                        print(e)

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=2)