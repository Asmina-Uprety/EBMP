import os
import librosa
import math
import json
import matplotlib.pyplot as plt
import numpy as np

dataset_path = r"C:\Users\asmin\PycharmProjects\EBMP\archive"
json_path = r"data.json"
SAMPLE_RATE = 22050
DURATION = 9
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048,
              hop_length=512, num_segments=5):
    # Data storage dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }
    samples_ps = int(SAMPLES_PER_TRACK / num_segments)  # ps = per segment
    expected_vects_ps = math.ceil(samples_ps / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensuring not at root
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}")

            # process files for specific genre
            for f in filenames:
                # if (f == str("jazz.00054.wav")(sad)
                #     # As librosa only read files <1Mb
                #     continue
                # else:
                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    for s in range(num_segments):
                        start_sample = samples_ps * s
                        finish_sample = start_sample + samples_ps

                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length)

                        mfcc = mfcc.T

                        # store mfcc if it has expected length
                        if len(mfcc) == expected_vects_ps:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)
                            print(f"{file_path}, segment: {s + 1}")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


from IPython.display import clear_output

save_mfcc(dataset_path, json_path, num_segments=10)
clear_output()
filepath = r"C:\Users\asmin\PycharmProjects\EBMP\archive\Happy\Happy\1.wav"
for i in range(2):
    audio, sfreq = librosa.load(filepath)
    time = np.arange(0, len(audio)) / sfreq
    plt.plot(time, audio)
    plt.xlabel("Time")
    plt.ylabel("Sound Amplitude")
    plt.show()