import numpy as np
from scipy.io.wavfile import read   
import scipy.io.wavfile as wav
import os

def read_and_split_audio(path, segment_duration):
    for file in os.listdir(path):
        if file.endswith(".wav"):
            fs, data = read(f'{path}/{file}')
            data = data.astype(float)
            data = data/32768
            duration = len(data) / fs
            n_samples_per_segment = int(fs * segment_duration)
            segments = np.array_split(data, data.shape[0]/n_samples_per_segment)
            
            for i in range(int(duration//2)):
                start = i * 2 * fs
                end = (i+1) * 2 * fs
                segment = data[int(start):int(end)]

                # Save the new wav file
                new_filename = f'{file.split(".")[0]}{i}.wav'
                wav.write(f'{"Data/Trucks_sliced/"}/{new_filename}', fs, segment)
            
            for i, segment in enumerate(segments):
                print(f'Segment {i} of file {file} : {segment}')

read_and_split_audio("Data/Trucks", 2)