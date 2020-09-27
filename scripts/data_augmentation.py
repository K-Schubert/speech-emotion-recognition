import os
import librosa
from librosa.effects import time_stretch, pitch_shift
from scipy.io import wavfile
import pandas as pd
import numpy as np

class DataAug:

	def __init__(self):
		pass

	def vary_time(self, rates, files):

		for rate in rates:
			for file in files:
			    y, sr = librosa.load(f'../data/wav/{file}')
			    y_changed = time_stretch(y, rate=rate)
			    wavfile.write(filename=f'../data/aug/{file.split(".")[0]}_{rate}.wav', rate=sr, data=y_changed)

		return

	def vary_pitch(self, steps, files):

		for n_step in steps:
		    for file in files:
		        y, sr = librosa.load(f'../data/wav/{file}')
		        y_changed = pitch_shift(y, sr, n_steps=n_step)
		        wavfile.write(filename=f'../data/aug/{file.split(".")[0]}_pitch_{n_step}.wav', rate=sr, data=y_changed)

		return


	def downsample(self, freq, files):

		for file in files:
		    y, sr = librosa.load(f'../data/wav/{file}', sr=freq)
		    wavfile.write(filename=f'../data/aug/{file}_downsampled_{freq/1000}khz.wav', rate=sr, data=y)

	def envelope(self, y, rate, threshold):
	    mask = []
	    y = pd.Series(y).apply(np.abs)
	    y_mean = y.rolling(window=int(0.1*rate), min_periods=1, center=True).mean()
	    for mean in y_mean:
	        if mean > threshold:
	            mask.append(True)
	        else:
	            mask.append(False)
	    return mask

	def remove_deadspace(self, files_aug):

		for file in files_aug:
		    signal, sr = librosa.load(f'../data/aug/{file}', sr=16000)
		    mask = self.envelope(signal, sr, 0.0005)
		    signal = signal[mask]
		    wavfile.write(filename=f'../data/aug/{file}', rate=sr, data=signal)


if __name__ == '__main__':

	os.mkdir('../data/aug')

	aug = DataAug()

	files = os.listdir('../data/wav')

	rates = [0.8, 1.2]
	aug.vary_time(rates, files)

	steps = [-2.5, -2, -1.5, -1, -0.5, 0.5,  1, 1.5, 2, 2.5]
	aug.vary_pitch(steps, files)

	freq = 8000
	aug.downsample(freq, files)

	files_aug = os.listdir('../data/aug')

	aug.remove_deadspace(files_aug)

