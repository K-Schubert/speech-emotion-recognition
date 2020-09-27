from glob import glob
from sklearn.model_selection import train_test_split
from pickledataset import PickleDataset
import os
import librosa
import numpy as np
import pickle


class ComputeMFCCS:

	def __init__(self):
		pass

	def create_trn_val_test(self):

		# test data
		lst = []
		for file in glob('../data/wav/*.wav'):
			data, sampling_rate = librosa.load(file, res_type='kaiser_fast')
			mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40), axis=1)
			file = file.split('/')[-1]
			arr = mfccs, file[5]
			lst.append(arr)

		X, y = zip(*lst)
		y_test = np.asarray(y)
		X_test = np.asarray(X)

		# trn and val data
		lst = []
		for file in glob('../data/aug/*.wav'):
			
			data, sampling_rate = librosa.load(file, res_type='kaiser_fast')
			mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40), axis=1)
			file = file.split('/')[-1]
			arr = mfccs, file[5]
			lst.append(arr)

		X, y = zip(*lst)
		y = np.asarray(y)
		X = np.asarray(X)

		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=42)

		return X_train, y_train, X_val, y_val, X_test, y_test




if __name__=='__main__':

	mfcc = ComputeMFCCS()

	X_train, y_train, X_val, y_val, X_test, y_test = mfcc.create_trn_val_test()
	
	pickle_data = PickleDataset()
	pickle_data.train = (X_train, y_train)
	pickle_data.val = (X_val, y_val)
	pickle_data.test = (X_test, y_test)

	os.mkdir('../data/pickles/')
	with open('../data/pickles/data.pkl', 'wb') as handle:
		# for compatibility with python 2: protocol=2
		pickle.dump(pickle_data, handle, protocol=2)



