from flask import Flask, render_template, request, url_for, session
import io
import base64
import os
import glob
import librosa
import numpy as np
import pickle
import pickledataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = "super_secret_key"

@app.route('/pred', methods=['GET', 'POST'])
def pred():

	if request.method == 'POST':
		
		# get user input
		selected_file = request.form['wav']

		# process audio file
		X = process_audio(selected_file)

		# load model
		with open('../models/knn.pkl', 'rb') as model_pkl:
			model = pickle.load(model_pkl)

		# make predictions
		prediction = model.predict(X.reshape(1,-1))

		emotions = {'W': 'Arger (Wut)',
			'L': 'Langeweile',
			'E': 'Ekel',
			'A': 'Angst',
			'F': 'Freude',
			'T': 'Trauer',
			'N': 'Neutral'}

		# get list of audio files
		#file_list = glob('./static/wav/*.wav')
		file_list = glob.glob('../data/wav/*.wav')
		file_list = [s.split('/')[-1] for s in file_list] 

		return render_template('pred.html', result=f'Predicted emotion for file "{selected_file}" is {prediction[0]} ({emotions[prediction[0]]}).', 
												 true_value=f'True emotion is {selected_file[5]} ({emotions[selected_file[5]]}).', file_list=file_list)

	else:
		#file_list = glob('./static/wav/*.wav')
		file_list = glob.glob('../data/wav/*.wav')
		file_list = [s.split('/')[-1] for s in file_list] 
		return render_template('pred.html', file_list = file_list)

def process_audio(selected_file):
	
	data, sampling_rate = librosa.load(f'../data/wav/{selected_file}', res_type='kaiser_fast')
	mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
	selected_file = selected_file.strip('.wav')
	X = mfccs
	X = np.asarray(X)

	return X

@app.route('/train', methods=['GET', 'POST'])
def train():
	if request.method == 'POST':

		# load trn/val/test sets
		X_train, y_train, X_val, y_val, X_test, y_test = load_data()

		# get user input
		nn, w, a, lf, m = get_user_input()

		# fit model
		knn = KNeighborsClassifier(n_neighbors=nn, weights=w,
							 algorithm=a, leaf_size=lf,
							 metric=m).fit(X_train, y_train)
		
		# compute scores
		trn_score, val_score, test_score = compute_scores(knn, X_train, y_train, X_val, y_val, X_test, y_test)

		class_report = classification_report(y_test, knn.predict(X_test), output_dict=True)
		class_report = np.around(pd.DataFrame(class_report), 3).T.to_html()

		# plot confusion matrix
		pngImageB64String = plot_cm(knn, X_test, y_test)

		return render_template('train.html', trn_score = trn_score, val_score = val_score,
											 test_score = test_score, class_report=class_report,
											 image = pngImageB64String)

	else:

		return render_template('train.html') 

def load_data():
	
	with open('../data/pickles/data.pkl', 'rb') as handle:
		data = pickle.load(handle)

	X_train, y_train = zip(data.train)
	X_train, y_train = np.array(X_train)[0], np.array(y_train)[0]
	X_val, y_val = zip(data.val)
	X_val, y_val = np.array(X_val)[0], np.array(y_val)[0]
	X_test, y_test = zip(data.test)
	X_test, y_test = np.array(X_test)[0], np.array(y_test)[0]

	return X_train, y_train, X_val, y_val, X_test, y_test

def get_user_input():

	session.clear()
	session['nn'] = int(request.form['n_neighbors'])
	session['w'] = request.form['weights']
	session['a'] = request.form['algorithm']
	session['lf'] = int(request.form['leaf_size'])
	session['m'] = request.form['metric']

	return session['nn'], session['w'], session['a'], session['lf'], session['m']

def compute_scores(estimator, X_train, y_train, X_val, y_val, X_test, y_test):

	trn_score = estimator.score(X_train, y_train)
	val_score = estimator.score(X_val, y_val)
	test_score = estimator.score(X_test, y_test)

	return np.around(trn_score, 3), np.around(val_score, 3), np.around(test_score, 3)

def plot_cm(est, x, y):
	fig, ax = plt.subplots(1)
	plot_confusion_matrix(est, x, y, ax=ax)
	pngImage = io.BytesIO()
	FigureCanvas(fig).print_png(pngImage)
	pngImageB64String = "data:image/png;base64,"
	pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

	return pngImageB64String

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')



