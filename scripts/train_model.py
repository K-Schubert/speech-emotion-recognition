from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

# load data
with open('../data/pickles/data.pkl', 'rb') as handle:
		data = pickle.load(handle)

X_train, y_train = zip(data.train)
X_train, y_train = np.array(X_train)[0], np.array(y_train)[0]

# train model
knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto',
						  leaf_size=5, metric='euclidean').fit(X_train, y_train)

# save model
with open('../models/knn.pkl', 'wb') as model_pkl:
	pickle.dump(knn, model_pkl)