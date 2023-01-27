# -*- coding: utf-8 -*-
"""
Example script

Script to perform some corrections in the brief audio project

Created on Fri Jan 27 09:08:40 2023

@author: ValBaron10
"""

# Import
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from features_functions import compute_features

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report

# Set the paths to the files 
data_path = "Data/"

# Names of the classes
classes_paths = ["Cars/", "Trucks/"]
classes_names = ["car", "truck"]
nbr_of_obs = 28

# Go to search for the files
learning_labels = []
for i in range(nbr_of_obs):
    if i < nbr_of_obs//2:
        name = f"{classes_names[0]}{i}.wav"
        learning_labels.append(classes_names[0])
        class_path = classes_paths[0]
    else:
        name = f"{classes_names[1]}{i - nbr_of_obs//2}.wav"
        learning_labels.append(classes_names[1])
        class_path = classes_paths[1]

    fs, data = sio.wavfile.read(data_path + class_path + name)
    data = data.astype(float)
    data = data/32768

    # Compute the signal in three domains
    sig_sq = data**2
    sig_t = data / np.sqrt(sig_sq.sum())
    sig_f = np.absolute(np.fft.fft(sig_t))
    sig_c = np.absolute(np.fft.fft(sig_f))

    # Compute the features and store them
    features_list = []
    N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], fs)
    features_vector = np.array(features_list)[np.newaxis,:]

    if i == 0:
        learning_features = features_vector
    else:
        learning_features = np.vstack((learning_features, features_vector))

# Separate data in train and test
X_train, X_test, y_train, y_test = train_test_split(learning_features, learning_labels, test_size=0.33, random_state=42)

# Standardize the labels
labelEncoder = preprocessing.LabelEncoder().fit(y_train)
learningLabelsStd = labelEncoder.transform(y_train)
testLabelsStd = labelEncoder.transform(y_test)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
learningFeatures_scaled = scaler.transform(X_train)

model.fit(learningFeatures_scaled, learningLabelsStd)

# Test the model
testFeatures_scaled = scaler.transform(X_test)

# Compute predictions
y_pred = model.predict(testFeatures_scaled)

# Compute and print the classification report
cr = classification_report(testLabelsStd, y_pred)
print(cr)

# Matrix confusion
plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
plt.show()

