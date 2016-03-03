from Experiments.load_files import load_files
from numpy.random import RandomState
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.legend as legend
from sklearn.decomposition import PCA
file_name = 'cadata'
train_size = 0.8
path_prefix = "/home/zjia/kernelsubsampling/DataSets/"
prng_load_files = RandomState(8888)
X_train, y_train, X_test, y_test = load_files(file_name, train_size, path_prefix,prng_load_files)
X = X_train
y = y_train
if y.ndim == 1:
    y = y[:, np.newaxis]
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)

X_std[X_std == 0.] = 1.
y_std[y_std == 0.] = 1.

X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

cova = np.cov(X.T)
n_points, n_features = np.shape(X)
#print(np.mean(y_train))
"""
for i in range(n_features):
    print("%dth row is: "%i)
    print(cova[i])
pca = PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
plt.figure()
"""
plt.scatter(X[:,2],X[:,3])
plt.figure()
plt.scatter(X[:,4],y)
plt.show()

