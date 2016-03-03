# Uniformly sampling n_SR points from training sets.
# Return indices of selected points
import numpy as np
#import random
from numpy.random import RandomState
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from time import time
def subsampling(X, n_SR, method):
    
    if method == 'None':
        return np.arange(n_SR)
    if method == 'uniform':
        return uni_sampling(X, n_SR)
    if method == 'D2_weighting':
        return DX_weighting(X, n_SR,'sqeuclidean')
    if method == 'D1_weighting':
        return DX_weighting(X, n_SR,'euclidean')
    if 'uniform_D2_mixed' in method:
        p_uniform = float(method.split('_')[3])
        return mixed_weighting(X, n_SR, 'sqeuclidean', p_uniform)
    if 'uniform_D1_mixed' in method:
        p_uniform = float(method.split('_')[3])
        return mixed_weighting(X, n_SR, 'euclidean', p_uniform)

def uni_sampling(X, n_SR):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples,_ = X.shape
    # Select n_SR points from X. Order doesn't matter
    idx_SR = np.random.choice(a=np.arange(n_samples),size=n_SR,replace=False) # high exclusive
    return idx_SR

def DX_weighting(X, n_SR, metric):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_features = X.shape    
    centers = np.empty([n_SR, n_features])
    centers_idx = []    
    init_center_idx = np.random.randint(low=0,high=n_samples,size=1)
    centers_idx.append(init_center_idx[0])
    centers[0] = X[init_center_idx]        
    # Store shortest distances from centers to X
    last_dist = cdist(X, np.array([centers[0]]), metric=metric)
    for n in range(1,n_SR): # Already drew initial center, so start from 1. # iter=n_SR-1
        prob, last_dist = update_prob(X, centers[:n], last_dist,metric) # arrays' order matters
        idx = np.random.choice(a=np.arange(n_samples),size=1,replace=False,p=prob)
        centers_idx.append(idx[0])
        centers[n] = X[idx]       
    return np.asarray(centers_idx)

def mixed_weighting(X, n_SR, metric, p_uniform):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_features = X.shape    
    centers = np.empty([n_SR, n_features])
    centers_idx = []    
    init_center_idx = np.random.randint(low=0,high=n_samples,size=1)
    centers_idx.append(init_center_idx[0])
    centers[0] = X[init_center_idx]    
    # Store shortest distances from centers to X
    last_dist = cdist(X, np.array([centers[0]]), metric=metric)
    states = np.random.binomial(1,p_uniform,n_SR-1) #Already drew initial center, so n_SR -1
    for n in range(1,n_SR): # Already drew initial center, so start from 1. # iter=n_SR-1
        if states[n-1] == 1:
            prob = np.ones(n_samples)
            prob[centers_idx] = 0
            prob = prob/np.sum(prob)
            idx = np.random.choice(a=np.arange(n_samples),size=1,replace=False,p=prob) 
        else:
            prob, last_dist = update_prob(X, centers[:n], last_dist,metric) # arrays' order matters    
            idx = np.random.choice(a=np.arange(n_samples),size=1,replace=False,p=prob) 
        centers_idx.append(idx[0])
        centers[n] = X[idx]
    return np.asarray(centers_idx) 

def update_prob(X, centers, last_dist, metric):
    # Compute distance between new drawn center (last column of ndarray centers) with X
    sq_dist = cdist(X, np.array([centers[-1]]),metric=metric)
    sq_dist = np.minimum(sq_dist, last_dist)[:,0]
    prob = sq_dist/np.sum(sq_dist)
    return prob, sq_dist[:,np.newaxis]

