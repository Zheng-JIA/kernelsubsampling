# Uniformly sampling n_SR points from training sets.
# Return indices of selected points
import numpy as np

def uni_sampling(X, n_SR):
    n_samples,_ = X.shape
    # Select n_SR points from X. Order doesn't matter
    idx_SR = np.random.permutation(n_samples)
    idx_SR = idx_SR[0:n_SR] 
    return idx_SR

