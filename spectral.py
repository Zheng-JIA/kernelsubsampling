# spectral components of rank-k k_approx (k < l) 
import numpy as np

def spectral(k, loss_time, idx_method, idx_sampling_sizes):
    U_SR = loss_time.U_SR[idx_method]
    S_SR = loss_time.S_SR[idx_method]
    V_SR = loss_time.V_SR[idx_method]
    indices_SR = loss_time.indices_SR[idx_method]
    k_train = loss_time.k_train[idx_method]
    U_SR_single = U_SR[idx_sampling_sizes]
    S_SR_single = S_SR[idx_sampling_sizes]
    V_SR_single = V_SR[idx_sampling_sizes]
    idx_SR = indices_SR[idx_sampling_sizes]
    k_train_SR = k_train[:,idx_SR]
    pseudo_inv_W = np.dot(U_SR_single[:,0:k]*1.0/S_SR_single[:k], V_SR_single[0:k,:])
    k_approx = np.dot(k_train_SR, np.dot(pseudo_inv_W, k_train_SR.T))
    _,S,_ = np.linalg.svd(k_approx)
    return S[0:k]
