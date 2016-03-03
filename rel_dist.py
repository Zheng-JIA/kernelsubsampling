import numpy as np

def rel_dist(sampling_sizes, loss_time, idx_method):
    n = len(sampling_sizes)
    U_SR = loss_time.U_SR[idx_method]
    S_SR = loss_time.S_SR[idx_method]
    V_SR = loss_time.V_SR[idx_method]
    k_train = loss_time.k_train[idx_method]
    U_train = loss_time.U_train[idx_method]
    S_train = loss_time.S_train[idx_method]
    V_train = loss_time.V_train[idx_method]
    indices_SR = loss_time.indices_SR[idx_method]
    rel_spectral_dist = []
    for i in range(n):
        U_SR_single = U_SR[i]
        S_SR_single = S_SR[i]
        V_SR_single = V_SR[i]
        idx_SR = indices_SR[i]
        
        top_k = len(idx_SR)
        k_train_bestk = np.dot(U_train[:,0:top_k]*S_train[:top_k], V_train[0:top_k,:])
        k_train_SR = k_train[:,idx_SR]
        pseudo_inv_W = np.dot(U_SR_single[:,0:top_k]*1.0/S_SR_single[:top_k], V_SR_single[0:top_k,:])
        k_approx = np.dot(k_train_SR, np.dot(pseudo_inv_W, k_train_SR.T))
        k_train_2norm = np.linalg.norm(k_train,ord='fro')
        diff_k_train_k_approx = np.linalg.norm(k_train-k_approx,ord='fro')
        rel_spectral_dist.append(diff_k_train_k_approx/k_train_2norm)
       
    return rel_spectral_dist
