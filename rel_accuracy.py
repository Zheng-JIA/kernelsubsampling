import numpy as np
import sys 
import scipy as sp
import matplotlib.pyplot as plt 
import matplotlib.legend as legend
import math
import cPickle
import os
import matplotlib.gridspec as gridspec
from rel_accuracy import rel_accuracy
from rel_dist import rel_dist
from spectral import spectral
from load_results import load_results
from multiprocessing import Pool
from Framework.framework import framework
from Algorithms.subsampling import subsampling
from sklearn.datasets import load_svmlight_file
from numpy.random import RandomState
from Experiments.load_files import load_files

def rel_accuracy(sampling_sizes, k, loss_time, idx_method):
    n = len(sampling_sizes)
    U_SR = loss_time.U_SR[idx_method]
    S_SR = loss_time.S_SR[idx_method]
    V_SR = loss_time.V_SR[idx_method]
    k_train = loss_time.k_train[idx_method]
    k_train_SR = loss_time.k_train_SR[idx_method]
    U_train = loss_time.U_train[idx_method]
    S_train = loss_time.S_train[idx_method]
    V_train = loss_time.V_train[idx_method]
    rel_accuracy = []
    for i in range(n):
        U_SR_single = U_SR[i]
        S_SR_single = S_SR[i]
        V_SR_single = V_SR[i]
        k_train_SR_single = k_train_SR[i]    
        pseudo_inv_W = np.dot(U_SR_single[:,0:k]*1.0/S_SR_single[:k], V_SR_single[0:k,:])
        k_approx = np.dot(k_train_SR[i], np.dot(pseudo_inv_W, k_train_SR[i].T) )
        k_train_bestk = np.dot(U_train[:,0:k]*S_train[:k], V_train[0:k,:])
        print("computing matrix norm")
        diff_k_train_k_bestk = np.linalg.norm(k_train-k_train_bestk,ord='fro')
        diff_k_train_k_approx = sp.linalg.norm(k_train-k_approx,ord='fro')
        rel_accuracy.append(diff_k_train_k_bestk/diff_k_train_k_approx)
    return rel_accuracy
