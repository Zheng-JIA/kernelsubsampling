from multiprocessing import Pool
from Experiments.experiments import execute_exper
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist
import os
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from load_results import load_results
if __name__ == '__main__':
    init_seed = 1024
    file_name = raw_input('File name: ')
    algorithm = raw_input('Input algorithm gp, svm_hinge, svm_log: ')
    log_lin = 'log'
    n_cores = int(raw_input('Number of cores you want to use: '))

    sampling_methods = ['None']
    rounds = 1
    n_subsets = 1

    pool = Pool(processes=n_cores)

    # initialize different seed for each call to execute_exper
    n_folds = 5 # five fold cross validation
    
    path_prefix = "/home/zjia/kernelsubsampling/DataSets/"
    path = path_prefix + file_name + "_scale.txt"
    print("Loading file %s_scale.txt" % file_name)
    
    # Load files
    data_X, data_y = load_svmlight_file(path)
    data_X = scale(data_X, with_mean=False)
    data_X = data_X.toarray()
    # Shuffle loaded data and sample partial data if data size is large to compute k_train 
    shuffle_prng = np.random.RandomState(1000)
    n_samples = len(data_y)
    idx_data = range(n_samples)
    shuffle_prng.shuffle(idx_data)
    data_X, data_y = data_X[idx_data], data_y[idx_data]
    print("Number of data is %d"%n_samples)
    reduce_data_size = raw_input('Reduce data size, yes/no?: ') # use part of data set to train
    if reduce_data_size == 'yes':
        max_n_samples = int(raw_input('Input maximum size of data set: '))
    else:
        max_n_samples = n_samples
    data_X = data_X[:max_n_samples] # use part of data set to train
    data_y = data_y[:max_n_samples]
    
    #avg_data_X = np.array([np.mean(data_X.toarray(),0)])
    #avg_dist_squared = np.mean(cdist(data_X.toarray(),avg_data_X,'sqeuclidean'),0)
    
    min_gamma = float(raw_input('Input minimum of gamma: '))
    max_gamma = float(raw_input('Input maximum of gamma: '))
    n_gamma = int(raw_input('Input number of gammas: '))
    min_reg = float(raw_input('Input minimum of regularization constant: '))
    max_reg = float(raw_input('Input maximum of regularization constant: '))
    n_reg = int(raw_input('Input number of regularization constants: '))

    gamma_range = np.logspace(np.log2(min_gamma), np.log2(max_gamma), n_gamma,base=2,endpoint=True)
    reg_const_range = np.logspace(np.log2(min_reg), np.log2(max_reg), n_reg,base=2,endpoint=True)
    kf = cross_validation.KFold(len(data_y),shuffle=False, n_folds=n_folds, random_state=shuffle_prng)
    fold = 0
    for train_idx, test_idx in kf:
        fold += 1
        X_train, y_train, X_test, y_test = data_X[train_idx], data_y[train_idx], data_X[test_idx], data_y[test_idx]
        k_train, U_train, S_train, V_train = None,None,None,None
        data = [X_train, y_train, X_test, y_test, k_train, U_train, S_train, V_train]
        min_SR_size = len(train_idx)
        max_SR_size = len(train_idx)
        parameters = [[gamma, reg_const, fold, data, init_seed, file_name, min_SR_size, max_SR_size, log_lin, n_subsets,algorithm,sampling_methods,'no'] for gamma in gamma_range for reg_const in reg_const_range] 
        #execute_exper(parameters[0])
        pool.map(execute_exper,parameters)
    pool.close()
    pool.join()
