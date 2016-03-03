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
from sklearn.datasets import load_svmlight_file
from numpy.random import RandomState
from Experiments.load_files import load_files
from Algorithms.subsampling import subsampling
from multiprocessing import Pool
from Experiments.experiments import execute_exper
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist

def quan_error(X_train, X_SR):
    sq_dist = cdist(X_train, X_SR, 'sqeuclidean')
    min_dist = np.min(sq_dist, axis=1)
    error = np.sum(min_dist)
    print("quan error")
    print(error)
    return error

def rel_approx_error(k_train, USV_k_SR, idx_SR):
    error = []
    U_SR, S_SR, V_SR = USV_k_SR
    k_train_SR = k_train[:,idx_SR]
    pseudo_inv_W = np.dot(U_SR*1.0/S_SR, V_SR)
    k_approx = np.dot(k_train_SR, np.dot(pseudo_inv_W, k_train_SR.T))
    diff_k_train_k_approx = np.linalg.norm(k_train-k_approx,ord='fro')/np.linalg.norm(k_train,ord='fro')
    print("diff matrix")
    print(diff_k_train_k_approx.tolist())
    error.append(diff_k_train_k_approx.tolist())
    return error, [U_SR,S_SR,V_SR]

def rel_approx_acc(k_train,USV_k_train,k_SR,USV_k_SR,idx_SR,k):
    acc = []
    U_SR, S_SR, V_SR = USV_k_SR
    U_train, S_train, V_train = USV_k_train
    k_train_SR = k_train[:,idx_SR]
    pseudo_inv_W = np.dot(U_SR[:,0:k]*1.0/S_SR[:k], V_SR[0:k,:])
    k_approx = np.dot(k_train_SR, np.dot(pseudo_inv_W, k_train_SR.T))
    k_train_k = np.dot(U_train[:,0:k]*S_train[:k],V_train[0:k,:])
    diff_k_train_k_approx = np.linalg.norm(k_train-k_train_k)/np.linalg.norm(k_train-k_approx,ord='fro')
    acc.append(diff_k_train_k_approx.tolist())
    return acc

def compute_err(parameters):
    X,k_train,USV_k_train,n_SR,method,gamma,seed,k = parameters
    np.random.seed(seed)
    idx_SR = subsampling(X,int(n_SR),method)
    X_SR = X[idx_SR]
    k_SR = rbf_kernel(X_SR,X_SR,gamma=gamma)
    U_SR, S_SR, V_SR = np.linalg.svd(k_SR)
    USV_k_SR = [U_SR, S_SR, V_SR]
    k_train_SR = rbf_kernel(X,X_SR,gamma=gamma)
    rel_err, USV_k_SR = rel_approx_error(k_train, USV_k_SR, idx_SR)
    rel_acc = rel_approx_acc(k_train,USV_k_train,k_SR,USV_k_SR,idx_SR,k)
    quan_err = quan_error(X,X_SR)
    return [rel_err,quan_err,rel_acc]

if __name__ == '__main__':
    init_seed = 2024
    file_name = raw_input('File name: ')
    n_cores = int(raw_input('Number of cores to use: '))
    pool = Pool(processes=n_cores)
    # Load files
    path_prefix = "/home/zjia/kernelsubsampling/DataSets/"
    path = path_prefix + file_name + "_scale.txt"
    data_X, data_y = load_svmlight_file(path)
    data_X = scale(data_X,with_mean=False)
    data_X = data_X.toarray() 
    
    #sampling_methods = ['uniform','D2_weighting','uniform_D2_mixed_0.5']
    sampling_methods = ['uniform','D2_weighting','D1_weighting','uniform_D2_mixed_0.5','uniform_D1_mixed_0.5']
    # Shuffle loaded data and sample partial data if data size is large to compute k_train  
    shuffle_prng = np.random.RandomState(1000)
    n_samples = len(data_y)
    idx_data = range(n_samples)
    shuffle_prng.shuffle(idx_data)
    data_X, data_y = data_X[idx_data], data_y[idx_data]
    print("The number of examples is %d"%n_samples)
    max_n_samples = 2000
    
    X = data_X[:max_n_samples] # use part of data set to train
    gamma = float(raw_input('Input gamma: '))
    #n_SR = int(raw_input('Input number of regressors: ')) 
    #min_SR_size= float(raw_input('Input minimum size of subset: '))              
    max_SR_size = float(raw_input('Input maximum size of subset: '))
    min_SR_size = 10
    #max_SR_size = 600
    #n_subsets = float(raw_input('Input number of different sizes of subsets: '))
    n_subsets = 5 
    n_rounds = int(raw_input('Number of repetitions: '))
    k = int(raw_input('Input top k approximation: '))
    seed_range = [init_seed + 100*i for i in range(n_rounds)]
    sampling_sizes = np.round(np.logspace(np.log(min_SR_size)/np.log(max_SR_size), 1, n_subsets,base=max_SR_size)) 
    k_train = rbf_kernel(X,X,gamma=gamma)
    print('compute svd of k_train')
    U_train, S_train, V_train = np.linalg.svd(k_train)
    USV_k_train = [U_train, S_train, V_train] 
     
    rel_err_fig = plt.figure()
    #quan_err_fig = plt.figure()
    rel_acc_fig = plt.figure()
    rel_quan_err_fig = plt.figure()
    idx_color = 0 
    idx_marker = 0
    markers = ['o','x','s','h','*']
    colors = ['blue','green','red','cyan','black'] 
    for method in sampling_methods:
        print('method is %s: '%method)
        rel_err = []
        quan_err = []
        rel_acc = []
        std_rel_err = []
        std_quan_err = []
        std_rel_acc = []
        rel_err_all = []
        quan_err_all = []
        for n_SR in sampling_sizes:
            rel_err_SR = []
            quan_err_SR = []
            rel_acc_SR = []
            parameters = [[X,k_train,USV_k_train,n_SR,method,gamma,seed,k] for seed in seed_range]
            #results = compute_err(parameters[0])
            results = pool.map(compute_err,parameters)
            
            for result in results:
                rel_err_SR.append(result[0])
                quan_err_SR.append(result[1])
                rel_acc_SR.append(result[2])
            std_rel_err.append(np.std(rel_err_SR))   
            std_quan_err.append(np.std(quan_err_SR)) 
            std_rel_acc.append(np.std(rel_acc_SR))
            rel_err.append(np.mean(rel_err_SR))   
            quan_err.append(np.mean(quan_err_SR))
            rel_acc.append(np.mean(rel_acc_SR))
            # store all rel_err and quan_err for plotting rel_err vs quan_err, each element represents one SR
            rel_err_all.append(rel_err_SR)
            quan_err_all.append(quan_err_SR) 
        
        #======================== dump files
        results_prefix = "/home/zjia/kernelsubsampling/Results/"
        path = os.path.join(results_prefix, file_name)
        f = file(os.path.join(path,file_name+'_'+method+'_'+str(gamma)+'_top_'+str(k)),'wb')
        dumped_results = [rel_err,quan_err,rel_acc,std_rel_err,std_quan_err,std_rel_acc,rel_err_all,quan_err_all]
        cPickle.dump(dumped_results, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close() 
        
        #======================== plot
        rel_err_plot = rel_err_fig.add_subplot(111)
        #rel_err_plot.set_title(file_name+'--gamma = '+str(gamma),fontsize=20)
        rel_err_plot.errorbar(sampling_sizes, rel_err, yerr=std_rel_err,color=colors[idx_color],marker=markers[idx_marker], label=method) 
        rel_err_plot.legend(loc='best',fontsize='xx-large')    
        rel_err_plot.tick_params(labelsize=20)
        rel_err_plot.set_xlabel('Number of Columns Sampled',fontsize='xx-large')
        rel_err_plot.set_ylabel('Relative Approximation Error',fontsize='xx-large')
        # relative accuracy: best k
        rel_acc_plot = rel_acc_fig.add_subplot(111)
        #rel_acc_plot.set_title(file_name+'--gamma = '+str(gamma),fontsize=20)
        rel_acc_plot.errorbar(sampling_sizes, rel_acc,yerr=std_rel_acc,color=colors[idx_color],marker=markers[idx_marker], label=method+'--k: '+str(k))
        rel_acc_plot.legend(loc='lower right',fontsize='xx-large')
        rel_acc_plot.tick_params(labelsize=20)
        rel_acc_plot.set_xlabel('Number of Columns Sampled',fontsize='xx-large')
        rel_acc_plot.set_ylabel('Best k Relative Accuracy',fontsize='xx-large') 
        # distribution of quantization and approxiamtion error
        rel_quan_plot = rel_quan_err_fig.add_subplot(111,yscale='log')
        #rel_quan_plot.set_title('Num of Cols: '+str(int(sampling_sizes[3])),fontsize=20)
        rel_quan_plot.scatter(quan_err_all[n_subsets-1], rel_err_all[n_subsets-1], color=colors[idx_color],marker=markers[idx_marker],label=method+'--Cols='+str(int(sampling_sizes[n_subsets-1])))
        rel_quan_plot.legend(loc='upper left',fontsize='xx-large')
        rel_quan_plot.tick_params(labelsize=20)
        rel_quan_plot.set_xlabel('Quantization Error',fontsize='xx-large')
        rel_quan_plot.set_ylabel('Relative Approximation Error',fontsize='xx-large')
        idx_color += 1
        idx_marker += 1
    plt.show()
