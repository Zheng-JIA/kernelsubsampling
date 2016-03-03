import numpy as np
import os
import cPickle
from rel_accuracy import rel_accuracy
from rel_dist import rel_dist
from spectral import spectral

def load_results(parameters):
    path, files, method,k,idx_sampling_sizes,compute_k_train = parameters
    f = file(os.path.join(path,files), 'rb')
    [gamma,reg_const,sampling_methods,sampling_sizes,loss_time] = cPickle.load(f)
    f.close()
    idx_method = loss_time.method_dic[method]
    loss_train = loss_time.loss_train[idx_method]
    loss_test = loss_time.loss_test[idx_method]
    dist_train = loss_time.dist_train[idx_method]
    dist_test = loss_time.dist_test[idx_method]
    #dist_train = [None]
    #dist_test = [None]
    if compute_k_train == 'True':
        print("Computing Relative Accuracy")
        relative_accuracy = rel_accuracy(sampling_sizes, k, loss_time, idx_method)
        print(relative_accuracy)
        print("Computing rel dist")
        relative_dist = rel_dist(sampling_sizes, loss_time, idx_method)
        S_train = loss_time.S_train[idx_method]
        print("Computing Spectral")
        #spectral_approx = spectral(k,loss_time,idx_method,idx_sampling_sizes)
        spectral_approx = None
    else:
        relative_accuracy = None
        relative_dist = None
        S_train = None
        spectral_approx = None
    sampling_time = loss_time.time_sampling[idx_method]
    fitting_time = loss_time.time_train[idx_method]

    return [loss_train,loss_test,relative_accuracy, relative_dist, S_train,spectral_approx,sampling_time,fitting_time, dist_train, dist_test]
