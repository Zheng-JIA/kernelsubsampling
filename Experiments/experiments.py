import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.legend as legend
import math
import os
import cPickle
from sklearn.metrics.pairwise import pairwise_kernels
from Framework.framework import framework
from Algorithms.subsampling import subsampling
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from numpy.random import RandomState
from sklearn.preprocessing import scale
from Experiments.load_files import load_files
import random

def execute_exper(parameters):
    #=============Parameters=============
    gamma, reg_const, fold, data, random_seed, file_name, min_SR_size, max_SR_size, log_lin, n_subsets, algorithm, sampling_methods, compute_k_train= parameters
    dataset_names = [file_name]#'bodyfat','housing','mpg','abalone'
    #path_prefix = "/home/zjia/kernelsubsampling/DataSets/"
    #Results_prefix = "/home/zjia/kernelsubsampling/Results/"
    path_prefix = "/Users/Allen_92/ETHZ/SemesterThesis/kernelsubsampling/DataSets"
    Results_prefix = "/Users/Allen_92/ETHZ/SemesterThesis/kernelsubsampling/Results"
    #=============Run Experiments=============
    np.random.seed(random_seed)
    path = path_prefix + file_name + "_scale.txt"
    for file_name in dataset_names:
        loss_time, sampling_sizes=framework(gamma, reg_const, data, sampling_methods,min_SR_size, max_SR_size, log_lin, n_subsets, algorithm,compute_k_train)
        print("%d fold is finised"%fold)
        # retrieve data
        f = file(os.path.join(Results_prefix,file_name)+"/"+file_name+"_"+ sampling_methods[0] +'_'+str(gamma)+"_"+str(reg_const)+"_"+str(fold)+'_'+str(random_seed), 'wb')
        result = [gamma, reg_const, sampling_methods, sampling_sizes, loss_time]
        print("Dumping Results")
        cPickle.dump(result, f, protocol=cPickle.HIGHEST_PROTOCOL)
        print("Dumping Finished")
        f.close()
