from multiprocessing import Pool
from Experiments.experiments import execute_exper
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
import numpy as np
import cPickle
import matplotlib.pyplot as plt
if __name__ == '__main__':
    file_name = raw_input('File name: ')
    max_SR_size = float(raw_input('Input maximum size of subset in percentage: '))
    log_lin = raw_input('log or linear? ')
    n_subsets = float(raw_input('Input number of different sizes of subsets: '))
    theta0 = float(raw_input('Input theta0: '))
    var_n = float(raw_input('Input var_n: '))
    rounds = 1
    init_seed = 1024
    seed_range = [init_seed + 100*i for i in range(rounds)] 
    n_folds = 5 # five fold cross validation
    path_prefix = "/home/zjia/kernelsubsampling/DataSets/"
    path = path_prefix + file_name + "_scale.txt"
    print("Loading file %s_scale.txt" % file_name)
    # Load files
    data_X, data_y = load_svmlight_file(path)
    # Convert from sparse to dense format
    kf = cross_validation.KFold(int(len(data_y)*0.01),n_folds=n_folds, random_state=8888)
    fold = 0 
    for train_idx, test_idx in kf: 
        fold += 1
        execute_exper([theta0, var_n, fold, train_idx, test_idx, 1024, 1, file_name, max_SR_size,log_lin,n_subsets])
