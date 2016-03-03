from multiprocessing import Pool
from Experiments.experiments import execute_exper
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist
import sys
import os
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from load_results import load_results
def search_opt_params(file_name, folder):
    
    n_cores = int(raw_input('Number of cores (= Number of rounds of repetitions): '))
    pool = Pool(processes=n_cores)
    # load results and find optimal parameters
    path_prefix = "/home/zjia/kernelsubsampling/Results/"
    path = os.path.join(path_prefix, file_name)
    path = os.path.join(path, folder)
    gamma_list = []
    reg_const_list = []
    parameter_list = []
    idx_gamma = 2
    idx_reg = 3
    for files in os.listdir(path):
        if os.path.isfile(os.path.join(path,files)) == True:
            file_split = files.split('_')
            print(file_split)
            try:
                if file_split[idx_gamma] not in gamma_list: 
                    gamma_list.append(file_split[idx_gamma]) 
                if file_split[idx_reg] not in reg_const_list:
                    reg_const_list.append(file_split[idx_reg])
            except:
                print("Files corrupted")
    idx_reg_const = []
    gamma_list = np.sort([float(gamma) for gamma in gamma_list])
    gamma_list = [str(gamma) for gamma in gamma_list]

    compute_k_train = 'False'
    for gamma in gamma_list:
        try:
            avg_loss_test = []
            for reg_const in reg_const_list:
                params = gamma+'_'+reg_const
                loss_test = []
                method = 'None'
                parameters = [[path, files, method,0, 0, compute_k_train] for files in os.listdir(path) if params in files]
                results = pool.map(load_results, parameters)
                #results = load_results(parameters[0])
                for result in results:
                    loss_test.append(result[1])
                    #print(result[1])
                avg_loss_test.append(np.mean(loss_test,0))
            opt_loss = np.min(avg_loss_test)
            opt_idx = np.argmin(avg_loss_test)
            idx_reg_const.append(opt_idx)
            print('gamma: %s'%gamma,'--reg_const: %s'%reg_const_list[opt_idx],'--loss is: %f'%opt_loss)
        except:
            print("Files corrupted")
    pool.close()
    pool.join()

if __name__ == '__main__':
    search_opt_params(sys.argv[1],sys.argv[2])
