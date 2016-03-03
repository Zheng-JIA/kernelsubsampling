from multiprocessing import Pool
from Experiments.experiments import execute_exper
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist
import numpy as np
import cPickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    init_seed = 1024
    file_name = raw_input('File name: ')
    algorithm = raw_input('gp, svm_hinge, or svm_log?: ')
    log_lin = raw_input('Use log or linear scale to generate sampling sizes?: ')
    n_cores = int(raw_input('Number of cores you want to use: '))
    rounds = int(raw_input('Input number of rounds: '))
    min_SR_size= float(raw_input('Input minimum size of subset: ')) 
    max_SR_size = float(raw_input('Input maximum size of subset: '))
    n_subsets = float(raw_input('Input number of different sizes of subsets: '))
    gamma = float(raw_input('Input gamma: '))
    reg_const = float(raw_input('Input reg_const: '))
    #-----------------------------------------------------------------
    p_uni_range = [0.25, 0.5, 0.75]
    sampling_methods = ['uniform','D2_weighting','D1_weighting']
    for p_uniform in p_uni_range:
        sampling_methods.append('uniform_D2_mixed_'+str(p_uniform))
        sampling_methods.append('uniform_D1_mixed_'+str(p_uniform))
    #sampling_methods = ['D1_weighting','uniform_D1_mixed_0.25','uniform_D1_mixed_0.5','uniform_D2_mixed_0.25','uniform_D2_mixed_0.5','uniform_D1_mixed_0.75','uniform_D2_mixed_0.75']
    #sampling_methods = ['uniform','D2_weighting']
    #sampling_methods = ['D1_weighting','uniform_D2_mixed_0.25','uniform_D2_mixed_0.5','uniform_D2_mixed_0.75','uniform_D1_mixed_0.25','uniform_D1_mixed_0.5','uniform_D1_mixed_0.75']
    sampling_methods = ['uniform_D2_mixed_0.5','uniform_D1_mixed_0.5']
    sampling_methods = ['uniform']
    print(sampling_methods)
    #---------------------------------------------------------------
    # Load files and scale input 
    print("Loading file %s_scale.txt" % file_name)
    #path_prefix = "/home/zjia/kernelsubsampling/DataSets/"
    path_prefix = "/Users/Allen_92/ETHZ/SemesterThesis/kernelsubsampling/DataSets/" 
    path = path_prefix + file_name + "_scale.txt"
    data_X, data_y = load_svmlight_file(path)
    data_X = scale(data_X,with_mean=False)
    data_X = data_X.toarray()
    n_samples = len(data_y)
    print("The number of data is %d. "%n_samples)

    # Shuffle loaded data and sample partial data if data size is large to compute k_train 
    shuffle_prng = np.random.RandomState(1000)
    idx_data = range(n_samples)
    shuffle_prng.shuffle(idx_data)
    data_X, data_y = data_X[idx_data], data_y[idx_data] 
    reduce_data_size = raw_input('Reduce data size ? yes/no: ') # use part of data set to train
    if reduce_data_size == 'yes':
        max_n_samples = int(raw_input('Input maximum size of data set: '))
    else:
        max_n_samples = n_samples
    data_X = data_X[:max_n_samples] 
    data_y = data_y[:max_n_samples]
    compute_k_train = raw_input('compute k_train? yes/no: ')

    #avg_data_X = np.array([np.mean(X_train,0)])
    #avg_dist_squared = np.mean(cdist(X_train,avg_data_X,'sqeuclidean'),0)
    #gamma = 4/avg_dist_squared
    #print(avg_dist_squared)

    # train one parameter multiple times on multiple cores
    pool = Pool(processes=n_cores)
   
    # initialize different seed for each call to execute_exper
    seed_range = [init_seed + 100*i for i in range(rounds)]
    crossval = raw_input('cross validation?: yes/no: ')
    n_folds = int(raw_input('Number of fold: ')) # five fold cross validation

    if crossval == 'yes':
        kf = cross_validation.KFold(len(data_y),shuffle=False, n_folds=n_folds)
        for method in sampling_methods:
            fold = 0
            for train_idx, test_idx in kf:
                print(len(train_idx))
                X_train, y_train, X_test, y_test = data_X[train_idx], data_y[train_idx], data_X[test_idx], data_y[test_idx]
                fold += 1
                if compute_k_train == 'yes':
                    k_train = pairwise_kernels(X_train,X_train,metric='rbf',gamma=gamma)
                    print("compute svd of k_train")
                    U_train,S_train,V_train = np.linalg.svd(k_train)
                    print("svd finished")
                else:
                    k_train, U_train, S_train, V_train = None,None,None,None
                data = [X_train, y_train, X_test, y_test, k_train, U_train, S_train, V_train]
                parameters = [[gamma, reg_const, fold, data, random_seed, file_name,min_SR_size,max_SR_size,log_lin,n_subsets,algorithm,[method],compute_k_train] for random_seed in seed_range]
                #execute_exper(parameters[0])        
                pool.map(execute_exper,parameters)
    else:
        print("Maximum data size is : %d"%max_n_samples)
        n_test = int(raw_input('Input number of test examples: '))
        train_idx = range(n_samples)[:max_n_samples-n_test]
        test_idx = range(n_samples)[max_n_samples-n_test:max_n_samples]
        X_train, y_train, X_test, y_test = data_X[train_idx], data_y[train_idx], data_X[test_idx], data_y[test_idx]
        fold = 0
        for method in sampling_methods:
            if compute_k_train == 'yes':
                k_train = pairwise_kernels(X_train,X_train,metric='rbf',gamma=gamma)
                print("compute svd of k_train")
                U_train,S_train,V_train = np.linalg.svd(k_train)
                print("svd finished")
            else:
                k_train, U_train, S_train, V_train = None,None,None,None 
            data = [X_train, y_train, X_test, y_test, k_train, U_train, S_train, V_train]
            parameters = [[gamma, reg_const, fold, data, random_seed, file_name,min_SR_size,max_SR_size,log_lin,n_subsets,algorithm,[method],compute_k_train] for random_seed in seed_range]
            #execute_exper(parameters[0])
            pool.map(execute_exper,parameters)        
    pool.close()
    pool.join()


