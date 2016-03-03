from Algorithms.gp import GaussianProcess
from Algorithms.svm import kernelsvm
from sklearn.metrics import mean_squared_error
from time import time
from Algorithms.subsampling import subsampling
from sklearn.metrics.pairwise import pairwise_kernels

from pylab import pcolor, show, colorbar, yticks, xticks
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from Framework.hinge_loss import hinge_loss
from Framework.log_loss import log_loss
# Tune and pass hyperparameters to GP model
# GaussianProcess(gamma, reg_const)
# k(x1,x2) = exp(-gamma*|x1 - x2|^2) + reg_const*delta(x1,x2)
def framework(gamma, reg_const, data, sampling_methods,min_SR_size, max_SR_size,log_lin, n_subsets, algorithm,compute_k_train):
    X_train, y_train, X_test, y_test, k_train, U_train, S_train, V_train = data
    n_train = len(y_train)

    if log_lin == 'linear':
        sampling_sizes = np.round(np.linspace(min_SR_size,max_SR_size,n_subsets))
    else:
        sampling_sizes = np.round(np.logspace(np.log(min_SR_size)/np.log(max_SR_size), 1, n_subsets,base=max_SR_size))
        
    # Pass hyper-parameters to models	
    if algorithm == 'gp':
        print("training gp")
        learner = GaussianProcess(gamma,reg_const) # choosing signal variance is important.
    if algorithm == 'svm_hinge':
        print("training svm_hinge")
        learner = kernelsvm(gamma, reg_const, 'hinge')
    if algorithm == 'svm_log':
        print("training svm log")
        learner = kernelsvm(gamma, reg_const, 'log')
    loss_time = store_loss_time(sampling_methods)
    for method in sampling_methods:
        loss_train=[]
        loss_test=[]
        time_train=[]
        time_sampling=[]
        
        U_SR = []
        S_SR = []
        V_SR = []
        dist_train = []
        dist_test = []
        indices_SR = []
        print("method is",method)
        for n_SR in sampling_sizes:
            print(sampling_sizes)
            n_SR = int(n_SR)# Convert float number to integer, otherwise kmeans will fail
            print("n_SR is %d"%n_SR)
            print("Sampling...")
            start = time()
            idx_SR = subsampling(X_train, n_SR, method)
            sampling_time = time() - start
            print("Sampling finished")
            # Fit using sampled points
            print("training")
            start = time()
            learner.fit(X_train, y_train, idx_SR)
            training_time = time() - start
            print("training finished")
            # store results
            n_test = len(y_test)
            if algorithm == 'gp':
                single_pred_train = learner.predict(X_train[:n_test])
                single_pred_test = learner.predict(X_test)
                dist_train.append(np.absolute(single_pred_train[:,0]-y_train[:n_test])) # single_pred_train has two dimenions, whereas y_train has one
                dist_test.append(np.absolute(single_pred_test[:,0]-y_test))
                loss_train.append(learner.loss(y_train[:n_test],single_pred_train))
                loss_test.append(learner.loss(y_test,single_pred_test))
                if compute_k_train=='yes':
                    U_SR_single, S_SR_single, V_SR_single = np.linalg.svd(learner.k_SR)
                else:
                    U_SR_single, S_SR_single, V_SR_single = None,None,None
                U_SR.append(U_SR_single)
                S_SR.append(S_SR_single)
                V_SR.append(V_SR_single)
                
            if algorithm == 'svm_hinge' or algorithm == 'svm_log':
                print("computing loss")
                single_pred_train, X_train_transform = learner.predict(X_train[:n_test])
                single_pred_test, X_test_transform = learner.predict(X_test)
                if algorithm == 'svm_hinge': 
                    dist_train.append(hinge_loss(y_train[:n_test],learner.decision_function(X_train_transform)))
                    dist_test.append(hinge_loss(y_test, learner.decision_function(X_test_transform)))
                if algorithm == 'svm_log':
                    dist_train.append(log_loss(y_train[:n_test],learner.decision_function(X_train_transform)))
                    dist_test.append(log_loss(y_test, learner.decision_function(X_test_transform)))
                loss_train.append(learner.err_rate(y_train[:n_test],single_pred_train))
                loss_test.append(learner.err_rate(y_test,single_pred_test))
                if compute_k_train=='yes':
                    U_SR_single = learner.feature_map_nystroem.U
                    S_SR_single = learner.feature_map_nystroem.S
                    V_SR_single = learner.feature_map_nystroem.V
                else:
                    U_SR_single = None
                    S_SR_single = None
                    V_SR_single = None
                U_SR.append(U_SR_single)
                S_SR.append(S_SR_single)
                V_SR.append(V_SR_single)
            time_train.append(training_time)
            time_sampling.append(sampling_time)
            
            indices_SR.append(idx_SR)
        #loss_time.store(method, loss_train, loss_test, time_train, time_sampling, k_train, U_train, S_train, V_train, U_SR, S_SR, V_SR,[None],[None],indices_SR,n_train)
        loss_time.store(method, loss_train, loss_test, time_train, time_sampling, k_train, U_train, S_train, V_train, U_SR, S_SR, V_SR,dist_train,dist_test, indices_SR, n_train)
    
    return loss_time, sampling_sizes

class store_loss_time():
    def __init__(self, sampling_methods):
        self.n_methods = len(sampling_methods)        
        self.method_dic = {method:i for method,i in zip(sampling_methods,np.arange(self.n_methods))}
        self.loss_train = [None]*self.n_methods
        self.loss_test = [None]*self.n_methods
        self.time_train = [None]*self.n_methods
        self.time_sampling = [None]*self.n_methods
        self.k_train = [None]*self.n_methods
        self.U_train = [None]*self.n_methods
        self.S_train = [None]*self.n_methods
        self.V_train = [None]*self.n_methods
        self.U_SR = [None]*self.n_methods
        self.S_SR = [None]*self.n_methods
        self.V_SR = [None]*self.n_methods
        self.dist_train= [None]*self.n_methods
        self.dist_test= [None]*self.n_methods
        self.indices_SR = [None]*self.n_methods
        self.n_train = [None]*self.n_methods
    def store(self,method,loss_train,loss_test,time_train,time_sampling,k_train,U_train,S_train,V_train,U_SR,S_SR, V_SR,dist_train,dist_test, indices_SR, n_train):
        idx_method = self.method_dic[method]
        self.loss_train[idx_method] = loss_train
        self.loss_test[idx_method] = loss_test
        self.time_train[idx_method] = time_train
        self.time_sampling[idx_method] = time_sampling
        self.k_train[idx_method] = k_train
        self.U_train[idx_method] = U_train
        self.S_train[idx_method] = S_train
        self.V_train[idx_method] = V_train
        self.U_SR[idx_method] = U_SR
        self.S_SR[idx_method] = S_SR
        self.V_SR[idx_method] = V_SR
        self.dist_train[idx_method] = dist_train
        self.dist_test[idx_method] = dist_test
        self.indices_SR[idx_method] = indices_SR
        self.n_train[idx_method] = n_train

