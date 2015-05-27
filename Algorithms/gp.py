
# Gaussian Process 
# Linear Predictor
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

class f_GP():

    def __init__(self, theta0, var_noise, var_signal):
        """
        Initialize hyperparameters of kernels and noise variance
        k(x1,x2) = var_f*exp(-theta0*|x1 - x2|^2) + var_n*delta(x1,x2)
        Parameters
        ----------
        theta0: 1/var
        var_n: noise variance
        var_f: signal variance
        """
        
        self.theta0 = theta0
        self.var_n = var_noise
        self.var_f = var_signal
        
    def fit(self, X, y, idx_SR):
        """
        The Gaussian Process model fitting method
        
        Parameters
        ----------
        X: Input of observations, an array with shape (n_samples, n_features) 
        
        y: Output of observations, an array with shape (n_samples, )
        
        idx_SR: Selected input index for "subset of regressors (SR)" approximation method, an 
                array with shape (m_samples, ). m_samples is the number of data selected.
                X[idx_SR,:] is the selected input
        """
        if y.ndim == 1:
            y = y[:, np.newaxis]
            
        # Check shapes of input and output
        n_samples, n_features = X.shape
        _, n_targets = y.shape
        
        # data normalization: center and scale X
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        X_std[X_std == 0.] = 1.
        y_std[y_std == 0.] = 1.

        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std    
        X_SR = X[idx_SR,:]
        
        # Compute Kernel Matrix 
        k_train = self.kernel(X,X)   #k_nn:=K(X_train, X_train)
        k_SR = self.kernel(X_SR,X_SR)
        k_SR_train = self.kernel(X_SR, X)   #k_mn:=K(X_SR, X_train) 
        k_train_SR = k_SR_train.T   #k_nm:=K(X_train, K_SR)
            
        # Set Attribute
        self.n_samples = n_samples
        self.X = X
        self.y = y
        self.X_SR = X_SR
        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std
        self.k_train = k_train
        self.k_SR = k_SR
        self.k_SR_train = k_SR_train
        self.k_train_SR = k_train_SR
        
        # Use full data points
        self.alpha = np.dot(np.linalg.inv(k_train + self.var_n * np.identity(self.n_samples)),y)

        # Use subset of data points: 
        #self.alpha_SR = np.dot(np.linalg.inv(np.dot(k_SR_train,k_train_SR) + self.var_n*k_SR), np.dot(k_SR_train,y)) #!!!!This causes numerical problem!!!!
	self.alpha_SR = np.linalg.solve(np.dot(k_SR_train,k_train_SR) + self.var_n*k_SR, np.dot(k_SR_train,y))
	return self.alpha
    
    def predict(self, X_test, eval_MSE=False, SR=True):
        # Normalize input
        X_test = (X_test - self.X_mean) / self.X_std
        # Kernelized input 
        k_train_test = self.kernel(self.X,X_test)
        # kernel matrix K(X_test,X_test)
        k_test = self.kernel(X_test,X_test)
        
        # Compute and denormalize Predictions and 95% Confidence Interval       
        if SR:
            k_SR_test = self.kernel(self.X_SR, X_test)
            pred_SR = np.dot(k_SR_test.T,self.alpha_SR)
            pred_SR = (pred_SR * self.y_std) + self.y_mean
            inv_temp_SR = np.linalg.inv(np.dot(self.k_SR_train,self.k_train_SR) + self.var_n*self.k_SR)
            #cov_SR = self.var_n*np.dot(k_SR_test.T, np.dot(inv_temp_SR, k_SR_test))
            if eval_MSE:
                return pred_SR, cov_SR
            else:
                return pred_SR
        else:
            pred = np.dot(k_train_test.T, self.alpha)
            pred = (pred * self.y_std) + self.y_mean   # mistakes: use X_mean X_std  
            print(self.n_samples)      
            inv_temp = np.linalg.inv(self.k_train + self.var_n * np.identity(self.n_samples))
            #cov = k_test - np.dot(self.k_train_test.T, np.dot(inv_temp, self.k_train_test))
            if eval_MSE:
                return pred, cov
            else:
                return pred
        
    def kernel(self, X1,X2):
        pairwise_dists = cdist(X1,X2,'sqeuclidean')
        self.kernel_train = self.var_f * np.exp(-self.theta0*pairwise_dists) # kernel matrix
        return self.kernel_train


