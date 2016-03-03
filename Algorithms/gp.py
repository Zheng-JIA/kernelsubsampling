# Gaussian Process 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import mean_squared_error
from time import time
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
class GaussianProcess():

    def __init__(self, theta0, var_noise):
        """
        Initialize hyperparameters of kernels and noise variance
        k(x1,x2) = exp(-theta0*|x1 - x2|^2) + var_n*delta(x1,x2)
        Parameters
        ----------
        theta0: 1/var
        var_n: noise variance
        """
        
        self.theta0 = theta0
        self.var_n = var_noise
  
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
        start = time()
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
        self.k_SR = k_SR
        self.k_SR_train = k_SR_train
        self.k_train_SR = k_train_SR
        
        # Use subset of data points: 
        start = time()
        #self.alpha_SR = np.linalg.solve(np.dot(k_SR_train,k_train_SR) + self.var_n*k_SR + np.diag([10*MACHINE_EPSILON]*X_SR.shape[0]), np.dot(k_SR_train,y))
        try: 
            self.L = cho_factor( np.dot(self.k_SR_train,self.k_train_SR) + self.var_n*(self.k_SR+np.diag([10**-7]*self.X_SR.shape[0])))
        except:
            self.L = cho_factor( np.dot(self.k_SR_train,self.k_train_SR) + self.var_n*(self.k_SR+np.diag([10**-5]*self.X_SR.shape[0])))
        self.alpha_SR = cho_solve(self.L,np.dot(self.k_SR_train,y))

    def predict(self, X_test, eval_MSE=False):
        # Normalize input
        X_test = (X_test - self.X_mean) / self.X_std
        # kernel matrix K(X_test,X_test)
        k_test = self.kernel(X_test,X_test)
        
        # Compute and denormalize Predictions and 95% Confidence Interval       
        k_SR_test = self.kernel(self.X_SR, X_test)
        pred_SR = np.dot(k_SR_test.T,self.alpha_SR)
        pred_SR = (pred_SR * self.y_std) + self.y_mean
        return pred_SR
        
    def loss(self, y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    def kernel(self, X1,X2):
        kernel = rbf_kernel(X1,X2,gamma=self.theta0)
        return kernel


