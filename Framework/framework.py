from Algorithms.gp import f_GP
from sklearn.metrics import mean_squared_error
from time import time
import matplotlib.pyplot as plt
import numpy as np
# Tune and pass hyperparameters to GP model
# f_GP(theta0, var_n, var_f)
# k(x1,x2) = var_f*exp(-theta0*|x1 - x2|^2) + var_n*delta(x1,x2)

def framework(X, y_noisy, idx_SR, x, y_real_noisy): 
    gp_time = []
    gp_loss = []
    start = time()
    var_n = 0.1
    var_f = 0.1
    theta0 = 0.1
    #================================================================================
    # Stupid way for selecting hyperparameters
    loss = []
    myrange = np.arange(0.1,10,0.1)
    for theta0 in myrange:
        # Pass hyper-parameters to models	
        gp = f_GP(theta0, var_n, var_f) # choosing signal variance is important. 
        # Fit GP model using selected points
        gp.fit(X, y_noisy,idx_SR) # default SR=True: use subset of data to train model       
        # Make predictions
        gp_predictions = gp.predict(x)
        # Compute prediction error
        gp_loss = mean_squared_error(y_real_noisy, gp_predictions)
        loss.append(gp_loss)
    smallestlossidx = loss.index(min(loss))
    theta0 = 0.1 + smallestlossidx*0.1

    loss = []
    for var_f in myrange:
        # Pass hyper-parameters to models	
        gp = f_GP(theta0, var_n, var_f) # choosing signal variance is important. 
        # Fit GP model using selected points
        gp.fit(X, y_noisy,idx_SR) # default SR=True: use subset of data to train model
        # Make predictions
        gp_predictions = gp.predict(x)
        # Compute prediction error
        gp_loss = mean_squared_error(y_real_noisy, gp_predictions)
        loss.append(gp_loss)
    smallestlossidx = loss.index(min(loss))
    var_f = 0.1 + smallestlossidx*0.1
    
    loss = []
    for var_n in myrange:
        # Pass hyper-parameters to models	
        gp = f_GP(theta0, var_n, var_f) # choosing signal variance is important. 
        # Fit GP model using selected points
        gp.fit(X, y_noisy,idx_SR) # default SR=True: use subset of data to train model
        # Make predictions
        gp_predictions = gp.predict(x)
        # Compute prediction error
        gp_loss = mean_squared_error(y_real_noisy, gp_predictions)
        loss.append(gp_loss)
    smallestlossidx = loss.index(min(loss))
    var_n = 0.1 + smallestlossidx*0.1     
    
    #================================================================================
    # Pass Best hyper-parameters to models	
    gp = f_GP(theta0, var_n, var_f) # choosing signal variance is important. 
    # Fit GP model using selected points
    gp.fit(X, y_noisy,idx_SR) # default SR=True: use subset of data to train model
    gp_time = time() - start
    # Make predictions
    gp_predictions = gp.predict(x)
    # Compute prediction error
    gp_loss = mean_squared_error(y_real_noisy, gp_predictions)
    
    return gp_time, gp_loss, gp_predictions