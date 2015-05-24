import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.legend as legend
import math
from Framework.framework import framework
from Algorithms.subsampling import uni_sampling

#=============Parameters=============
var_n = 1
sample_sizes = [1,3,5,7,10,20,30,49] # size of subset of training points  
n_subsets = len(sample_sizes) # number of subsets to evaluate

#=============Generate training data from function y=x*sin(x)=============
def f(x):
     return x * np.sin(x)
X = np.atleast_2d(np.linspace(1,10,num=50)).T # ten training points
y = f(X).ravel()
n_samples,_ = X.shape

# Add noise to y, zero mean and variance = var_n
dy = np.sqrt(var_n) * (np.random.random(n_samples)-0.5)
y_noisy = y + dy
# Test input
x = np.atleast_2d(np.linspace(0, 10, 1000)).T # 1000 testing points 
n_test,_ = x.shape
# ground_truth output
y_real = f(x).ravel() 
# Add noise to the output, zero mean and variance = var_n
y_real_noisy = y_real + np.sqrt(var_n) * (np.random.random(n_test)-0.5)

#=============Run Experiments=============
gp_time = []
gp_loss = []
gp_predictions = []
idx_SR_all = []
for n_SR in sample_sizes:
    print("Number of regressors: %d" % n_SR)
    # Uniformly sampling n_SR points from training sets
    idx_SR = uni_sampling(X, n_SR)
    # Test on framework, return fitting time, loss, and predictions
    single_time, single_loss, single_predictions = framework(X, y_noisy, idx_SR, x, y_real_noisy) 
    # Store time, loss, predictions and selected training points' indices for later plotting
    gp_time.append(single_time)
    gp_loss.append(single_loss)
    gp_predictions.append(single_predictions)
    idx_SR_all.append(idx_SR)

#====================Plot Results===================================================
# (a)========predictions and ground truth==========
# pred_real_fig: a figure plotting predictions and ground truth (real data)
pred_real_fig = plt.figure()
for n in range(0, n_subsets):
    # Retrieve nth selected training points
    idx_SR_nth = idx_SR_all[n]
    X_nth = X[idx_SR_nth]
    y_noisy_nth = y_noisy[idx_SR_nth]
    # Retrieve nth predictions
    gp_predictions_nth = gp_predictions[n]
    
    # 2 columns Subplots
    pred_real_fig.add_subplot(math.ceil(n_subsets/2.0),2, n+1)
    # plot noisy training data
    plt.scatter(X_nth, y_noisy_nth) 
    # plot noisy test data: red
    plt.plot(x,y_real_noisy,c=u'r', label="Ground_Truth") 
    # plot my predictions: blue curve
    plt.plot(x,gp_predictions_nth,'b',linewidth=2, label="Predictions") 
    # indicate selected points: blue crosses
    ymin, ymax = plt.gca().get_ylim()   
    plt.scatter(X[idx_SR_nth],[ymin]*len(idx_SR_nth), label="Chosen Data Loc", marker=u'x', s=30, linewidths=2)
    plt.legend(loc='upper left', fontsize=6, markerscale=0.5) 
    plt.title("Number of Regressors: %d"%len(idx_SR_nth))      
plt.tight_layout()

# (b)=======loss, time and # of regressors===========
gp_loss_full = [gp_loss[-1]]*n_subsets # prediction loss using all training data
gp_time_full = [gp_time[-1]]*n_subsets # training time using all training data

# plot loss vs # regressors
plt.figure()
loss = plt.subplot(2,1,1)
loss.set_title("Prediction Loss")
loss.plot(sample_sizes, gp_loss, label='Uniform', linewidth=2)
loss.plot(sample_sizes, gp_loss_full, label='Full', linewidth=2) 
loss.set_xlabel("Number of Regressors")
loss.set_ylabel("Prediction Loss")
#loss.set_xticks(())
plt.legend(loc='best')

# plot time vs # regressors
time = plt.subplot(2,1,2)
time.set_title("Training Time")
time.plot(sample_sizes, gp_time, label='Uniform', linewidth=2)
time.plot(sample_sizes, gp_time_full, label='Full', linewidth=2)
time.set_ylabel("Training Time")
time.set_xlabel("Number of Regressors")
plt.legend(loc='best')
plt.tight_layout()

# plot loss, time vs # regressors
plt.figure()
loss_time = plt.subplot(111)
loss_time.plot(sample_sizes, gp_loss, linewidth=2, label='Prediction Loss')
loss_time.set_xlabel("Number of Regressors")
loss_time.set_ylabel("Prediction Loss")
loss_time.legend(loc='upper left')
loss_time.twinx().plot(sample_sizes, gp_time, 'g', linewidth=2, label='Training Time')
plt.ylabel("Training Time")
plt.legend(loc='upper right')
plt.show()

