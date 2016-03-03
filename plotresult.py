import numpy as np
import sys
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.legend as legend
import math
import cPickle
import os
import matplotlib.gridspec as gridspec
from rel_accuracy import rel_accuracy
from rel_dist import rel_dist
from spectral import spectral
from load_results import load_results
from multiprocessing import Pool
from Framework.framework import framework
from Algorithms.subsampling import subsampling
from sklearn.datasets import load_svmlight_file
from numpy.random import RandomState
from Experiments.load_files import load_files

def plotresult(file_name, folder):
    path_prefix = "/home/zjia/kernelsubsampling/Results/"
    # absolute path of the folder
    path = path_prefix + os.path.join(file_name, folder)
    print(path)
    for files in os.listdir(path):  
        if '1024' in files or '2024' in files or '3024' in files:
            f = file(os.path.join(path,files),'rb')
            [gamma, reg_const,sampling_methods,sampling_sizes,loss_time] = cPickle.load(f)
            f.close()
            if '_0_' in files:
                fold_range = range(1)
            else:
                fold_range = range(1,6)
    print([[i,int(sampling_sizes[i])] for i in range(len(sampling_sizes))])
    idx_sampling_sizes = int(raw_input('Input idx of sampling_sizes: '))
    k = int(raw_input('Input rank k for approximation. k = '))
    
    # k_train computed?
    if loss_time.k_train[0] != None:
        compute_k_train = 'True'
        n_train = len(loss_time.k_train[0])
    else:
        compute_k_train = 'False'

    #compute_k_train = 'False'
    # define figures
    loss_plot_linear = plt.figure()
    loss_plot_log = plt.figure()
    loss_hist_fig = plt.figure()
    two_loss_hist_fig = plt.figure()
    two_loss_hist_fig_avg = plt.figure()
    std_fig = plt.figure()
    #rel_acc_fig = plt.figure()
    #std_rel_acc_fig = plt.figure()
    #rel_dist_fig = plt.figure()
    #std_rel_dist_fig = plt.figure()
    
    #S_fig = plt.figure()
    #sampling_time_fig = plt.figure()
    #fitting_time_fig = plt.figure()
    total_time_fig = plt.figure()
    loss_vs_time_fig = plt.figure()
    
    # line styles
    line_style = ['-','--',':','-.','']
    marker_style = ['o','x','s','h']
    colors = ['blue','green','red','cyan']
    ecolors = ['magenta','green','black','blue']
    idx_method = 0 
    idx_line = 0
    idx_marker = 0
    idx_colors = 0
    idx_ecolors = 0
    # number of cores used to process data and plot 
    n_cores = int(raw_input('Number of cores: '))
    pool = Pool(processes=n_cores)

    # iterate over all sampling methods, load results and plot
    sampling_methods = ['uniform','D2_weighting','D1_weighting']
    #sampling_methods = ['uniform','D2_weighting','D1_weighting','uniform_D2_mixed_0.5','uniform_D1_mixed_0.5']
    #sampling_methods = ['uniform','D2_weighting']
    sampling_methods = ['uniform']
    #sampling_methods = ['D1_weighting'] 
    #sampling_methods = ['uniform','D1_weighting','uniform_D1_mixed_0.5']
    dist_loss_all = []
    for method in sampling_methods:
        print(sampling_methods)
        loss_train = []
        loss_test = []
        relative_accuracy = []
        relative_dist = []
        S_train = []
        spectral_approx = []
        sampling_time = []
        fitting_time = []
        total_time = []
        print("loading results")
       
        dist_loss = np.array([])
        avg_dist_loss = np.array([])
        std_dist_loss = np.array([])
        for fold in fold_range:
            parameters = [[path, files, method,k,idx_sampling_sizes,compute_k_train] for files in os.listdir(path) if method+'_'+str(gamma) in files and '_'+str(fold)+'_' in files]
            print(parameters)
            results = pool.map(load_results, parameters) 
            #results = [load_results(parameters[0])]
            # load results with the same fold 
            abs_loss_train = []
            abs_loss_test = []
            for result in results:
                loss_train.append(result[0])
                loss_test.append(result[1])
                relative_accuracy.append(result[2])
                relative_dist.append(result[3])
                S_train.append(result[4])
                spectral_approx.append(result[5])
                sampling_time.append(result[6])
                fitting_time.append(result[7])
                abs_loss_train.append(result[8])
                abs_loss_test.append(result[9])
#======================================================Plot Figures======================================================
            avg_dist_loss_single_fold = []
            for j in range(len(results)):
                abs_loss_file_j = abs_loss_test[j]
                single = abs_loss_file_j[idx_sampling_sizes]
                dist_loss = np.concatenate((dist_loss,single)) 
                avg_dist_loss_single_fold.append(single)
            std_dist_loss_single_fold = np.std(avg_dist_loss_single_fold,axis=0)
            avg_dist_loss_single_fold = np.mean(avg_dist_loss_single_fold,axis=0)
            avg_dist_loss = np.concatenate((avg_dist_loss,avg_dist_loss_single_fold))
            std_dist_loss = np.concatenate((std_dist_loss,std_dist_loss_single_fold))
        print(dist_loss) 
        # -------------histogram of loss-----------------
        #bins = np.arange(0,80,5) 
        bins = range(45) 
        loss_hist_plot = loss_hist_fig.add_subplot(111)
        loss_hist_plot.ticklabel_format(style='sci',scilimits=(0,0),fontsize=20)
        loss_hist_plot.set_xlabel('Absolute Value of Loss',fontsize=20)
        loss_hist_plot.set_ylabel('Frequency',fontsize=20)
        dist_loss_all.append(dist_loss)
        if method == sampling_methods[-1]:
            loss_hist_plot.hist(dist_loss_all,log=True,bins=bins,alpha=0.5,ec='none',label=sampling_methods,histtype='bar')
            #loss_hist_plot.set_title(file_name+'--'+': '+str(gamma)+' and '+'penalty: '+str(reg_const)+'--sampled columns: '+str(int(sampling_sizes[idx_sampling_sizes]))) 
            loss_hist_plot.legend(loc='best',fontsize='xx-large')
            loss_hist_plot.tick_params(labelsize=20)
        if method == 'uniform':
            avg_dist_loss_uniform= avg_dist_loss
            std_dist_loss_uniform = std_dist_loss 
            dist_loss_uniform = dist_loss
        
        two_hist_plot = two_loss_hist_fig.add_subplot(111,aspect='equal',xscale='linear',yscale='linear')
        two_hist_plot.ticklabel_format(style='sci',scilimits=(0,0))
        two_hist_plot.set_xlabel('Test Error: Uniform Sampling',fontsize=20)
        two_hist_plot.set_ylabel('Test Error: Other Sampling Methods',fontsize=20)
        two_hist_plot.axis('equal')
        two_hist_plot.tick_params(labelsize=20)
        two_hist_plot.scatter(dist_loss_uniform,dist_loss,label=method,edgecolors='black',alpha=0.75,color=colors[idx_colors])
        two_hist_plot.legend(loc='upper center',fontsize='xx-large')

        two_hist_plot_avg = two_loss_hist_fig_avg.add_subplot(111,aspect='equal',xscale='linear',yscale='linear') 
        two_hist_plot_avg.ticklabel_format(style='sci',scilimits=(0,0))
        two_hist_plot_avg.set_xlabel('Test Error: Uniform Sampling',fontsize=20)
        two_hist_plot_avg.set_ylabel('Test Error: Other Sampling Methods',fontsize=20)
        two_hist_plot_avg.axis('equal') 
        two_hist_plot_avg.tick_params(labelsize=20) 
        if method == 'uniform':
            two_hist_plot_avg.errorbar(avg_dist_loss_uniform,avg_dist_loss,fmt='o',alpha=0.75,label=method)
        else:
            #two_hist_plot_avg.errorbar(avg_dist_loss_uniform,avg_dist_loss,xerr=std_dist_loss_uniform,yerr=std_dist_loss,fmt='*',alpha=0.75,label=method)
            two_hist_plot_avg.errorbar(avg_dist_loss_uniform,avg_dist_loss,fmt='*',markersize=8,label=method)
        two_hist_plot_avg.legend(loc='best',fontsize='xx-large')

        #----------prediction error/inaccuracy----------
        avg_loss_train = np.mean(loss_train,0)
        avg_loss_test = np.mean(loss_test,0)
        std_loss_train = np.std(loss_train,0)
        std_loss_test = np.std(loss_test,0)
        original_plot = loss_plot_linear.add_subplot(111)
        #original_plot.set_title(file_name+'--'+': '+str(gamma)+' and '+'penalty: '+str(reg_const))
        original_plot.ticklabel_format(style='sci',scilimits=(0,0))
        original_plot.set_xlabel('Number of Columns Sampled',fontsize=20)
        original_plot.set_ylabel('Prediction Error',fontsize=20)
        original_plot.plot(sampling_sizes,avg_loss_train,color='blue',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label='loss_train_'+method,linewidth=2)
        original_plot.plot(sampling_sizes,avg_loss_test,color='red',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label='loss_test_'+method,linewidth=2)
        original_plot.legend(loc='best',fontsize='x-large')
        original_plot.tick_params(labelsize=20) 
        log_plot = loss_plot_log.add_subplot(111,xscale='log',yscale='linear')
        log_plot.set_xlabel('Number of Columns Sampled',fontsize=20)
        log_plot.set_ylabel('Prediction Error',fontsize=20)
        log_plot.errorbar(sampling_sizes,avg_loss_train,yerr=std_loss_train,ecolor=ecolors[idx_ecolors],elinewidth=2,capsize=5,capthick=2,color='blue',
                                                        linestyle=line_style[idx_line],marker=marker_style[idx_marker],label='loss_train_'+method,linewidth=2)
        log_plot.errorbar(sampling_sizes,avg_loss_test,yerr=std_loss_test,ecolor=ecolors[idx_ecolors],elinewidth=2,capsize=5,capthick=2,color='red',
                                                        linestyle=line_style[idx_line],marker=marker_style[idx_marker], label='loss_test_'+method,linewidth=2)
        log_plot.autoscale_view(tight=False,scalex=True,scaley=False)
        log_plot.legend(loc='best',fontsize='large')
        log_plot.tick_params(labelsize=20) 
        # ---------standard deviation plot 
        std_plot = std_fig.add_subplot(111)
        std_plot.set_xlabel('Number of Columns Sampled',fontsize=20)
        std_plot.set_ylabel('Deviation of Prediction Error',fontsize=20)
        std_plot.plot(sampling_sizes,std_loss_train,color='blue',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label='std_train_'+method)
        std_plot.plot(sampling_sizes,std_loss_test,color='red',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label='std_test_'+method)
        std_plot.legend(loc='best',fontsize='xx-large')
        std_plot.tick_params(labelsize=20) 
        #---------kernel approximation ---------
        # optional: find k such that it has 0.9 of spectral sum of k_train
        ratio = 0.9
        if compute_k_train == 'True':
            # rank-k approximation relative accuracy
            print("computing rel acc")
            std_rel_acc = np.std(relative_accuracy,0)
            print("std_rel_acc ")
            print(std_rel_acc)
            relative_accuracy = np.mean(relative_accuracy,0)
            rel_acc_plot = rel_acc_fig.add_subplot(111,xscale='linear',yscale='linear',xlabel='Number of Columns Sampled (l)', ylabel='Relative Accuracy')
            #rel_acc_plot.set_title(file_name+':'+' Besk Rank-'+str(k) +' Approximation'+ '--gamma: '+str(gamma))
            rel_acc_plot.errorbar(sampling_sizes,relative_accuracy,yerr=std_rel_acc,ecolor=ecolors[idx_ecolors],elinewidth=2,capsize=5,capthick=2,color='blue',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label=method)
            rel_acc_plot.legend(loc='best',fontsize='medium')
            
            # relative approximation distance
            std_rel_dist = np.std(relative_dist,0)
            relative_dist = np.mean(relative_dist,0)
            rel_dist_plot = rel_dist_fig.add_subplot(111,xscale='linear',yscale='log',xlabel='Number of Columns Sampled (l)',ylabel='Relative Distance') 
            #rel_dist_plot.set_title(file_name+':'+'Relative Distance'+ '--gamma: '+str(gamma))
            rel_dist_plot.errorbar(sampling_sizes,relative_dist,yerr=std_rel_dist,ecolor=ecolors[idx_ecolors],elinewidth=2,capsize=5,capthick=2,color='blue',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label=method)
            rel_dist_plot.legend(loc='best',fontsize='medium')
            
            # std plot
            std_rel_acc_plot = std_rel_acc_fig.add_subplot(111,xlabel='Number of Columns Sampled (l)',ylabel='Deviation of Approximation Accuracy')
            #std_rel_acc_plot.set_title(file_name+'--'+': '+str(gamma)+' and '+'penalty: '+str(reg_const))
            std_rel_acc_plot.plot(sampling_sizes,std_rel_acc,color='blue',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label=method)
            std_rel_acc_plot.legend(loc='best',fontsize='medium') 
            
            std_rel_dist_plot = std_rel_dist_fig.add_subplot(111,xlabel='Number of Columns Sampled (l)',ylabel='Deviation of Approximation Error')
            #std_rel_dist_plot.set_title(file_name+'--'+': '+str(gamma)+' and '+'penalty: '+str(reg_const))
            std_rel_dist_plot.plot(sampling_sizes,std_rel_dist,color='blue',linestyle=line_style[idx_line],marker=marker_style[idx_marker],label=method)
            std_rel_acc_plot.legend(loc='best',fontsize='medium')
            # spectral components
            """
            avg_S_train = np.mean(S_train,0)
            ratio_S_train = np.cumsum(avg_S_train/np.sum(avg_S_train))
            k_top = np.searchsorted(ratio_S_train, ratio)
            S_plot = S_fig.add_subplot(111,xscale='log',yscale='log',xlabel='rank',ylabel='spectral')
            S_plot.set_title(file_name+':'+'Top '+str(k)+ ' spectrals of k_approx approximated by ' + str(int(sampling_sizes[idx_sampling_sizes])) +' sampled points' +'-gamma: '+str(gamma))
            
            # Plot spectral components of matrices. 
            # During 5 fold cross validation, sometimes training data sizes are different in each fold. Enforce same sizes of training points cross each fold 
            shortest = min([len(S_train_single) for S_train_single in S_train])
            S_train = [S_train_single[:shortest] for S_train_single in S_train]
            S_plot.plot(np.arange(1,1+shortest)[:k],np.mean(S_train,0)[:k],label='k_train',c='red') # top k spectral of k_train
            S_plot.axvline(x=k_top,color='red')
            S_plot.plot(np.arange(1,1+len(spectral_approx[0])),np.mean(spectral_approx,0),c=colors[idx_colors],linestyle=line_style[idx_line],marker=marker_style[idx_marker],label='S_SR_'+method)
            S_plot.legend(loc='best',fontsize='small')
            """
        avg_sampling_time = np.mean(sampling_time,0)
        avg_fitting_time = np.mean(fitting_time,0)
        avg_total_time = avg_sampling_time + avg_fitting_time
        """
        # sampling time
        sampling_time_plot = sampling_time_fig.add_subplot(111, xlabel='Sampling sizes', ylabel='Sampling time in seconds', title='Sampling time')  
        sampling_time_plot.plot(sampling_sizes, np.mean(sampling_time,0),linestyle=line_style[idx_line], marker=marker_style[idx_marker],label=method)
        # fitting time
        fitting_time_plot = fitting_time_fig.add_subplot(111, xlabel='Sampling sizes', ylabel='Fitting time', title='Fitting time')
        fitting_time_plot.plot(sampling_sizes, np.mean(fitting_time,0), linestyle=line_style[idx_line], marker=marker_style[idx_marker],label=method)
        """
        
        # Total time
        total_time_plot = total_time_fig.add_subplot(111)
        total_time_plot.plot(sampling_sizes, np.mean(sampling_time,0)+np.mean(fitting_time,0), linestyle=line_style[idx_line], marker=marker_style[idx_marker],label=method)
        total_time_plot.legend(loc='upper left',fontsize='xx-large')
        total_time_plot.set_xlabel('Sampling_sizes',fontsize='xx-large')
        total_time_plot.set_ylabel('Sampling + fitting time',fontsize='xx-large')
        total_time_plot.tick_params(labelsize=20)
        error_time_plot = loss_vs_time_fig.add_subplot(111,xscale='log')
        error_time_plot.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        #error_time_plot.set_title(file_name+'--'+'gamma: '+str(gamma)+' and '+'penalty: '+str(reg_const))
        error_time_plot.errorbar(avg_total_time, avg_loss_train, yerr=std_loss_train, ecolor=ecolors[idx_ecolors],elinewidth=2,capsize=5,capthick=2, color='blue',linestyle=line_style[idx_line], marker=marker_style[idx_marker],label='loss_train ' + method)
        error_time_plot.errorbar(avg_total_time, avg_loss_test,yerr=std_loss_test,ecolor=ecolors[idx_ecolors],elinewidth=2,capsize=5,capthick=2, color='red',linestyle=line_style[idx_line], marker=marker_style[idx_marker],label='loss_test ' + method)
        error_time_plot.legend(loc='upper right',fontsize='large')
        error_time_plot.set_xlabel('time',fontsize=20)
        error_time_plot.set_ylabel('Prediction Error',fontsize=20)
        error_time_plot.tick_params(labelsize=20)
        idx_method += 1
        idx_marker += 1
        idx_line += 1
        idx_colors += 1
        idx_ecolors += 1
    
    plt.show() 
if __name__ == '__main__':
    plotresult(sys.argv[1],sys.argv[2])
