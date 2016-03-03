# Load data and split data into training and testing sets
from sklearn.datasets import load_svmlight_file
#from sklearn.cross_validation import train_test_split
import random
from sklearn.preprocessing import scale
def load_files(file_name, train_size, path_prefix, prng):
    # Set up path to files
    path = path_prefix + file_name + "_scale.txt"
    print("Loading file %s_scale.txt" % file_name)
    # Load files
    data_X, data_y = load_svmlight_file(path)
    # Convert from sparse to dense format
    X = data_X.toarray()
    y = data_y    
    if file_name == 'cadata':
        X = scale(X)
        y = scale(y)
    #print("data is scaled")
    # Split data into training, validation and test sets
    if file_name == 'covtype':
        X_train, X_test, y_train, y_test = split_data_for_covtype(X,y,train_size,prng)
    else:
        X_train, X_test, y_train, y_test = split_data(X,y,train_size,prng)
    print(len(y_test))
    return X_train, y_train, X_test, y_test

def split_data(X,y,train_size,prng):
    n=len(X)
    idx=prng.permutation(n)
    idx_train = idx[:int(n*train_size)]
    idx_test = idx[int(n*train_size):]
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    
    return X_train, X_test, y_train, y_test

def split_data_for_covtype(X,y,train_size,prng):
    n=len(X)
    idx=prng.permutation(n)[:int(n*0.01)]
    idx_train = idx[:int(len(idx)*train_size)]
    idx_test = idx[int(len(idx)*train_size):]
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
  
    return X_train, X_test, y_train, y_test
