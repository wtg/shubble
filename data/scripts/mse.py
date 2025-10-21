import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#the starting point is (0,0) - > we drop the bias term

def create_vandermonde(training, degree):
    training = np.array(training, dtype = "float")
    cols = []
    for i in range(1, degree+1):
        col = []
        for j in range(len(training)):
            col.append(training[j] ** i)
        cols.append(col)
    x = np.column_stack(cols) 
    
    return x
        
        
def create_vandermonde_fast(training, degree):
    t = np.asarray(training, dtype=float).ravel()
    X_full = np.vander(t, N=degree+1, increasing=True) 
    X = X_full[:, 1:]                                   
    return X

def fit_polynomial(time_tr, distance_tr, degree, ridge = None):
    t = np.asarray(time_tr, dtype="float").ravel()
    y = np.asarray(distance_tr, dtype = "float").ravel()
    
    X = create_vandermonde(t, degree)
    
    #beta = SVD[(X.T@X)^-1]X.T@y 
    if ridge is None: 
        beta, *_ =  np.linalg.lstsq(X, y, rcond=None)
    
    else:
        XX = X.T @ X
        Xy = X.T @ y
        A = XX + ridge * np.eye(X.shape[1])
        beta = np.linalg.solve(A, Xy)
        
    return beta

def prediction(beta, time):
    time = float(time)
    b = np.asarray(beta, dtype=float)  
    powers = []
    
    for i in range(1, len(b)+1):
        powers.append(time**i)
    powers = np.array(powers, dtype="float")
    
    return(powers @ b)

def prediction_horners(beta, time):
    t = float(time)
    acc = 0.0
    for b in reversed(beta): 
        acc = acc * t + b
    return acc * t    




    
    