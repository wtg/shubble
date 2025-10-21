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

def evaluate_window(t1, y1, t2, y2, t3, y3, t4, y4, degree, ridge, scale_time = True):
    t = np.array([t1,t2,t3,t4], dtype="float")
    y = np.array([y1,y2,y3,y4], dtype="float")
    
    t_local = t - t[0]
    y_local = y - y[0]
    
    if degree < 1:
        raise ValueError("degree must be >= 1 for through-origin model")
    if t_local[1] == 0 or t_local[2] == 0:
        raise ValueError("Need two distinct nonzero training times after re-anchoring.")
    
    if scale_time:
        denom = t_local[2] if t_local[2] != 0 else (t_local[1] if t_local[1] != 0 else 1.0)
    else: 
        denom = 1.0
        
    t2_s, t3_s, t4_s = t_local[1]/denom, t_local[2]/denom, t_local[3]/denom
    
    lam = None if ridge is None else float(ridge)
    
    beta = fit_polynomial([t2_s, t3_s], [y_local[1], y_local[2]], degree, ridge = lam)
    
    y_prediction_local = prediction(beta, t4_s)
    y_hat = y_prediction_local +  y[0]

    
    error = y[3] - y_hat
    return y_hat, error

def mse_across_data(df, degree, ridge = None, scale_time = True, return_details = True):
    predictions = []
    errors = []
    
    for _, r in df.iterrows():
        y_hat, error = evaluate_window(r["Timedelta1"], r["Distance1"], r["Timedelta2"], 
                                       r["Distance2"], r["Timedelta3"], r["Distance3"], 
                                       r["Timedelta4"], r["Distance4"], degree, ridge, scale_time)
        
        predictions.append(y_hat)
        errors.append(error)
        
    predictions = np.array(predictions, dtype="float")
    errors = np.array(errors, dtype="float")
    mse = float(np.mean(errors**2))
    
    if return_details:  
        return mse, predictions, errors
    
    return mse

def plot_actual_vs_pred(df, preds, label):
    y_true = df["Distance4"].to_numpy()
    y_pred = np.asarray(preds, dtype=float)

    plt.figure()
    plt.scatter(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual Distance4")
    plt.ylabel("Predicted Distance4")
    plt.title(f"Actual vs Predicted — {label}")
    plt.tight_layout()
    plt.show()




    
    