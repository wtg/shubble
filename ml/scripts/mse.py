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

def plot_residuals_vs_pred(preds, errs, label):
    y_pred = np.asarray(preds, dtype=float)
    e = np.asarray(errs, dtype=float)

    plt.figure()
    plt.scatter(y_pred, e)
    plt.axhline(0.0)
    plt.xlabel("Predicted Distance4")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residuals vs Predicted — {label}")
    plt.tight_layout()
    plt.show()

def plot_residual_hist(errs, label):
    e = np.asarray(errs, dtype=float)

    plt.figure()
    plt.hist(e, bins=20)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution — {label}")
    plt.tight_layout()
    plt.show()
    
    
def summarize_models(df, configs, scale_time=True):

    rows, details = [], {}
    for cfg in configs:
        mse, preds, errs = mse_across_data(
            df,
            degree=cfg["degree"],
            ridge=cfg.get("ridge", None),
            scale_time=scale_time,
            return_details=True
        )
        rmse = float(np.sqrt(mse))
        lab = f"deg={cfg['degree']}" + ("" if cfg.get("ridge") is None else f" λ={cfg['ridge']}")
        rows.append({"label": lab, "degree": cfg["degree"], "ridge": cfg.get("ridge", None),
                     "MSE": mse, "RMSE": rmse})
        details[lab] = {"preds": preds, "errs": errs}
    summary = pd.DataFrame(rows).sort_values(["degree", "ridge"], ascending=[True, True])
    return summary, details

def plot_rmse_bars(summary, title="RMSE by degree / ridge"):
    labels = summary["label"].tolist()
    rmse = summary["RMSE"].to_numpy()

    plt.figure()
    x = np.arange(len(labels))
    bars = plt.bar(x, rmse)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("RMSE")
    plt.title(title)
    for xi, bi, val in zip(x, bars, rmse):
        plt.text(xi, bi.get_height(), f"{val:.4g}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

def plot_window(df, row_index, degree, ridge=None, scale_time=True, show_curve=True, show=True):
    """
    Plot one 4-point window and return the matplotlib Figure.
    """
    r = df.iloc[row_index]

    # Extract local (already relative) times/distances from the row
    t = np.array([r[f"Timedelta{i}"] for i in range(1, 5)], dtype=float)
    y = np.array([r[f"Distance{i}"] for i in range(1, 5)], dtype=float)

    # Get prediction + error using your pipeline (this respects scaling)
    y_hat, err = evaluate_window(
        t1=t[0], y1=y[0],
        t2=t[1], y2=y[1],
        t3=t[2], y3=y[2],
        t4=t[3], y4=y[3],
        degree=degree, ridge=ridge, scale_time=scale_time
    )

    # Reconstruct the scaled time and beta used inside evaluate_window (for optional curve)
    t_local = t - t[0]
    y_local = y - y[0]
    if scale_time:
        denom = t_local[2] if t_local[2] != 0 else (t_local[1] if t_local[1] != 0 else 1.0)
    else:
        denom = 1.0
    t2_s, t3_s = t_local[1]/denom, t_local[2]/denom

    beta = fit_polynomial([t2_s, t3_s], [y_local[1], y_local[2]],
                          degree=degree, ridge=ridge)

    # --- Plot on axes ---
    fig, ax = plt.subplots()

    # Training points (1..3)
    ax.plot(t[:3], y[:3], marker="o", label="Train pts 1–3")

    # Actual 4th
    ax.scatter([t[3]], [y[3]], marker="x", s=80, label="Actual 4th")

    # Predicted 4th
    ax.scatter([t[3]], [y_hat], marker="^", s=80, label="Predicted 4th")

    # Optional fitted curve from 0..t4 using the same scaling
    if show_curve:
        tt = np.linspace(0.0, t[3]-t[0], 200)   # local time
        tt_s = tt / denom
        yy_local = np.array([prediction(beta, u) for u in tt_s])
        yy = yy_local + y[0]                    # back to global coords
        ax.plot(tt + t[0], yy, linestyle="--", label="Fitted curve")

    title = f"Window {row_index} — deg={degree}" + ("" if ridge is None else f", λ={ridge}")
    ax.set_title(title)
    ax.set_xlabel("Time (s) (relative to point 1)")
    ax.set_ylabel("Distance from start (km)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()

    return fig

from matplotlib.backends.backend_pdf import PdfPages
import os

def plot_all_windows(df, degree, ridge=None, scale_time=True, show_curve=True,
                     limit=None, show=False, save_pdf=None, save_dir=None, dpi=120):
    
    n = len(df) if limit is None else min(limit, len(df))

    pdf = PdfPages(save_pdf) if save_pdf else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(n):
        fig = plot_window(df, i, degree=degree, ridge=ridge,
                          scale_time=scale_time, show_curve=show_curve, show=show)

        if pdf:
            pdf.savefig(fig)
        if save_dir:
            fname = f"window_{i:04d}_deg{degree}" + ("" if ridge is None else f"_lam{ridge}") + ".png"
            fig.savefig(os.path.join(save_dir, fname), dpi=dpi)

        plt.close(fig)

    if pdf:
        pdf.close()



df = pd.read_csv("sequences_100.csv")


mse1, _, _ = mse_across_data(df, degree=1, ridge=None)
print(f"Linear (deg=1)     MSE={mse1:.6g}  RMSE={np.sqrt(mse1):.6g}")

mse2, pred2, err2 = mse_across_data(df, 2)
print(f"Quadratic (deg=2)  MSE={mse2:.6g}  RMSE={np.sqrt(mse2):.6g}")

for lam in (1e-4, 1e-3, 1e-2):
    mse3, _, _ = mse_across_data(df, degree=3, ridge=lam)
    print(f"Cubic ridge (deg=3) λ={lam:g}  MSE={mse3:.6g}  RMSE={np.sqrt(mse3):.6g}")
    

    
plot_all_windows(df, degree=2, ridge=None, save_pdf="all_windows_deg2.pdf")


    
    