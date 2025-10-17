# %% script


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- params -----------------------------------
INPUT_PATH  = "100.csv"                 
OUTPUT_PATH = "sequences_100.csv"     
L           = 4                         
T           = 15.0                       
PREVIEW_ROWS = 10                       

TIME_COL    = "timestamp"   
SECONDS_COL = None   #
DIST_COL    = None   
LAT_COL     = "latitude" 
LON_COL     = "longitude"   
VEH_COL     = "vehical_id"   

SHOW_PLOTS  = True   
# ----------------------------------------------

def _norm_key(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")

def _find_col(candidates, columns):
    cols_norm = {_norm_key(c): c for c in columns}
    for cand in candidates:
        key = _norm_key(cand)
        if key in cols_norm:
            return cols_norm[key]
    return None

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

