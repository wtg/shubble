# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 16:02:34 2025

@author: willi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def haversine_np(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two sets of GPS coordinates
    using the Haversine formula. Returns distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


df = pd.read_csv("100_pruned.csv", header=None)

df.columns = [
    "id", "vehicle_id", "route_id", "timestamp", 
    "latitude", "longitude", "heading", "speed", "status", 
    "address", "stop_id", "stop_name", "event_time"
]

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df["timestamp"] = pd.to_datetime(
    df["timestamp"], 
    format="%Y-%m-%d %H:%M:%S.%f", 
    errors="coerce"
)

df = df.dropna(subset=["timestamp", "latitude", "longitude"]).sort_values("timestamp").reset_index(drop=True)


