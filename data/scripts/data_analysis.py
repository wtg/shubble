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


lat1 = df["latitude"].values[:-1]
lon1 = df["longitude"].values[:-1]
lat2 = df["latitude"].values[1:]
lon2 = df["longitude"].values[1:]


distances = haversine_np(lat1, lon1, lat2, lon2)


df_dist = pd.DataFrame({
    "timestamp": df["timestamp"].iloc[1:].values,
    "distance_km": distances
})

#sort
df_dist = df_dist.sort_values("timestamp").reset_index(drop=True)


x = np.arange(len(df_dist))  # 0, 1, 2, ..., 99
labels = df_dist["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")  # full timestamp strings

plt.figure(figsize=(18,6))
plt.plot(x, df_dist["distance_km"], marker="o", linestyle="-")

plt.xticks(x, labels, rotation=90, fontsize=7)

plt.xlabel("Timestamp (all points)")
plt.ylabel("Distance between points (km)")
plt.title("Distance differences between consecutive GPS points (All 100 Timestamps)")
plt.tight_layout()
plt.show()

df_sorted = df.sort_values("timestamp").reset_index(drop=True)

time_diffs = df_sorted["timestamp"].diff().dt.total_seconds().dropna()

import numpy as np
import matplotlib.pyplot as plt

df_sorted = df.sort_values("timestamp").reset_index(drop=True)
time_diffs = df_sorted["timestamp"].diff().dt.total_seconds().dropna().astype(int)

max_bin = 60
time_diffs_capped = time_diffs.clip(upper=max_bin)

bins = np.arange(1, max_bin+2)
labels = [str(i) for i in range(1, max_bin)] + [f">{max_bin}"]

plt.figure(figsize=(14,6))
plt.hist(time_diffs_capped, bins=bins, edgecolor='black', align='left')

plt.xticks(np.arange(1, max_bin+1), labels, rotation=90)
plt.xlabel("Gap between consecutive timestamps (seconds)")
plt.ylabel("Count")
plt.title("Histogram of GPS Time Gaps (1s resolution, >60s grouped)")
plt.tight_layout()
plt.show()

print("Gap summary (seconds):")
print(time_diffs.describe())
print(f"Gaps > {max_bin}s: {(time_diffs > max_bin).sum()}")



