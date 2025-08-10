import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from datetime import timedelta
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings("ignore")

start = time.time()

# Load dan persiapan
df = pd.read_pickle('data/maritim_selat_sunda_500k.pkl')
df = df.dropna(subset=['mmsi', 'lat', 'lon', 'created_at', 'sog'])
df['utc'] = pd.to_datetime(df['created_at'])  # rename supaya konsisten
df = df.sort_values(by='utc')

# Parameter aturan
PROXIMITY_THRESHOLD_KM = 1.0
DURATION_THRESHOLD_MIN = 30
SOG_THRESHOLD = 0.5  # ≈ 0 knot

# Bikin list hasil anomali
anomalies = []

# Bikin interval waktu per 1 menit
df['utc_rounded'] = df['utc'].dt.floor('1min')
grouped = df.groupby('utc_rounded')

for time, group in grouped:
    if len(group) < 2:
        continue
    
    coords = np.radians(group[['lat', 'lon']])
    tree = BallTree(coords, metric='haversine')
    indices = tree.query_radius(coords, r=PROXIMITY_THRESHOLD_KM / 6371.0)

    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i >= j:
                continue

            row_i = group.iloc[i]
            row_j = group.iloc[j]

            # Cek kecepatan rendah
            if row_i['sog'] >= SOG_THRESHOLD or row_j['sog'] >= SOG_THRESHOLD:
                continue

            # Simpan hasil awal
            anomalies.append({
                'mmsi_1': row_i['mmsi'],
                'mmsi_2': row_j['mmsi'],
                'utc': time,
                'lat': (row_i['lat'] + row_j['lat']) / 2,
                'lon': (row_i['lon'] + row_j['lon']) / 2,
            })

# Masuk ke tahap agregasi durasi per pasangan
anom_df = pd.DataFrame(anomalies)
if anom_df.empty:
    print("Tidak ada interaksi mencurigakan.")
else:
    grouped_pairs = anom_df.groupby(['mmsi_1', 'mmsi_2'])
    final_anomalies = []

    for (m1, m2), group in grouped_pairs:
        time_diff = (group['utc'].max() - group['utc'].min()).total_seconds() / 60
        if time_diff >= DURATION_THRESHOLD_MIN:
            final_anomalies.append({
                'mmsi_1': m1,
                'mmsi_2': m2,
                'start_time': group['utc'].min(),
                'end_time': group['utc'].max(),
                'duration_min': round(time_diff, 2),
                'lat': group['lat'].mean(),
                'lon': group['lon'].mean(),
            })

    final_df = pd.DataFrame(final_anomalies)
    final_df.to_csv("output_tabel_anomali.csv", index=False)

    # Plot grafik
    plt.figure(figsize=(10, 6))
    plt.scatter(final_df['lon'], final_df['lat'], c='red', label='Anomali')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Lokasi Anomali Transhipment")
    plt.grid(True)
    plt.legend()
    plt.savefig("output_grafik_anomali.png", dpi=300)
    plt.show()

end = time.time()
print(f"Waktu eksekusi: {round((end - start)/60, 2)} menit")