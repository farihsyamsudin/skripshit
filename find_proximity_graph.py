import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from datetime import timedelta
from haversine import haversine
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

start = time.time()

# Load data awal
df = pd.read_pickle('data/maritim_selat_sunda.pkl')

# Filter area Selat Sunda dan data penting
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)]
df = df.dropna(subset=['mmsi', 'lat', 'lon', 'created_at', 'sog'])
df['utc'] = pd.to_datetime(df['created_at'])
df = df.sort_values(by='utc')

# Parameter
PROXIMITY_THRESHOLD_KM = 1.0
DURATION_THRESHOLD_MIN = 30
SOG_THRESHOLD = 0.5

# Buat interval waktu per 1 menit
df['utc_rounded'] = df['utc'].dt.floor('1min')
grouped = df.groupby('utc_rounded')

# Cari anomali proximity
anomalies = []

for time_key, group in grouped:
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

            if row_i['sog'] >= SOG_THRESHOLD or row_j['sog'] >= SOG_THRESHOLD:
                continue

            anomalies.append({
                'mmsi_1': min(row_i['mmsi'], row_j['mmsi']),
                'mmsi_2': max(row_i['mmsi'], row_j['mmsi']),
                'utc': time_key,
                'lat': (row_i['lat'] + row_j['lat']) / 2,
                'lon': (row_i['lon'] + row_j['lon']) / 2,
            })

# Buat dataframe hasil proximity
anom_df = pd.DataFrame(anomalies)

# Proses jika ada data mencurigakan
if not anom_df.empty:
    final_anomalies = []
    grouped_pairs = anom_df.groupby(['mmsi_1', 'mmsi_2'])

    for (m1, m2), group in grouped_pairs:
        unique_minutes = group['utc'].nunique()
        if unique_minutes >= DURATION_THRESHOLD_MIN:
            coords = list(zip(group['lat'], group['lon']))
            if len(coords) >= 2:
                distances = [
                    haversine(coords[i], coords[i + 1])
                    for i in range(len(coords) - 1)
                ]
                avg_distance = np.mean(distances)
            else:
                avg_distance = 0

            final_anomalies.append({
                'mmsi_1': m1,
                'mmsi_2': m2,
                'start_time': group['utc'].min(),
                'end_time': group['utc'].max(),
                'duration_min': unique_minutes,
                'lat': group['lat'].mean(),
                'lon': group['lon'].mean(),
                'proximity_km': avg_distance
            })

    final_df = pd.DataFrame(final_anomalies)
    final_df.to_pickle("output_anomali_proximity.pkl")
    print("Hasil proximity disimpan di output_anomali_proximity.pkl")

    # Buat grafik hubungan proximity dan durasi
    plt.figure(figsize=(10, 6))
    plt.scatter(final_df['proximity_km'], final_df['duration_min'],
                alpha=0.7, c='blue', edgecolors='k')
    plt.title('Hubungan Proximity dan Durasi Interaksi Kapal')
    plt.xlabel('Proximity (km)')
    plt.ylabel('Durasi Interaksi (menit)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafik_proximity_durasi.png", dpi=300)
    plt.show()
    print("Grafik disimpan di grafik_proximity_durasi.png")

else:
    print("Tidak ditemukan interaksi mencurigakan.")

end = time.time()
print(f"Waktu eksekusi: {round((end - start)/60, 2)} menit")
