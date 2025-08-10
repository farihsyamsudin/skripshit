import pandas as pd
import numpy as np
from haversine import haversine
from datetime import timedelta
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# 1. Load Data
df = pd.read_pickle('data/maritim_selat_sunda.pkl')
df = df.dropna(subset=['mmsi', 'lat', 'lon', 'utc', 'sog'])
df['utc'] = pd.to_datetime(df['utc'])

# 2. Sort data
df = df.sort_values(by=['mmsi', 'utc'])

# 3. Atur parameter
PROXIMITY_THRESHOLD_KM = 1.0
DURATION_THRESHOLD_MIN = 30
SOG_THRESHOLD = 0.5  # speed ≈ 0 knot

# 4. Ambil kombinasi pasangan kapal unik
mmsi_list = df['mmsi'].unique()
pairs = list(combinations(mmsi_list, 2))

anomalies = []

for mmsi1, mmsi2 in pairs:
    df1 = df[df['mmsi'] == mmsi1]
    df2 = df[df['mmsi'] == mmsi2]
    
    merged = pd.merge_asof(
        df1.sort_values('utc'), 
        df2.sort_values('utc'), 
        on='utc', 
        direction='nearest', 
        tolerance=pd.Timedelta('1min'),
        suffixes=('_1', '_2')
    ).dropna(subset=['lat_2', 'lon_2'])
    
    # Hitung jarak
    merged['distance_km'] = merged.apply(lambda row: haversine(
        (row['lat_1'], row['lon_1']), (row['lat_2'], row['lon_2'])
    ), axis=1)
    
    # Deteksi proximity < 1km dan sog mendekati 0
    condition = (merged['distance_km'] < PROXIMITY_THRESHOLD_KM) & \
                (merged['sog_1'] < SOG_THRESHOLD) & (merged['sog_2'] < SOG_THRESHOLD)

    if condition.sum() == 0:
        continue

    # Cek durasi interaksi
    grouped = merged[condition]
    time_diff = (grouped['utc'].max() - grouped['utc'].min()).total_seconds() / 60

    if time_diff >= DURATION_THRESHOLD_MIN:
        anomalies.append({
            'mmsi_1': mmsi1,
            'mmsi_2': mmsi2,
            'start_time': grouped['utc'].min(),
            'end_time': grouped['utc'].max(),
            'duration_min': round(time_diff, 2),
            'min_distance_km': round(grouped['distance_km'].min(), 3),
            'avg_sog_1': round(grouped['sog_1'].mean(), 2),
            'avg_sog_2': round(grouped['sog_2'].mean(), 2),
            'lat': grouped['lat_1'].mean(),
            'lon': grouped['lon_1'].mean()
        })

# 5. Buat DataFrame hasil anomali
anomalies_df = pd.DataFrame(anomalies)

# 6. Simpan tabel
anomalies_df.to_csv("output_tabel_anomali.csv", index=False)

# 7. Buat grafik
plt.figure(figsize=(10, 6))
plt.scatter(anomalies_df['min_distance_km'], anomalies_df['duration_min'], c='red')
plt.xlabel("Jarak Minimum (km)")
plt.ylabel("Durasi Interaksi (menit)")
plt.title("Grafik hubungan proximity dan durasi kapal pada kasus anomali")
plt.grid(True)
plt.savefig("output_grafik_anomali.png", dpi=300)
plt.show()
