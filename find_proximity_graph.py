import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from datetime import timedelta
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")

# Waktu mulai
start = time.time()

# Load data
df = pd.read_pickle('data/maritim_selat_sunda.pkl')
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)]
df = df.dropna(subset=['mmsi', 'lat', 'lon', 'created_at', 'sog'])
df['utc'] = pd.to_datetime(df['created_at'])
df = df.sort_values(by='utc')

# Parameter
PROXIMITY_THRESHOLD_KM = 0.05  # 50 meter
DURATION_THRESHOLD_MIN = 30
SOG_THRESHOLD = 0.5
TIME_GAP_MINUTES = 10
PORT_DISTANCE_THRESHOLD_KM = 10.0

# Pelabuhan penting
ports = [
    {"name": "Pelabuhan Merak", "lat": -5.8933, "lon": 106.0086},
    {"name": "Pelabuhan Ciwandan", "lat": -5.9525, "lon": 106.0358},
    {"name": "Pelabuhan Bojonegara", "lat": -5.8995, "lon": 106.0657},
    {"name": "Pelabuhan Bakauheni", "lat": -5.8711, "lon": 105.7421},
    {"name": "Pelabuhan Panjang", "lat": -5.4558, "lon": 105.3134},
    {"name": "Pelabuhan Ciwandan 2", "lat": -6.02147, "lon": 105.95485},
]

def is_far_from_ports(lat, lon, ports, min_distance_km=PORT_DISTANCE_THRESHOLD_KM):
    for port in ports:
        dist = haversine((lat, lon), (port['lat'], port['lon']), unit=Unit.KILOMETERS)
        if dist < min_distance_km:
            return False
    return True

# Buat interval waktu per 1 menit
df['utc_rounded'] = df['utc'].dt.floor('1min')
grouped = df.groupby('utc_rounded')

# Cari proximity
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

# Analisis hubungan proximity–durasi
anom_df = pd.DataFrame(anomalies)
final_anomalies = []

if not anom_df.empty:
    for (m1, m2), group in anom_df.groupby(['mmsi_1', 'mmsi_2']):
        group = group.sort_values('utc')
        group['time_diff'] = group['utc'].diff().fillna(pd.Timedelta(seconds=0))
        group['gap'] = (group['time_diff'] > pd.Timedelta(minutes=TIME_GAP_MINUTES)).cumsum()

        for _, session in group.groupby('gap'):
            duration = session['utc'].nunique()
            lat_mean = session['lat'].mean()
            lon_mean = session['lon'].mean()

            if duration >= DURATION_THRESHOLD_MIN and is_far_from_ports(lat_mean, lon_mean, ports):
                coords = list(zip(session['lat'], session['lon']))
                if len(coords) >= 2:
                    distances = [haversine(coords[i], coords[i+1], unit=Unit.KILOMETERS)
                                 for i in range(len(coords) - 1)]
                    avg_distance = np.mean(distances)
                else:
                    avg_distance = 0

                final_anomalies.append({
                    'mmsi_1': m1,
                    'mmsi_2': m2,
                    'start_time': session['utc'].min(),
                    'end_time': session['utc'].max(),
                    'duration_min': duration,
                    'lat': lat_mean,
                    'lon': lon_mean,
                    'proximity_km': avg_distance
                })

# Simpan dan tampilkan hasil
final_df = pd.DataFrame(final_anomalies)
if not final_df.empty:
    final_df.to_pickle("output_anomali_proximity_FIX.pkl")
    print("✅ Hasil proximity disimpan di output_anomali_proximity_FIX.pkl")

    # Grafik hubungan proximity vs durasi
    plt.figure(figsize=(10, 6))
    plt.scatter(final_df['proximity_km'], final_df['duration_min'],
                alpha=0.7, c='darkred', edgecolors='k')
    plt.title('Hubungan Rata-rata Proximity dan Durasi Interaksi Kapal', fontsize=13)
    plt.xlabel('Rata-rata Proximity (km)')
    plt.ylabel('Durasi Interaksi (menit)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("grafik_proximity_durasi_FIX.png", dpi=300)
    plt.show()
    print("✅ Grafik disimpan di grafik_proximity_durasi_FIX.png")
else:
    print("⚠️ Tidak ditemukan interaksi mencurigakan.")

end = time.time()
print(f"⏱️ Waktu eksekusi: {round((end - start)/60, 2)} menit")
