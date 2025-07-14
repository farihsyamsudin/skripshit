import pandas as pd
import numpy as np
from haversine import haversine, haversine_vector, Unit
from datetime import timedelta
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import BallTree
import folium
from folium.plugins import MarkerCluster
import warnings
import hashlib
warnings.filterwarnings("ignore")

start = time.time()

# Load dan persiapan
df = pd.read_pickle('data/maritim_selat_sunda.pkl')
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)].reset_index(drop=True)
df = df.dropna(subset=['mmsi', 'lat', 'lon', 'created_at', 'sog'])
df['utc'] = pd.to_datetime(df['created_at'])
df = df.sort_values(by='utc')

# Parameter aturan
PROXIMITY_THRESHOLD_KM = 0.05
DURATION_THRESHOLD_MIN = 30
SOG_THRESHOLD = 0.5
PORT_DISTANCE_THRESHOLD_KM = 10.0

# Daftar pelabuhan besar & kecil di Selat Sunda
ports = [
    {"name": "Pelabuhan Merak", "lat": -5.8933, "lon": 106.0086},
    {"name": "Pelabuhan Ciwandan", "lat": -5.9525, "lon": 106.0358},
    {"name": "Pelabuhan Bojonegara", "lat": -5.8995, "lon": 106.0657},
    {"name": "Pelabuhan Bakauheni", "lat": -5.8711, "lon": 105.7421},
    {"name": "Pelabuhan Panjang", "lat": -5.4558, "lon": 105.3134},
    {"name": "Pelabuhan Ciwandan", "lat": -6.02147, "lon": 105.95485},
]

def is_far_from_ports(lat, lon, ports, min_distance_km=PORT_DISTANCE_THRESHOLD_KM):
    for port in ports:
        dist = haversine((lat, lon), (port['lat'], port['lon']))
        if dist < min_distance_km:
            return False
    return True

def get_color_hex(mmsi_1, mmsi_2):
    pair_str = f"{mmsi_1}-{mmsi_2}"
    hex_digest = hashlib.md5(pair_str.encode()).hexdigest()
    return f"#{hex_digest[:6]}"  # ambil 6 digit awal buat warna hex

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

            if row_i['sog'] >= SOG_THRESHOLD or row_j['sog'] >= SOG_THRESHOLD:
                continue

            anomalies.append({
                'mmsi_1': min(row_i['mmsi'], row_j['mmsi']),
                'mmsi_2': max(row_i['mmsi'], row_j['mmsi']),
                'utc': time,
                'lat': (row_i['lat'] + row_j['lat']) / 2,
                'lon': (row_i['lon'] + row_j['lon']) / 2,
            })

# Agregasi berdasarkan pasangan
anom_df = pd.DataFrame(anomalies)

if anom_df.empty:
    print("Tidak ada interaksi mencurigakan.")
else:
    final_anomalies = []
    grouped_pairs = anom_df.groupby(['mmsi_1', 'mmsi_2'])

    for (m1, m2), group in grouped_pairs:
        unique_minutes = group['utc'].nunique()
        lat_mean = group['lat'].mean()
        lon_mean = group['lon'].mean()

        # Cek jarak dari pelabuhan terdekat
        if unique_minutes >= DURATION_THRESHOLD_MIN and is_far_from_ports(lat_mean, lon_mean, ports):
            final_anomalies.append({
                'mmsi_1': m1,
                'mmsi_2': m2,
                'start_time': group['utc'].min(),
                'end_time': group['utc'].max(),
                'duration_min': unique_minutes,
                'lat': lat_mean,
                'lon': lon_mean,
            })

    final_df = pd.DataFrame(final_anomalies)
    final_df.to_csv("output_tabel_anomali_FIX.csv", index=False)

    # ======= Plot dengan warna berbeda per pasangan ========
    plt.figure(figsize=(12, 8))
    for idx, row in final_df.iterrows():
        key = (row['mmsi_1'], row['mmsi_2'])
        color = get_color_hex(row['mmsi_1'], row['mmsi_2'])
        plt.scatter(row['lon'], row['lat'], color=color, label=f'{key[0]}-{key[1]}', s=40)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Lokasi Anomali Transhipment")
    plt.grid(True)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    plt.savefig("output_grafik_anomali_warna_FIX.png", dpi=300)

    # Folium Map
    map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=9)

    for idx, row in final_df.iterrows():
        key = (row['mmsi_1'], row['mmsi_2'])
        color = get_color_hex(row['mmsi_1'], row['mmsi_2'])

        popup_text = f"{row['mmsi_1']} & {row['mmsi_2']}<br>Durasi: {row['duration_min']} menit"
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    m.save("output_peta_anomali_FIX_not_cluster.html")
    print("Peta disimpan di output_peta_anomali_FIX_not_cluster.html")

end = time.time()
print(f"Waktu eksekusi: {round((end - start)/60, 2)} menit")