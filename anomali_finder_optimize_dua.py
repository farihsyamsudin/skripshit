import pandas as pd
import numpy as np
from haversine import haversine, Unit
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
PROXIMITY_THRESHOLD_KM = 0.05  # 50 meter
DURATION_THRESHOLD_MIN = 30    # minimal 30 menit
SOG_THRESHOLD = 0.5            # kapal hampir diam
PORT_DISTANCE_THRESHOLD_KM = 10.0  # minimal 10 km dari pelabuhan
TIME_GAP_MINUTES = 10  # batas waktu antar interaksi dianggap sesi baru

# Daftar pelabuhan
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

def get_color_hex(mmsi_1, mmsi_2):
    pair_str = f"{mmsi_1}-{mmsi_2}"
    hex_digest = hashlib.md5(pair_str.encode()).hexdigest()
    
    # Ambil hanya warna di spektrum hangat
    r = int(hex_digest[0:2], 16) % 200 + 50   # 50–250
    g = int(hex_digest[2:4], 16) % 160        # 0–160
    b = int(hex_digest[4:6], 16) % 60         # 0–60 (hindari biru dominan)
    return f"#{r:02x}{g:02x}{b:02x}"

# Deteksi proximity
df['utc_rounded'] = df['utc'].dt.floor('1min')
anomalies = []

for time, group in df.groupby('utc_rounded'):
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

# Agregasi final dengan pemisahan sesi interaksi
anom_df = pd.DataFrame(anomalies)
final_anomalies = []

if not anom_df.empty:
    for (m1, m2), group in anom_df.groupby(['mmsi_1', 'mmsi_2']):
        group = group.sort_values('utc')
        group['time_diff'] = group['utc'].diff().fillna(pd.Timedelta(seconds=0))
        group['gap'] = (group['time_diff'] > pd.Timedelta(minutes=TIME_GAP_MINUTES)).cumsum()

        for _, session in group.groupby('gap'):
            lat_mean = session['lat'].mean()
            lon_mean = session['lon'].mean()
            start_time = session['utc'].min()
            end_time = session['utc'].max()
            duration_minutes = (end_time - start_time).total_seconds() / 60

            if duration_minutes >= DURATION_THRESHOLD_MIN and is_far_from_ports(lat_mean, lon_mean, ports):
                final_anomalies.append({
                    'mmsi_1': m1,
                    'mmsi_2': m2,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_min': round(duration_minutes, 2),
                    'lat': lat_mean,
                    'lon': lon_mean,
                })

# Simpan ke CSV
final_df = pd.DataFrame(final_anomalies)
if not final_df.empty:
    final_df['start_time'] = pd.to_datetime(final_df['start_time']).dt.strftime("%d-%m-%Y %H:%M")
    final_df['end_time'] = pd.to_datetime(final_df['end_time']).dt.strftime("%d-%m-%Y %H:%M")
    final_df.to_csv("output_tabel_anomali_FIX_after_change.csv", index=False)

    # Plot Matplotlib
    plt.figure(figsize=(12, 8))
    for idx, row in final_df.iterrows():
        color = get_color_hex(row['mmsi_1'], row['mmsi_2'])
        plt.scatter(row['lon'], row['lat'], color=color, label=f"{row['mmsi_1']}-{row['mmsi_2']}", s=40)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Lokasi Anomali Transhipment")
    plt.grid(True)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output_grafik_anomali_warna_FIX_after_change.png", dpi=300)

    # Peta Folium
    map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=9)

    for idx, row in final_df.iterrows():
        color = get_color_hex(row['mmsi_1'], row['mmsi_2'])
        popup_text = (
            f"<b>{row['mmsi_1']} & {row['mmsi_2']}</b><br>"
            f"Durasi: {row['duration_min']} menit<br>"
            f"Start: {row['start_time']}<br>"
            f"End: {row['end_time']}"
        )
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    m.save("output_peta_anomali_FIX_not_cluster_after_change.html")
    print("Peta disimpan di output_peta_anomali_FIX_not_cluster_after_change.html")
else:
    print("Tidak ada interaksi mencurigakan.")

end = time.time()
print(f"Waktu eksekusi: {round((end - start)/60, 2)} menit")
