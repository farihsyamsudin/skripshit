import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from datetime import timedelta
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import BallTree
import folium
from folium.plugins import MarkerCluster
import warnings
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
PROXIMITY_THRESHOLD_KM = 1.0
DURATION_THRESHOLD_MIN = 30
SOG_THRESHOLD = 0.5

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
        if unique_minutes >= DURATION_THRESHOLD_MIN:
            final_anomalies.append({
                'mmsi_1': m1,
                'mmsi_2': m2,
                'start_time': group['utc'].min(),
                'end_time': group['utc'].max(),
                'duration_min': unique_minutes,
                'lat': group['lat'].mean(),
                'lon': group['lon'].mean(),
            })

    final_df = pd.DataFrame(final_anomalies)
    final_df.to_csv("output_tabel_anomali_FIX.csv", index=False)

    # ======= Plot dengan warna berbeda per pasangan ========
    plt.figure(figsize=(12, 8))
    r, g, b = 50, 30, 90
    color_map = {}

    for idx, row in final_df.iterrows():
        key = (row['mmsi_1'], row['mmsi_2'])
        if key not in color_map:
            color = (r / 255, g / 255, b / 255)
            color_map[key] = color
            r = (r + 10) % 256
            g = (g + 5) % 256
            b = (b + 15) % 256
        
        plt.scatter(row['lon'], row['lat'], color=color_map[key], label=f'{row["mmsi_1"]}-{row["mmsi_2"]}', s=40)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Lokasi Anomali Transhipment")
    plt.grid(True)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    plt.savefig("output_grafik_anomali_warna_FIX.png", dpi=300)

    # Membuat peta awal di titik tengah (bisa disesuaikan)
    map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=9)
    marker_cluster = MarkerCluster().add_to(m)

    # Warna dinamis pakai kombinasi RGB
    color_map = {}
    r, g, b = 50, 50, 50  # awal RGB

    for idx, row in final_df.iterrows():
        key = f"{row['mmsi_1']}_{row['mmsi_2']}"
        if key not in color_map:
            color = f"#{r:02x}{g:02x}{b:02x}"
            color_map[key] = color
            r = (r + 40) % 256
            g = (g + 25) % 256
            b = (b + 55) % 256

        popup_text = f"{row['mmsi_1']} & {row['mmsi_2']}<br>Durasi: {row['duration_min']} menit"
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=color_map[key],
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(marker_cluster)

    # Simpan ke file HTML
    m.save("output_peta_anomali_FIX.html")
    print("Peta disimpan di output_peta_anomali_FIX.html")

end = time.time()
print(f"Waktu eksekusi: {round((end - start)/60, 2)} menit")
