import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta, datetime
import folium
from folium.plugins import MarkerCluster

# --- Parameter aturan ---
PROXIMITY_THRESHOLD_KM = 0.05  # 50 meter
DURATION_THRESHOLD_MIN = 5     # minimal 5 menit (sudah diubah dari 30)
SOG_THRESHOLD = 0.5            # kapal hampir diam (knot)
PORT_DISTANCE_THRESHOLD_KM = 20.0  # minimal 10 km dari pelabuhan
TIME_GAP_MINUTES = 20          # Batas waktu antar interaksi dianggap sesi baru (sudah diubah dari 10)
TIME_WINDOW_HOURS = 1 # Ukuran jendela waktu untuk memproses data (misal: 1 jam data sekaligus)

# Daftar pelabuhan
ports = [
    {"name": "Pelabuhan Merak", "lat": -5.8933, "lon": 106.0086},
    {"name": "Pelabuhan Ciwandan", "lat": -5.9525, "lon": 106.0358},
    {"name": "Pelabuhan Bojonegara", "lat": -5.8995, "lon": 106.0657},
    {"name": "Pelabuhan Bakauheni", "lat": -5.8711, "lon": 105.7421},
    {"name": "Pelabuhan Panjang", "lat": -5.4558, "lon": 105.3134},
    {"name": "Pelabuhan Ciwandan 2", "lat": -6.02147, "lon": 105.95485},
]

# --- Fungsi Haversine untuk menghitung jarak dalam KM ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius bumi dalam kilometer

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return R * c

# --- Fungsi untuk mengecek apakah kapal jauh dari pelabuhan ---
def is_far_from_port(lat, lon, ports, threshold_km):
    for port in ports:
        distance = haversine_distance(lat, lon, port['lat'], port['lon'])
        if distance < threshold_km:
            return False  # Dekat dengan setidaknya satu pelabuhan
    return True  # Jauh dari semua pelabuhan

# --- Main Logic ---
def detect_illegal_transhipment(file_path):
    print(f"Memuat data dari: {file_path}...")
    try:
        df = pd.read_pickle(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Data dimuat. Jumlah baris: {len(df)}")

    # Pre-processing
    print("Melakukan pre-processing data...")
    # Pastikan 'created_at' adalah datetime object
    if isinstance(df['created_at'].iloc[0], dict) and '$date' in df['created_at'].iloc[0]:
        df['created_at'] = df['created_at'].apply(lambda x: pd.to_datetime(x['$date']))
    else:
        df['created_at'] = pd.to_datetime(df['created_at'])

    # Urutkan data berdasarkan MMSI dan waktu
    df = df.sort_values(by=['mmsi', 'created_at']).reset_index(drop=True)

    # Bersihkan data yang tidak valid atau null yang krusial
    initial_rows = len(df)
    df.dropna(subset=['mmsi', 'lon', 'lat', 'sog', 'created_at'], inplace=True)
    df = df[df['sog'] >= 0] # SOG tidak boleh negatif
    print(f"Data setelah membersihkan null/invalid: {len(df)} baris (dihapus {initial_rows - len(df)})")

    # Filter data awal untuk kapal yang kecepatannya rendah (berpotensi diam)
    df_slow = df[df['sog'] < SOG_THRESHOLD].copy()
    print(f"Jumlah entri dengan SOG < {SOG_THRESHOLD} knot: {len(df_slow)}")

    # Tambahkan kolom jarak dari pelabuhan
    print("Menghitung jarak dari pelabuhan...")
    df_slow['far_from_port'] = df_slow.apply(
        lambda row: is_far_from_port(row['lat'], row['lon'], ports, PORT_DISTANCE_THRESHOLD_KM), axis=1
    )
    df_slow_filtered = df_slow[df_slow['far_from_port']].copy()
    print(f"Jumlah entri SOG rendah dan jauh dari pelabuhan: {len(df_slow_filtered)}")

    if df_slow_filtered.empty:
        print("Tidak ada kapal yang memenuhi kriteria awal (SOG rendah dan jauh dari pelabuhan). Tidak ada anomali terdeteksi.")
        return pd.DataFrame(), pd.DataFrame()

    current_interactions = {}
    detected_anomalies = []
    all_potential_interactions_records = [] # Ini akan kita isi dengan benar

    min_time = df_slow_filtered['created_at'].min()
    max_time = df_slow_filtered['created_at'].max()
    current_window_start = min_time

    total_chunks = (max_time - min_time).total_seconds() / (TIME_WINDOW_HOURS * 3600)
    print(f"Memulai deteksi anomali dalam {total_chunks:.2f} jendela waktu ({TIME_WINDOW_HOURS} jam per jendela)...")
    
    window_count = 0
    while current_window_start <= max_time:
        window_end = current_window_start + timedelta(hours=TIME_WINDOW_HOURS)
        
        window_data = df_slow_filtered[
            (df_slow_filtered['created_at'] >= current_window_start - timedelta(minutes=TIME_GAP_MINUTES)) &
            (df_slow_filtered['created_at'] < window_end + timedelta(minutes=TIME_GAP_MINUTES))
        ].copy() 
        
        unique_timestamps_in_window = window_data['created_at'].unique()
        unique_timestamps_in_window = np.sort(unique_timestamps_in_window)

        for timestamp in unique_timestamps_in_window:
            snapshot = window_data[window_data['created_at'] == timestamp]
            
            vessels_in_snapshot = snapshot[['mmsi', 'lat', 'lon', 'sog']].to_dict(orient='records')
            
            current_snapshot_interactions = set() # Pasangan yang berinteraksi di snapshot ini

            for i in range(len(vessels_in_snapshot)):
                vessel1 = vessels_in_snapshot[i]
                for j in range(i + 1, len(vessels_in_snapshot)):
                    vessel2 = vessels_in_snapshot[j]

                    if vessel1['mmsi'] == vessel2['mmsi']:
                        continue
                    
                    pair_key = tuple(sorted((vessel1['mmsi'], vessel2['mmsi'])))

                    distance = haversine_distance(vessel1['lat'], vessel1['lon'], vessel2['lat'], vessel2['lon'])

                    if distance < PROXIMITY_THRESHOLD_KM:
                        current_snapshot_interactions.add(pair_key)
            
            # --- Perubahan Logika Utama untuk menghitung durasi potensi interaksi ---
            # 1. Identifikasi interaksi yang berakhir di snapshot ini (atau terputus)
            keys_to_remove = []
            for pair_key, data in current_interactions.items():
                if pair_key in current_snapshot_interactions:
                    time_diff = (timestamp - data['last_seen']).total_seconds() / 60
                    
                    if time_diff <= TIME_GAP_MINUTES:
                        # Interaksi berlanjut
                        data['current_duration_min'] += time_diff
                        data['last_seen'] = timestamp
                        
                        # Tambahkan ke potential_interactions_records setelah durasi diperbarui
                        # Ambil posisi kedua kapal di snapshot saat ini
                        v1_data = snapshot[snapshot['mmsi'] == pair_key[0]].iloc[0]
                        v2_data = snapshot[snapshot['mmsi'] == pair_key[1]].iloc[0]

                        all_potential_interactions_records.append({
                            'mmsi_1': pair_key[0],
                            'mmsi_2': pair_key[1],
                            'timestamp': timestamp,
                            'lat_1': v1_data['lat'],
                            'lon_1': v1_data['lon'],
                            'lat_2': v2_data['lat'],
                            'lon_2': v2_data['lon'],
                            'sog_1': v1_data['sog'],
                            'sog_2': v2_data['sog'],
                            'distance_km': distance,
                            'interaction_duration_min': data['current_duration_min'] # Durasi yang sudah terakumulasi
                        })

                        if data['current_duration_min'] >= DURATION_THRESHOLD_MIN:
                            # Jika durasi mencapai ambang batas, ini adalah anomali terdeteksi
                            detected_anomalies.append({
                                'mmsi_1': pair_key[0],
                                'mmsi_2': pair_key[1],
                                'start_time': data['start_time'],
                                'end_time': timestamp,
                                'duration_min': data['current_duration_min'],
                                'start_lat_1': data['start_lat_1'],
                                'start_lon_1': data['start_lon_1'],
                                'start_lat_2': data['start_lat_2'],
                                'start_lon_2': data['start_lon_2'],
                                'end_lat_1': v1_data['lat'], # Posisi akhir di snapshot ini
                                'end_lon_1': v1_data['lon'],
                                'end_lat_2': v2_data['lat'],
                                'end_lon_2': v2_data['lon'],
                                'avg_distance_km': distance # Jarak terakhir yang terdeteksi
                            })
                            keys_to_remove.append(pair_key) # Anomali terdeteksi, hapus dari current_interactions
                    else:
                        keys_to_remove.append(pair_key) # Jeda terlalu besar, putuskan sesi
                else:
                    keys_to_remove.append(pair_key) # Kapal tidak lagi berdekatan, putuskan sesi

            # Hapus interaksi yang sudah selesai atau terputus
            for key in keys_to_remove:
                if key in current_interactions:
                    del current_interactions[key]
            
            # 2. Identifikasi interaksi yang baru dimulai di snapshot ini
            for pair_key in current_snapshot_interactions:
                if pair_key not in current_interactions:
                    v1_data = snapshot[snapshot['mmsi'] == pair_key[0]].iloc[0]
                    v2_data = snapshot[snapshot['mmsi'] == pair_key[1]].iloc[0]
                    current_interactions[pair_key] = {
                        'start_time': timestamp,
                        'last_seen': timestamp,
                        'current_duration_min': 0, # Durasi awal untuk sesi baru
                        'start_lat_1': v1_data['lat'],
                        'start_lon_1': v1_data['lon'],
                        'start_lat_2': v2_data['lat'],
                        'start_lon_2': v2_data['lon'],
                    }
                    # Tambahkan juga ke potential_interactions_records (dengan durasi awal 0)
                    all_potential_interactions_records.append({
                        'mmsi_1': pair_key[0],
                        'mmsi_2': pair_key[1],
                        'timestamp': timestamp,
                        'lat_1': v1_data['lat'],
                        'lon_1': v1_data['lon'],
                        'lat_2': v2_data['lat'],
                        'lon_2': v2_data['lon'],
                        'sog_1': v1_data['sog'],
                        'sog_2': v2_data['sog'],
                        'distance_km': distance, # Ini adalah distance dari snapshot saat ini
                        'interaction_duration_min': 0 # Durasi awal karena baru dimulai
                    })
        
        current_window_start = window_end
        window_count += 1
        if window_count % 10 == 0:
            print(f"  Memproses jendela ke-{window_count} (hingga {current_window_start.strftime('%Y-%m-%d %H:%M:%S')})...")

    df_anomalies = pd.DataFrame(detected_anomalies)
    df_potential_interactions = pd.DataFrame(all_potential_interactions_records)

    print("\nDeteksi anomali selesai.")
    print(f"Total anomali terdeteksi: {len(df_anomalies)}")

    return df_anomalies, df_potential_interactions

# --- Fungsi untuk memvisualisasikan anomali di Folium ---
def visualize_anomalies_on_map(anomalies_df, ports, map_file_name="transhipment_anomalies_map_dua.html"):
    if anomalies_df.empty:
        print("Tidak ada anomali untuk divisualisasikan.")
        return

    map_center_lat = anomalies_df['start_lat_1'].mean()
    map_center_lon = anomalies_df['start_lon_1'].mean()

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=9)

    marker_cluster = MarkerCluster().add_to(m)

    for port in ports:
        folium.Marker(
            location=[port['lat'], port['lon']],
            popup=f"<b>Pelabuhan:</b> {port['name']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    for idx, row in anomalies_df.iterrows():
        lat = row['start_lat_1']
        lon = row['start_lon_1']

        popup_html = f"""
        <b>Illegal Transhipment Detected!</b><br>
        MMSI 1: {row['mmsi_1']}<br>
        MMSI 2: {row['mmsi_2']}<br>
        Start Time: {row['start_time'].strftime('%Y-%m-%d %H:%M:%S')}<br>
        End Time: {row['end_time'].strftime('%Y-%m-%d %H:%M:%S')}<br>
        Duration: {row['duration_min']:.2f} minutes<br>
        Start Location: ({row['start_lat_1']:.4f}, {row['start_lon_1']:.4f})<br>
        End Location 1: ({row['end_lat_1']:.4f}, {row['end_lon_1']:.4f})<br>
        End Location 2: ({row['end_lat_2']:.4f}, {row['end_lon_2']:.4f})<br>
        Average Proximity: {row['avg_distance_km'] * 1000:.2f} meters
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=popup_html,
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(marker_cluster)
    
    m.save(map_file_name)
    print(f"\nPeta anomali disimpan ke '{map_file_name}'")


# --- Jalankan deteksi dan visualisasi ---
if __name__ == "__main__":
    file_path = "data/maritim_selat_sunda.pkl"
    
    anomalies_df, potential_interactions_df = detect_illegal_transhipment(file_path)

    if not anomalies_df.empty:
        print("\n--- Detail Anomali Terdeteksi ---")
        print(anomalies_df.head())
        anomalies_df.to_csv("illegal_transhipment_anomalies_dua.csv", index=False)
        print("\nAnomali disimpan ke 'illegal_transhipment_anomalies_dua.csv'")

        visualize_anomalies_on_map(anomalies_df, ports)
    else:
        print("\nTidak ada anomali illegal transhipment yang terdeteksi.")
    
    if not potential_interactions_df.empty:
        print("\n--- Contoh Potensi Interaksi (dengan durasi) ---")
        print(potential_interactions_df.head())
        potential_interactions_df.to_csv("potential_vessel_interactions_with_duration.csv", index=False)
        print("\nPotensi interaksi disimpan ke 'potential_vessel_interactions_with_duration.csv'")