import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.spatial import KDTree
import time
import folium
from folium.plugins import MarkerCluster

# --- Parameter aturan ---
PROXIMITY_THRESHOLD_KM = 2.0  # 2000 meter
DURATION_THRESHOLD_MIN = 30   # minimal 30 menit
SOG_THRESHOLD = 0.5           # kapal hampir diam (Speed Over Ground < 0.5 knot)
PORT_DISTANCE_THRESHOLD_KM = 10.0 # minimal 10 km dari pelabuhan

# Daftar pelabuhan
ports = [
    {"name": "Pelabuhan Merak", "lat": -5.8933, "lon": 106.0086},
    {"name": "Pelabuhan Ciwandan", "lat": -5.9525, "lon": 106.0358},
    {"name": "Pelabuhan Bojonegara", "lat": -5.8995, "lon": 106.0657},
    {"name": "Pelabuhan Bakauheni", "lat": -5.8711, "lon": 105.7421},
    {"name": "Pelabuhan Panjang", "lat": -5.4558, "lon": 105.3134},
    {"name": "Pelabuhan Ciwandan 2", "lat": -6.02147, "lon": 105.95485},
]

# --- Fungsi Pembantu ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak Haversine antara dua titik koordinat (latitude, longitude)
    dalam kilometer.
    """
    R = 6371  # Radius bumi dalam kilometer
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def is_far_from_ports(lat, lon, ports, threshold_km):
    """
    Memeriksa apakah sebuah titik koordinat jauh dari semua pelabuhan yang ditentukan.
    Mengembalikan True jika jarak ke *semua* pelabuhan melebihi threshold_km.
    """
    min_dist_to_any_port = float('inf')
    for port in ports:
        dist = haversine_distance(lat, lon, port['lat'], port['lon'])
        if dist < min_dist_to_any_port:
            min_dist_to_any_port = dist
    return min_dist_to_any_port > threshold_km

# --- Fungsi Utama Deteksi Anomali ---

def detect_illegal_transhipment(file_path="data/maritim_selat_sunda_with_type.pkl"):
    print(f"[{time.ctime()}] Memulai deteksi anomali...")
    print(f"[{time.ctime()}] Memuat data dari: {file_path}")

    try:
        df = pd.read_pickle(file_path)
    except FileNotFoundError:
        print(f"[{time.ctime()}] ERROR: File '{file_path}' tidak ditemukan.")
        return pd.DataFrame() # Mengembalikan DataFrame kosong jika file tidak ditemukan
    except Exception as e:
        print(f"[{time.ctime()}] ERROR saat memuat data: {e}")
        return pd.DataFrame()

    initial_rows = len(df)
    print(f"[{time.ctime()}] Data dimuat. Jumlah baris awal: {initial_rows}")

    # 1. Preprocessing dan Filtering Awal
    print(f"[{time.ctime()}] Melakukan filtering berdasarkan area geografis...")
    df_filtered = df[
        (df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)
    ].copy() # Gunakan .copy() untuk menghindari SettingWithCopyWarning

    # Hapus baris dengan nilai NaN di kolom penting
    df_filtered.dropna(subset=['mmsi', 'created_at', 'lon', 'lat', 'sog'], inplace=True)

    # Konversi 'created_at' ke datetime objek
    # Penanganan kesalahan jika format tidak sesuai
    try:
        df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'], utc=True)
    except Exception as e:
        print(f"[{time.ctime()}] ERROR: Gagal mengkonversi 'created_at' ke datetime. Pastikan format '2024-06-01T00:00:00.000Z'. Error: {e}")
        return pd.DataFrame()

    df_filtered.sort_values(by=['mmsi', 'created_at'], inplace=True)
    
    filtered_rows = len(df_filtered)
    print(f"[{time.ctime()}] Data setelah filter area dan preprocessing: {filtered_rows} baris")
    if filtered_rows == 0:
        print(f"[{time.ctime()}] Tidak ada data setelah filtering awal. Selesai.")
        return pd.DataFrame()

    # Filter data untuk hanya yang memiliki kecepatan rendah (SOG < SOG_THRESHOLD)
    df_slow_ships = df_filtered[df_filtered['sog'] < SOG_THRESHOLD].copy()
    print(f"[{time.ctime()}] Jumlah records kapal dengan SOG < {SOG_THRESHOLD} knot: {len(df_slow_ships)}")

    if len(df_slow_ships) < 2:
        print(f"[{time.ctime()}] Tidak cukup data kapal lambat untuk mendeteksi pasangan. Selesai.")
        return pd.DataFrame()

    # 2. Deteksi Pasangan Kapal Berdekatan (Proximity)
    print(f"[{time.ctime()}] Memulai deteksi pasangan kapal berdekatan dan berkecepatan rendah...")
    
    # Membuat list untuk menyimpan anomali
    anomalies = []

    # Grouping by date for efficiency
    df_slow_ships['date'] = df_slow_ships['created_at'].dt.date
    
    processed_dates = 0
    total_dates = df_slow_ships['date'].nunique()
    
    for current_date in df_slow_ships['date'].unique():
        processed_dates += 1
        print(f"[{time.ctime()}] Memproses tanggal: {current_date} ({processed_dates}/{total_dates})")
        
        # Ambil data untuk tanggal saat ini
        df_today = df_slow_ships[df_slow_ships['date'] == current_date].copy()
        
        if len(df_today) < 2:
            continue

        # Buat KDTree untuk mencari tetangga terdekat pada tanggal ini
        coords = df_today[['lat', 'lon']].values
        tree = KDTree(coords)

        # Cari pasangan kapal dalam radius PROXIMITY_THRESHOLD_KM
        # Convert KM to degrees for KDTree query, roughly.
        # 1 degree lat ~ 111 km, 1 degree lon ~ 111 * cos(lat) km.
        # Using a conservative estimate for radius in degrees.
        # More accurate approach is to use Haversine after KDTree preliminary check.
        approx_radius_deg = PROXIMITY_THRESHOLD_KM / 111.0
        
        # Query pairs within the approximate radius
        pairs_indices = tree.query_pairs(approx_radius_deg)
        
        contact_points = []
        for i, j in pairs_indices:
            row_i = df_today.iloc[i]
            row_j = df_today.iloc[j]

            if row_i['mmsi'] != row_j['mmsi']:
                dist = haversine_distance(row_i['lat'], row_i['lon'], row_j['lat'], row_j['lon'])
                if dist <= PROXIMITY_THRESHOLD_KM:
                    contact_points.append({
                        'mmsi1': row_i['mmsi'],
                        'mmsi2': row_j['mmsi'],
                        'time1': row_i['created_at'],
                        'time2': row_j['created_at'],
                        'lat1': row_i['lat'],
                        'lon1': row_i['lon'],
                        'lat2': row_j['lat'],
                        'lon2': row_j['lon'],
                        'sog1': row_i['sog'],
                        'sog2': row_j['sog'],
                        'distance_km': dist,
                        'avg_lat': (row_i['lat'] + row_j['lat']) / 2,
                        'avg_lon': (row_i['lon'] + row_j['lon']) / 2,
                    })

        if not contact_points:
            continue

        df_contacts_today = pd.DataFrame(contact_points)
        
        df_contacts_today['mmsi_pair'] = df_contacts_today.apply(
            lambda x: tuple(sorted((x['mmsi1'], x['mmsi2']))), axis=1
        )
        
        df_contacts_today.sort_values(by=['mmsi_pair', 'time1'], inplace=True)

        current_pair_anomalies = {} 
        
        for _, row in df_contacts_today.iterrows():
            mmsi_pair = row['mmsi_pair']
            current_time = row['time1']
            current_lat = row['avg_lat']
            current_lon = row['avg_lon']

            if mmsi_pair not in current_pair_anomalies:
                current_pair_anomalies[mmsi_pair] = {
                    'start_time': current_time,
                    'last_contact_time': current_time,
                    'start_lat': current_lat,
                    'start_lon': current_lon,
                    'mmsi1': row['mmsi1'],
                    'mmsi2': row['mmsi2'],
                    'sog1_start': row['sog1'],
                    'sog2_start': row['sog2'],
                }
            else:
                contact_info = current_pair_anomalies[mmsi_pair]
                contact_info['last_contact_time'] = current_time

                duration = (current_time - contact_info['start_time']).total_seconds() / 60
                
                if duration >= DURATION_THRESHOLD_MIN:
                    dist_from_start_to_end = haversine_distance(
                        contact_info['start_lat'], contact_info['start_lon'],
                        current_lat, current_lon
                    )

                    if dist_from_start_to_end < PROXIMITY_THRESHOLD_KM:
                        anomalies.append({
                            'mmsi1': contact_info['mmsi1'],
                            'mmsi2': contact_info['mmsi2'],
                            'start_time': contact_info['start_time'],
                            'end_time': current_time,
                            'duration_min': duration,
                            'start_lat': contact_info['start_lat'],
                            'start_lon': contact_info['start_lon'],
                            'end_lat': current_lat,
                            'end_lon': current_lon,
                            'distance_from_start_to_end_km': dist_from_start_to_end,
                            'sog1_start': contact_info['sog1_start'],
                            'sog2_start': contact_info['sog2_start'],
                            'sog1_end': row['sog1'],
                            'sog2_end': row['sog2'],
                        })
                        del current_pair_anomalies[mmsi_pair]
    
    if not anomalies:
        print(f"[{time.ctime()}] Tidak ada anomali 'long duration proximity' terdeteksi.")
        return pd.DataFrame()

    df_anomalies = pd.DataFrame(anomalies)
    # Gunakan kombinasi mmsi_pair dan waktu awal sebagai identifikasi unik untuk duplikat
    df_anomalies['event_id'] = df_anomalies.apply(lambda x: f"{x['mmsi1']}-{x['mmsi2']}-{x['start_time']}", axis=1)
    df_anomalies.drop_duplicates(subset=['event_id'], inplace=True)
    df_anomalies.drop(columns=['event_id'], inplace=True) # Hapus kolom bantu
    
    print(f"[{time.ctime()}] Jumlah potensi anomali terdeteksi sebelum filter jarak pelabuhan: {len(df_anomalies)}")

    # 3. Filter Anomali Jauh dari Pelabuhan
    print(f"[{time.ctime()}] Melakukan filtering anomali berdasarkan jarak dari pelabuhan...")
    
    df_anomalies['is_far_from_port'] = df_anomalies.apply(
        lambda row: is_far_from_ports(row['start_lat'], row['start_lon'], ports, PORT_DISTANCE_THRESHOLD_KM),
        axis=1
    )
    
    final_anomalies = df_anomalies[df_anomalies['is_far_from_port']].copy()

    print(f"[{time.ctime()}] Deteksi anomali selesai.")
    print(f"[{time.ctime()}] Jumlah anomali transhipment ilegal yang terdeteksi: {len(final_anomalies)}")

    return final_anomalies

def visualize_anomalies_on_map(anomalies_df, map_output_path="anomalies_map.html"):
    """
    Membuat peta interaktif menggunakan Folium untuk memvisualisasikan anomali.
    """
    if anomalies_df.empty:
        print("Tidak ada anomali untuk divisualisasikan di peta.")
        return

    # Tentukan titik tengah peta berdasarkan rata-rata anomali
    center_lat = anomalies_df['start_lat'].mean()
    center_lon = anomalies_df['start_lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Tambahkan MarkerCluster untuk mengelompokkan marker jika ada banyak
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in anomalies_df.iterrows():
        popup_html = f"""
        <b>Anomali Transhipment Ilegal</b><br>
        MMSI Kapal 1: {row['mmsi1']}<br>
        MMSI Kapal 2: {row['mmsi2']}<br>
        Waktu Mulai: {row['start_time'].strftime('%Y-%m-%d %H:%M UTC')}<br>
        Waktu Selesai: {row['end_time'].strftime('%Y-%m-%d %H:%M UTC')}<br>
        Durasi: {row['duration_min']:.2f} menit<br>
        Lokasi Awal (Avg): ({row['start_lat']:.4f}, {row['start_lon']:.4f})<br>
        Jarak Pergeseran: {row['distance_from_start_to_end_km']:.2f} km
        """
        
        # Menggunakan lokasi awal rata-rata untuk penempatan marker
        folium.CircleMarker(
            location=[row['start_lat'], row['start_lon']],
            radius=5, # Radius marker
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(marker_cluster)
    
    # Tambahkan lokasi pelabuhan sebagai marker biru
    for port in ports:
        folium.Marker(
            location=[port['lat'], port['lon']],
            icon=folium.Icon(color='blue', icon='info-sign'),
            popup=f"<b>Pelabuhan:</b> {port['name']}"
        ).add_to(m)

    m.save(map_output_path)
    print(f"Peta anomali telah disimpan ke: {map_output_path}")

# --- Eksekusi Kode ---
if __name__ == "__main__":
    start_time_total = time.time()
    
    # Path ke file PKL Anda
    data_file_path = "data/maritim_selat_sunda_with_type.pkl" 

    anomalous_events_df = detect_illegal_transhipment(file_path=data_file_path)

    if not anomalous_events_df.empty:
        print("\n--- Detail Anomali Terdeteksi ---")
        # Pilih kolom yang ingin ditampilkan di output konsol dan CSV
        output_columns = [
            'mmsi1', 'mmsi2', 'start_time', 'end_time', 'duration_min', 
            'start_lat', 'start_lon', 'end_lat', 'end_lon', 
            'distance_from_start_to_end_km', 'sog1_start', 'sog2_start', 'sog1_end', 'sog2_end'
        ]
        print(anomalous_events_df[output_columns].head())
        
        # Simpan hasil ke CSV
        csv_output_path = "illegal_transhipment_anomalies_baruuu.csv"
        anomalous_events_df[output_columns].to_csv(csv_output_path, index=False)
        print(f"\nHasil anomali telah disimpan ke '{csv_output_path}'")

        # Visualisasikan anomali di peta
        map_output_path = "illegal_transhipment_anomalies_map_baruuu.html"
        visualize_anomalies_on_map(anomalous_events_df, map_output_path)

    else:
        print("\nTidak ada anomali yang terdeteksi berdasarkan kriteria yang diberikan.")

    end_time_total = time.time()
    print(f"\n[{time.ctime()}] Total waktu eksekusi: {end_time_total - start_time_total:.2f} detik")