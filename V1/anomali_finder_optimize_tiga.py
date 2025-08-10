import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta, datetime

# --- Parameter aturan ---
PROXIMITY_THRESHOLD_KM = 0.05  # 50 meter
DURATION_THRESHOLD_MIN = 30    # minimal 30 menit
SOG_THRESHOLD = 0.5            # kapal hampir diam (knot)
PORT_DISTANCE_THRESHOLD_KM = 10.0  # minimal 10 km dari pelabuhan
TIME_GAP_MINUTES = 10          # Batas waktu antar interaksi dianggap sesi baru
# CHUNK_SIZE = 100000 # Contoh jika membaca CSV/JSON per chunk, tapi PKL biasanya dibaca sekaligus
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
    return distance

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
    # Cek apakah 'created_at' berupa dictionary dengan '$date' key
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
    # Ini membantu mengurangi jumlah perbandingan
    df_slow = df[df['sog'] < SOG_THRESHOLD].copy()
    print(f"Jumlah entri dengan SOG < {SOG_THRESHOLD} knot: {len(df_slow)}")

    # Tambahkan kolom jarak dari pelabuhan
    print("Menghitung jarak dari pelabuhan...")
    df_slow['far_from_port'] = df_slow.apply(
        lambda row: is_far_from_port(row['lat'], row['lon'], ports, PORT_DISTANCE_THRESHOLD_KM), axis=1
    )
    # Kita hanya tertarik pada kapal yang jauh dari pelabuhan untuk transshipment
    df_slow_filtered = df_slow[df_slow['far_from_port']].copy()
    print(f"Jumlah entri SOG rendah dan jauh dari pelabuhan: {len(df_slow_filtered)}")

    if df_slow_filtered.empty:
        print("Tidak ada kapal yang memenuhi kriteria awal (SOG rendah dan jauh dari pelabuhan). Tidak ada anomali terdeteksi.")
        return pd.DataFrame(), pd.DataFrame()

    # Inisialisasi untuk melacak sesi interaksi
    # Format: {(mmsi1, mmsi2): {'start_time': datetime, 'last_seen': datetime, 'current_duration_min': float, 'start_lat': float, 'start_lon': float}}
    current_interactions = {}
    detected_anomalies = [] # Untuk menyimpan anomali yang ditemukan

    # Digunakan untuk menyimpan semua potensi interaksi (untuk analisis lebih lanjut)
    all_potential_interactions = []

    # Iterasi data per jendela waktu
    # Kita akan membagi data berdasarkan waktu untuk mengurangi beban komputasi
    min_time = df_slow_filtered['created_at'].min()
    max_time = df_slow_filtered['created_at'].max()
    current_window_start = min_time

    total_chunks = (max_time - min_time).total_seconds() / (TIME_WINDOW_HOURS * 3600)
    print(f"Memulai deteksi anomali dalam {total_chunks:.2f} jendela waktu ({TIME_WINDOW_HOURS} jam per jendela)...")
    
    window_count = 0
    while current_window_start <= max_time:
        window_end = current_window_start + timedelta(hours=TIME_WINDOW_HOURS)
        
        # Filter data untuk jendela waktu saat ini
        # Tambahkan sedikit buffer untuk memastikan tidak ada data yang terlewat di batas jendela
        window_data = df_slow_filtered[
            (df_slow_filtered['created_at'] >= current_window_start - timedelta(minutes=TIME_GAP_MINUTES)) &
            (df_slow_filtered['created_at'] < window_end + timedelta(minutes=TIME_GAP_MINUTES))
        ]
        
        # Ambil hanya data unik per MMSI dalam jendela ini (posisi terakhir dalam window)
        # Atau ambil data yang paling dekat dengan start of window
        # Untuk kasus ini, karena kita melacak 'terus menerus', penting untuk memproses setiap timestamp
        # Solusi yang lebih baik adalah mengindeks data berdasarkan waktu dan MMSI

        # Untuk efisiensi, ambil snapshot dari setiap MMSI di jendela ini
        # Menggunakan groupby().last() untuk mendapatkan entri terbaru per MMSI dalam window
        # Ini adalah trade-off performa vs. presisi untuk mengidentifikasi pasangan di *setiap* timestamp
        # Pendekatan yang lebih akurat (tapi lebih lambat): iterasi per timestamp unik dalam window_data
        
        unique_timestamps_in_window = window_data['created_at'].unique()
        unique_timestamps_in_window.sort()

        for timestamp in unique_timestamps_in_window:
            snapshot = window_data[window_data['created_at'] == timestamp]
            
            # Buat list of dictionaries untuk memudahkan perulangan
            vessels_in_snapshot = snapshot[['mmsi', 'lat', 'lon', 'sog']].to_dict(orient='records')
            
            # Identifikasi interaksi dalam snapshot ini
            current_snapshot_interactions = set() # Untuk melacak pasangan yang berinteraksi di timestamp ini

            for i in range(len(vessels_in_snapshot)):
                vessel1 = vessels_in_snapshot[i]
                
                # Hanya pertimbangkan kapal yang SOG-nya rendah (sudah difilter di df_slow_filtered)
                # dan jauh dari pelabuhan (sudah difilter di df_slow_filtered)

                for j in range(i + 1, len(vessels_in_snapshot)):
                    vessel2 = vessels_in_snapshot[j]

                    # Pastikan kedua kapal tidak sama
                    if vessel1['mmsi'] == vessel2['mmsi']:
                        continue
                    
                    # Sort MMSI untuk kunci unik (mmsi1, mmsi2) selalu sama
                    pair_key = tuple(sorted((vessel1['mmsi'], vessel2['mmsi'])))

                    # Hitung jarak antar kapal
                    distance = haversine_distance(vessel1['lat'], vessel1['lon'], vessel2['lat'], vessel2['lon'])

                    if distance < PROXIMITY_THRESHOLD_KM:
                        # Kapal berdekatan dan SOG sudah rendah (karena datang dari df_slow_filtered)
                        current_snapshot_interactions.add(pair_key)
                        
                        # Tambahkan ke semua potensi interaksi (untuk debugging/analisis)
                        all_potential_interactions.append({
                            'mmsi_1': vessel1['mmsi'],
                            'mmsi_2': vessel2['mmsi'],
                            'timestamp': timestamp,
                            'lat_1': vessel1['lat'],
                            'lon_1': vessel1['lon'],
                            'lat_2': vessel2['lat'],
                            'lon_2': vessel2['lon'],
                            'sog_1': vessel1['sog'],
                            'sog_2': vessel2['sog'],
                            'distance_km': distance
                        })

            # Update status interaksi berdasarkan snapshot ini
            keys_to_remove = []
            for pair_key, data in current_interactions.items():
                if pair_key in current_snapshot_interactions:
                    # Interaksi berlanjut atau baru dimulai
                    time_diff = (timestamp - data['last_seen']).total_seconds() / 60 # dalam menit
                    
                    if time_diff <= TIME_GAP_MINUTES: # Jika gap waktu masih dalam toleransi
                        data['current_duration_min'] += time_diff
                        data['last_seen'] = timestamp
                        # Check for anomaly
                        if data['current_duration_min'] >= DURATION_THRESHOLD_MIN:
                            detected_anomalies.append({
                                'mmsi_1': pair_key[0],
                                'mmsi_2': pair_key[1],
                                'start_time': data['start_time'],
                                'end_time': timestamp,
                                'duration_min': data['current_duration_min'],
                                'start_lat': data['start_lat'],
                                'start_lon': data['start_lon'],
                                'end_lat_1': vessel1['lat'] if vessel1['mmsi'] == pair_key[0] else vessel2['lat'], # Lintang akhir kapal 1
                                'end_lon_1': vessel1['lon'] if vessel1['mmsi'] == pair_key[0] else vessel2['lon'], # Bujur akhir kapal 1
                                'end_lat_2': vessel1['lat'] if vessel1['mmsi'] == pair_key[1] else vessel2['lat'], # Lintang akhir kapal 2
                                'end_lon_2': vessel1['lon'] if vessel1['mmsi'] == pair_key[1] else vessel2['lon'], # Bujur akhir kapal 2
                                'avg_distance_km': distance # Ambil jarak terakhir sebagai representasi
                            })
                            keys_to_remove.append(pair_key) # Anomali terdeteksi, reset interaksi ini
                    else:
                        # Gap terlalu besar, sesi interaksi terputus, reset durasi
                        keys_to_remove.append(pair_key)
                else:
                    # Pasangan ini tidak lagi berdekatan di snapshot ini, reset durasi
                    keys_to_remove.append(pair_key)

            # Hapus interaksi yang sudah selesai atau terputus
            for key in keys_to_remove:
                if key in current_interactions:
                    del current_interactions[key]
            
            # Tambahkan interaksi baru atau yang berlanjut
            for pair_key in current_snapshot_interactions:
                if pair_key not in current_interactions:
                    # Ambil data vessel1 dan vessel2 dari snapshot untuk mendapatkan lat/lon awal
                    v1_data = snapshot[snapshot['mmsi'] == pair_key[0]].iloc[0]
                    v2_data = snapshot[snapshot['mmsi'] == pair_key[1]].iloc[0]
                    current_interactions[pair_key] = {
                        'start_time': timestamp,
                        'last_seen': timestamp,
                        'current_duration_min': 0, # Durasi dihitung dari detik pertama
                        'start_lat': v1_data['lat'],
                        'start_lon': v1_data['lon'],
                        'start_lat_2': v2_data['lat'],
                        'start_lon_2': v2_data['lon'],
                    }
        
        # Pindah ke jendela waktu berikutnya
        current_window_start = window_end
        window_count += 1
        if window_count % 10 == 0:
            print(f"  Memproses jendela ke-{window_count} (hingga {current_window_start.strftime('%Y-%m-%d %H:%M:%S')})...")


    df_anomalies = pd.DataFrame(detected_anomalies)
    df_potential_interactions = pd.DataFrame(all_potential_interactions)

    print("\nDeteksi anomali selesai.")
    print(f"Total anomali terdeteksi: {len(df_anomalies)}")

    return df_anomalies, df_potential_interactions

# --- Jalankan deteksi ---
if __name__ == "__main__":
    file_path = "data/maritim_selat_sunda.pkl"
    
    anomalies_df, potential_interactions_df = detect_illegal_transhipment(file_path)

    if not anomalies_df.empty:
        print("\n--- Detail Anomali Terdeteksi ---")
        print(anomalies_df.head())
        # Anda bisa menyimpan hasilnya ke CSV atau Excel
        anomalies_df.to_csv("illegal_transhipment_anomalies.csv", index=False)
        print("\nAnomali disimpan ke 'illegal_transhipment_anomalies.csv'")
    else:
        print("\nTidak ada anomali illegal transhipment yang terdeteksi.")
    
    if not potential_interactions_df.empty:
        print("\n--- Contoh Potensi Interaksi (untuk debugging/analisis) ---")
        print(potential_interactions_df.head())
        # Simpan juga potensi interaksi untuk analisis lebih lanjut
        potential_interactions_df.to_csv("potential_vessel_interactions.csv", index=False)
        print("\nPotensi interaksi disimpan ke 'potential_vessel_interactions.csv'")