import pandas as pd
import numpy as np
from datetime import timedelta
from geopy.distance import geodesic # Menggunakan geodesic untuk perhitungan jarak akurat
import folium
from folium.plugins import MarkerCluster

# --- Parameter aturan (Bisa diubah) ---
PROXIMITY_THRESHOLD_KM = 0.2  # 200 meter
DURATION_THRESHOLD_MIN = 30  # minimal 30 menit
SOG_THRESHOLD = 0.5  # kecepatan < 0.5 knot (kapal hampir diam)
PORT_DISTANCE_THRESHOLD_KM = 30.0  # minimal 10 km dari pelabuhan

# Daftar pelabuhan
ports = [
    {"name": "Pelabuhan Merak", "lat": -5.8933, "lon": 106.0086},
    {"name": "Pelabuhan Ciwandan", "lat": -5.9525, "lon": 106.0358},
    {"name": "Pelabuhan Bojonegara", "lat": -5.8995, "lon": 106.0657},
    {"name": "Pelabuhan Bakauheni", "lat": -5.8711, "lon": 105.7421},
    {"name": "Pelabuhan Panjang", "lat": -5.4558, "lon": 105.3134},
    {"name": "Pelabuhan Ciwandan 2", "lat": -6.02147, "lon": 105.95485},
]

# Path ke file data
DATA_PATH = "data/maritim_selat_sunda_with_type.pkl"
OUTPUT_CSV_PATH = "illegal_transhipment_anomalies_dua.csv"
OUTPUT_MAP_PATH = "illegal_transhipment_map_dua.html"

print("Memulai proses deteksi anomali illegal transhipment...")

# --- 1. Memuat Data ---
try:
    df = pd.read_pickle(DATA_PATH)
    print(f"Data berhasil dimuat dari {DATA_PATH}. Jumlah baris: {len(df)}")
except FileNotFoundError:
    print(f"ERROR: File data tidak ditemukan di {DATA_PATH}. Pastikan path sudah benar.")
    exit()
except Exception as e:
    print(f"ERROR saat memuat data: {e}")
    exit()

# --- 2. Preprocessing Data ---
print("Melakukan preprocessing data...")
# Filter awal berdasarkan batas geografis
initial_rows = len(df)
df = df[
    (df['lat'].between(-6.5, -5.5)) & # Menggunakan .between() bisa lebih rapi
    (df['lon'].between(105.0, 106.0))
].copy() # Gunakan .copy() untuk menghindari SettingWithCopyWarning
print(f"Data setelah filter geografis: {len(df)} baris (filter {initial_rows - len(df)} baris).")

# Konversi 'created_at' ke datetime
# Pastikan ini mengkonversi dengan benar, bahkan jika ada objek datetime dari MongoDB
df['created_at'] = pd.to_datetime(df['created_at'])

# Urutkan data berdasarkan waktu untuk memudahkan deteksi durasi
df = df.sort_values(by='created_at').reset_index(drop=True)
print("Data berhasil diurutkan berdasarkan waktu.")

# --- 3. Implementasi Rule-Based Filtering ---

# Rule 1: Kecepatan < 0.5 knot (SOG_THRESHOLD)
# Pastikan 'sog' adalah numerik dan tangani nilai NaN jika ada
# Menggunakan .dropna() dan .astype(float) untuk robustness
df['sog'] = pd.to_numeric(df['sog'], errors='coerce') # Ubah ke numerik, non-numerik jadi NaN
df_slow = df[df['sog'] < SOG_THRESHOLD].copy()
df_slow.dropna(subset=['sog'], inplace=True) # Hapus baris dengan sog NaN setelah konversi
print(f"Jumlah titik data dengan kecepatan rendah (< {SOG_THRESHOLD} knot): {len(df_slow)}")

# Rule 4: Kapal jauh dari pelabuhan (PORT_DISTANCE_THRESHOLD_KM)
print("Menghitung jarak dari pelabuhan...")
def is_far_from_ports(row, ports, threshold_km):
    point = (row['lat'], row['lon'])
    for port in ports:
        port_point = (port['lat'], port['lon'])
        if geodesic(point, port_point).km < threshold_km:
            return False # Dekat dengan setidaknya satu pelabuhan
    return True # Jauh dari semua pelabuhan

# Menerapkan fungsi ini ke DataFrame bisa lambat untuk data besar.
# Jika ini terlalu lambat, pertimbangkan untuk menggunakan pendekatan spatial indexing (KDTree)
# untuk pelabuhan jika memungkinkan, atau pre-calculate untuk area grid.
# Namun, karena jumlah pelabuhan sedikit, apply mungkin masih OK.
df_far_from_ports = df_slow[
    df_slow.apply(lambda row: is_far_from_ports(row, ports, PORT_DISTANCE_THRESHOLD_KM), axis=1)
].copy()
print(f"Jumlah titik data yang kecepatan rendah DAN jauh dari pelabuhan: {len(df_far_from_ports)}")

# Rule 2 & 3: 2 Kapal berada dalam jarak berdekatan < 200 meter DAN dalam waktu yang cukup lama (30 menit)
anomalies = []
processed_pairs = set() # Untuk melacak pasangan yang sudah diproses agar tidak duplikat

# Mengelompokkan data berdasarkan interval waktu untuk efisiensi
TIME_GROUP_INTERVAL = timedelta(minutes=5) # Sesuaikan jika perlu

# Buat kolom 'time_group'
df_far_from_ports['time_group'] = df_far_from_ports['created_at'].dt.floor(TIME_GROUP_INTERVAL)

print(f"Mulai deteksi kedekatan antar kapal dalam kelompok waktu...")
unique_time_groups = df_far_from_ports['time_group'].unique()
total_groups = len(unique_time_groups)

for i, time_group in enumerate(unique_time_groups):
    if (i + 1) % 100 == 0:
        print(f"Memproses grup waktu {i + 1}/{total_groups}...")

    # Ambil data untuk jendela waktu yang relevan
    # Ini harus mencakup time_group saat ini dan durasi ke depan
    time_window_start = time_group
    time_window_end = time_group + timedelta(minutes=DURATION_THRESHOLD_MIN)

    # Filter data yang masuk dalam jendela waktu ini dan sudah memenuhi rule kecepatan & jauh dari pelabuhan
    # Ini adalah subset data yang akan kita cari pasangan di dalamnya
    relevant_data_window = df_far_from_ports[
        (df_far_from_ports['created_at'] >= time_window_start) &
        (df_far_from_ports['created_at'] < time_window_end)
    ].copy()

    if len(relevant_data_window) < 2:
        continue # Tidak mungkin ada pasangan kapal jika kurang dari 2 entri

    # Iterasi melalui setiap titik data di relevant_data_window
    # Kita bandingkan setiap titik dengan titik-titik lain dalam jendela waktu yang sama
    # Ini adalah pendekatan O(N^2) dalam subset data, tapi diharapkan subsetnya kecil
    for idx1, row1 in relevant_data_window.iterrows():
        point1 = (row1['lat'], row1['lon'])
        time1 = row1['created_at']

        # Loop kedua dimulai dari idx1 + 1 untuk menghindari duplikasi (A-B vs B-A) dan self-comparison
        for idx2, row2 in relevant_data_window.loc[idx1+1:].iterrows():
            point2 = (row2['lat'], row2['lon'])
            time2 = row2['created_at']

            distance_km = geodesic(point1, point2).km

            if distance_km < PROXIMITY_THRESHOLD_KM:
                time_difference = abs(time1 - time2)
                
                # Check if the time difference between the two observations is within the duration threshold
                if time_difference <= timedelta(minutes=DURATION_THRESHOLD_MIN):
                    
                    # Pastikan _id adalah string. Jika di .pkl _id masih dictionary,
                    # Anda perlu menambahkan penanganan seperti:
                    # id1 = row1['_id']['$oid'] if isinstance(row1['_id'], dict) else row1['_id']
                    # id2 = row2['_id']['$oid'] if isinstance(row2['_id'], dict) else row2['_id']
                    # Namun, berdasarkan error yang Anda alami, kemungkinan _id sudah string.
                    id1 = row1['_id']
                    id2 = row2['_id']

                    pair_key = tuple(sorted((str(id1), str(id2)))) # Pastikan dikonversi ke string untuk hashing
                    
                    if pair_key not in processed_pairs:
                        anomalies.append({
                            'timestamp1': time1,
                            'lat1': row1['lat'],
                            'lon1': row1['lon'],
                            'sog1': row1['sog'],
                            'timestamp2': time2,
                            'lat2': row2['lat'],
                            'lon2': row2['lon'],
                            'sog2': row2['sog'],
                            'distance_km': distance_km,
                            'time_difference_minutes': time_difference.total_seconds() / 60
                        })
                        processed_pairs.add(pair_key)

print(f"Total anomali yang terdeteksi: {len(anomalies)}")

# --- 4. Menyimpan Hasil ke CSV ---
if anomalies:
    anomalies_df = pd.DataFrame(anomalies)
    anomalies_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Data anomali disimpan ke {OUTPUT_CSV_PATH}")
else:
    print("Tidak ada anomali yang terdeteksi.")

# --- 5. Membuat Peta Folium ---
print("Membuat peta Folium...")
# Tentukan titik pusat peta (rata-rata dari data anomali atau area Selat Sunda)
if anomalies:
    avg_lat_anom = np.mean([a['lat1'] for a in anomalies] + [a['lat2'] for a in anomalies])
    avg_lon_anom = np.mean([a['lon1'] for a in anomalies] + [a['lon2'] for a in anomalies])
    map_center = [avg_lat_anom, avg_lon_anom]
else:
    # Fallback ke pusat area filter jika tidak ada anomali
    map_center = [-6.0, 105.5] # Kira-kira tengah Selat Sunda sesuai filter

m = folium.Map(location=map_center, zoom_start=9)

# Tambahkan marker untuk pelabuhan
port_marker_cluster = MarkerCluster().add_to(m)
for port in ports:
    folium.Marker(
        location=[port['lat'], port['lon']],
        popup=f"<b>Pelabuhan:</b> {port['name']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(port_marker_cluster)

# Tambahkan marker untuk anomali
if anomalies:
    anom_marker_cluster = MarkerCluster().add_to(m)
    for anom in anomalies:
        # Ambil titik tengah dari kedua observasi anomali untuk menempatkan marker
        avg_lat = (anom['lat1'] + anom['lat2']) / 2
        avg_lon = (anom['lon1'] + anom['lon2']) / 2
        
        popup_html = f"""
        <b>Anomali Transhipment Terdeteksi!</b><br>
        <hr>
        <b>Observasi 1:</b><br>
        Waktu: {anom['timestamp1']}<br>
        Lokasi: ({anom['lat1']:.4f}, {anom['lon1']:.4f})<br>
        SOG: {anom['sog1']:.2f} knot<br>
        <hr>
        <b>Observasi 2:</b><br>
        Waktu: {anom['timestamp2']}<br>
        Lokasi: ({anom['lat2']:.4f}, {anom['lon2']:.4f})<br>
        SOG: {anom['sog2']:.2f} knot<br>
        <hr>
        Jarak antar observasi: {anom['distance_km']:.2f} km<br>
        Perbedaan waktu: {anom['time_difference_minutes']:.2f} menit
        """
        folium.Marker(
            location=[avg_lat, avg_lon],
            popup=popup_html,
            icon=folium.Icon(color='red', icon='warning-sign')
        ).add_to(anom_marker_cluster)

m.save(OUTPUT_MAP_PATH)
print(f"Peta anomali disimpan ke {OUTPUT_MAP_PATH}")

print("Proses selesai.")