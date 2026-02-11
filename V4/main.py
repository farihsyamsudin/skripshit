# HTR: python V4/main.py real_case

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import hashlib
import time
import argparse
import warnings

from folium.plugins import MarkerCluster
# Pastikan file anomaly_logic.py ada di folder yang sama
from anomaly_logic import detect_anomalies 

warnings.filterwarnings("ignore")

# ==============================================================================
# KONFIGURASI PELABUHAN (BATAM & SINGAPURA)
# ==============================================================================
PORTS = [
    # --- BATAM, INDONESIA ---
    {"name": "Batu Ampar (Cargo)", "lat": 1.1617, "lon": 104.0047},
    {"name": "Kabil (Citranusa/Oil)", "lat": 1.1108, "lon": 104.1403},
    {"name": "Sekupang (Ferry/Intl)", "lat": 1.1261, "lon": 103.9278},
    {"name": "Tanjung Uncang (Shipyard)", "lat": 1.0750, "lon": 103.9050},
    {"name": "Nongsa Pura", "lat": 1.1960, "lon": 104.0830},
    {"name": "Telaga Punggur", "lat": 1.0370, "lon": 104.1480},
    {"name": "Batam Centre", "lat": 1.1320, "lon": 104.0520},
    {"name": "Harbour Bay", "lat": 1.1550, "lon": 103.9950},
    
    # --- BINTAN (Sisi Timur) ---
    {"name": "Tanjung Uban (Oil)", "lat": 1.0713, "lon": 104.2209},
    
    # --- SINGAPURA (Filter Antrean Legal) ---
    {"name": "Jurong Port", "lat": 1.2604, "lon": 103.6888},
    {"name": "Pasir Panjang", "lat": 1.2761, "lon": 103.7914},
    {"name": "Keppel Terminal", "lat": 1.2600, "lon": 103.8300},
    {"name": "Brani Terminal", "lat": 1.2630, "lon": 103.8350},
    {"name": "Tanjong Pagar", "lat": 1.2670, "lon": 103.8450},
    {"name": "Marina South Pier", "lat": 1.2700, "lon": 103.8640},
    {"name": "Changi Naval Base", "lat": 1.3200, "lon": 104.0200},
    {"name": "Changi Cargo", "lat": 1.3500, "lon": 104.0300},
    {"name": "Tuas Mega Port", "lat": 1.2900, "lon": 103.6200},
    {"name": "Sembawang", "lat": 1.4550, "lon": 103.8250},

    # --- TAMBAHAN: JOHOR, MALAYSIA (Supaya Cluster 11 Hilang) ---
    {"name": "Tanjung Pelepas (PTP)", "lat": 1.3600, "lon": 103.5500},
    {"name": "Tanjung Bin (Power/Coal)", "lat": 1.3300, "lon": 103.5400},
    {"name": "Kukup Anchorage", "lat": 1.3200, "lon": 103.4500},
    {"name": "Johor Port (Pasir Gudang)", "lat": 1.4300, "lon": 103.9000},
    {"name": "Tanjung Langsat", "lat": 1.4500, "lon": 104.0100},
]

def get_color_hex(id_str):
    """Generate warna konsisten berdasarkan string ID."""
    hex_digest = hashlib.md5(str(id_str).encode()).hexdigest()
    r = int(hex_digest[0:2], 16) % 200 + 50
    g = int(hex_digest[2:4], 16) % 160
    b = int(hex_digest[4:6], 16) % 60
    return f"#{r:02x}{g:02x}{b:02x}"

# ==============================================================================
# VISUALISASI
# ==============================================================================

def visualize_anomalies(final_df, output_prefix):
    """Hanya visualisasi Transhipment (Tanpa Ghost Ship)."""
    
    print("🎨 Membuat Peta Sebaran Anomali...")
    
    if not final_df.empty:
        map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
    else:
        map_center = [1.20, 104.0] # Default Batam

    # Gunakan peta dasar terang agar titik terlihat jelas
    m = folium.Map(location=map_center, zoom_start=10, tiles='CartoDB positron')
    
    # Gunakan MarkerCluster agar peta tidak berat jika titiknya banyak
    marker_cluster = MarkerCluster(name="Transhipment Cluster").add_to(m)
    
    for _, row in final_df.iterrows():
        color = get_color_hex(f"{row['mmsi_1']}-{row['mmsi_2']}")
        
        # Format popup info
        popup_text = (
            f"<div style='width:200px'>"
            f"<b>⚠️ SUSPECTED TRANSHIPMENT</b><br><hr>"
            f"<b>Kapal 1:</b> {row['mmsi_1']}<br>"
            f"<b>Kapal 2:</b> {row['mmsi_2']}<br>"
            f"<b>Durasi:</b> {row['duration_min']} menit<br>"
            f"<b>Mulai:</b> {row['start_time']}<br>"
            f"<b>Selesai:</b> {row['end_time']}"
            f"</div>"
        )
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']], 
            radius=6, 
            color=color, 
            fill=True, 
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"MMSI: {row['mmsi_1']} & {row['mmsi_2']}"
        ).add_to(marker_cluster)
    
    # Tambahkan radius pelabuhan (Opsional, untuk validasi visual)
    for port in PORTS:
        folium.Circle(
            location=[port['lat'], port['lon']],
            radius=3000, # 3000 meter = 3 KM (sesuai threshold kode)
            color="gray",
            weight=1,
            fill=True,
            fill_opacity=0.1,
            tooltip=f"Zona Pelabuhan: {port['name']}"
        ).add_to(m)

    # Simpan Peta
    output_html = f"results/{output_prefix}_map.html"
    m.save(output_html)
    print(f"🗺️  Peta interaktif disimpan di: {output_html}")


# ==============================================================================
# MODE EKSEKUSI UTAMA (REAL CASE)
# ==============================================================================

def run_real_case_mode():
    print("--- Running in REAL CASE mode (Transhipment Only) ---")
    start = time.time()

    # --- 1. PARAMETER FINAL (BATAM) ---
    PROXIMITY_THRESHOLD_KM = 0.5    # Jarak antar kapal < 500m
    DURATION_THRESHOLD_MIN = 30     # Nempel minimal 30 menit
    CANDIDATE_DURATION_THRESHOLD_MIN = 20 # Kandidat minimal 20 menit
    
    SOG_THRESHOLD = 1.5             # Kecepatan < 1.5 knot (Hanyut/Diam)
    PORT_DISTANCE_THRESHOLD_KM = 10.0 # Minimal 8 KM dari pelabuhan terdaftar
    TIME_GAP_MINUTES = 30           # Toleransi data bolong 30 menit
    
    INPUT_FILE = 'V4/data/maritim_batam.pkl' # Pastikan path ini benar

    # --- 2. LOAD & CLEAN DATA ---
    print(f"📂 Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File {INPUT_FILE} tidak ditemukan.")
        return

    df = pd.read_pickle(INPUT_FILE)

    # Konversi tipe data
    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
    df['utc'] = pd.to_datetime(df['utc'])
    
    # Filter Wilayah (Kotak Batam-Singapura)
    df = df[(df['lat'] >= 1.0) & (df['lat'] <= 1.5) &
            (df['lon'] >= 103.5) & (df['lon'] <= 104.5)].reset_index(drop=True)
    
    # Buang data sampah
    before = len(df)
    df = df[(df['mmsi'] > 0) & (df['mmsi'] < 999999999)] 
    df = df.dropna(subset=['mmsi', 'lat', 'lon', 'utc'])
    print(f"🧹 Data Cleaning: Dibuang {before - len(df)} baris data sampah.")
    
    df = df.sort_values(by='utc')
    print(f"✅ Data siap: {len(df)} baris.")

    # --- 3. DETEKSI TRANSHIPMENT ---
    print("\n🔍 Memulai Analisis Deteksi Transhipment...")
    final_df, candidate_df = detect_anomalies(
        df, PROXIMITY_THRESHOLD_KM, DURATION_THRESHOLD_MIN, CANDIDATE_DURATION_THRESHOLD_MIN,
        SOG_THRESHOLD, PORT_DISTANCE_THRESHOLD_KM, TIME_GAP_MINUTES, PORTS
    )

    # --- 4. EXPORT HASIL ---
    os.makedirs("results", exist_ok=True)
    
    if not final_df.empty:
        print(f"\n✅ DITEMUKAN {len(final_df)} KEJADIAN TRANSHIPMENT!")
        # Simpan CSV
        csv_file = "results/transhipment_anomalies.csv"
        final_df.to_csv(csv_file, index=False)
        print(f"   --> Data disimpan di {csv_file}")
        
        # Buat Visualisasi
        visualize_anomalies(final_df, "transhipment_batam")
    else:
        print("\n❌ Tidak ditemukan anomali transhipment dengan parameter ini.")

    # Export Kandidat (Opsional, buat analisa manual)
    if not candidate_df.empty:
        candidate_df.to_csv("results/transhipment_candidates.csv", index=False)
        print(f"ℹ️  Juga ditemukan {len(candidate_df)} kandidat (durasi 20-30 menit).")

    end = time.time()
    print(f"\n⏱️ Waktu eksekusi: {round((end - start)/60, 2)} menit")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deteksi Illegal Transhipment")
    parser.add_argument('mode', choices=['real_case'], help="Mode: real_case")
    args = parser.parse_args()

    if args.mode == 'real_case':
        run_real_case_mode()

if __name__ == "__main__":
    main()