# HTR: python V4/main.py real_case

import os
import pandas as pd
import numpy as np
import folium
import time
import argparse
import warnings

# Hapus import MarkerCluster karena user minta tidak di-grouping
# from folium.plugins import MarkerCluster 
from folium import Element

# Pastikan file anomaly_logic.py ada di folder yang sama
from anomaly_logic import detect_anomalies 

warnings.filterwarnings("ignore")

# ==============================================================================
# KONFIGURASI PELABUHAN LENGKAP (BATAM, SG, JOHOR)
# ==============================================================================
# Filter ini krusial untuk membuang antrean legal di dermaga.
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
    
    # --- SINGAPURA ---
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

    # --- JOHOR, MALAYSIA ---
    {"name": "Tanjung Pelepas (PTP)", "lat": 1.3600, "lon": 103.5500},
    {"name": "Tanjung Bin (Power/Coal)", "lat": 1.3300, "lon": 103.5400},
    {"name": "Kukup Anchorage", "lat": 1.3200, "lon": 103.4500},
    {"name": "Johor Port (Pasir Gudang)", "lat": 1.4300, "lon": 103.9000},
    {"name": "Tanjung Langsat", "lat": 1.4500, "lon": 104.0100},
]

# ==============================================================================
# VISUALISASI (TANPA CLUSTERING + LEGENDA)
# ==============================================================================

def visualize_anomalies(final_df, output_prefix, buffer_km):
    """Visualisasi Peta Transhipment Bersih & Jelas."""
    
    print("🎨 Membuat Peta Sebaran Anomali (High Detail)...")
    
    if not final_df.empty:
        map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
    else:
        map_center = [1.25, 103.9] 

    # Base Map: CartoDB Positron (Terang & Bersih)
    m = folium.Map(location=map_center, zoom_start=11, tiles='CartoDB positron')
    
    # 1. GAMBAR RADIUS PELABUHAN (ZONA AMAN)
    # Digambar duluan biar ada di lapisan bawah
    vis_radius = buffer_km * 1000 # convert km to meters
    
    for port in PORTS:
        folium.Circle(
            location=[port['lat'], port['lon']],
            radius=vis_radius, 
            color="#999999",      # Abu-abu
            weight=1,
            fill=True,
            fill_color="#cccccc",
            fill_opacity=0.3,     # Transparan
            tooltip=f"Zona Aman Pelabuhan: {port['name']} ({buffer_km} km)"
        ).add_to(m)

    # 2. GAMBAR TITIK ANOMALI (MERAH MENYALA)
    # Tidak pakai Cluster, langsung add_to(m)
    for _, row in final_df.iterrows():
        
        # Isi Popup HTML yang rapi
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 220px;">
            <h4 style="margin: 0; color: #d9534f;">⚠️ SUSPECTED ACTIVITY</h4>
            <hr style="margin: 5px 0;">
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>MMSI A:</b></td><td>{row['mmsi_1']}</td></tr>
                <tr><td><b>MMSI B:</b></td><td>{row['mmsi_2']}</td></tr>
                <tr><td><b>Durasi:</b></td><td>{row['duration_min']} menit</td></tr>
                <tr><td><b>Waktu:</b></td><td>{row['start_time']}</td></tr>
            </table>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']], 
            radius=5,           # Ukuran titik pas (tidak terlalu kecil/besar)
            color='#ff0000',    # Outline Merah
            weight=1,
            fill=True,
            fill_color='#ff0000', # Isi Merah Solid
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Suspect: {row['mmsi_1']} & {row['mmsi_2']}"
        ).add_to(m)

    # 3. TAMBAHKAN LEGENDA (HTML OVERLAY)
    legend_html = f'''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: 110px; 
     border:2px solid grey; z-index:9999; font-size:13px;
     background-color:white; opacity:0.95; padding: 10px;
     border-radius: 5px; box-shadow: 3px 3px 5px #888888;
     font-family: Arial, sans-serif;
     ">
     <b>Keterangan Peta</b><br>
     <div style="margin-top:8px;">
       <span style="height: 10px; width: 10px; background-color: #ff0000; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
       <b>Indikasi Transhipment</b><br>
       <span style="font-size:11px; color:#555; margin-left:22px;">(Kapal nempel di tengah laut)</span>
     </div>
     <div style="margin-top:5px;">
       <span style="height: 10px; width: 10px; background-color: #cccccc; border: 1px solid #999; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
       <b>Zona Buffer Pelabuhan</b><br>
       <span style="font-size:11px; color:#555; margin-left:22px;">(Area Filter {buffer_km} KM)</span>
     </div>
     </div>
     '''
    m.get_root().html.add_child(Element(legend_html))

    # Simpan
    output_html = f"results/{output_prefix}_map.html"
    m.save(output_html)
    print(f"🗺️  Peta interaktif disimpan di: {output_html}")


# ==============================================================================
# MODE EKSEKUSI UTAMA
# ==============================================================================

def run_real_case_mode():
    print("--- Running in REAL CASE mode (Visualisasi Fix) ---")
    start = time.time()

    # --- PARAMETER ---
    PROXIMITY_THRESHOLD_KM = 0.5    
    DURATION_THRESHOLD_MIN = 30     
    CANDIDATE_DURATION_THRESHOLD_MIN = 20
    SOG_THRESHOLD = 1.5             
    PORT_DISTANCE_THRESHOLD_KM = 10.0 # Filter 10 KM (Cukup untuk buang kapal antre)
    TIME_GAP_MINUTES = 30           
    
    INPUT_FILE = 'V4/data/maritim_batam.pkl'

    # --- LOAD DATA ---
    print(f"📂 Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File {INPUT_FILE} tidak ditemukan.")
        return

    df = pd.read_pickle(INPUT_FILE)

    # Convert Data Types
    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
    df['utc'] = pd.to_datetime(df['utc'])
    
    # Filter Area
    df = df[(df['lat'] >= 1.0) & (df['lat'] <= 1.5) &
            (df['lon'] >= 103.5) & (df['lon'] <= 104.5)].reset_index(drop=True)
    
    # Clean Data
    df = df[(df['mmsi'] > 0) & (df['mmsi'] < 999999999)] 
    df = df.dropna(subset=['mmsi', 'lat', 'lon', 'utc'])
    df = df.sort_values(by='utc')
    
    print(f"✅ Data siap: {len(df)} baris.")

    # --- DETEKSI ---
    print("\n🔍 Memulai Analisis Deteksi Transhipment...")
    final_df, candidate_df = detect_anomalies(
        df, PROXIMITY_THRESHOLD_KM, DURATION_THRESHOLD_MIN, CANDIDATE_DURATION_THRESHOLD_MIN,
        SOG_THRESHOLD, PORT_DISTANCE_THRESHOLD_KM, TIME_GAP_MINUTES, PORTS
    )

    # --- EXPORT & VISUALISASI ---
    os.makedirs("results", exist_ok=True)
    
    if not final_df.empty:
        print(f"\n✅ DITEMUKAN {len(final_df)} KEJADIAN TRANSHIPMENT!")
        
        # Export CSV Format Cantik
        export_df = final_df.copy()
        export_df['start_time'] = export_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        export_df['end_time'] = export_df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cols = ['mmsi_1', 'mmsi_2', 'start_time', 'end_time', 'duration_min', 'lat', 'lon']
        export_df[cols].to_csv("results/transhipment_anomalies_formatted.csv", index=False)
        
        # VISUALISASI TANPA CLUSTERING
        visualize_anomalies(final_df, "transhipment_batam", PORT_DISTANCE_THRESHOLD_KM)
    else:
        print("\n❌ Tidak ditemukan anomali transhipment (mungkin tertutup radius pelabuhan).")
        # Opsional: Jika kosong, coba turunkan radius pelabuhan manual di script
        print("Saran: Coba turunkan PORT_DISTANCE_THRESHOLD_KM jika hasil terlalu sedikit.")

    end = time.time()
    print(f"\n⏱️ Waktu eksekusi: {round((end - start)/60, 2)} menit")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['real_case'])
    args = parser.parse_args()
    if args.mode == 'real_case':
        run_real_case_mode()

if __name__ == "__main__":
    main()