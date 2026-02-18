import pandas as pd
import folium
from folium import Element
import json
import os
from dateutil import parser

# ================= CONFIGURATION =================
# Pastikan nama file sesuai dengan hasil export Compass kamu
INPUT_JSON = "V4/data/study-case.ais.json"
OUTPUT_HTML = "results/peta_jalur_lengkap_mongo_legend.html"

# Lokasi Kejadian (Untuk Center Peta & Highlight)
INCIDENT_COORDS = [1.3077345, 104.11867]

# Definisi Kapal
MMSI_A = 352800000
MMSI_B = 538002666
names = {MMSI_A: 'Kapal A', MMSI_B: 'Kapal B'}
colors = {MMSI_A: 'blue', MMSI_B: 'red'}

def load_compass_json(filepath):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            f.seek(0)
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
            return data

def visualize_mongo_data():
    print(f"🗺️  Membaca file ekspor Compass: {INPUT_JSON}...")

    if not os.path.exists(INPUT_JSON):
        print("❌ File JSON tidak ditemukan!")
        return

    data = load_compass_json(INPUT_JSON)
    df = pd.DataFrame(data)

    if df.empty:
        print("❌ Data kosong!")
        return

    print("⏳ Sedang memproses format waktu...")
    
    def parse_mongo_date(x):
        try:
            if isinstance(x, dict) and '$date' in x:
                return pd.to_datetime(x['$date'])
            return pd.to_datetime(x)
        except:
            return pd.NaT

    df['time_obj'] = df['created_at'].apply(parse_mongo_date)
    df = df.dropna(subset=['time_obj', 'lat', 'lon'])
    df = df.sort_values('time_obj')

    print(f"✅ Data Loaded: {len(df)} titik dari MongoDB Raw.")

    # Setup Peta
    m = folium.Map(location=INCIDENT_COORDS, zoom_start=7, tiles='CartoDB positron')

    # Gambar Jalur
    mmsi_list = df['mmsi'].unique()
    
    for mmsi_str in mmsi_list:
        # Pastikan tipe data mmsi konsisten (int) untuk lookup dictionary
        try:
            mmsi_int = int(mmsi_str)
        except:
            continue

        track = df[df['mmsi'] == mmsi_str]
        
        # Gambar Garis
        points = track[['lat', 'lon']].values.tolist()
        color = colors.get(mmsi_int, 'black')
        name = names.get(mmsi_int, str(mmsi_int))

        # Polyline (Garis Jalur)
        folium.PolyLine(
            points, color=color, weight=2.5, opacity=0.8, 
            tooltip=f"Jalur {name}"
        ).add_to(m)

        # Marker Awal (Start)
        start_pt = track.iloc[0]
        folium.Marker(
            [start_pt['lat'], start_pt['lon']],
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            tooltip=f"<b>Awal Terdeteksi ({name})</b><br>{start_pt['time_obj'].strftime('%Y-%m-%d %H:%M')}<br>Lat/Lon: {start_pt['lat']:.3f}, {start_pt['lon']:.3f}"
        ).add_to(m)

        # Marker Akhir (End)
        end_pt = track.iloc[-1]
        folium.Marker(
            [end_pt['lat'], end_pt['lon']],
            icon=folium.Icon(color='black', icon='stop', prefix='fa'),
            tooltip=f"<b>Akhir Terdeteksi ({name})</b><br>{end_pt['time_obj'].strftime('%Y-%m-%d %H:%M')}"
        ).add_to(m)

    # Highlight Area Kejadian
    folium.Marker(
        INCIDENT_COORDS,
        icon=folium.Icon(color='orange', icon='exclamation-triangle', prefix='fa'),
        tooltip="<b>LOKASI TRANSHIPMENT (51 Menit)</b>"
    ).add_to(m)

    # ================= TAMBAHAN LEGENDA =================
    legend_html = f'''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 250px; height: 200px; 
     border:2px solid grey; z-index:9999; font-size:12px;
     background-color:white; opacity:0.95; padding: 10px;
     border-radius: 5px; font-family: Arial, sans-serif;
     box-shadow: 3px 3px 5px #888888;
     ">
     <b>Legenda Pergerakan (Data Mentah)</b><br><hr>
     
     <div style="margin-bottom: 5px;">
       <i class="fa fa-circle" style="color:blue"></i> 
       <b>{names[MMSI_A]}</b> <br>
       <span style="color:grey;">(MMSI: {MMSI_A})</span>
     </div>
     
     <div style="margin-bottom: 10px;">
       <i class="fa fa-circle" style="color:red"></i> 
       <b>{names[MMSI_B]}</b> <br>
       <span style="color:grey;">(MMSI: {MMSI_B})</span>
     </div>

     <i class="fa fa-play-circle" style="color:green"></i> Titik Awal Terdeteksi<br>
     <i class="fa fa-stop-circle" style="color:black"></i> Titik Akhir Terdeteksi<br>
     <br>
     <i class="fa fa-exclamation-triangle" style="color:orange"></i> <b>Lokasi Transhipment</b>
     </div>
     '''
    m.get_root().html.add_child(Element(legend_html))
    # ====================================================

    # Simpan
    m.fit_bounds(m.get_bounds())
    m.save(OUTPUT_HTML)
    print(f"✅ Peta DENGAN LEGENDA berhasil disimpan di: {OUTPUT_HTML}")
    print("👉 Silakan buka file HTML-nya, dijamin lebih jelas!")

if __name__ == "__main__":
    visualize_mongo_data()