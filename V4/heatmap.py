import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.element import Template, MacroElement # Library bawaan folium untuk elemen HTML
import os

# --- KONFIGURASI ---
INPUT_FILE = 'V4/data/maritim_batam.pkl'
OUTPUT_HTML = 'V4/res/output_heatmap_batam_light.html'

def generate_heatmap_pro():
    print(f"📂 Memuat data dari: {INPUT_FILE}...")
    try:
        df = pd.read_pickle(INPUT_FILE)
    except FileNotFoundError:
        print("❌ File .pkl tidak ditemukan! Pastikan path benar.")
        return

    # 1. Sampling Data (Biar Ringan & Tidak Hang)
    # Ambil sampel cukup banyak biar polanya halus
    MAX_POINTS = 60000 
    if len(df) > MAX_POINTS:
        print(f"⚠️ Data terlalu besar, mengambil sampel {MAX_POINTS} titik...")
        df_vis = df.sample(n=MAX_POINTS, random_state=42)
    else:
        df_vis = df

    # Data untuk Heatmap
    heat_data = df_vis[['lat', 'lon']].values.tolist()

    print("🗺️ Sedang merender peta...")

    # 2. Buat Base Map (TEMA TERANG/PROFESIONAL)
    # 'CartoDB positron' adalah standar emas untuk visualisasi data geospatial di jurnal.
    # Bersih, minimalis, dan kontras dengan warna heatmap.
    m = folium.Map(
        location=[1.20, 103.90], # Sedikit digeser biar center di perbatasan
        zoom_start=11,
        tiles='CartoDB positron' 
    )

    # 3. Konfigurasi Warna Heatmap
    # Kita definisikan warnanya biar sinkron sama legenda
    gradient_map = {
        0.2: '#0000FF', # Biru (Rendah)
        0.4: '#00FF00', # Hijau
        0.6: '#FFFF00', # Kuning
        0.8: '#FFA500', # Oranye
        1.0: '#FF0000'  # Merah (Tinggi)
    }

    HeatMap(
        heat_data,
        name='Densitas Lalu Lintas',
        min_opacity=0.3,
        max_zoom=13,
        radius=14,       # Radius disesuaikan biar "buntut" heatmapnya nyatu
        blur=20,         # Blur diperhalus
        gradient=gradient_map
    ).add_to(m)

    # 4. MEMBUAT LEGENDA KUSTOM (HTML Floating Box)
    # Ini trik untuk nampilin legenda di pojok kanan bawah
    legend_html = '''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 180px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity:0.9; padding: 10px;
     border-radius: 5px; box-shadow: 3px 3px 5px #888888;
     ">
     <b>Intensitas Pergerakan</b><br>
     <div style="background: linear-gradient(to right, blue, lime, yellow, orange, red); height: 15px; width: 100%; margin-top:5px; margin-bottom:5px; border:1px solid #ccc;"></div>
     <div style="display: flex; justify-content: space-between; font-size: 12px;">
        <span>Rendah<br>(Sepi)</span>
        <span>Tinggi<br>(Padat)</span>
     </div>
     </div>
     '''
    
    # Masukkan legenda ke dalam peta
    m.get_root().html.add_child(folium.Element(legend_html))

    # 5. Simpan
    m.save(OUTPUT_HTML)
    print(f"✅ Peta Heatmap TERANG berhasil disimpan: {OUTPUT_HTML}")
    print("👉 Buka file, lalu Screenshot untuk Bab 4!")

if __name__ == "__main__":
    generate_heatmap_pro()