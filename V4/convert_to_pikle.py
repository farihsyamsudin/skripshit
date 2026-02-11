import pandas as pd
import os

# --- KONFIGURASI NAMA FILE (Sesuaikan dengan path kamu) ---
INPUT_JSON = 'V4/data/maritim-batam.ais.json'  
OUTPUT_PKL = 'V4/data/maritim_batam.pkl' 

def convert_json_to_pickle():
    print(f"🔄 Membaca file {INPUT_JSON}...")
    
    try:
        # Coba baca format standar JSON Array
        df = pd.read_json(INPUT_JSON)
    except ValueError:
        # Kalau gagal, coba baca format line-delimited
        print("⚠️ Format array gagal, mencoba format lines=True...")
        df = pd.read_json(INPUT_JSON, lines=True)

    print(f"✅ Data berhasil dimuat! Total baris: {len(df)}")
    
    # --- [FIX] MEMBERSIHKAN FORMAT TANGGAL MONGODB ---
    print("🛠️ Memperbaiki format tanggal MongoDB ({$date: ...})...")
    
    def fix_mongo_date(x):
        # Jika datanya berbentuk dict {'$date': '...'}, ambil isinya saja
        if isinstance(x, dict) and '$date' in x:
            return x['$date']
        return x

    if 'created_at' in df.columns:
        # 1. Kupas dictionary-nya dulu
        df['created_at'] = df['created_at'].apply(fix_mongo_date)
        # 2. Baru convert ke datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        # 3. Copy ke kolom 'utc' (dibutuhkan main.py)
        df['utc'] = df['created_at']
    
    # --- PEMBERSIHAN LAINNYA ---
    print("🧹 Membersihkan tipe data angka...")
    
    # Pastikan angka adalah angka (Float/Int)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
    # Int64 biar aman kalau ada data kosong
    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce').astype('Int64') 

    # Buang data kosong/rusak
    df = df.dropna(subset=['lat', 'lon', 'created_at', 'mmsi'])
    
    # Sorting berdasarkan waktu (WAJIB buat analisis pergerakan)
    df = df.sort_values(by=['mmsi', 'created_at']).reset_index(drop=True)

    # --- SIMPAN KE PICKLE ---
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    df.to_pickle(OUTPUT_PKL)
    
    print(f"🎉 SUKSES! Data tersimpan di: {OUTPUT_PKL}")
    print(f"📊 Info Data:")
    print(df.info())
    print("-" * 30)
    print(df.head())

if __name__ == "__main__":
    convert_json_to_pickle()