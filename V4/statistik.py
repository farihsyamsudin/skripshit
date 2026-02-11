import pandas as pd

# --- KONFIGURASI ---
# Pastikan ini mengarah ke file .pkl hasil convert kamu yang BARU (Batam)
INPUT_FILE = 'V4/data/maritim_batam.pkl' 

def hitung_statistik_data():
    print(f"📂 Memuat data dari: {INPUT_FILE}...")
    
    try:
        df = pd.read_pickle(INPUT_FILE)
    except FileNotFoundError:
        print("❌ File tidak ditemukan! Pastikan path-nya benar.")
        return

    # 1. Total Data Mentah (Baris)
    total_entri = len(df)
    
    # 2. Total Kapal Unik (MMSI)
    total_kapal_unik = df['mmsi'].nunique()
    
    # --- SIMULASI PEMBERSIHAN (Sesuai Logika Skripsi Kamu) ---
    # Di skripsi biasanya ada tahap "Cleaning". Kita hitung statistik setelah filter dasar.
    # Filter: Hapus data duplikat, speed tidak masuk akal (> 50 knot), atau koordinat error (0,0)
    
    df_bersih = df.copy()
    
    # a. Hapus duplikat
    df_bersih = df_bersih.drop_duplicates(subset=['mmsi', 'created_at'])
    
    # b. Hapus data dengan kecepatan error (misal > 60 knot itu mustahil buat kapal dagang)
    #    atau SOG < 0 (error alat)
    df_bersih = df_bersih[(df_bersih['sog'] >= 0) & (df_bersih['sog'] <= 60)]
    
    # c. Hapus koordinat 0,0 (biasanya error GPS)
    df_bersih = df_bersih[(df_bersih['lat'] != 0) & (df_bersih['lon'] != 0)]

    # 3. Total Data Setelah Pembersihan
    total_entri_bersih = len(df_bersih)
    
    # 4. Total Kapal Unik Setelah Pembersihan
    total_kapal_unik_bersih = df_bersih['mmsi'].nunique()

    # --- TAMPILKAN HASIL ---
    print("\n" + "="*40)
    print("📊 STATISTIK DATA AIS (UNTUK BAB 3)")
    print("="*40)
    print(f"1. Total Data Mentah (Entri)      : {total_entri:,}".replace(',', '.'))
    print(f"2. Total Kapal Unik (Awal)        : {total_kapal_unik:,}".replace(',', '.'))
    print("-" * 40)
    print(f"3. Data Setelah Pembersihan       : {total_entri_bersih:,}".replace(',', '.'))
    print(f"4. Kapal Unik Setelah Pembersihan : {total_kapal_unik_bersih:,}".replace(',', '.'))
    print("="*40)
    
    # --- CONTOH PARAGRAF JADI ---
    print("\n📝 [DRAFT PARAGRAF BARU]")
    print(f'"Data dikumpulkan melalui observasi dan permintaan data dari Laboratorium AIS UPI Serang, '
          f'dalam bentuk data historis pergerakan kapal di wilayah perairan perbatasan Batam-Singapura. '
          f'Data terdiri atas elemen-elemen penting seperti MMSI, Posisi Geografis, Waktu, dan Kecepatan. '
          f'Total data mentah yang dikumpulkan sebanyak {total_entri:,} entri dari {total_kapal_unik:,} kapal unik, '
          f'yang kemudian dibersihkan dan disortir menjadi {total_entri_bersih:,} entri dari {total_kapal_unik_bersih:,} MMSI unik."'.replace(',', '.'))

if __name__ == "__main__":
    hitung_statistik_data()