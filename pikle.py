import pandas as pd

try:
    # 1) Baca file
    df = pd.read_pickle("data/maritim_selat_sunda.pkl")

    # 2) Info + contoh baris
    print(df.info(show_counts=True, memory_usage='deep'))  # lebih jelas
    print(df.head())

    # 3) Cek kolom aistype dan tampilkan 30 kode pertama yang unik
    if 'aistype' in df.columns:
        unique_codes = df['aistype'].dropna().unique()   # buang NaN dulu
        print("First 100 unique aistype codes:", unique_codes[:100])
    else:
        print("[!] Kolom 'aistype' tidak ditemukan.")

except FileNotFoundError:
    print("File tidak ditemukan. Pastikan path‑nya benar.")
except Exception as e:
    print(f"Error baca Pickle: {e}")
