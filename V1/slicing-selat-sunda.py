import pandas as pd

# Load data utama
df = pd.read_pickle('data/maritim.pkl')

# Filter berdasarkan koordinat Selat Sunda
df_sunda = df[(df['lat'] >= -8.0) & (df['lat'] <= -4.5) &
              (df['lon'] >= 104.0) & (df['lon'] <= 107.0)]

# Simpan hasilnya ke file .pkl baru
df_sunda.to_pickle('data/maritim_selat_sunda_dua.pkl')

print(f"Data tersimpan. Jumlah entri: {len(df_sunda)}")
