import pandas as pd

# 1. Load data AIS
df = pd.read_pickle('data/maritim_selat_sunda.pkl')

# 2. Load hasil scraped vessel_type
df_type = pd.read_csv('scraped_vessel_type.csv', dtype={'mmsi': str})

# 3. Normalisasi vessel_type
df_type['vessel_type'] = df_type['vessel_type'].replace(['ERROR', 'NOT_FOUND'], 'UNKNOWN')

# 4. Pastikan MMSI cocok (string)
df['mmsi'] = df['mmsi'].astype(str)

# 5. Merge dengan left join
df_merged = df.merge(df_type, on='mmsi', how='left')

# 6. Simpan ke file baru
df_merged.to_pickle('data/maritim_selat_sunda_with_type.pkl')

print("✅ Merge selesai. Semua 'ERROR' & 'NOT_FOUND' sudah jadi 'UNKNOWN'.")
print("💾 File disimpan ke: data/maritim_selat_sunda_with_type.pkl")
