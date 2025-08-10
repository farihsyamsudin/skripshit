import pandas as pd

# Load data asli (tidak diubah)
df = pd.read_pickle("data/maritim_selat_sunda.pkl")

df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)].reset_index(drop=True)

# Ambil subset untuk test (contoh: 500.000 baris pertama)
df_test = df.head(500_000).copy()

# Simpan sebagai file baru untuk eksperimen
df_test.to_pickle("data/maritim_selat_sunda_500k.pkl")

print("Test data saved as 'maritim_selat_sunda_500k.pkl' with", len(df_test), "rows.")
