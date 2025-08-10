import pandas as pd

# Load data dari file .pkl
df = pd.read_pickle("data/maritim.pkl")

# Hitung jumlah nilai unik pada kolom mmsi (yang tidak null)
jumlah_kapal_unik = df['mmsi'].nunique()

print(f"Jumlah kapal unik berdasarkan MMSI: {jumlah_kapal_unik}")
