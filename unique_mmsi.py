import pandas as pd

# Load dataset AIS kamu
df = pd.read_pickle('data/maritim_selat_sunda.pkl')

# Ambil daftar MMSI unik (hapus null, sort, reset index)
mmsi_unique = df['mmsi'].dropna().astype(str).unique()
print(f"Jumlah kapal unik (MMSI): {len(mmsi_unique)}")

# Simpan ke CSV untuk scraping
pd.DataFrame(mmsi_unique, columns=["mmsi"]).to_csv("mmsi_list_unique.csv", index=False)
