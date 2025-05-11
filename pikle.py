import pandas as pd

try:
    df = pd.read_pickle("data/maritim.pkl")
    print(df.info())  # Cek apakah datanya lengkap
    print(df.head())  # Lihat sample datanya
except Exception as e:
    print(f"Error baca Pickle: {e}")
