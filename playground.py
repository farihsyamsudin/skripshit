import json
import pandas as pd

# Buka file JSON
with open("data/maritim.ais.json", "r") as f:
    data = json.load(f)  # Load semua data

# Fungsi untuk mengubah MongoDB Extended JSON ke format standar
def convert_mongodb_json(record):
    record["_id"] = record["_id"]["$oid"]  # Ubah _id dari BSON ke string
    record["created_at"] = record["created_at"]["$date"]  # Ubah tanggal ke string
    return record

# Konversi semua data
converted_data = [convert_mongodb_json(entry) for entry in data]

# Masukkan ke DataFrame
df = pd.DataFrame(converted_data)

# Simpan ke Pickle (biar ga perlu load JSON lagi nanti)
df.to_pickle("data/maritim.pkl")

print("Dataset berhasil disimpan sebagai Pickle (.pkl)")
