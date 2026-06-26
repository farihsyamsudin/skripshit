import pandas as pd

# 1. Load data
df = pd.read_csv('V4/scraped_vessel_type.csv')

# 2. Hitung jumlah tiap jenis dan urutkan dari yang terbanyak
# value_counts() secara default sudah mengurutkan secara descending (terbanyak ke terkecil)
summary = df['vessel_type'].value_counts().reset_index()

# 3. Ganti nama kolom agar lebih informatif
summary.columns = ['Jenis', 'Jumlah']

# 4. Tampilkan hasil di terminal
print(summary.to_string(index=False))

# 5. Simpan hasilnya ke file CSV baru
summary.to_csv('V4/summary_vessel_sorted.csv', index=False)
print("\n✅ Ringkasan sudah disimpan ke 'V4/summary_vessel_sorted.csv'")