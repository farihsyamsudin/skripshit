import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('V4/scraped_vessel_type.csv')

# 2. FILTERING: Hapus UNKNOWN dan NOT_FOUND dari analisis
# Langkah ini penting untuk meminimalkan noise dari data statis AIS yang tidak lengkap [cite: 1203, 1206]
exclude_list = []
df_filtered = df[~df['vessel_type'].isin(exclude_list)].copy()

# 3. BIKIN TABEL LAMPIRAN FULL
tabel_full = df_filtered['vessel_type'].value_counts().reset_index()
tabel_full.columns = ['Jenis Kapal', 'Jumlah (Unit)']

# Simpan ke CSV untuk lampiran skripsi
tabel_full.to_csv('V4/tabel_lampiran_full.csv', index=False)
print("Berhasil: File 'V4/tabel_lampiran_full.csv' (Lampiran) telah dibuat.")

# 4. PERSIAPAN DATA GRAFIK (Top 5 + Others)
top_5 = tabel_full.head(5).copy()
others_count = tabel_full.iloc[5:]['Jumlah (Unit)'].sum()

# Gabungkan kategori sisa menjadi 'OTHERS'
others_row = pd.DataFrame([{'Jenis Kapal': 'OTHERS', 'Jumlah (Unit)': others_count}])
data_grafik = pd.concat([top_5, others_row], ignore_index=True)

# 5. BIKIN GRAFIK YANG RAPIH
plt.figure(figsize=(12, 7))
# Gunakan warna steelblue untuk data utama dan abu-abu untuk OTHERS
colors = ['#4682B4' if x != 'OTHERS' else '#A9A9A9' for x in data_grafik['Jenis Kapal']]
bars = plt.bar(data_grafik['Jenis Kapal'], data_grafik['Jumlah (Unit)'], color=colors, edgecolor='black', alpha=0.85)

# Tambahkan label jumlah di atas bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(data_grafik['Jumlah (Unit)']) * 0.01), 
             f'{int(yval)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Pengaturan visual agar siap masuk Bab 4
plt.title('Distribusi 5 Jenis Kapal Terbanyak di Perbatasan Batam-Singapura\n(Eksklusi Data Statis Tidak Lengkap)', 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Kategori Jenis Kapal', fontsize=11)
plt.ylabel('Jumlah Kapal (Unit)', fontsize=11)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Simpan dengan resolusi tinggi (300 DPI) agar tidak pecah saat diprint
plt.tight_layout()
plt.savefig('V4/grafik_top5_vessel.png', dpi=300)
print("Berhasil: File 'V4/grafik_top5_vessel.png' (Bab 4) telah dibuat.")

plt.show()