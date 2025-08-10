import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_anomali = pd.read_csv("output_tabel_anomali_FIX.csv")
df_total = pd.read_pickle("data/maritim_selat_sunda.pkl")

# Gabungkan kedua kolom mmsi dan hitung kapal unik
kapal_anomali = pd.concat([df_anomali["mmsi_1"], df_anomali["mmsi_2"]]).unique()
jumlah_kapal_anomali = len(kapal_anomali)

# Kapal unik dari data utama
kapal_normal = df_total["mmsi"].unique()
jumlah_kapal_normal = len(kapal_normal)

# Hitung kapal normal yang tidak terlibat anomali
jumlah_kapal_non_anomali = jumlah_kapal_normal - jumlah_kapal_anomali

# Plot grafik
labels = ['Terdeteksi Anomali', 'Normal']
values = [jumlah_kapal_anomali, jumlah_kapal_non_anomali]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, values, color=['red', 'green'])
ax.set_ylabel('Jumlah Kapal')
ax.set_title('Perbandingan Jumlah Kapal Normal vs Terdeteksi Anomali')

# Tampilkan jumlah di atas bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("gambar_4_4_perbandingan_kapal.png", dpi=300)
