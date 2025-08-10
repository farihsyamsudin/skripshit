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

# Donut chart
labels = ['Terdeteksi Anomali', 'Normal']
values = [jumlah_kapal_anomali, jumlah_kapal_non_anomali]
colors = ['#FF4C4C', '#4CAF50']
total = sum(values)

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    values, labels=labels, colors=colors, startangle=90,
    autopct=lambda p: f'{p:.1f}%\n({int(p*total/100):,})', pctdistance=0.8
)

# Tambahkan lubang untuk donut chart
centre_circle = plt.Circle((0, 0), 0.65, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title('Distribusi Kapal Berdasarkan Deteksi Anomali', fontsize=14)
plt.tight_layout()
plt.savefig("gambar_donut_kapal.png", dpi=300)
plt.show()

# Tabel ringkasan (bisa print ke terminal atau export)
summary = pd.DataFrame({
    'Kategori': labels,
    'Jumlah Kapal': values,
    'Persentase': [f'{(v/total)*100:.2f}%' for v in values]
})
print("\nRingkasan Kapal Anomali vs Normal:")
print(summary.to_string(index=False))
