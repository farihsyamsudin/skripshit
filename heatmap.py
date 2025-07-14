import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import contextily as ctx
import time
import random

# Load data
df = pd.read_pickle('data/maritim_selat_sunda.pkl')
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['lat', 'lon', 'created_at'])
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)]

# Sampling 2%
df_sample = df.sample(frac=0.02, random_state=42).copy()

# Buat GeoDataFrame
df_sample['geometry'] = [Point(xy) for xy in zip(df_sample['lon'], df_sample['lat'])]
gdf = gpd.GeoDataFrame(df_sample, geometry='geometry', crs='EPSG:4326').to_crs(epsg=3857)

# Ambil koordinat
coords = np.vstack(gdf['geometry'].apply(lambda p: (p.x, p.y)))

# Heatmap lebih menarik + estetik
fig, ax = plt.subplots(figsize=(12, 10))

# KDE plot dengan colormap lebih hidup
sns.kdeplot(
    x=coords[:, 0], y=coords[:, 1],
    cmap="inferno", fill=True,
    bw_adjust=0.6, thresh=0.05, levels=60,
    ax=ax
)

# Tambah peta dasar
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set batas wilayah studi
ax.set_xlim(gdf.total_bounds[[0, 2]])
ax.set_ylim(gdf.total_bounds[[1, 3]])

# Tambah grid koordinat
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xticks(np.linspace(*ax.get_xlim(), num=6))
ax.set_yticks(np.linspace(*ax.get_ylim(), num=6))

# Tambahkan judul dan subjudul
ax.text(0.5, 1.03, "Heatmap Pergerakan Kapal di Selat Sunda berdasarkan Data AIS Agustus–Desember 2024", fontsize=11, ha='center', transform=ax.transAxes)

# Hapus axis label
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(left=False, bottom=False)

# Simpan
plt.tight_layout()
plt.savefig("heatmap_pergerakan_kapal_estetik_1.png", dpi=400)
plt.close()
print("[✅] Heatmap versi estetik berhasil disimpan!")
