import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import numpy as np

# Load data
df = pd.read_pickle("data/maritim_selat_sunda_with_type.pkl")

# Pastikan waktu & lokasi valid
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['lat', 'lon', 'created_at'])

# Filter wilayah Selat Sunda dan bulan Agustus–Desember 2024
df = df[
    (df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
    (df['lon'] >= 105.0) & (df['lon'] <= 106.0) &
    (df['created_at'].dt.year == 2024) &
    (df['created_at'].dt.month >= 8) & (df['created_at'].dt.month <= 12)
].copy()

# Convert ke GeoDataFrame
df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326').to_crs(epsg=3857)

# Ambil koordinat x, y dalam satuan meter (crs 3857)
x = gdf.geometry.x
y = gdf.geometry.y

# Buat figure
fig, ax = plt.subplots(figsize=(12, 12))

# Buat heatmap pakai histogram 2D
heatmap = ax.hist2d(
    x, y,
    bins=1000,
    cmap='jet',
    norm='log'  # log scale biar gradasi makin jelas
)

# Tambahkan basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)

# Labeling
plt.colorbar(heatmap[3], ax=ax, label='Kepadatan Titik AIS')
ax.set_title('Heatmap Kepadatan Posisi Kapal di Selat Sunda\n(Agustus–Desember 2024)', fontsize=14)
ax.set_axis_off()

# Save
plt.savefig("heatmap_kepadatan_kapal_dengan_map.png", dpi=300, bbox_inches='tight')
plt.show()
