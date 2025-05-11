import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

# 1. Load data dari file pickle
df = pd.read_pickle('data/maritim.pkl')  # ganti sesuai file kamu

# 2. Filter data yang ada koordinat
df = df.dropna(subset=['lat', 'lon'])

# 3. Filter data wilayah Selat Sunda
# Selat Sunda kira-kira sekitar:
# lat: -7.5 sampai -5.0
# lon: 105.5 sampai 106.5
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)].reset_index(drop=True)

# 4. Buat GeoDataFrame
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# 5. Konversi ke Web Mercator
gdf = gdf.to_crs(epsg=3857)

# 6. Plot visualisasi
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, markersize=1, color='blue', alpha=0.5)

# 7. Tambah basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)

# 8. Style
ax.set_title('Sebaran Posisi Kapal di Selat Sunda (Agustus–Desember 2024)', fontsize=14)
ax.axis('off')

# 9. Simpan hasil
plt.savefig('gambar_4_1_sebaran_kapal_selat_sunda_2.png', dpi=300, bbox_inches='tight')
plt.show()
