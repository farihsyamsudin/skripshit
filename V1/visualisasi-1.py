import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

# 1. Load data dari file pickle
df = pd.read_pickle('data/maritim.pkl')  # ganti nama file sesuai punyamu

# 2. Filter data yang ada koordinat
df = df.dropna(subset=['lat', 'lon'])

# 3. Buat GeoDataFrame
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')  # WGS84

# 4. Konversi ke Web Mercator
gdf = gdf.to_crs(epsg=3857)

# 5. Plot visualisasi
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, markersize=0.5, color='blue', alpha=0.3)

# 6. Tambah basemap (OpenStreetMap)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# 7. Style
ax.set_title('Sebaran Posisi Kapal di Selat Sunda (Agustus–Desember 2024)', fontsize=14)
ax.axis('off')

# 8. Simpan hasilnya
plt.savefig('gambar_4_1_sebaran_kapal.png', dpi=300, bbox_inches='tight')
plt.show()
