import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. Load data AIS hasil merge
df = pd.read_pickle('data/maritim_selat_sunda_with_type.pkl')

# 2. Pastikan kolom waktu valid
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# 3. Filter posisi Selat Sunda
df = df.dropna(subset=['lat', 'lon', 'created_at', 'vessel_type'])
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)]

# 4. Filter hanya data Agustus–Desember 2024
df = df[(df['created_at'].dt.year == 2024) &
        (df['created_at'].dt.month >= 8) &
        (df['created_at'].dt.month <= 12)]

# 5. Ubah vessel_type ke kategori (hemat memori)
df['vessel_type'] = df['vessel_type'].astype('category')

# 6. Buat GeoDataFrame
df['geometry'] = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
gdf = gdf.to_crs(epsg=3857)

# 7. Buat colormap untuk tiap vessel_type
vessel_types = sorted(gdf['vessel_type'].unique())
colormap = cm.get_cmap('tab20', len(vessel_types))
color_dict = {vtype: mcolors.rgb2hex(colormap(i)) for i, vtype in enumerate(vessel_types)}

# 8. Plot per vessel_type
fig, ax = plt.subplots(figsize=(12, 12))
for vtype in vessel_types:
    subset = gdf[gdf['vessel_type'] == vtype]
    subset.plot(ax=ax, markersize=1, color=color_dict[vtype], label=vtype, alpha=0.5)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)
ax.set_title('Sebaran Posisi Kapal Berdasarkan Jenis (Agustus–Desember 2024)', fontsize=14)
ax.axis('off')
ax.legend(title="Jenis Kapal", fontsize=8, loc='lower left', markerscale=5)

# 9. Simpan
plt.savefig('sebaran_kapal_per_jenis_5bulan.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("✅ Gambar gabungan disimpan sebagai 'sebaran_kapal_per_jenis_5bulan.png'")
