import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

# 1. Load data
df = pd.read_pickle('data/maritim.pkl')

# 2. Konversi waktu
df['created_at'] = pd.to_datetime(df['created_at'])

# 3. Filter kapal yang hampir berhenti
df_slow = df[df['sog'] <= 0.5].copy()

# 4. Bikin geometry
df_slow['geometry'] = df_slow.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
gdf = gpd.GeoDataFrame(df_slow, geometry='geometry', crs='EPSG:4326')

# 5. Ubah ke meter untuk jarak (Web Mercator)
gdf = gdf.to_crs(epsg=3857)

# 6. Cari dua kapal berbeda yang dekat & waktunya hampir sama
found = None
for i, row1 in gdf.iterrows():
    for j, row2 in gdf.iterrows():
        if i >= j: continue
        if row1['mmsi'] == row2['mmsi']: continue
        if abs((row1['created_at'] - row2['created_at']).total_seconds()) > 600: continue  # beda waktu >10 menit skip
        if row1['geometry'].distance(row2['geometry']) < 1000:  # <1 km
            found = (row1, row2)
            break
    if found:
        break

# 7. Plot kalau ketemu
if found:
    kapal1, kapal2 = found
    gdf_pair = gpd.GeoDataFrame([kapal1, kapal2], geometry='geometry', crs='EPSG:3857')

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_pair.plot(ax=ax, color=['red', 'blue'], markersize=100, alpha=0.8)

    # Tambahin lingkaran radius 1 km
    for geom in gdf_pair.geometry:
        circle = geom.buffer(1000)
        gpd.GeoSeries([circle], crs=gdf_pair.crs).plot(ax=ax, facecolor='none', edgecolor='gray', linestyle='--')

    # Anotasi MMSI dan waktu
    for i, row in gdf_pair.iterrows():
        label = f"MMSI: {row['mmsi']}\n{row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
        ax.annotate(label, (row.geometry.x + 300, row.geometry.y + 300), fontsize=9)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
    ax.set_title('Dua Kapal Berhenti Bersamaan di Lokasi Terpencil\n(Potensi Transhipment Ilegal)', fontsize=14)
    ax.axis('off')

    plt.savefig('transhipment_dua_kapal.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Tidak ditemukan pasangan kapal yang memenuhi kriteria.")
