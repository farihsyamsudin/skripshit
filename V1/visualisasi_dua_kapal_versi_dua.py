import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from itertools import combinations

# Load data
df = pd.read_pickle('data/maritim.pkl')
df['created_at'] = pd.to_datetime(df['created_at'])
df_slow = df[df['sog'] <= 0.5].copy()

# GeoDataFrame
df_slow['geometry'] = df_slow.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
gdf = gpd.GeoDataFrame(df_slow, geometry='geometry', crs='EPSG:4326').to_crs(epsg=3857)

# Proses pasangan
results = []
image_saved = False

for i, j in combinations(gdf.index, 2):
    kapal1 = gdf.loc[i]
    kapal2 = gdf.loc[j]

    if kapal1['mmsi'] == kapal2['mmsi']:
        continue

    time_diff = abs((kapal1['created_at'] - kapal2['created_at']).total_seconds())
    if time_diff > 600:
        continue

    distance = kapal1.geometry.distance(kapal2.geometry)
    if distance < 1000:
        # Simpan ke list
        results.append({
            'mmsi_1': kapal1['mmsi'],
            'lat_1': kapal1['lat'],
            'lon_1': kapal1['lon'],
            'waktu_1': kapal1['created_at'],
            'mmsi_2': kapal2['mmsi'],
            'lat_2': kapal2['lat'],
            'lon_2': kapal2['lon'],
            'waktu_2': kapal2['created_at'],
            'jarak_meter': distance,
            'selisih_waktu_detik': time_diff
        })

        # Simpan gambar hanya untuk pasangan pertama
        if not image_saved:
            gdf_pair = gpd.GeoDataFrame([kapal1, kapal2], geometry='geometry', crs='EPSG:3857')

            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_pair.plot(ax=ax, color=['red', 'blue'], markersize=100, alpha=0.8)

            for geom in gdf_pair.geometry:
                buffer = geom.buffer(1000)
                gpd.GeoSeries([buffer], crs=gdf_pair.crs).plot(ax=ax, facecolor='none', edgecolor='gray', linestyle='--')

            for _, row in gdf_pair.iterrows():
                label = (
                    f"MMSI: {row['mmsi']}\n"
                    f"Lat, Lon: ({row['lat']:.4f}, {row['lon']:.4f})\n"
                    f"Waktu: {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                ax.annotate(label, (row.geometry.x + 300, row.geometry.y + 300), fontsize=9)

            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
            ax.set_title('Dua Kapal Berhenti Bersamaan di Lokasi Terpencil\n(Potensi Transhipment Ilegal)', fontsize=13)
            ax.axis('off')
            plt.savefig('transhipment_dua_kapal_laut.png', dpi=300, bbox_inches='tight')
            plt.close()
            image_saved = True

# Simpan CSV
if results:
    df_output = pd.DataFrame(results)
    df_output.to_csv('potensi_transhipment.csv', index=False)
    print(f"Berhasil simpan {len(results)} pasangan ke 'potensi_transhipment.csv'")
else:
    print("Tidak ditemukan pasangan kapal yang memenuhi kriteria.")
