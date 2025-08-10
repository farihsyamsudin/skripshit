import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime

# 1. Load data
df = pd.read_pickle('data/maritim_selat_sunda.pkl')

# 2. Pastikan kolom datetime
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# 3. Filter berdasarkan lokasi (Selat Sunda)
df = df.dropna(subset=['lat', 'lon', 'created_at'])
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)].reset_index(drop=True)

# 4. Loop per bulan Agustus - Desember 2024
months = ['Agustus', 'September', 'Oktober', 'November', 'Desember']
for month_num, month_name in zip(range(8, 13), months):
    # Filter per bulan
    df_bulan = df[(df['created_at'].dt.year == 2024) &
                  (df['created_at'].dt.month == month_num)].copy()
    
    if df_bulan.empty:
        print(f"[Info] Tidak ada data untuk bulan {month_name} 2024.")
        continue

    # Buat GeoDataFrame
    df_bulan['geometry'] = [Point(xy) for xy in zip(df_bulan['lon'], df_bulan['lat'])]
    gdf = gpd.GeoDataFrame(df_bulan, geometry='geometry', crs='EPSG:4326')
    gdf = gdf.to_crs(epsg=3857)

    # Buat color map
    unique_mmsi = gdf['mmsi'].unique()
    num_mmsi = len(unique_mmsi)

    # Gunakan colormap dengan sampling agar berbeda-beda warnanya
    colormap = cm.get_cmap('tab20', num_mmsi)  # bisa diganti dengan 'nipy_spectral', 'gist_rainbow', dll
    color_dict = {mmsi: mcolors.rgb2hex(colormap(i % colormap.N)) for i, mmsi in enumerate(unique_mmsi)}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    for mmsi in unique_mmsi:
        subset = gdf[gdf['mmsi'] == mmsi]
        subset.plot(ax=ax, markersize=1, color=color_dict[mmsi], alpha=0.6)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)
    ax.set_title(f'Sebaran Posisi Kapal di Selat Sunda ({month_name} 2024)', fontsize=14)
    ax.axis('off')

    # Save
    filename = f'gambar_sebaran_kapal_selat_sunda_{month_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # supaya tidak menampilkan dan tidak menahan script
    print(f"[Sukses] Gambar bulan {month_name} disimpan ke '{filename}'")
