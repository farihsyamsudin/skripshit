import pandas as pd
import numpy as np
from haversine import haversine
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import folium
import hashlib
import time

start = time.time()

# Load AIS data
df = pd.read_pickle('data/maritim_selat_sunda.pkl')
df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) & 
        (df['lon'] >= 105.0) & (df['lon'] <= 106.0)].reset_index(drop=True)
df = df.dropna(subset=['mmsi', 'lat', 'lon', 'created_at', 'sog'])
df['utc'] = pd.to_datetime(df['created_at'])

# Step 1: Time binning (10-minute bins)
df['bin10'] = df['utc'].dt.floor('10T')

# Step 2: Select one position per ship per bin (first position)
df_bin = df.sort_values('utc').groupby(['bin10', 'mmsi']).first().reset_index()

# Step 3: Detect close ship pairs per bin
proximity_threshold_km = 0.05
pairs = []

for ts, g in df_bin.groupby('bin10'):
    coords = np.radians(g[['lat', 'lon']])
    tree = BallTree(coords, metric='haversine')
    idxs = tree.query_radius(coords, r=proximity_threshold_km / 6371.0)
    mmsis = g['mmsi'].values
    for i, neighbors in enumerate(idxs):
        for j in neighbors:
            if i < j:
                pairs.append({
                    'm1': min(mmsis[i], mmsis[j]),
                    'm2': max(mmsis[i], mmsis[j]),
                    'bin10': ts
                })

pairs_df = pd.DataFrame(pairs)
print("🔍 Jumlah proximity pairs ditemukan:", len(pairs_df))

# Step 4: Detect STS candidates based on streak
sts_candidates = []

for (m1, m2), group in pairs_df.groupby(['m1', 'm2']):
    bins = sorted(group['bin10'])
    streak = 1
    start_bin = bins[0]
    for i in range(1, len(bins)):
        if (bins[i] - bins[i-1]) == pd.Timedelta('10min'):
            streak += 1
        else:
            if streak >= 12:
                sts_candidates.append({
                    'm1': m1,
                    'm2': m2,
                    'start': start_bin,
                    'end': bins[i-1],
                    'duration_min': streak * 10
                })
            streak = 1
            start_bin = bins[i]
    if streak >= 12:
        sts_candidates.append({
            'm1': m1,
            'm2': m2,
            'start': start_bin,
            'end': bins[-1],
            'duration_min': streak * 10
        })

sts_df = pd.DataFrame(sts_candidates)
print(f"✅ Jumlah kandidat STS (durasi ≥ 2 jam): {len(sts_df)}")

# Optional: Calculate mean lat/lon for mapping
if not sts_df.empty:
    locs = []
    for idx, row in sts_df.iterrows():
        subset = df[(df['mmsi'].isin([row['m1'], row['m2']])) & 
                    (df['utc'] >= row['start']) & (df['utc'] <= row['end'])]
        lat_mean = subset['lat'].mean()
        lon_mean = subset['lon'].mean()
        locs.append({'lat': lat_mean, 'lon': lon_mean})

    locs_df = pd.DataFrame(locs)
    sts_df = pd.concat([sts_df, locs_df], axis=1)
    sts_df.to_csv("output_sts_candidates.csv", index=False)

    # Folium map
    m = folium.Map(location=[sts_df['lat'].mean(), sts_df['lon'].mean()], zoom_start=9)
    
    def get_color_hex(m1, m2):
        pair_str = f"{m1}-{m2}"
        return f"#{hashlib.md5(pair_str.encode()).hexdigest()[:6]}"

    for _, row in sts_df.iterrows():
        popup_text = f"{row['m1']} & {row['m2']}<br>Durasi: {row['duration_min']} menit"
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=get_color_hex(row['m1'], row['m2']),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    m.save("output_sts_map.html")
    print("🗺️ Peta disimpan sebagai output_sts_map.html")

end = time.time()
print(f"🕒 Waktu eksekusi: {round((end - start)/60, 2)} menit")
