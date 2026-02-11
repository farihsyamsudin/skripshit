# HTR: python V3/main.py test_case
# HTR: python V3/main.py real_case

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import hashlib
import time
import argparse
import warnings

from folium.plugins import MarkerCluster
from anomaly_logic import detect_anomalies

warnings.filterwarnings("ignore")

# ==============================================================================
# SHARED CONFIGURATION AND HELPERS
# ==============================================================================

PORTS = [
    {"name": "Pelabuhan Merak", "lat": -5.8933, "lon": 106.0086},
    {"name": "Pelabuhan Ciwandan", "lat": -5.9525, "lon": 106.0358},
    {"name": "Pelabuhan Bojonegara", "lat": -5.8995, "lon": 106.0657},
    {"name": "Pelabuhan Bakauheni", "lat": -5.8711, "lon": 105.7421},
    {"name": "Pelabuhan Panjang", "lat": -5.4558, "lon": 105.3134},
    {"name": "Pelabuhan Ciwandan 2", "lat": -6.02147, "lon": 105.95485},
    {"name": "Labuan", "lat": -6.395829, "lon": 105.807895},
    {"name": "Citeureup", "lat": -6.491586, "lon": 105.725007},
    {"name": "Tarahan", "lat": -5.565000, "lon": 105.372998},
]

def get_color_hex(mmsi_1, mmsi_2):
    """Generates a consistent color for a pair of MMSIs."""
    pair_str = f"{mmsi_1}-{mmsi_2}"
    hex_digest = hashlib.md5(pair_str.encode()).hexdigest()
    r = int(hex_digest[0:2], 16) % 200 + 50
    g = int(hex_digest[2:4], 16) % 160
    b = int(hex_digest[4:6], 16) % 60
    return f"#{r:02x}{g:02x}{b:02x}"

# ==============================================================================
# REAL CASE MODE
# ==============================================================================

def run_real_case_mode():
    """Runs the anomaly detection process for a single, large dataset."""
    print("--- Running in REAL CASE mode ---")
    start = time.time()

    # 1. Parameters
    PROXIMITY_THRESHOLD_KM = 0.2
    DURATION_THRESHOLD_MIN = 30
    CANDIDATE_DURATION_THRESHOLD_MIN = 22
    SOG_THRESHOLD = 0.5
    PORT_DISTANCE_THRESHOLD_KM = 10.0
    TIME_GAP_MINUTES = 10
    INPUT_FILE = 'data/maritim_selat_sunda.pkl'

    # 2. Load and Prepare Data (OPTIMIZED)
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_pickle(INPUT_FILE)

    # Reduce memory usage
    df['mmsi'] = df['mmsi'].astype('int32')
    df['sog'] = df['sog'].astype('float32')
    df['lat'] = df['lat'].astype('float32')
    df['lon'] = df['lon'].astype('float32')

    df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
            (df['lon'] >= 105.0) & (df['lon'] <= 106.0)].reset_index(drop=True)
    df = df.dropna(subset=['mmsi', 'lat', 'lon', 'created_at', 'sog'])
    df['utc'] = pd.to_datetime(df['created_at'])
    df = df.sort_values(by='utc')
    print(f"Data loaded and prepared. Shape: {df.shape}")

    # 3. Run Core Algorithm
    print("Finding anomalies...")
    final_df, candidate_df = detect_anomalies(
        df, PROXIMITY_THRESHOLD_KM, DURATION_THRESHOLD_MIN, CANDIDATE_DURATION_THRESHOLD_MIN,
        SOG_THRESHOLD, PORT_DISTANCE_THRESHOLD_KM, TIME_GAP_MINUTES, PORTS
    )

    # 4. Generate Outputs for Confirmed Anomalies
    if final_df.empty:
        print("Tidak ada anomali terkonfirmasi yang ditemukan.")
    else:
        print(f"Ditemukan {len(final_df)} anomali terkonfirmasi.")
        final_df_display = final_df.copy()
        final_df_display['start_time'] = pd.to_datetime(final_df_display['start_time']).dt.strftime("%d-%m-%Y %H:%M")
        final_df_display['end_time'] = pd.to_datetime(final_df_display['end_time']).dt.strftime("%d-%m-%Y %H:%M")
        final_df_display.to_csv("output_tabel_anomali_FIX_after_change.csv", index=False)

        # Matplotlib (OPTIMIZED: sample max 1000 points to avoid memory crash)
        plt.figure(figsize=(12, 8))
        sample_df = final_df if len(final_df) <= 1000 else final_df.sample(1000, random_state=42)
        for _, row in sample_df.iterrows():
            color = get_color_hex(row['mmsi_1'], row['mmsi_2'])
            plt.scatter(row['lon'], row['lat'], color=color, label=f"{row['mmsi_1']}-{row['mmsi_2']}", s=40)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Lokasi Anomali Transhipment")
        plt.grid(True)
        plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("output_grafik_anomali_warna_FIX_after_change.png", dpi=300)
        plt.close()

        # Folium Map (OPTIMIZED: use MarkerCluster)
        map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=9)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in final_df_display.iterrows():
            color = get_color_hex(row['mmsi_1'], row['mmsi_2'])
            popup_text = (
                f"<b>{row['mmsi_1']} & {row['mmsi_2']}</b><br>"
                f"Durasi: {row['duration_min']} menit<br>"
                f"Start: {row['start_time']}<br>"
                f"End: {row['end_time']}"
            )
            folium.CircleMarker(
                location=[row['lat'], row['lon']], radius=5, color=color, fill=True,
                fill_opacity=0.7, popup=folium.Popup(popup_text, max_width=300)
            ).add_to(marker_cluster)
        m.save("output_peta_anomali_FIX_cluster_after_change.html")
        print("Peta anomali terkonfirmasi disimpan di output_peta_anomali_FIX_cluster_after_change.html")

    # 5. Generate Output for Candidate Anomalies
    if not candidate_df.empty:
        print(f"Ditemukan {len(candidate_df)} calon anomali.")
        candidate_df['start_time'] = pd.to_datetime(candidate_df['start_time']).dt.strftime("%d-%m-%Y %H:%M")
        candidate_df['end_time'] = pd.to_datetime(candidate_df['end_time']).dt.strftime("%d-%m-%Y %H:%M")
        candidate_df.to_csv("output_tabel_calon_anomali.csv", index=False)
        print("Tabel calon anomali disimpan di output_tabel_calon_anomali.csv")
    else:
        print("Tidak ada calon anomali yang ditemukan.")

    end = time.time()
    print(f"Waktu eksekusi: {round((end - start)/60, 2)} menit")

# ==============================================================================
# TEST CASE MODE (no heavy optimization needed)
# ==============================================================================

def run_test_case_mode():
    """Runs the anomaly detection process on all files in the test_case directory."""
    print("--- Running in TEST CASE mode ---")
    
    # 1. Parameters
    PROXIMITY_THRESHOLD_KM = 0.2
    DURATION_THRESHOLD_MIN = 30
    CANDIDATE_DURATION_THRESHOLD_MIN = 22
    SOG_THRESHOLD = 0.5
    PORT_DISTANCE_THRESHOLD_KM = 10.0
    TIME_GAP_MINUTES = 10
    TEST_CASE_DIR = "test_case/"
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("res_case", exist_ok=True)

    # 2. Loop through all test files
    for file in sorted(os.listdir(TEST_CASE_DIR)):
        if not file.endswith(".pkl"):
            continue
        
        input_file = os.path.join(TEST_CASE_DIR, file)
        output_prefix = os.path.splitext(file)[0]
        print(f"\n>>> Processing: {input_file}")

        # 3. Load and Prepare Data
        df = pd.read_pickle(input_file)
        df['mmsi'] = df['mmsi'].astype('int32')
        df['sog'] = df['sog'].astype('float32')
        df['lat'] = df['lat'].astype('float32')
        df['lon'] = df['lon'].astype('float32')

        df = df[(df['lat'] >= -6.5) & (df['lat'] <= -5.5) &
                (df['lon'] >= 105.0) & (df['lon'] <= 106.0)].reset_index(drop=True)
        df = df.dropna(subset=['mmsi', 'lat', 'lon', 'created_at', 'sog'])
        df['utc'] = pd.to_datetime(df['created_at'])
        df = df.sort_values(by='utc')

        # 4. Run Core Algorithm
        final_df, candidate_df = detect_anomalies(
            df, PROXIMITY_THRESHOLD_KM, DURATION_THRESHOLD_MIN, CANDIDATE_DURATION_THRESHOLD_MIN,
            SOG_THRESHOLD, PORT_DISTANCE_THRESHOLD_KM, TIME_GAP_MINUTES, PORTS
        )

        # 5. Generate Outputs for Confirmed Anomalies
        if final_df.empty:
            print(f"[{output_prefix}] ❌ Tidak ada anomali terkonfirmasi.")
        else:
            final_df_display = final_df.copy()
            final_df_display['start_time'] = pd.to_datetime(final_df_display['start_time']).dt.strftime("%d-%m-%Y %H:%M")
            final_df_display['end_time'] = pd.to_datetime(final_df_display['end_time']).dt.strftime("%d-%m-%Y %H:%M")
            csv_file = f"results/{output_prefix}_anomali.csv"
            final_df_display.to_csv(csv_file, index=False)

            plt.figure(figsize=(10, 7))
            for _, row in final_df.iterrows():
                color = get_color_hex(row['mmsi_1'], row['mmsi_2'])
                plt.scatter(row['lon'], row['lat'], color=color, s=40, label=f"{row['mmsi_1']}-{row['mmsi_2']}")
            plt.xlabel("Longitude"); plt.ylabel("Latitude")
            plt.title(f"Lokasi Anomali - {output_prefix}")
            plt.grid(True); plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"res_case/{output_prefix}_anomali.png", dpi=300)
            plt.close()

            map_center = [final_df['lat'].mean(), final_df['lon'].mean()]
            m = folium.Map(location=map_center, zoom_start=9)
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in final_df_display.iterrows():
                color = get_color_hex(row['mmsi_1'], row['mmsi_2'])
                popup_text = (f"<b>{row['mmsi_1']} & {row['mmsi_2']}</b><br>"
                              f"Durasi: {row['duration_min']} menit<br>"
                              f"Start: {row['start_time']}<br>"
                              f"End: {row['end_time']}")
                folium.CircleMarker(
                    location=[row['lat'], row['lon']], radius=5, color=color, fill=True, fill_opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(marker_cluster)
            html_file = f"results/{output_prefix}_anomali.html"
            m.save(html_file)
            print(f"[{output_prefix}] ✅ {len(final_df)} anomali terkonfirmasi → {csv_file}, {html_file}")

        # 6. Generate Output for Candidate Anomalies
        if not candidate_df.empty:
            print(f"[{output_prefix}] ℹ️  Ditemukan {len(candidate_df)} calon anomali.")
            candidate_df['start_time'] = pd.to_datetime(candidate_df['start_time']).dt.strftime("%d-%m-%Y %H:%M")
            candidate_df['end_time'] = pd.to_datetime(candidate_df['end_time']).dt.strftime("%d-%m-%Y %H:%M")
            calon_csv_file = f"results/{output_prefix}_calon_anomali.csv"
            candidate_df.to_csv(calon_csv_file, index=False)
            print(f"[{output_prefix}] → Tabel calon anomali disimpan di {calon_csv_file}")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deteksi anomali transhipment dari data AIS.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'mode',
        choices=['test_case', 'real_case'],
        help="Pilih mode eksekusi:\n"
             "test_case - Proses semua file di folder 'test_case/'\n"
             "real_case - Proses file data besar tunggal (hardcoded)"
    )
    args = parser.parse_args()

    if args.mode == 'test_case':
        run_test_case_mode()
    elif args.mode == 'real_case':
        run_real_case_mode()

if __name__ == "__main__":
    main()