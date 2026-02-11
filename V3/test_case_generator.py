import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_case(case, duration=40, gap=False, near_port=False, high_speed=False, far_proximity=False, multi=False):
    start_time = datetime(2023, 8, 1, 10, 0, 0)
    rows = []

    # jumlah PAIRS (bukan individual vessels)
    num_pairs = 5 if multi else 1

    for i in range(duration):
        t = start_time + timedelta(minutes=i)
        
        # Tambah gap kosong
        if gap and (10 <= i < 25):
            continue

        # generate untuk tiap PAIR mmsi yang berbeda
        for k in range(num_pairs):
            # FIXED: Buat pair MMSI yang unik untuk setiap k
            mmsi1 = 111111111 + (k * 1000)  # 111111111, 111112111, 111113111, dst
            mmsi2 = 222222222 + (k * 1000)  # 222222222, 222223222, 222224222, dst
            
            # FIXED: Buat koordinat base yang berbeda untuk setiap pair
            base_lat = -6.0 - (k * 0.01)  # -6.0, -6.01, -6.02, dst
            base_lon = 105.5 + (k * 0.01)  # 105.5, 105.51, 105.52, dst
            
            # Kondisi proximity untuk pair ini
            lat1, lon1 = base_lat, base_lon
            if far_proximity:
                # FIXED: Jarak yang benar-benar jauh (>1km)
                lat2, lon2 = base_lat + 0.02, base_lon + 0.02  # ~2.8 km
            else:
                # Proximity dekat (~40m)
                lat2, lon2 = base_lat + 0.0003, base_lon + 0.0003  # ~40 m

            # Kondisi lokasi
            if near_port:
                lat1, lon1 = -5.89 - (k * 0.001), 106.01 + (k * 0.001)  # dekat Merak tapi berbeda per pair
                if far_proximity:
                    lat2, lon2 = lat1 + 0.02, lon1 + 0.02
                else:
                    lat2, lon2 = lat1 + 0.0003, lon1 + 0.0003

            # Kondisi speed
            sog1 = sog2 = 0.3
            if high_speed:
                sog1 = sog2 = 1.2

            # Tambahkan kedua vessel dari pair ini
            rows.append({"mmsi": mmsi1, "lat": lat1, "lon": lon1, "sog": sog1, "created_at": t})
            rows.append({"mmsi": mmsi2, "lat": lat2, "lon": lon2, "sog": sog2, "created_at": t})

    df = pd.DataFrame(rows)
    suffix = "_multi_data" if multi else ""
    filename = f"test_case/ais_test_case_{case}{suffix}.pkl"
    df.to_pickle(filename)
    print(f"Test case {case}{suffix} saved. Shape={df.shape}")
    
    # Debug info
    if multi:
        unique_mmsi = sorted(df['mmsi'].unique())
        unique_coords = df[['lat', 'lon']].drop_duplicates().shape[0]
        print(f"  -> {len(unique_mmsi)} unique MMSI: {unique_mmsi}")
        print(f"  -> {unique_coords} unique coordinate pairs")

# ==============================
# Generate FIXED test cases
# ==============================

print("=== Generating FIXED test cases ===")

generate_test_case("valid")                     
generate_test_case("valid", multi=True)

generate_test_case("short_duration", duration=20)  
generate_test_case("short_duration", duration=20, multi=True)

generate_test_case("far_proximity", far_proximity=True)  
generate_test_case("far_proximity", far_proximity=True, multi=True)

generate_test_case("high_speed", high_speed=True)       
generate_test_case("high_speed", high_speed=True, multi=True)

generate_test_case("near_port", near_port=True)         
generate_test_case("near_port", near_port=True, multi=True)

generate_test_case("gap", gap=True)                     
generate_test_case("gap", gap=True, multi=True)

generate_test_case("borderline", duration=31)           
generate_test_case("borderline", duration=31, multi=True)

generate_test_case("with_noise")                        
generate_test_case("with_noise", multi=True)

print("\n=== FIXED Test cases generated! ===")
print("Key changes:")
print("1. Each pair gets unique MMSI (111111111+k*1000, 222222222+k*1000)")
print("2. Each pair gets different base coordinates")
print("3. Far proximity now creates genuine >2km distance")
print("4. Multi-pair scenarios create independent vessel pairs, not vessel clusters")