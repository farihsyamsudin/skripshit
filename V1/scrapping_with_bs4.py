import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

CSV_SOURCE = "mmsi_list_unique.csv"
CSV_OUTPUT = "scraped_vessel_type.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115 Safari/537.36"
}

# Load MMSI
df_mmsi = pd.read_csv(CSV_SOURCE)
mmsi_list = df_mmsi["mmsi"].dropna().astype(str).unique().tolist()
total_mmsi = len(mmsi_list)

# Load hasil sebelumnya (kalau ada)
if os.path.exists(CSV_OUTPUT):
    df_existing = pd.read_csv(CSV_OUTPUT)
    done_mmsi = set(df_existing["mmsi"].astype(str))
else:
    df_existing = pd.DataFrame(columns=["mmsi", "vessel_type"])
    done_mmsi = set()

print(f"🚀 Total MMSI: {total_mmsi}")
print(f"✅ Sudah diproses: {len(done_mmsi)}")

results = []

for idx, mmsi in enumerate(mmsi_list):
    if mmsi in done_mmsi:
        print(f"[{idx}] Skip (done): {mmsi}")
        continue

    print(f"[{idx}] Fetching: {mmsi}")

    try:
        search_url = f"https://www.vesselfinder.com/vessels?name={mmsi}"
        res = requests.get(search_url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(res.text, "html.parser")

        link_tag = soup.select_one("a.ship-link")
        if not link_tag:
            print(f"❌ {mmsi} Not Found")
            results.append({"mmsi": mmsi, "vessel_type": "NOT_FOUND"})
            continue

        detail_url = "https://www.vesselfinder.com" + link_tag["href"]
        detail_res = requests.get(detail_url, headers=HEADERS, timeout=30)
        detail_soup = BeautifulSoup(detail_res.text, "html.parser")

        rows = detail_soup.select("tr")
        vessel_type = "UNKNOWN"

        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 2 and "Ship Type" in cols[0].text:
                vessel_type = cols[1].text.strip()
                break

        print(f"✅ {mmsi} → {vessel_type}")
        results.append({"mmsi": mmsi, "vessel_type": vessel_type})

    except Exception as e:
        print(f"❌ {mmsi} ERROR: {str(e)[:100]}")
        results.append({"mmsi": mmsi, "vessel_type": "ERROR"})

    # Optional: delay untuk hindari rate limit
    time.sleep(random.uniform(1.5, 3.5))

    # Simpan bertahap tiap 20 hasil
    if len(results) % 20 == 0:
        df_temp = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_temp], ignore_index=True).drop_duplicates(subset="mmsi")
        df_combined.to_csv(CSV_OUTPUT, index=False)
        print("💾 Disimpan sementara...")

# Final save
df_temp = pd.DataFrame(results)
df_combined = pd.concat([df_existing, df_temp], ignore_index=True).drop_duplicates(subset="mmsi")
df_combined.to_csv(CSV_OUTPUT, index=False)
print("\n✅ Selesai semua! File tersimpan di scraped_vessel_type.csv")
