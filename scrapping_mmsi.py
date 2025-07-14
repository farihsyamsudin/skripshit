import pandas as pd
import time
import random
import os
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ======================== CONFIG =========================
CSV_SOURCE = "mmsi_list_unique.csv"
CSV_OUTPUT = "scraped_vessel_type.csv"
LOG_FILE = "scraper_log.txt"
BATCH_SIZE = 100
HEADLESS_MODE = True  # Ubah ke False kalau mau debugging
# =========================================================

# Load MMSI list
df_mmsi = pd.read_csv(CSV_SOURCE)
mmsi_list = df_mmsi["mmsi"].dropna().astype(str).unique().tolist()
total_mmsi = len(mmsi_list)

# Load previous data
if os.path.exists(CSV_OUTPUT):
    df_existing = pd.read_csv(CSV_OUTPUT)
    done_mmsi = set(df_existing["mmsi"].astype(str))
else:
    df_existing = pd.DataFrame(columns=["mmsi", "vessel_type"])
    done_mmsi = set()

print(f"🚀 Total MMSI: {total_mmsi}")
print(f"✅ Sudah diproses sebelumnya: {len(done_mmsi)}")

current_idx = 0

with sync_playwright() as p:
    browser = p.chromium.launch(headless=HEADLESS_MODE)
    context = browser.new_context()
    page = context.new_page()

    while current_idx < total_mmsi:
        results = []
        print(f"\n🔁 Batch mulai dari index {current_idx}")

        for i in range(current_idx, min(current_idx + BATCH_SIZE, total_mmsi)):
            mmsi = mmsi_list[i]
            if mmsi in done_mmsi:
                print(f"[{i}] Skip (sudah ada): {mmsi}")
                continue

            print(f"[{i}] Scraping MMSI: {mmsi}")
            try:
                # Step 1: buka halaman pencarian
                search_url = f"https://www.vesselfinder.com/vessels?name={mmsi}"
                page.goto(search_url, timeout=60000)
                page.wait_for_selector("a.ship-link", timeout=10000)

                # Step 2: klik ke detail kapal
                ship_link = page.query_selector("a.ship-link")
                if not ship_link:
                    raise Exception("Link kapal tidak ditemukan.")
                href = ship_link.get_attribute("href")
                page.goto(href, timeout=60000)
                page.wait_for_selector("tr", timeout=10000)

                # Step 3: ekstrak Ship Type
                vessel_type = ""
                rows = page.query_selector_all("tr")
                for row in rows:
                    try:
                        th = row.query_selector(".tpc1")
                        td = row.query_selector(".tpc2")
                        if th and th.inner_text().strip() == "Ship Type":
                            vessel_type = td.inner_text().strip()
                            break
                    except:
                        continue

                print(f"✅ {mmsi} → {vessel_type}")
                result = {"mmsi": mmsi, "vessel_type": vessel_type}
                results.append(result)
                done_mmsi.add(mmsi)

            except PlaywrightTimeout as e:
                print(f"❌ {mmsi} GAGAL (timeout) → {str(e)[:100]}")
                result = {"mmsi": mmsi, "vessel_type": "ERROR"}
                results.append(result)
                done_mmsi.add(mmsi)

            except Exception as e:
                print(f"❌ {mmsi} GAGAL (lainnya) → {str(e)[:100]}")
                result = {"mmsi": mmsi, "vessel_type": "ERROR"}
                results.append(result)
                done_mmsi.add(mmsi)

            # Simpan log
            with open(LOG_FILE, "a") as f:
                f.write(f"{mmsi},{result['vessel_type']}\n")

            time.sleep(random.uniform(3, 6))

        # Simpan hasil ke CSV
        df_batch = pd.DataFrame(results)
        df_existing = pd.concat([df_existing, df_batch], ignore_index=True)
        df_existing.drop_duplicates(subset="mmsi", inplace=True)
        df_existing.to_csv(CSV_OUTPUT, index=False)
        print(f"💾 Batch disimpan ({len(df_existing)} total hasil)")

        current_idx += BATCH_SIZE

    browser.close()

print("\n✅ Semua MMSI sudah diproses.")
