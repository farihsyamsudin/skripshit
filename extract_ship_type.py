from pathlib import Path
import gc
import pandas as pd
from pyais import decode              # <‑‑ API baru pyais ≥ 2.0 :contentReference[oaicite:0]{index=0}

SRC = Path("data/maritim_selat_sunda.pkl")
DST = Path("data/maritim_with_ship_type.pkl")
CHUNK = 300_000                        # sesuaikan kalau masih berat

# ────────────────────────── helper ──────────────────────────
def ship_type_code(nmea_str: str | bytes) -> int | None:
    """
    Decode kalimat NMEA.
    Return integer 0-99 (ship_type) hanya untuk Msg-5 / Msg-24.
    """
    try:
        msg = decode(nmea_str)         # pyais.decode → objek MessageTypeN
        if msg.msg_type in (5, 24):
            return msg.asdict().get("ship_type")   # field 'ship_type' di dict hasil
    except Exception:
        pass                            # payload rusak → abaikan
    return None


def ship_group(code):
    if code == 30:
        return "Fishing"
    if 60 <= code <= 69:
        return "Passenger"
    if 70 <= code <= 79:
        return "Cargo"
    if 80 <= code <= 89:
        return "Tanker"
    return "Other"

# ───────────────────── fase‑A: buat mapping MMSI↦ship_type ─────────────────────
print("➜  Bangun mapping MMSI → ship_type_code ...")
df_all = pd.read_pickle(SRC, compression="infer")          # ~2 GB di RAM

mapper_parts = []                                          # kumpul pecahan

total = len(df_all)
for start in range(0, total, CHUNK):
    end = min(start + CHUNK, total)
    part = df_all.iloc[start:end, :][['mmsi', 'aistype', 'original']]

    static = part[part['aistype'].isin([5, 24])].copy()
    if static.empty:
        continue

    static['ship_type_code'] = static['original'].apply(ship_type_code)
    static = (
        static[['mmsi', 'ship_type_code']]
        .dropna(subset=['ship_type_code'])
        .astype({'ship_type_code': 'uint8'})
    )
    mapper_parts.append(static)

    del part, static
    gc.collect()

if not mapper_parts:
    raise RuntimeError("Dataset tidak mengandung message type 5 / 24 sama sekali!")

mapper = (
    pd.concat(mapper_parts, ignore_index=True)
      .drop_duplicates('mmsi', keep='last')
)
del mapper_parts
gc.collect()

print(f"   ✔  {len(mapper):,} kapal mendapat ship_type_code")

# ───────────────────── fase‑B: merge ke dataset penuh ──────────────────────
print("➜  Merge kolom baru ke seluruh baris ...")
df_all = df_all.merge(mapper, on='mmsi', how='left')
df_all['ship_group'] = df_all['ship_type_code'].apply(ship_group)

# ────────────────────────── save ──────────────────────────
df_all.to_pickle(DST, compression="infer")
print(f"✅  File selesai ditulis →  {DST}")
