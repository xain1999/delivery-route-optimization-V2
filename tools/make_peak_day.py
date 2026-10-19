"""Build the peak-day instances of Chapter 9 (data/chicago_peak*.csv).

The Wednesday before Thanksgiving: Tuesday's regular customers order
bigger baskets, and holiday orders arrive from new addresses.  The file
adds three business fields to each order:

    value_usd    basket value in dollars
    tier         "club" (Larder Club subscriber) or "regular"
    postponed    1 if the order was already postponed once (from Tuesday)

    python tools/make_peak_day.py          # deterministic (fixed seed)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 20261125
BASKET_GROWTH = 1.12           # holiday baskets are larger
N_HOLIDAY = 10                 # new holiday orders
USD_PER_TOTE = 32.0
CLUB_SHARE = 0.35
N_POSTPONED = 3


def main() -> None:
    rng = np.random.default_rng(SEED)
    with (DATA / "chicago_tuesday.csv").open() as fh:
        tues = list(csv.DictReader(fh))
    with (DATA / "chicago_tuesday_slots.csv").open() as fh:
        pref = {int(r["id"]): int(r["preferred_hour"]) for r in csv.DictReader(fh)}
    with (DATA / "chicago_ch02.csv").open() as fh:
        hoods = list(csv.DictReader(fh))[1:]

    rows = [dict(tues[0], preferred_hour="", value_usd=0, tier="", postponed=0)]
    for r in tues[1:]:
        q = max(1, round(int(r["demand"]) * BASKET_GROWTH))
        rows.append({**r, "demand": q,
                     "weight_lb": round(int(r["weight_lb"]) * q / int(r["demand"])),
                     "preferred_hour": pref[int(r["id"])]})
    for k, h in enumerate(rng.choice(len(hoods), N_HOLIDAY, replace=False)):
        h = hoods[h]
        q = int(rng.integers(6, 19))
        rows.append({"id": len(rows), "name": f"Holiday order H{k + 1:02d}",
                     "district": h["name"],
                     "lat": f"{float(h['lat']) + rng.uniform(-0.006, 0.006):.5f}",
                     "lon": f"{float(h['lon']) + rng.uniform(-0.006, 0.006):.5f}",
                     "demand": q, "weight_lb": round(q * rng.uniform(20, 26)),
                     "preferred_hour": int(rng.choice([7, 9, 11, 13]))})
    regular = [r for r in rows[1:]]
    for r in regular:
        r["value_usd"] = round(int(r["demand"]) * USD_PER_TOTE * rng.lognormal(0, 0.15))
        new = r["name"].startswith("Holiday")
        r["tier"] = "regular" if new else ("club" if rng.random() < CLUB_SHARE else "regular")
        r["postponed"] = 0
    tuesday_ids = [r for r in regular if not r["name"].startswith("Holiday")]
    for k in rng.choice(len(tuesday_ids), N_POSTPONED, replace=False):
        tuesday_ids[k]["postponed"] = 1

    fields = ["id", "name", "district", "lat", "lon", "demand", "weight_lb",
              "preferred_hour", "value_usd", "tier", "postponed"]
    with (DATA / "chicago_peak.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # late orders for an evening top-up run (Section 9.5): small baskets,
    # placed Wednesday morning for delivery the same evening
    late = [{"id": 0, "name": "Lakeshore Larder warehouse", "district": "McKinley Park",
             "lat": "41.83000", "lon": "-87.68000", "demand": 0, "value_usd": 0}]
    for k, h in enumerate(hoods):
        q = int(rng.integers(2, 9))
        late.append({"id": k + 1, "name": f"Late order L{k + 1:02d}", "district": h["name"],
                     "lat": f"{float(h['lat']) + rng.uniform(-0.008, 0.008):.5f}",
                     "lon": f"{float(h['lon']) + rng.uniform(-0.008, 0.008):.5f}",
                     "demand": q,
                     "value_usd": round(q * USD_PER_TOTE * rng.lognormal(0, 0.25))})
    with (DATA / "chicago_peak_late.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "name", "district", "lat", "lon",
                                           "demand", "value_usd"])
        w.writeheader()
        w.writerows(late)
    print(f"{len(late) - 1} late orders, {sum(r['demand'] for r in late)} totes")

    tot = sum(int(r["demand"]) for r in rows)
    wt = sum(int(r["weight_lb"]) for r in rows)
    print(f"{len(rows) - 1} orders, {tot} totes, {wt} lb, "
          f"{sum(r['tier'] == 'club' for r in rows)} club, "
          f"{sum(int(r['postponed']) for r in rows)} postponed")


if __name__ == "__main__":
    main()
