"""Build the raw order export of Hearthline Home (Chapter 16), with the kinds
of mess a real export has: duplicate rows, several time formats, items as
free text, notes about stairs, and a few missing coordinates.

    python tools/make_hearthline.py        # deterministic (fixed seed)
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 1606
AREAS = {   # metro Atlanta neighborhoods and suburbs (approximate centers)
    "Buckhead": (33.8390, -84.3800), "Midtown": (33.7810, -84.3830),
    "Decatur": (33.7748, -84.2963), "Sandy Springs": (33.9304, -84.3733),
    "Marietta": (33.9526, -84.5499), "Smyrna": (33.8840, -84.5144),
    "Alpharetta": (34.0754, -84.2941), "Dunwoody": (33.9462, -84.3346),
    "Roswell": (34.0232, -84.3616), "East Point": (33.6795, -84.4394),
    "Kirkwood": (33.7540, -84.3180), "Brookhaven": (33.8651, -84.3366),
    "Tucker": (33.8545, -84.2171), "Kennesaw": (34.0234, -84.6155),
    "Peachtree Corners": (33.9701, -84.2216), "West End": (33.7370, -84.4140)}
ITEMS = ["sofa", "sectional", "armchair", "dining table", "bed frame", "mattress",
         "dresser", "washer", "dryer", "refrigerator", "dishwasher", "range"]
SLOT_FORMATS = ["{a}-{b}", "{a}:00-{b}:00", "{a12}-{b12}"]


def h12(h: int) -> str:
    return f"{(h - 1) % 12 + 1}{'am' if h < 12 else 'pm'}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows, oid = [], 1000
    day = date(2026, 9, 14)                        # four weeks of weekdays
    days = [day + timedelta(d) for d in range(28) if (day + timedelta(d)).weekday() < 5]
    names = list(AREAS)
    weights = np.array([3, 3, 2, 3, 2, 2, 2, 2, 2, 1, 2, 3, 1, 1, 2, 1], float)
    for d in days:
        n = int(rng.integers(42, 58))
        for _ in range(n):
            oid += 1
            area = names[rng.choice(len(names), p=weights / weights.sum())]
            lat, lon = AREAS[area]
            lat += rng.normal(0, 0.015)
            lon += rng.normal(0, 0.018)
            k = int(rng.choice([1, 1, 1, 2, 2, 3]))
            items = "; ".join(rng.choice(ITEMS, size=k, replace=False))
            a = int(rng.choice([8, 10, 12, 14, 16], p=[0.25, 0.25, 0.2, 0.18, 0.12]))
            fmt = SLOT_FORMATS[int(rng.integers(3))]
            slot = fmt.format(a=a, b=a + 2, a12=h12(a), b12=h12(a + 2))
            note = rng.choice(["", "", "", "", "2nd floor, no elevator", "3rd floor walk-up",
                               "call 30 min ahead", "gate code 4411"])
            row = {"order_id": f"HH-{oid}", "delivery_date": d.isoformat(),
                   "customer_area": area, "lat": f"{lat:.5f}", "lon": f"{lon:.5f}",
                   "items": items, "slot": slot, "notes": note}
            if rng.random() < 0.02:                    # a few addresses never geocoded
                row["lat"] = row["lon"] = ""
            rows.append(row)
            if rng.random() < 0.03:                    # exported twice
                rows.append(dict(row))
    with (DATA / "atlanta_orders_raw.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows,", len(days), "days")


if __name__ == "__main__":
    main()
