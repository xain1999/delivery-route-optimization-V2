"""Sample requests built from the book's data (Chapter 15)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def tuesday_request(time_limit_s: float = 5.0, vans: int = 8) -> dict:
    """Lakeshore's Tuesday with two-hour slots (Chapter 6) as a JSON request."""
    with (DATA / "chicago_tuesday.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    with (DATA / "chicago_tuesday_slots.csv").open() as fh:
        pref = {int(r["id"]): int(r["preferred_hour"]) for r in csv.DictReader(fh)}
    stops = []
    for r in rows[1:]:
        a = 7 + 2 * ((pref[int(r["id"])] - 7) // 2)
        q = int(r["demand"])
        stops.append({"id": f"order-{r['id']}",
                      "location": {"lat": float(r["lat"]), "lon": float(r["lon"])},
                      "demand": q, "service_min": round(3 + 0.4 * q, 1),
                      "window_start": f"{a:02d}:00", "window_end": f"{min(a + 2, 15):02d}:00",
                      "priority": 500})
    return {"depot": {"lat": float(rows[0]["lat"]), "lon": float(rows[0]["lon"])},
            "stops": stops,
            "vehicles": [{"id": f"van-{k}", "capacity": 100, "shift_start": "06:30",
                          "shift_end": "15:30", "fixed_cost": 50} for k in range(1, vans + 1)],
            "time_limit_s": time_limit_s}


if __name__ == "__main__":
    (DATA / "requests").mkdir(exist_ok=True)
    (DATA / "requests" / "tuesday.json").write_text(json.dumps(tuesday_request(), indent=1))
