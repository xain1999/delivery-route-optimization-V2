"""From an order export to a routing instance (Chapter 3).

The pipeline has five stages, each of which records what it did as an
Issue, so that every order that does not reach the plan can be explained:

1. de-duplicate and filter (status, cut-off time),
2. validate fields (service area, address form, quantities),
3. geocode, accepting only results precise enough to drive to,
4. merge orders that go to the same customer at the same address and unit,
5. split orders too large for one vehicle.

A malformed field never stops the pipeline: it becomes an error Issue for
that order, and the order goes to the review queue.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .addresses import AddressError, ParsedAddress, parse
from .geocode import Geocoder
from .instances import Instance, Site


# <<issue>>
@dataclass(frozen=True)
class Issue:
    order_id: str
    severity: str      # "error": order held for review, "warning": kept, flagged.
                       # "info": order intentionally dropped or changed
    code: str
    message: str
# <<end_issue>>


@dataclass
class Stop:
    stop_id: int
    customer_id: str
    order_ids: list[str]
    address: str
    lat: float
    lon: float
    totes: int
    weight_lb: int
    geocode_source: str
    part: str = ""          # "1/2", "2/2" for split deliveries
    unit: str = ""          # apartment or suite, for the driver
    precision: str = ""     # of the geocode, kept as provenance


@dataclass
class CleaningResult:
    stops: list[Stop]
    issues: list[Issue]
    rows_in: int
    deferred: list[dict] = field(default_factory=list)

    def counts(self) -> Counter:
        return Counter((i.severity, i.code) for i in self.issues)

    def review_queue(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]


# <<rules>>
@dataclass(frozen=True)
class Rules:
    """Business rules for accepting an order into tomorrow's plan."""
    cutoff: datetime                      # orders placed later wait a day
    service_city: str = "CHICAGO"
    service_zip_prefix: str = "606"
    default_lb_per_tote: float = 25.0     # used when weight is missing
    max_lb_per_tote: float = 60.0         # above this, weight is suspect
    vehicle_totes: int = 100              # larger orders are split
    precise_enough: tuple[str, ...] = (   # coarser geocodes go to review
        "verified", "rooftop", "parcel", "interpolated")
# <<end_rules>>


def normalize_unit(text: str) -> str:
    """'Apt. 3' and 'APT 3' are the same unit."""
    return " ".join((text or "").upper().replace(".", " ").replace(",", " ").split())


def _count(text: str) -> int:
    """A whole number of totes. Raises ValueError for anything else."""
    x = float((text or "0").strip())
    if not x.is_integer():
        raise ValueError(f"{text!r} is not a whole number")
    return int(x)


def load_raw(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


REQUIRED = ("order_id", "placed_at", "customer_id", "address", "city", "zip", "status")


def clean_orders(rows: list[dict], rules: Rules, geocoder: Geocoder) -> CleaningResult:
    issues: list[Issue] = []
    deferred: list[dict] = []

    def flag(row, severity, code, msg):
        issues.append(Issue(row.get("order_id") or "?", severity, code, msg))

    # 1. duplicates, status, cut-off ------------------------------------
    seen: set[tuple] = set()
    live = []
    for row in rows:
        signature = tuple(row.values())
        if signature in seen:
            flag(row, "info", "duplicate_row", "exact duplicate of an earlier row, dropped")
            continue
        seen.add(signature)
        missing = [k for k in REQUIRED if not (row.get(k) or "").strip()]
        if missing:
            flag(row, "error", "missing_field", f"missing {', '.join(missing)}")
            continue
        if row["status"].strip().lower() != "paid":
            flag(row, "info", "not_paid", f"status {row['status']!r}, dropped")
            continue
        try:
            placed = datetime.fromisoformat(row["placed_at"].strip())
        except ValueError:
            flag(row, "error", "bad_timestamp", f"placed_at {row['placed_at']!r} is not a time")
            continue
        if placed > rules.cutoff:
            flag(row, "info", "after_cutoff", "placed after cut-off, moved to next day")
            deferred.append(row)
            continue
        live.append(row)

    # 2-3. validate, then geocode -------------------------------------------
    accepted = []   # (row, parsed address, geocode result, totes, weight)
    for row in live:
        errors_before = len(issues)
        city, zipc = row["city"].strip().upper(), row["zip"].strip()
        if city != rules.service_city or not zipc.startswith(rules.service_zip_prefix):
            flag(row, "error", "outside_area", f"{row['city']} {zipc} is outside the service area")
            continue                       # no point checking further
        addr: ParsedAddress | None = None
        try:
            addr = parse(row["address"])
        except AddressError as e:
            flag(row, "error", e.code, str(e))
        try:
            totes = _count(row.get("totes", ""))
        except ValueError:
            totes = None
            flag(row, "error", "bad_quantity", f"totes {row.get('totes')!r} is not a number")
        if totes is not None and totes < 0:
            flag(row, "error", "bad_quantity", f"{totes} totes")
        elif totes == 0:
            flag(row, "error", "empty_order", "order has no totes")
        weight_txt = (row.get("weight_lb") or "").strip()
        weight = 0
        if totes is None or totes <= 0:
            pass                           # already an error, weight cannot be judged
        elif not weight_txt:
            weight = round(totes * rules.default_lb_per_tote)
            flag(row, "warning", "weight_estimated",
                 f"weight missing, estimated {weight} lb from {totes} totes")
        else:
            try:
                weight = round(float(weight_txt))
            except ValueError:
                weight = None
                flag(row, "error", "bad_weight", f"weight {weight_txt!r} is not a number")
            if weight is not None and weight <= 0:
                flag(row, "error", "bad_weight", f"{weight_txt} lb for {totes} totes")
            elif weight is not None and weight / totes > rules.max_lb_per_tote:
                flag(row, "error", "implausible_weight",
                     f"{weight} lb for {totes} totes ({weight / totes:.0f} lb per tote)")
        if len([i for i in issues[errors_before:] if i.severity == "error"]):
            continue
        result = geocoder.geocode(addr)
        if result is None:
            flag(row, "error", "not_geocoded", f"could not locate {addr.key!r}")
            continue
        if result.precision not in rules.precise_enough:
            flag(row, "error", "coarse_geocode",
                 f"{addr.key!r} located only to {result.precision!r} precision")
            continue
        if result.corrected_street:
            flag(row, "warning", "street_corrected",
                 f"street {addr.street!r} read as {result.corrected_street!r}")
            addr = ParsedAddress(addr.number, addr.direction, result.corrected_street,
                                 addr.suffix)
        accepted.append((row, addr, result, totes, weight))

    # 4. merge same customer at same address and unit ------------------------
    groups: dict[tuple[str, str, str], list] = {}
    for item in accepted:
        row, addr = item[0], item[1]
        unit = normalize_unit(row.get("unit", ""))
        groups.setdefault((row["customer_id"], addr.key, unit), []).append(item)

    # 5. build stops, splitting oversized deliveries --------------------------
    stops: list[Stop] = []
    for (cust, key, unit), items in groups.items():
        order_ids = [it[0]["order_id"] for it in items]
        if len(items) > 1:
            for oid in order_ids[1:]:
                issues.append(Issue(oid, "info", "merged",
                                    f"merged with {order_ids[0]} (same customer, address and unit)"))
        totes = sum(it[3] for it in items)
        weight = sum(it[4] for it in items)
        res = items[0][2]
        parts = math.ceil(totes / rules.vehicle_totes)
        if parts > 1:
            issues.append(Issue(order_ids[0], "warning", "split_delivery",
                                f"{totes} totes exceed one van, split into {parts} deliveries"))
        done_t = done_w = 0
        for p in range(parts):
            t = totes // parts + (1 if p < totes % parts else 0)
            w = round(weight * (done_t + t) / totes) - done_w    # parts add up to the total
            done_t, done_w = done_t + t, done_w + w
            stops.append(Stop(len(stops) + 1, cust, order_ids, key, res.lat, res.lon, t, w,
                              res.source, f"{p + 1}/{parts}" if parts > 1 else "",
                              unit=unit, precision=res.precision))
    return CleaningResult(stops, issues, len(rows), deferred)


def to_instance(result: CleaningResult, depot: dict, n_vehicles: int, capacity: int,
                name: str = "chicago-tuesday") -> tuple[Instance, list[int]]:
    """Return the instance (demand = totes) and the matching weights."""
    sites = [Site(0, depot["name"], depot.get("neighborhood", ""), depot["lat"], depot["lon"])]
    demands, weights = [0], [0]
    for s in result.stops:
        label = s.address + (f" ({s.part})" if s.part else "")
        sites.append(Site(s.stop_id, label, s.customer_id, s.lat, s.lon))
        demands.append(s.totes)
        weights.append(s.weight_lb)
    return Instance(name, sites, demands, n_vehicles, capacity), weights


def write_instance_csv(inst: Instance, weights: list[int], path: str | Path,
                       units: list[str] | None = None) -> None:
    """The instance as CSV. Units (one per site, "" for none) travel with
    it so that the driver's manifest can show them."""
    units = units or [""] * len(inst.sites)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "district", "lat", "lon", "demand", "weight_lb", "unit"])
        for s, q, wt, u in zip(inst.sites, inst.demands, weights, units):
            w.writerow([s.id, s.name, s.district, f"{s.lat:.5f}", f"{s.lon:.5f}", q, wt, u])
