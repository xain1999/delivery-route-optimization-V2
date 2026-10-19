"""Hearthline Home: from a raw order export to evaluated daily plans (Chapter 16).

Every step reuses the book: cleaning rules as in Chapter 3, costs as in
Chapter 1, the planner through the service of Chapter 15, and an
independent evaluation of every plan, the optimizer's and the dispatcher's.
"""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from routing_service.models import PlanRequest
from routing_service.solver import matrices, solve, to_s

DATA = Path(__file__).resolve().parent.parent / "data"
DEPOTS = {"Doraville": (33.8980, -84.2830), "Fulton Industrial": (33.7800, -84.5500),
          "Forest Park": (33.6220, -84.3690)}

# <<catalog>>
CATALOG = {   # item: (cubic feet on the truck, minutes on site to place or install)
    "sofa": (60, 15), "sectional": (110, 30), "armchair": (25, 10),
    "dining table": (40, 30), "bed frame": (45, 40), "mattress": (30, 10),
    "dresser": (35, 10), "washer": (30, 45), "dryer": (30, 40),
    "refrigerator": (45, 40), "dishwasher": (12, 60), "range": (25, 45)}
STAIRS = {"2nd floor, no elevator": 15, "3rd floor walk-up": 25}     # extra minutes
ARRIVE_MIN = 10           # park, unload, walk in, paperwork
# <<end_catalog>>


@dataclass
class Fleet:
    trucks: int = 8
    capacity_cuft: int = 800        # a 16-foot box truck
    shift: tuple = ("07:00", "19:00")
    crew_per_hour: float = 61.62    # two people at Chapter 1's loaded wage
    per_mile: float = 1.10
    per_day: float = 120.0          # lease of each truck, paid whether it moves or not
    speed_mph: float = 24.0


@dataclass
class Order:
    id: str
    day: str
    area: str
    lat: float
    lon: float
    cuft: int
    minutes: float
    window: tuple                   # ("08:00", "10:00")


# <<clean>>
def parse_slot(text: str) -> tuple[int, int]:
    """'8-10', '8:00-10:00' and '8am-10am' all mean 8:00 to 10:00, '12am'
    is midnight and '12pm' noon.  Anything else raises ValueError, which
    clean() records as an issue."""
    parts = text.lower().replace(" ", "").split("-")
    if len(parts) != 2:
        raise ValueError(f"a slot needs exactly two times: {text!r}")
    out = []
    for end, part in enumerate(parts):
        m = re.fullmatch(r"(\d{1,2})(?::00)?(am|pm)?", part)
        if not m:
            raise ValueError(f"unreadable slot {text!r}")
        h, half = int(m.group(1)), m.group(2)
        if half and not 1 <= h <= 12:
            raise ValueError(f"no hour {part!r} on a 12-hour clock: {text!r}")
        if half:
            h = h % 12 + (12 if half == "pm" else 0)      # 12am -> 0, 12pm -> 12
        if end and h == 0:
            h = 24                                        # ends at midnight
        out.append(h)
    a, b = out
    if not (0 <= a < b <= 24):
        raise ValueError(f"not a slot within one day: {text!r}")
    return a, b


def clean(rows: list[dict]) -> tuple[list[Order], dict[str, list[str]]]:
    """Orders ready to plan, and the issues found, by kind."""
    issues: dict[str, list[str]] = defaultdict(list)
    seen, orders = set(), []
    for r in rows:
        key = tuple(r.values())
        if key in seen:
            issues["duplicate row"].append(r["order_id"])
            continue
        seen.add(key)
        if not r["lat"] or not r["lon"]:
            issues["no coordinates"].append(r["order_id"])
            continue
        try:
            a, b = parse_slot(r["slot"])
        except ValueError:
            issues["unreadable slot"].append(r["order_id"])
            continue
        items = [i.strip() for i in r["items"].split(";") if i.strip()]
        unknown = [i for i in items if i not in CATALOG]
        if unknown:
            issues["unknown item"].append(r["order_id"])
            continue
        cuft = sum(CATALOG[i][0] for i in items)
        minutes = ARRIVE_MIN + sum(CATALOG[i][1] for i in items) + STAIRS.get(r["notes"], 0)
        orders.append(Order(r["order_id"], r["delivery_date"], r["customer_area"],
                            float(r["lat"]), float(r["lon"]), cuft, minutes,
                            (f"{a:02d}:00", f"{b:02d}:00")))
    return orders, dict(issues)
# <<end_clean>>


def load(path: Path = DATA / "atlanta_orders_raw.csv"):
    with path.open() as fh:
        return clean(list(csv.DictReader(fh)))


# <<request>>
def to_request(orders: list[Order], fleet: Fleet, depot: str = "Doraville",
               time_limit_s: float = 5.0, widen_h: int = 0) -> PlanRequest:
    """One day's orders as a request to the routing service.  widen_h widens
    every window by that many hours on each side (Section 16.4).  The lease
    is paid whatever the plan, so it is no cost of dispatch: fixed_cost is
    zero, and evaluate() adds the lease of every truck in the fleet."""
    lat, lon = DEPOTS[depot]

    def widen(hhmm, dh):
        return f"{min(max(int(hhmm[:2]) + dh, 0), 23):02d}:00"
    stops = [{"id": o.id, "location": {"lat": o.lat, "lon": o.lon}, "demand": o.cuft,
              "service_min": o.minutes, "priority": 1000.0,
              "window_start": widen(o.window[0], -widen_h),
              "window_end": widen(o.window[1], widen_h)} for o in orders]
    vehicles = [{"id": f"truck-{k + 1}", "capacity": fleet.capacity_cuft,
                 "shift_start": fleet.shift[0], "shift_end": fleet.shift[1],
                 "cost_per_hour": fleet.crew_per_hour, "cost_per_mile": fleet.per_mile,
                 "fixed_cost": 0.0} for k in range(fleet.trucks)]
    return PlanRequest(depot={"lat": lat, "lon": lon}, stops=stops, vehicles=vehicles,
                       time_limit_s=time_limit_s, speed_mph=fleet.speed_mph)
# <<end_request>>


# <<manual>>
def manual_plan(req: PlanRequest) -> list[list[int]]:
    """How Hearthline's dispatcher plans: cut the map into one wedge per truck
    around the warehouse, with about the same volume in each, then drive each
    wedge slot by slot, nearest stop first within a slot."""
    dist, _ = matrices(req)
    d0 = req.depot
    stops = req.stops
    ang = [math.atan2(s.location.lat - d0.lat, (s.location.lon - d0.lon)
                      * math.cos(math.radians(d0.lat))) for s in stops]
    order = sorted(range(len(stops)), key=lambda k: ang[k])
    total = sum(s.demand for s in stops)
    target = total / len(req.vehicles)
    wedges, cur, vol = [], [], 0
    for k in order:
        if cur and (vol + stops[k].demand > req.vehicles[0].capacity or
                    (vol >= target and len(wedges) < len(req.vehicles) - 1)):
            wedges.append(cur)
            cur, vol = [], 0
        cur.append(k)
        vol += stops[k].demand
    wedges.append(cur)
    routes = []
    for w in wedges:
        route, here = [], 0
        for slot in sorted({stops[k].window_start for k in w}):
            todo = [k for k in w if stops[k].window_start == slot]
            while todo:
                k = min(todo, key=lambda k: dist[here, k + 1])
                route.append(k + 1)
                todo.remove(k)
                here = k + 1
        routes.append(route)
    return routes
# <<end_manual>>


# <<evaluate>>
def evaluate(req: PlanRequest, routes: list[list[int]], departs: list[int] | None = None,
             fleet: Fleet = Fleet()) -> dict:
    """Cost, service and workload of any plan, from the data alone.  Nodes
    are 1 + the index of the stop, a truck leaves at its departure (default:
    start of shift) and waits when it is early.  Every truck in the request
    is leased, so every one pays its lease, used or idle."""
    dist, secs = matrices(req)
    hours, miles, late, late_min, used, loads = [], 0.0, 0, 0.0, 0, []
    for v, r in enumerate(routes):
        if not r:
            continue
        used += 1
        t = departs[v] if departs else to_s(req.vehicles[v].shift_start)
        start, prev = t, 0
        for k in r:
            s = req.stops[k - 1]
            t = max(t + secs[prev, k], to_s(s.window_start))
            if t > to_s(s.window_end):
                late += 1
                late_min += (t - to_s(s.window_end)) / 60
            t += 60 * s.service_min
            prev = k
        t += secs[prev, 0]
        hours.append((t - start) / 3600)
        miles += sum(dist[a, b] for a, b in zip([0] + r, r + [0])) / 1609.344
        loads.append(sum(req.stops[k - 1].demand for k in r))
    lease = fleet.per_day * len(req.vehicles)
    cost = fleet.crew_per_hour * sum(hours) + fleet.per_mile * miles + lease
    served = sum(len(r) for r in routes)
    return {"cost": cost, "lease": lease, "miles": miles, "crew_hours": sum(hours),
            "trucks": used,
            "late": late, "late_min": late_min, "served": served,
            "longest_h": max(hours, default=0), "shortest_h": min(hours, default=0),
            "overloaded": sum(1 for x in loads if x > req.vehicles[0].capacity),
            "cost_per_stop": cost / max(1, served)}
# <<end_evaluate>>


def optimize(req: PlanRequest):
    """The routing service's plan, as node lists and departure times."""
    res = solve(req)
    pos = {s.id: k + 1 for k, s in enumerate(req.stops)}
    by_vehicle = {r.vehicle_id: r for r in res.routes}
    routes, departs = [], []
    for v in req.vehicles:
        r = by_vehicle.get(v.id)
        routes.append([pos[x.stop_id] for x in r.visits] if r else [])
        departs.append(to_s(r.depart) if r else 0)
    return routes, departs, res


def by_day(orders: list[Order]) -> dict[str, list[Order]]:
    out: dict[str, list[Order]] = defaultdict(list)
    for o in orders:
        out[o.day].append(o)
    return dict(sorted(out.items()))


def summary(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
