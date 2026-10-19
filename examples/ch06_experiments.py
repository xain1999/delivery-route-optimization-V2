"""Chapter 6: time windows and service times.

    python examples/ch06_experiments.py --book ../book      # about 15 minutes

Uses Tuesday's cleaned instance (Chapter 3), the customers' preferred
delivery hours (data/chicago_tuesday_slots.csv) and the cost model of
Chapter 1.  Travel times assume 15 mph on the Chapter 2 distance
estimates. Chapter 4 replaces both with road-network data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vrpbook import distance_matrix, load_csv, travel_time_matrix
from vrpbook import plotting
from vrpbook.formulations import vrptw_mip
from vrpbook.timewindows import (HOUR, TWData, best_departure, hhmm, load_preferences,
                                 precheck, schedule, service_seconds, slot_windows,
                                 solve_vrptw, validate_schedule)

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "results" / "ch06"
M_PER_MI = 1609.344
SPEED_MPH = 15.0
DEPOT_HOURS = (int(6.5 * HOUR), int(15.5 * HOUR))    # shifts from fleet_tuesday.json
WIDTHS = [8, 4, 2, 1]               # 8 = any time 7:00-15:00
MAX_VANS = 8                        # six of Lakeshore's plus two more rentals
FIXED_PER_VAN = 50.0                # dollars per van used per day (assumption)
DRIVER_PER_HOUR = 30.81             # Chapter 1
VEHICLE_PER_MILE = 0.76             # Chapter 1
LIMIT = 30
STRATEGIES = ["PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION",
              "LOCAL_CHEAPEST_INSERTION", "GLOBAL_CHEAPEST_ARC"]


def tuesday(width: int, n_vehicles: int = MAX_VANS) -> tuple[TWData, object]:
    inst = load_csv(DATA / "chicago_tuesday.csv", n_vehicles, 100)
    with (DATA / "chicago_tuesday.csv").open() as fh:
        weights = [int(r["weight_lb"]) for r in csv.DictReader(fh)]
    pref = load_preferences(DATA / "chicago_tuesday_slots.csv")
    dist = distance_matrix(inst.sites)
    ttime = travel_time_matrix(dist, SPEED_MPH * 1.609344)
    service = [0] + [service_seconds(q) for q in inst.demands[1:]]
    windows = [DEPOT_HOURS] + slot_windows([pref[i] for i in range(1, len(inst.sites))], width)
    data = TWData(dist, ttime, service, windows, inst.demands, 100, n_vehicles,
                  {"Weight": (weights, 2400)})
    return data, inst


def summarize(routes, data) -> dict:
    T = validate_schedule(routes, data)
    vans = sum(1 for r in routes if len(r) > 2)
    miles = T["meters"] / M_PER_MI
    paid = T["duration"] / HOUR
    cost = DRIVER_PER_HOUR * paid + VEHICLE_PER_MILE * miles + FIXED_PER_VAN * vans
    return {"vans": vans, "miles": miles, "drive_h": T["drive"] / HOUR,
            "service_h": T["service"] / HOUR, "wait_h": T["wait"] / HOUR,
            "paid_h": paid, "cost": cost, "cost_per_stop": cost / (len(data.demands) - 1)}


def solve(data, objective="cost", strategy="PARALLEL_CHEAPEST_INSERTION", limit=LIMIT):
    return solve_vrptw(data, objective=objective, time_limit_s=limit,
                       driver_per_hour=DRIVER_PER_HOUR, vehicle_per_mile=VEHICLE_PER_MILE,
                       fixed_per_van=FIXED_PER_VAN, first_solution=strategy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--from-json", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if args.from_json:
        R = json.loads(path.read_text())
    else:
        R = run()
        path.write_text(json.dumps(R, indent=1))
    if args.book:
        write_book_assets(args.book, R)


def run() -> dict:
    R: dict = {"widths": {}, "strategies": {}, "objectives": {}, "mip": []}
    # ---- 1. window widths -------------------------------------------------
    for w in WIDTHS:
        data, _ = tuesday(w)
        pc = precheck(data)
        sol = solve(data)
        row = summarize(sol.routes, data)
        row.update({"blocked_pct": 100 * pc["blocked_arcs"] / pc["arcs"],
                    "unreachable": pc["unreachable"], "routes": sol.routes})
        R["widths"][str(w)] = row
        print(w, {k: v for k, v in row.items() if k != "routes"}, flush=True)

    # ---- 2. first-solution strategies with 2-hour windows ----------------
    data, inst = tuesday(2)
    for st in STRATEGIES:
        sol = solve(data, strategy=st, limit=10)
        R["strategies"][st] = summarize(sol.routes, data)["cost"] if sol.routes else None
        print(st, R["strategies"][st], flush=True)

    # ---- 3. objectives with 2-hour windows ----------------------------------
    for obj in ("distance", "time", "cost"):
        sol = solve(data, objective=obj)
        R["objectives"][obj] = summarize(sol.routes, data)
        print(obj, R["objectives"][obj], flush=True)

    # ---- 4. worked example: one route's schedule -------------------------------
    routes = R["widths"]["2"]["routes"]
    r = min((r for r in routes if len(r) > 2), key=len)
    vis = schedule(r, data)
    R["worked"] = {"route": r, "names": [inst.sites[v.node].name for v in vis],
                   "windows": [data.windows[v.node] for v in vis],
                   "service": [data.service[v.node] for v in vis],
                   "travel": [0] + [int(data.time[i, j]) for i, j in zip(r, r[1:])],
                   "visits": [[v.node, v.arrival, v.start, v.departure] for v in vis],
                   "earliest_departure": data.windows[0][0],
                   "best_departure": best_departure(r, data)}

    # ---- 5. exact MIP on small instances --------------------------------------
    rng = np.random.default_rng(6)
    for n in (8, 10, 12, 14):
        ids = sorted(rng.choice([i for i in range(1, 34)], n, replace=False).tolist())
        keep = [0] + ids
        sub = TWData(data.dist[np.ix_(keep, keep)], data.time[np.ix_(keep, keep)],
                     [data.service[i] for i in keep], [data.windows[i] for i in keep],
                     [data.demands[i] for i in keep], 100, 4)
        lp = vrptw_mip(sub.dist, sub.time, sub.service, sub.windows, sub.demands, 100, 4,
                       relax=True)
        ip = vrptw_mip(sub.dist, sub.time, sub.service, sub.windows, sub.demands, 100, 4,
                       time_limit_s=300)
        ot = solve_vrptw(sub, objective="distance", time_limit_s=5)
        if ot.routes:
            validate_schedule(ot.routes, sub)
        row = {"n": n, "status": ip.status, "seconds": ip.seconds,
               "opt_mi": ip.objective / M_PER_MI, "bound_mi": ip.bound / M_PER_MI,
               "lp_gap": 100 * (ip.objective - lp.objective) / ip.objective,
               "ortools_mi": ot.objective / M_PER_MI if ot.routes else None}
        R["mip"].append(row)
        print(row, flush=True)
    return R


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    W = R["widths"]
    base = W["8"]
    m = {"limit": LIMIT, "fixed-van": f"\\${FIXED_PER_VAN:.0f}", "max-vans": MAX_VANS}
    for w, row in W.items():
        m[f"w{w}-vans"] = row["vans"]
        m[f"w{w}-cost"] = f"\\${row['cost']:,.0f}"
        m[f"w{w}-cps"] = f"\\${row['cost_per_stop']:.2f}"
        m[f"w{w}-wait"] = f"{row['wait_h']:.1f}"
        m[f"w{w}-miles"] = f"{row['miles']:.1f}"
        m[f"w{w}-paid"] = f"{row['paid_h']:.1f}"
        m[f"w{w}-pct"] = f"{100 * (row['cost'] / base['cost'] - 1):.0f}"
        m[f"w{w}-blocked"] = f"{row['blocked_pct']:.0f}"
    O = R["objectives"]
    for k, row in O.items():
        m[f"obj-{k}-cost"] = f"\\${row['cost']:,.0f}"
        m[f"obj-{k}-miles"] = f"{row['miles']:.1f}"
        m[f"obj-{k}-paid"] = f"{row['paid_h']:.1f}"
        m[f"obj-{k}-wait"] = f"{row['wait_h']:.1f}"
        m[f"obj-{k}-vans"] = row["vans"]
    m["obj-dist-vs-cost"] = f"{100 * (O['distance']['cost'] / O['cost']['cost'] - 1):.0f}"
    S = R["strategies"]
    m["strat-failed"] = sum(1 for v in S.values() if v is None)
    m["strat-total"] = len(S)
    # worked example
    Wk = R["worked"]
    m["we-stops"] = len(Wk["route"]) - 2
    m["we-earliest"] = hhmm(Wk["earliest_departure"])
    m["we-best"] = hhmm(Wk["best_departure"])
    vis0 = Wk["visits"]
    m["we-return"] = hhmm(vis0[-1][1])
    a0 = Wk["earliest_departure"]
    first_arrival = a0 + Wk["travel"][1]
    m["we-early-wait"] = f"{(Wk['windows'][1][0] - first_arrival) / 60:.0f}"
    m["we-early-arrive"] = hhmm(first_arrival)
    m["we-duration"] = str(vis0[-1][1] // 60 - vis0[0][3] // 60)   # as printed by hhmm
    # MIP
    M = R["mip"]
    m["mip-max-lpgap"] = f"{max(r['lp_gap'] for r in M):.0f}"
    m["mip-min-lpgap"] = f"{min(r['lp_gap'] for r in M):.0f}"
    lines = ["% Generated by code/examples/ch06_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch06/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch06-results.tex", "\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    names = {"8": "Any time, 7--15", "4": "4 hours", "2": "2 hours", "1": "1 hour"}
    for w in ("8", "4", "2", "1"):
        r = W[w]
        rows.append(f"{names[w]} & {r['blocked_pct']:.0f} & {r['vans']} & {r['miles']:.1f} & "
                    f"{r['drive_h']:.1f} & {r['wait_h']:.1f} & {r['paid_h']:.1f} & "
                    f"\\${r['cost']:,.0f} & {100 * (r['cost'] / base['cost'] - 1):+.0f}\\% \\\\")
    plotting.write_tex(gen / "ch06-width-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for k, label in (("distance", "Distance"), ("time", "Travel time"),
                     ("cost", "Cost (Chapter 1)")):
        r = O[k]
        rows.append(f"{label} & {r['vans']} & {r['miles']:.1f} & {r['drive_h']:.1f} & "
                    f"{r['wait_h']:.1f} & {r['paid_h']:.1f} & \\${r['cost']:,.0f} \\\\")
    plotting.write_tex(gen / "ch06-objective-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for st, v in S.items():
        cell = f"\\${v:,.0f}" if v is not None else "no plan found"
        rows.append(f"\\py{{{st}}} & {cell} \\\\")
    plotting.write_tex(gen / "ch06-strategy-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for k, (vis, name, win, svc, trav) in enumerate(zip(Wk["visits"], Wk["names"],
                                                        Wk["windows"], Wk["service"],
                                                        Wk["travel"])):
        node, arr, start, dep = vis
        label = "Depot" if node == 0 else name.title().replace(" N ", " N ").replace("Ave", "Ave")
        if k == 0:
            rows.append(f"{label} & & & & & & {hhmm(dep)} \\\\")
            continue
        if node == 0:                       # back at the depot
            rows.append(f"{label} & & {trav // 60} & {hhmm(arr)} & & & \\\\")
            continue
        wtxt = f"{hhmm(win[0])}--{hhmm(win[1])}"
        rows.append(f"{label} & {wtxt} & {trav // 60} & {hhmm(arr)} & {(start - arr) // 60} & "
                    f"{hhmm(start)} & {hhmm(dep)} \\\\")
    plotting.write_tex(gen / "ch06-worked-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for r in M:
        t = (f"{r['seconds']:.1f}" if r["seconds"] >= 0.1 else "$<0.1$") \
            if r["status"] == "optimal" else \
            f"({100 * (r['opt_mi'] - r['bound_mi']) / r['opt_mi']:.1f}\\%)"
        ot = f"{r['ortools_mi']:.1f}" if r["ortools_mi"] else "--"
        rows.append(f"{r['n']} & {r['opt_mi']:.1f} & {r['lp_gap']:.0f} & {t} & {ot} \\\\")
    plotting.write_tex(gen / "ch06-mip-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    _plot_widths(W, book / "figures" / "ch06" / "widths.pdf")
    print("book assets written")


def _plot_widths(W, path: Path) -> None:
    """Stacked paid hours by activity, one bar per window width."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    order = ["8", "4", "2", "1"]
    labels = ["Any time", "4-hour", "2-hour", "1-hour"]
    drive = [W[w]["drive_h"] for w in order]
    serv = [W[w]["service_h"] for w in order]
    wait = [W[w]["wait_h"] for w in order]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    y = np.arange(len(order))[::-1]
    kw = dict(height=0.56, edgecolor="white", linewidth=1.5)
    ax.barh(y, drive, color=plotting.ROUTE_COLORS[0], label="Driving", **kw)
    ax.barh(y, serv, left=drive, color=plotting.ROUTE_COLORS[2], label="At customers",
            hatch="....", **kw)
    left2 = [a + b for a, b in zip(drive, serv)]
    ax.barh(y, wait, left=left2, color=plotting.ROUTE_COLORS[1], label="Waiting",
            hatch="////", **kw)
    for yi, w in zip(y, order):
        total = W[w]["paid_h"]
        ax.annotate(f"{total:.1f} h, \\${W[w]['cost']:,.0f}".replace("\\$", "$"),
                    (total, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=plotting.INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel("paid driver hours (all vans)")
    ax.set_xlim(0, max(W[w]["paid_h"] for w in order) * 1.32)
    ax.grid(True, axis="x", color=plotting.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    plotting.save(fig, path)
    plt.close(fig)


if __name__ == "__main__":
    main()
