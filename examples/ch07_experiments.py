"""Chapter 7: shifts, breaks, a mixed fleet and refrigerated orders.

    python examples/ch07_experiments.py --book ../book      # about 4 minutes

Tuesday with two-hour slots (Chapter 6), now with Lakeshore's real fleet:
four cargo vans and one refrigerated van with salaried drivers, and up to
three rental vans with agency drivers paid by the hour.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch06_experiments as ch06  # noqa: E402  (same Tuesday instance)

from vrpbook import plotting
from vrpbook.fleet import (BreakRule, VehicleType, add_reload_nodes, check_plan, expand,
                           solve_rich, true_cost)
from vrpbook.timewindows import HOUR

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch07"
LIMIT = 30


def H(h: float) -> int:
    return int(h * HOUR)


# <<fleet>>
FLEET = [
    VehicleType("Cargo van", 4, totes=100, payload_lb=2400, per_mile=0.76, fixed=0,
                hourly=0, shift=(H(6.5), H(15.5))),
    VehicleType("Refrigerated van", 1, totes=80, payload_lb=2000, per_mile=0.95, fixed=0,
                hourly=0, shift=(H(6.5), H(15.5)), refrigerated=True),
    VehicleType("Rental van", 3, totes=100, payload_lb=2400, per_mile=0.76, fixed=50,
                hourly=38, shift=(H(7), H(15.5)), max_hours=7.0),
]
# <<end_fleet>>
OVERTIME_RATE = 1.5 * 30.81


def with_overtime(fleet, hours=1.0):
    return [VehicleType(**{**vt.__dict__, "overtime_h": hours, "overtime_rate": OVERTIME_RATE})
            if vt.hourly == 0 else vt for vt in fleet]


def load():
    data, inst = ch06.tuesday(2)
    weights = data.extra["Weight"][0]
    with (HERE / "data" / "chicago_tuesday_slots.csv").open() as fh:
        cold = [False] + [bool(int(r["needs_cold"])) for r in csv.DictReader(fh)]
    return data, inst, weights, cold


def run() -> dict:
    data, inst, weights, cold = load()
    rule = BreakRule()
    variants = {
        "full": dict(fleet=FLEET, cold=cold, breaks=rule),
        "no_breaks": dict(fleet=FLEET, cold=cold, breaks=None),
        "no_cold": dict(fleet=FLEET, cold=None, breaks=rule),
        "price_all_time": dict(fleet=FLEET, cold=cold, breaks=rule, price=30.81),
        "overtime": dict(fleet=with_overtime(FLEET), cold=cold, breaks=rule),
        "two_trips": dict(fleet=FLEET, cold=cold, breaks=rule, reloads=True),
    }
    R = {"variants": {}, "cold_stops": sum(cold), "cold_totes": sum(
        q for q, c in zip(data.demands, cold) if c)}
    for name, v in variants.items():
        vehicles = expand(v["fleet"])
        d, w, c, rel = data, weights, v["cold"], []
        if v.get("reloads"):
            d, w, c, rel = add_reload_nodes(data, weights, v["cold"], len(vehicles))
        plan = solve_rich(d, vehicles, w, cold=c, breaks=v["breaks"], reload_nodes=rel,
                          price_salaried_time=v.get("price", 0.0), time_limit_s=LIMIT)
        all_cold = cold + [False] * len(rel)
        chk = check_plan(plan, d, w, all_cold, rule, rel)   # always check every rule
        cost = true_cost(plan, d, rel)
        used = [k for k, r in enumerate(plan.routes) if len(r) > 2]
        R["variants"][name] = {
            "status": plan.status, "violations": chk["violations"], "cost": cost,
            "vans": chk["vans"], "rentals": sum(1 for k in used if vehicles[k].fixed > 0),
            "routes": [plan.routes[k] for k in used], "starts": [plan.starts[k] for k in used],
            "breaks": [plan.breaks[k] for k in used],
            "vehicles": [vehicles[k].name for k in used],
            "stops_per_rental": [len(plan.routes[k]) - 2 for k in used if vehicles[k].fixed > 0],
            "reload_nodes": rel, "trips": chk.get("trips", chk["vans"]),
            "n_reloads": sum(1 for k in used for i in plan.routes[k] if i in set(rel)),
        }
        print(name, plan.status, chk["vans"], round(cost["total"], 1), chk["violations"][:3],
              flush=True)

    # the Chapter 6 plan (identical vans) checked against this chapter's rules
    ch6 = json.loads((HERE / "results" / "ch06" / "results.json").read_text())
    routes6 = [r for r in ch6["widths"]["2"]["routes"] if len(r) > 2]
    cold_per_route = [sum(cold[i] for i in r[1:-1]) for r in routes6]
    fridge = int(np.argmax(cold_per_route))
    R["ch6_check"] = {"vans": len(routes6), "cold_elsewhere": sum(cold_per_route) -
                      cold_per_route[fridge],
                      "routes_with_cold": sum(1 for c in cold_per_route if c)}
    return R


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    V = R["variants"]
    full = V["full"]["cost"]["total"]
    m = {"limit": LIMIT, "cold-stops": R["cold_stops"], "cold-totes": R["cold_totes"],
         "ot-rate": f"\\${OVERTIME_RATE:.2f}",
         "ch6-vans": R["ch6_check"]["vans"], "ch6-cold-elsewhere": R["ch6_check"]["cold_elsewhere"],
         "ch6-routes-with-cold": R["ch6_check"]["routes_with_cold"]}
    for k, v in V.items():
        key = k.replace("_", "")
        c = v["cost"]
        m[f"{key}-cost"] = f"\\${c['total']:,.0f}"
        m[f"{key}-pct"] = f"{100 * (c['total'] / full - 1):+.0f}"
        m[f"{key}-pctabs"] = f"{abs(100 * (c['total'] / full - 1)):.0f}"
        m[f"{key}-vans"] = v["vans"]
        m[f"{key}-rentals"] = v["rentals"]
        m[f"{key}-salh"] = f"{c['salaried_h']:.1f}"
        m[f"{key}-agencyh"] = f"{c['hourly_h']:.1f}"
        m[f"{key}-oth"] = f"{c['overtime_h']:.1f}"
        m[f"{key}-miles"] = f"{c['miles']:.1f}"
        m[f"{key}-viol"] = len(v["violations"])
    m["full-rental-stops"] = sum(V["full"]["stops_per_rental"])
    m["twotrips-reloads"] = V["two_trips"]["n_reloads"]
    k = V["two_trips"]["rentals"]
    m["twotrips-rentals-phrase"] = {0: "no rental van", 1: "one rental van"}.get(k, f"{k} rental vans")
    m["twotrips-trips"] = V["two_trips"]["trips"]
    m["twotrips-saving"] = f"\\${full - V['two_trips']['cost']['total']:,.0f}"
    m["nocold-saving"] = f"\\${full - V['no_cold']['cost']['total']:,.0f}"
    lines = ["% Generated by code/examples/ch07_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch07/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch07-results.tex", "\n".join(lines) + "\n", encoding="utf-8")

    labels = {"full": "All rules (this chapter)", "no_breaks": "Without the meal break",
              "no_cold": "Without the refrigeration rule",
              "price_all_time": "Salaried time priced as in Chapter 6",
              "overtime": "With up to 1 hour of overtime",
              "two_trips": "Vans may reload at the depot"}
    rows = []
    for k in ("full", "no_breaks", "no_cold", "price_all_time", "overtime", "two_trips"):
        v, c = V[k], V[k]["cost"]
        viol = "--" if not v["violations"] else str(len(v["violations"]))
        rows.append(f"{labels[k]} & {v['vans']} ({v['rentals']}) & {c['miles']:.0f} & "
                    f"{c['hourly_h']:.1f} & {c['overtime_h']:.1f} & \\${c['total']:,.0f} & "
                    f"{100 * (c['total'] / full - 1):+.0f}\\% & {viol} \\\\")
    plotting.write_tex(gen / "ch07-variant-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    _gantt(R, "full", book / "figures" / "ch07" / "gantt.pdf")
    _gantt(R, "two_trips", book / "figures" / "ch07" / "gantt-reload.pdf")
    print("book assets written")


def _gantt(R: dict, variant: str, path: Path) -> None:
    """One row per vehicle: driving, service, waiting and the meal break."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    data, inst, weights, cold = load()
    v = R["variants"][variant]
    rel = v["reload_nodes"]
    if rel:
        data, weights, cold, rel = add_reload_nodes(data, weights, cold, len(rel))
    plotting.use_book_style()
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    C = {"drive": plotting.ROUTE_COLORS[0], "service": plotting.ROUTE_COLORS[2],
         "wait": plotting.ROUTE_COLORS[1], "break": "#3d3c39", "reload": "#a3a29c"}
    seen = set()

    def seg(y, a, b, kind, hatch=None):
        lab = {"drive": "Driving", "service": "At customer", "wait": "Waiting",
               "break": "Meal break", "reload": "Reloading"}[kind] if kind not in seen else None
        seen.add(kind)
        ax.barh(y, (b - a) / HOUR, left=a / HOUR, height=0.6, color=C[kind], hatch=hatch,
                edgecolor="white", linewidth=0.6, label=lab)

    rows = list(zip(v["vehicles"], v["routes"], v["starts"], v["breaks"]))
    for y, (name, r, s, brk) in enumerate(reversed(rows)):
        for k, (i, j) in enumerate(zip(r, r[1:])):
            dep = s[k] + data.service[i]
            if i != 0:
                seg(y, s[k], dep, "reload" if i in rel else "service")
            arrive = dep + int(data.time[i, j])
            if brk and dep <= brk[0] < s[k + 1]:
                if brk[0] < arrive:              # break during the drive
                    seg(y, dep, brk[0], "drive")
                    seg(y, brk[0], brk[1], "break")
                    seg(y, brk[1], arrive + (brk[1] - brk[0]), "drive")
                    arrive += brk[1] - brk[0]
                else:                            # break while waiting
                    seg(y, dep, arrive, "drive")
                    seg(y, arrive, brk[0], "wait", "////")
                    seg(y, brk[0], brk[1], "break")
                    arrive = brk[1]
            else:
                seg(y, dep, arrive, "drive")
            if s[k + 1] > arrive:
                seg(y, arrive, s[k + 1], "wait", "////")
        if brk and not (s[0] <= brk[0] < s[-1]):  # break before or after the route
            seg(y, brk[0], brk[1], "break")
    count, labels = {}, []
    for n, *_ in rows:                      # "Cargo van 1", "Cargo van 2", ...
        count[n] = count.get(n, 0) + 1
        labels.append(n if n == "Refrigerated van" else f"{n} {count[n]}")
    ax.set_yticks(range(len(rows)), labels[::-1], fontsize=7.5)
    for h in (7, 9, 11, 13, 15):
        ax.axvline(h, color=plotting.GRID, linewidth=0.8, zorder=0)
    ax.set_xlim(6.5, 15.75)
    ax.set_xticks(range(7, 16), [f"{h}:00" for h in range(7, 16)], fontsize=7.5)
    ax.set_xlabel("time of day (two-hour slots begin at 7, 9, 11 and 13)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False,
              columnspacing=1.0, handlelength=1.6)
    plotting.save(fig, path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--from-json", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    R = json.loads(path.read_text()) if args.from_json else run()
    if not args.from_json:
        path.write_text(json.dumps(R, indent=1))
    if args.book:
        write_book_assets(args.book, R)


if __name__ == "__main__":
    main()
