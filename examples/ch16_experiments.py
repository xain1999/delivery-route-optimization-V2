"""Chapter 16: the Hearthline Home case study, end to end.

    python tools/make_hearthline.py                              # the raw export
    python examples/ch16_experiments.py --book ../book         # about 7 minutes

Four weeks of orders: cleaned, planned every day through the routing
service of Chapter 15, and compared with the dispatcher's manual method.
Then sensitivity to fleet size, window width and depot location.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from case_study.hearthline import (DEPOTS, Fleet, by_day, evaluate, load, manual_plan,
                                 to_request)
from routing_service.app import app
from routing_service.solver import matrices, solve, to_s
from vrpbook import plotting

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch16"
SECONDS = 5
SENS_DAYS = 5
FLEETS = [6, 7, 8, 10]
WIDEN = [0, 1, 12]               # 2-hour slots, 4-hour slots, any time of day
DAYS_PER_YEAR = 250


def via_service(client, req):
    """Plan through the HTTP API, exactly as the order platform would."""
    plan = client.post("/plans/solve", json=req.model_dump()).json()
    pos = {s.id: k + 1 for k, s in enumerate(req.stops)}
    by_vehicle = {r["vehicle_id"]: r for r in plan["routes"]}
    routes, departs = [], []
    for v in req.vehicles:
        r = by_vehicle.get(v.id)
        routes.append([pos[x["stop_id"]] for x in r["visits"]] if r else [])
        departs.append(to_s(r["depart"]) if r else 0)
    return routes, departs, plan


def run() -> dict:
    orders, issues = load()
    days = by_day(orders)
    client = TestClient(app)
    fleet = Fleet()
    R = {"issues": {k: len(v) for k, v in issues.items()}, "orders": len(orders),
         "days": len(days), "per_day": [len(v) for v in days.values()],
         "cuft": [o.cuft for o in orders], "minutes": [o.minutes for o in orders],
         "baseline": []}
    for day, todays in days.items():
        req = to_request(todays, fleet, time_limit_s=SECONDS)
        man = evaluate(req, manual_plan(req), fleet=fleet)
        routes, departs, plan = via_service(client, req)
        opt = evaluate(req, routes, departs, fleet=fleet)
        opt["unassigned"] = len(plan["unassigned"])
        R["baseline"].append({"day": day, "orders": len(todays), "manual": man, "opt": opt,
                              "left_out": plan["unassigned"],
                              "impossible": impossible(req)})
        print(day, round(man["cost"]), round(opt["cost"]), man["late"], opt["unassigned"],
              flush=True)
    sample = list(days.items())[:SENS_DAYS]
    R["fleet"], R["widen"], R["depot"] = {}, {}, {}
    for k in FLEETS:
        R["fleet"][k] = _scenario(client, sample, Fleet(trucks=k))
    for w in WIDEN:
        R["widen"][w] = _scenario(client, sample, fleet, widen_h=w)
    for name in DEPOTS:
        R["depot"][name] = _scenario(client, sample, fleet, depot=name)
    return R


def _scenario(client, sample, fleet, **kw):
    rows, left_out = [], []
    for _, todays in sample:
        req = to_request(todays, fleet, time_limit_s=SECONDS, **kw)
        routes, departs, plan = via_service(client, req)
        e = evaluate(req, routes, departs, fleet=fleet)
        e["unassigned"] = len(plan["unassigned"])
        e["impossible"] = len(impossible(req))
        rows.append(e)
        left_out += plan["unassigned"]
    out = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    print(kw, fleet.trucks, {k: round(v, 1) for k, v in out.items()}, flush=True)
    out["left_out"] = left_out
    return out


def diagnose(R) -> list[dict]:
    """Why were the other orders left out?  Re-plan every day that left out an
    order some truck could serve, (a) with the same eight trucks and six times
    the time, (b) with ten trucks and the same time.  Served in (a): the search
    ran out of time. Served only in (b): the fleet was short."""
    days = by_day(load()[0])
    out = []
    for b in R["baseline"]:
        imp = {o["id"] for o in b["impossible"]}
        other = [u["stop_id"] for u in b["left_out"] if u["stop_id"] not in imp]
        if not other:
            continue
        row = {"day": b["day"], "left": other}
        for label, fleet, secs in (("longer", Fleet(), 6 * SECONDS),
                                   ("ten", Fleet(trucks=10), SECONDS)):
            plan = solve(to_request(days[b["day"]], fleet, time_limit_s=secs))
            row[label] = [u.stop_id for u in plan.unassigned if u.stop_id in other]
        print("diagnosis", row, flush=True)
        out.append(row)
    return out


def impossible(req) -> list[dict]:
    """Orders that no truck could serve even if it served nothing else: the
    earliest a dedicated truck can start is the window's opening (or its
    arrival, if later), and it must be back by the end of the shift."""
    _, secs = matrices(req)
    out = []
    for k, s in enumerate(req.stops, start=1):
        for v in req.vehicles:
            begin = max(to_s(v.shift_start) + secs[0, k], to_s(s.window_start))
            back = begin + 60 * s.service_min + secs[k, 0]
            if begin <= to_s(s.window_end) and back <= to_s(v.shift_end):
                break
        else:
            begin = max(to_s(req.vehicles[0].shift_start) + secs[0, k], to_s(s.window_start))
            out.append({"id": s.id, "window": [s.window_start, s.window_end],
                        "service_min": s.service_min,
                        "back": float(begin + 60 * s.service_min + secs[k, 0])})
    return out


# ---------------------------------------------------------------- book assets
def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    B = R["baseline"]
    n = len(B)
    m = {"days": R["days"], "orders": R["orders"], "seconds": SECONDS,
         "sens-days": SENS_DAYS, "per-year": DAYS_PER_YEAR,
         "dupes": R["issues"].get("duplicate row", 0),
         "nocoords": R["issues"].get("no coordinates", 0),
         "per-day-min": min(R["per_day"]), "per-day-max": max(R["per_day"]),
         "per-day-mean": f"{sum(R['per_day']) / len(R['per_day']):.0f}",
         "cuft-mean": f"{sum(R['cuft']) / len(R['cuft']):.0f}",
         "minutes-mean": f"{sum(R['minutes']) / len(R['minutes']):.0f}"}
    m["rows"] = m["orders"] + m["dupes"] + m["nocoords"]

    def avg(side, k):
        return sum(b[side][k] for b in B) / n

    def pooled(side):                     # total cost over total orders delivered
        return sum(b[side]["cost"] for b in B) / sum(b[side]["served"] for b in B)

    rows = []
    for label, k, fmt in (("Cost per day (\\$)", "cost", "{:,.0f}"),
                          ("Cost per order delivered (\\$)", "pooled", "{:,.2f}"),
                          ("Miles per day", "miles", "{:,.0f}"),
                          ("Crew hours per day", "crew_hours", "{:,.1f}"),
                          ("Trucks used", "trucks", "{:.1f}"),
                          ("Late deliveries per day", "late", "{:.1f}"),
                          ("Minutes late per day", "late_min", "{:,.0f}"),
                          ("Longest crew day (h)", "longest_h", "{:.1f}")):
        a, b = (pooled("manual"), pooled("opt")) if k == "pooled" else (avg("manual", k),
                                                                           avg("opt", k))
        ch = f"{100 * (b / a - 1):+.0f}\\%" if a else "--"
        rows.append(f"{label} & {fmt.format(a)} & {fmt.format(b)} & {ch} \\\\")
    plotting.write_tex(gen / "ch16-baseline-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    man_c, opt_c = avg("manual", "cost"), avg("opt", "cost")
    m["save-day"] = f"\\${man_c - opt_c:,.0f}"
    m["save-pct"] = f"{100 * (1 - opt_c / man_c):.0f}"
    m["save-year"] = f"\\${(man_c - opt_c) * DAYS_PER_YEAR:,.0f}"
    m["manual-late"] = f"{avg('manual', 'late'):.1f}"
    m["manual-late-days"] = sum(1 for b in B if b["manual"]["late"] > 0)
    m["opt-late"] = f"{avg('opt', 'late'):.1f}"
    m["opt-unassigned"] = sum(b["opt"]["unassigned"] for b in B)
    m["manual-longest"] = f"{max(b['manual']['longest_h'] for b in B):.1f}"
    m["opt-longest"] = f"{max(b['opt']['longest_h'] for b in B):.1f}"
    m["opt-better-days"] = sum(1 for b in B if b["opt"]["cost"] < b["manual"]["cost"])
    m["cps-pct"] = f"{100 * (1 - pooled('opt') / pooled('manual')):.0f}"
    m["lease-day"] = f"\\${B[0]['opt']['lease']:,.0f}"
    # why orders were left out: impossible for any truck, or a choice of the plan
    left = [u for b in B for u in b["left_out"]]
    D = R.get("diagnosis", [])
    m["diag-n"] = sum(len(d["left"]) for d in D)
    m["diag-longer"] = sum(1 for d in D for x in d["left"] if x not in d["longer"])
    m["diag-ten"] = sum(1 for d in D for x in d["left"] if x not in d["ten"])
    m["diag-ten-only"] = sum(1 for d in D for x in d["left"]
                             if x not in d["ten"] and x in d["longer"])
    m["diag-neither"] = sum(1 for d in D for x in d["left"]
                            if x in d["ten"] and x in d["longer"])
    m["diag-days"] = len(D)
    imp = {o["id"]: o for b in B for o in b["impossible"]}
    m["imp-total"] = len(imp)
    m["left-imp"] = sum(1 for u in left if u["stop_id"] in imp)
    m["left-other"] = len(left) - m["left-imp"]
    if imp:
        ex = max(imp.values(), key=lambda o: o["service_min"])
        m["imp-ex-id"] = ex["id"]
        m["imp-ex-window"] = f"{ex['window'][0]}--{ex['window'][1]}"
        m["imp-ex-min"] = f"{ex['service_min']:.0f}"
        h, r = divmod(round(ex["back"]), 3600)
        m["imp-ex-back"] = f"{h}:{r // 60:02d}"
    m["miles-pct"] = f"{100 * (1 - avg('opt', 'miles') / avg('manual', 'miles')):.0f}"
    m["hours-pct"] = f"{100 * (1 - avg('opt', 'crew_hours') / avg('manual', 'crew_hours')):.0f}"

    def sens_rows(D, label):
        out = []
        for key, v in D.items():
            out.append(f"{label(key)} & \\${v['cost']:,.0f} & {v['miles']:,.0f} & "
                       f"{v['crew_hours']:.1f} & {v['trucks']:.1f} & {v['unassigned']:.1f} & "
                       f"{v['longest_h']:.1f} \\\\")
        return out
    rows = sens_rows(R["fleet"], lambda k: f"{k} trucks")
    rows.append("\\midrule")
    rows += sens_rows(R["widen"], lambda w: {"0": "2-hour slots", "1": "4-hour slots",
                                             "12": "any time"}[str(w)])
    rows.append("\\midrule")
    rows += sens_rows(R["depot"], lambda d: f"Depot: {d}")
    plotting.write_tex(gen / "ch16-sens-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    F, W, D = R["fleet"], R["widen"], R["depot"]
    m["w4-save"] = f"{100 * (1 - W['1']['cost'] / W['0']['cost']):.0f}"
    m["wany-save"] = f"{100 * (1 - W['12']['cost'] / W['0']['cost']):.0f}"
    m["w4-save-year"] = f"\\${(W['0']['cost'] - W['1']['cost']) * DAYS_PER_YEAR:,.0f}"
    best_depot = min(D, key=lambda k: D[k]["cost"])
    worst_depot = max(D, key=lambda k: D[k]["cost"])
    m["best-depot"], m["worst-depot"] = best_depot, worst_depot
    m["depot-spread"] = f"\\${D[worst_depot]['cost'] - D[best_depot]['cost']:,.0f}"
    m["depot-spread-pct"] = f"{100 * (D[worst_depot]['cost'] / D[best_depot]['cost'] - 1):.0f}"
    m["f6-unassigned"] = f"{F['6']['unassigned']:.1f}"
    m["f7-unassigned"] = f"{F['7']['unassigned']:.1f}"
    m["f8-unassigned"] = f"{F['8']['unassigned']:.1f}"
    m["f10-unassigned"] = f"{F['10']['unassigned']:.1f}"
    m["f8-impossible"] = f"{F['8']['impossible']:.1f}"
    m["f10-impossible"] = f"{F['10']['impossible']:.1f}"
    m["f10-extra"] = f"\\${F['10']['cost'] - F['8']['cost']:,.0f}"
    m["f10-trucks"] = f"{F['10']['trucks']:.1f}"
    m["f10-cost"] = f"\\${F['10']['cost']:,.0f}"
    m["f8-cost"] = f"\\${F['8']['cost']:,.0f}"
    lines = ["% Generated by code/examples/ch16_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch16/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch16-results.tex", "\n".join(lines) + "\n", encoding="utf-8")
    _plot_days(B, book / "figures" / "ch16" / "days.pdf")
    print("book assets written")


def _plot_days(B, path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = range(1, len(B) + 1)
    ax.plot(x, [b["manual"]["cost"] for b in B], "--s", color=plotting.ROUTE_COLORS[1],
            markersize=3.5, linewidth=1.2, label="dispatcher's method")
    ax.plot(x, [b["opt"]["cost"] for b in B], "-o", color=plotting.ROUTE_COLORS[0],
            markersize=3.5, linewidth=1.2, label="routing service")
    ax.set_xlabel("working day")
    ax.set_ylabel("cost of the day (\\$)")
    ax.set_xticks(list(x))
    ax.grid(True, color=plotting.GRID, linewidth=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    plotting.save(fig, path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--from-json", action="store_true")
    ap.add_argument("--diagnose-only", action="store_true",
                    help="add the left-out diagnosis to saved results")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    if args.from_json or args.diagnose_only:
        R = json.loads(path.read_text())
    else:
        R = run()
    if not args.from_json:
        R["diagnosis"] = diagnose(R)
        path.write_text(json.dumps(R, indent=1, default=float))
        R = json.loads(path.read_text())
    if args.book:
        write_book_assets(args.book, R)


if __name__ == "__main__":
    main()
