"""Chapter 9: when orders do not all fit.

    python examples/ch09_experiments.py --book ../book      # about 25 minutes

The Wednesday before Thanksgiving (data/chicago_peak.csv, built by
tools/make_peak_day.py): 47 orders, 20% more totes than Lakeshore's six
vans can carry, two-hour delivery slots and the cost model of Chapter 6.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from vrpbook import distance_matrix, load_csv, plotting, travel_time_matrix
from vrpbook.optional import (PenaltyPolicy, orienteering_mip, removal_savings,
                              solve_orienteering, solve_with_penalties)
from vrpbook.timewindows import (HOUR, TWData, service_seconds, slot_windows,
                                 solve_vrptw, validate_schedule)

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "results" / "ch09"
M_PER_MI = 1609.344
SPEED_MPH = 15.0
DEPOT_HOURS = (int(6.5 * HOUR), int(15.5 * HOUR))
N_VANS = 6                       # four owned vans and two rentals: all there is
CAPACITY, PAYLOAD = 100, 2400
FIXED_PER_VAN = 50.0
DRIVER_PER_HOUR = 30.81
VEHICLE_PER_MILE = 0.76
PER_MILE_ALL_IN = 2.81           # Chapter 1: vehicle plus driver time per mile
LIMIT = float(os.environ.get("CH09_LIMIT", 60))
LAMBDAS = [0, 10, 20, 40, 60, 80, 100, 150, 250, 500, 1000, 5000]
EVENING_HOURS = [2, 3, 4]
EVENING_RENTAL = 60.0            # a van for the evening, dollars (assumption)
LATE_MARGIN = 0.30               # gross margin on a late basket
LATE_FEE = 9.99                  # same-evening delivery fee
COST = dict(driver_per_hour=DRIVER_PER_HOUR, vehicle_per_mile=VEHICLE_PER_MILE,
            fixed_per_van=FIXED_PER_VAN)


def peak() -> tuple[TWData, list[dict], object]:
    path = DATA / "chicago_peak.csv"
    inst = load_csv(path, N_VANS, CAPACITY)
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    dist = distance_matrix(inst.sites)
    ttime = travel_time_matrix(dist, SPEED_MPH * 1.609344)
    service = [0] + [service_seconds(q) for q in inst.demands[1:]]
    windows = [DEPOT_HOURS] + slot_windows([int(r["preferred_hour"]) for r in rows[1:]], 2)
    weights = [int(r["weight_lb"]) for r in rows]
    data = TWData(dist, ttime, service, windows, inst.demands, CAPACITY, N_VANS,
                  {"Weight": (weights, PAYLOAD)})
    return data, rows, inst


def solve(data, penalties, limit=LIMIT):
    return solve_vrptw(data, objective="cost", time_limit_s=limit,
                       driver_per_hour=DRIVER_PER_HOUR, vehicle_per_mile=VEHICLE_PER_MILE,
                       fixed_per_van=FIXED_PER_VAN, penalties=penalties)


def evaluate(routes, data, rows, policy_pen) -> dict:
    """Routing cost from an independent schedule, plus what the dropped
    orders cost the business under the penalty policy."""
    T = validate_schedule(routes, data, allow_dropped=True)
    weights = data.extra["Weight"][0]
    for r in routes:
        assert sum(weights[i] for i in r[1:-1]) <= PAYLOAD, "payload"
    vans = sum(1 for r in routes if len(r) > 2)
    miles = T["meters"] / M_PER_MI
    paid = T["duration"] / HOUR
    routing = DRIVER_PER_HOUR * paid + VEHICLE_PER_MILE * miles + FIXED_PER_VAN * vans
    D = T["dropped"]
    pen = sum(policy_pen[i] for i in D)
    return {"vans": vans, "miles": miles, "paid_h": paid, "routing": routing,
            "dropped": D, "n_dropped": len(D), "served": len(data.demands) - 1 - len(D),
            "totes_dropped": sum(data.demands[i] for i in D),
            "totes_served": sum(data.demands) - sum(data.demands[i] for i in D),
            "club_dropped": sum(rows[i]["tier"] == "club" for i in D),
            "repeat_dropped": sum(int(rows[i]["postponed"]) for i in D),
            "value_dropped": sum(int(rows[i]["value_usd"]) for i in D),
            "penalty": pen, "total": routing + pen, "routes": routes}


def run() -> dict:
    data, rows, inst = peak()
    n = len(rows) - 1
    policy = PenaltyPolicy()
    pen = [0.0] + [policy.penalty(int(r["value_usd"]), r["tier"], r["postponed"] == "1")
                   for r in rows[1:]]
    R: dict = {"orders": n, "totes": sum(data.demands),
               "weight": sum(data.extra["Weight"][0]),
               "club": sum(r["tier"] == "club" for r in rows),
               "repeat": sum(r["postponed"] == "1" for r in rows),
               "value": sum(int(r["value_usd"]) for r in rows),
               "penalties": pen, "plans": {}}

    # ---- 1. every order mandatory -----------------------------------------
    sol = solve(data, None, limit=min(LIMIT, 30))
    R["mandatory_status"] = sol.status if not sol.routes else "solved"
    print("mandatory", R["mandatory_status"], flush=True)

    # ---- 2. three ways to set penalties, judged by the policy -------------
    mean_pen = float(np.mean(pen[1:]))
    R["mean_penalty"] = mean_pen
    designs = {"uniform": [None] + [mean_pen] * n,
               "value": [None] + [float(r["value_usd"]) for r in rows[1:]],
               "policy": [None] + pen[1:]}
    R["strategy_runs"] = {}
    for name, p in designs.items():
        best, runs = solve_with_penalties(data, p, time_limit_s=LIMIT, **COST)
        R["plans"][name] = evaluate(best.routes, data, rows, pen)
        R["strategy_runs"][name] = runs
        print(name, runs, {k: v for k, v in R["plans"][name].items() if k != "routes"},
              flush=True)

    # ---- 3. the price of service: one uniform penalty, swept ---------------
    R["sweep"] = []
    for lam in LAMBDAS:
        best, _ = solve_with_penalties(data, [None] + [float(lam)] * n,
                                       strategies=("SAVINGS", "GLOBAL_CHEAPEST_ARC"),
                                       time_limit_s=min(LIMIT, 15), **COST)
        e = evaluate(best.routes, data, rows, pen)
        R["sweep"].append({"lambda": lam, **{k: e[k] for k in
                           ("served", "totes_served", "routing", "vans", "club_dropped")}})
        print("lambda", R["sweep"][-1], flush=True)

    # ---- 4. an evening van for late orders: orienteering -------------------
    late = load_csv(DATA / "chicago_peak_late.csv", 1, CAPACITY)
    with (DATA / "chicago_peak_late.csv").open() as fh:
        lrows = list(csv.DictReader(fh))
    ldist = distance_matrix(late.sites)
    ltime = travel_time_matrix(ldist, SPEED_MPH * 1.609344)
    lserv = [0] + [service_seconds(q) for q in late.demands[1:]]
    prize = [0.0] + [LATE_MARGIN * int(r["value_usd"]) + LATE_FEE for r in lrows[1:]]
    R["late"] = {"orders": len(lrows) - 1, "totes": sum(late.demands),
                 "prize_all": sum(prize)}
    R["evening"] = []
    for h in EVENING_HOURS:
        b = int(h * HOUR)
        dflt = solve_orienteering(ltime, lserv, prize, late.demands, b, CAPACITY,
                                  time_limit_s=10, swap_moves=False,
                                  metaheuristic="GUIDED_LOCAL_SEARCH")
        ot = solve_orienteering(ltime, lserv, prize, late.demands, b, CAPACITY,
                                time_limit_s=10)
        ex = orienteering_mip(ltime, lserv, prize, late.demands, b, CAPACITY,
                              time_limit_s=600)
        r = ot["route"]
        miles = sum(int(ldist[i, j]) for i, j in zip(r, r[1:])) / M_PER_MI
        cost = EVENING_RENTAL + DRIVER_PER_HOUR * h + VEHICLE_PER_MILE * miles
        R["evening"].append({"hours": h, "stops": len(r) - 2, "prize": ot["prize"],
                             "default_prize": dflt["prize"],
                             "totes": sum(late.demands[i] for i in r),
                             "mip_prize": ex.get("prize"), "mip_rounds": ex.get("rounds"),
                             "mip_status": ex["status"], "mip_route": ex.get("route"),
                             "duration_h": ot["duration"] / HOUR, "miles": miles,
                             "cost": cost, "net": ot["prize"] - cost, "route": r})
        print("evening", R["evening"][-1], flush=True)

    # ---- 5. worked example: one van, three orders, one must go -------------
    R["worked"] = worked_example(policy)
    return R


def worked_example(policy: PenaltyPolicy) -> dict:
    """Depot and three customers on one street, miles east of the depot."""
    pos = {"D": 0.0, "A": 2.0, "B": 3.0, "C": 6.0}
    names = list(pos)
    cost = np.array([[abs(pos[a] - pos[b]) * PER_MILE_ALL_IN for b in names] for a in names])
    orders = {"A": (400, "club"), "B": (300, "regular"), "C": (350, "club")}
    pol = {k: policy.penalty(v, t, False) for k, (v, t) in orders.items()}
    full = [0, 1, 2, 3, 0]
    saving = removal_savings(full, cost)
    rows = []
    for k, c in enumerate("ABC", start=1):
        route = [i for i in full if i != k]
        rc = sum(cost[i, j] for i, j in zip(route, route[1:]))
        rows.append({"drop": c, "miles": rc / PER_MILE_ALL_IN, "route_cost": rc,
                     "saving": saving[k], "pen_uniform": 100.0, "pen_policy": pol[c],
                     "total_uniform": rc + 100.0, "total_policy": rc + pol[c]})
    return {"rows": rows, "full_miles": sum(cost[i, j] for i, j in zip(full, full[1:]))
            / PER_MILE_ALL_IN, "orders": orders, "pos": pos}


# ---------------------------------------------------------------- book assets
DESIGNS = (("uniform", "uni", "Same penalty for every order"),
           ("value", "val", "Penalty = basket value"),
           ("policy", "pol", "Penalties from the policy"))


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    pol = PenaltyPolicy()
    m = {"orders": R["orders"], "totes": R["totes"], "capacity": N_VANS * CAPACITY,
         "vans": N_VANS, "over-pct": f"{100 * (R['totes'] / (N_VANS * CAPACITY) - 1):.0f}",
         "weight": f"{R['weight']:,}", "payload": f"{N_VANS * PAYLOAD:,}",
         "club": R["club"], "repeat": R["repeat"], "limit": f"{LIMIT:.0f}",
         "mandatory": R["mandatory_status"].replace("_", r"\_"),
         "mean-penalty": f"\\${R['mean_penalty']:,.0f}",
         "voucher": f"\\${pol.voucher:.0f}", "cancel-pct": f"{100 * pol.cancel_prob:.0f}",
         "margin-pct": f"{100 * pol.margin:.0f}",
         "churn-club": f"{100 * pol.churn_prob['club']:.0f}",
         "churn-regular": f"{100 * pol.churn_prob['regular']:.0f}",
         "future-club": f"\\${pol.future_margin['club']:,.0f}",
         "future-regular": f"\\${pol.future_margin['regular']:,.0f}",
         "repeat-factor": f"{pol.repeat_factor:.0f}"}
    pen = R["penalties"][1:]
    m["pen-min"] = f"\\${min(pen):,.0f}"
    m["pen-max"] = f"\\${max(pen):,.0f}"
    m["pen-ratio"] = f"{max(pen) / min(pen):.1f}"
    m["big-orders"] = sum(1 for q in peak()[0].demands if q >= 50)
    m["big-totes"] = max(peak()[0].demands)
    for name, key, _ in DESIGNS:
        X = R["plans"][name]
        m[f"{key}-dropped"] = X["n_dropped"]
        m[f"{key}-totes"] = X["totes_dropped"]
        m[f"{key}-club"] = X["club_dropped"]
        m[f"{key}-repeat"] = X["repeat_dropped"]
        m[f"{key}-routing"] = f"\\${X['routing']:,.0f}"
        m[f"{key}-penalty"] = f"\\${X['penalty']:,.0f}"
        m[f"{key}-total"] = f"\\${X['total']:,.0f}"
        m[f"{key}-value"] = f"\\${X['value_dropped']:,.0f}"
        m[f"{key}-served"] = X["served"]
    U, V, P = (R["plans"][k] for k in ("uniform", "value", "policy"))
    m["pol-vs-uni"] = f"\\${U['total'] - P['total']:,.0f}"
    m["pol-vs-uni-pct"] = f"{100 * (1 - P['total'] / U['total']):.0f}"
    m["pol-vs-val"] = f"\\${V['total'] - P['total']:,.0f}"
    m["big-dropped-pol"] = sum(1 for i in P["dropped"] if peak()[0].demands[i] >= 50)
    runs = R["strategy_runs"]["policy"]
    m["spread-best"] = f"\\${min(runs.values()):,.0f}"
    m["spread-worst"] = f"\\${max(runs.values()):,.0f}"
    m["spread-pct"] = f"{100 * (max(runs.values()) / min(runs.values()) - 1):.0f}"
    m["spread-best-name"] = min(runs, key=runs.get).replace("_", r"\_")
    S = R["sweep"]
    m["sweep-max-served"] = max(s["served"] for s in S)
    m["sweep-nonmono"] = sum(1 for a, b in zip(S, S[1:]) if b["served"] < a["served"])
    first = next(s for s in S if s["served"] > 0)
    m["sweep-first-lambda"] = f"\\${first['lambda']:,}"
    m["sweep-first-served"] = first["served"]
    full = next(s for s in S if s["served"] == max(x["served"] for x in S))
    m["sweep-full-lambda"] = f"\\${full['lambda']:,}"
    m["sweep-per-order"] = f"\\${first['routing'] / first['served']:,.0f}"
    top = [s for s in S if s["served"] == full["served"]]
    m["sweep-top-lo"] = f"\\${min(s['routing'] for s in top):,.0f}"
    m["sweep-top-hi"] = f"\\${max(s['routing'] for s in top):,.0f}"
    L = R["late"]
    m["late-orders"] = L["orders"]
    m["late-totes"] = L["totes"]
    m["late-fee"] = f"\\${LATE_FEE:.2f}"
    m["late-margin-pct"] = f"{100 * LATE_MARGIN:.0f}"
    m["eve-rental"] = f"\\${EVENING_RENTAL:.0f}"
    E = {e["hours"]: e for e in R["evening"]}
    best = max(R["evening"], key=lambda e: e["net"])
    m["eve-best-hours"] = best["hours"]
    m["eve-best-stops"] = best["stops"]
    m["eve-best-net"] = f"\\${best['net']:,.0f}"
    m["eve-tuned-optimal"] = sum(abs(e["prize"] - e["mip_prize"]) < 0.01 for e in R["evening"])
    m["eve-default-optimal"] = sum(abs(e["default_prize"] - e["mip_prize"]) < 0.01
                                   for e in R["evening"])
    gaps = [100 * (1 - e["default_prize"] / e["mip_prize"]) for e in R["evening"]]
    m["eve-default-gap"] = f"{max(gaps):.0f}"
    m["eve-cases"] = len(R["evening"])
    m["eve-max-rounds"] = max(e["mip_rounds"] for e in R["evening"])
    m["eve-4-totes"] = E[4]["totes"]
    W = R["worked"]
    m["we-full-miles"] = f"{W['full_miles']:.0f}"
    m["we-per-mile"] = f"\\${PER_MILE_ALL_IN:.2f}"
    for r in W["rows"]:
        c = r["drop"].lower()
        m[f"we-{c}-saving"] = f"\\${r['saving']:.2f}"
        m[f"we-{c}-pen"] = f"\\${r['pen_policy']:.2f}"
        m[f"we-{c}-uni"] = f"\\${r['total_uniform']:.2f}"
        m[f"we-{c}-pol"] = f"\\${r['total_policy']:.2f}"
    m["we-best-uniform"] = min(W["rows"], key=lambda r: r["total_uniform"])["drop"]
    m["we-best-policy"] = min(W["rows"], key=lambda r: r["total_policy"])["drop"]
    lines = ["% Generated by code/examples/ch09_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch09/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch09-results.tex", "\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for name, _, label in DESIGNS:
        X = R["plans"][name]
        rows.append(f"{label} & {X['n_dropped']} ({X['totes_dropped']}) & "
                    f"{X['club_dropped']} & {X['repeat_dropped']} & "
                    f"\\${X['value_dropped']:,.0f} & \\${X['routing']:,.0f} & "
                    f"\\${X['penalty']:,.0f} & \\${X['total']:,.0f} \\\\")
    plotting.write_tex(gen / "ch09-plans-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = [f"\\py{{{st}}} & \\${v:,.0f} \\\\" for st, v in
            sorted(runs.items(), key=lambda kv: kv[1])]
    plotting.write_tex(gen / "ch09-strategy-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = [f"\\${s['lambda']:,} & {s['served']} & {s['totes_served']} & {s['vans']} & "
            f"\\${s['routing']:,.0f} \\\\" for s in S]
    plotting.write_tex(gen / "ch09-sweep-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for e in R["evening"]:
        rows.append(f"{e['hours']} h & \\${e['default_prize']:,.0f} & \\${e['prize']:,.0f} & "
                    f"\\${e['mip_prize']:,.0f} & {e['stops']} ({e['totes']}) & "
                    f"\\${e['cost']:,.0f} & \\${e['net']:,.0f} \\\\")
    plotting.write_tex(gen / "ch09-evening-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    bu, bp = m["we-best-uniform"], m["we-best-policy"]
    for r in W["rows"]:
        f = (lambda v, b: f"\\textbf{{\\${v:.2f}}}" if b else f"\\${v:.2f}")
        rows.append(f"{r['drop']} & {r['miles']:.0f} & \\${r['route_cost']:.2f} & "
                    f"\\${r['saving']:.2f} & {f(r['total_uniform'], r['drop'] == bu)} & "
                    f"\\${r['pen_policy']:.2f} & {f(r['total_policy'], r['drop'] == bp)} \\\\")
    plotting.write_tex(gen / "ch09-worked-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    _plot_plans(R, book / "figures" / "ch09" / "plans.pdf")
    _plot_frontier(R, book / "figures" / "ch09" / "frontier.pdf")
    print("book assets written")


def _plot_plans(R, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    data, rows, inst = peak()
    xs, ys = plotting.to_miles(inst)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 4.6), sharey=True)
    for ax, key, title in ((axes[0], "uniform", "Same penalty for every order"),
                           (axes[1], "policy", "Penalties from the policy")):
        X = R["plans"][key]
        for r in X["routes"]:
            if len(r) > 2:
                ax.plot([xs[i] for i in r], [ys[i] for i in r], "-",
                        color=plotting.ROUTE_COLORS[0], linewidth=0.9, alpha=0.75, zorder=2)
        served = [i for r in X["routes"] for i in r[1:-1]]
        for group, mk, face in (("regular", "o", plotting.ROUTE_COLORS[0]),
                                ("club", "s", plotting.ROUTE_COLORS[0])):
            idx = [i for i in served if rows[i]["tier"] == group]
            ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], marker=mk, s=18,
                       color=face, edgecolor="white", linewidth=0.6, zorder=3)
        for group, mk in (("regular", "o"), ("club", "s")):
            idx = [i for i in X["dropped"] if rows[i]["tier"] == group]
            ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], marker=mk, s=46,
                       facecolor="white", edgecolor=plotting.ROUTE_COLORS[1],
                       linewidth=1.6, zorder=5)
        rep = [i for i in X["dropped"] if rows[i]["postponed"] == "1"]
        ax.scatter([xs[i] for i in rep], [ys[i] for i in rep], marker="x", s=22,
                   color=plotting.ROUTE_COLORS[1], linewidth=1.2, zorder=6)
        plotting._depot(ax, xs[0], ys[0])
        plotting._axes(ax)
        ax.set_title(title, loc="left", fontsize=9, color=plotting.INK)
    axes[1].set_ylabel("")
    handles = [
        Line2D([], [], color=plotting.ROUTE_COLORS[0], marker="o", linestyle="-",
               markersize=4, label="served, regular"),
        Line2D([], [], color=plotting.ROUTE_COLORS[0], marker="s", linestyle="-",
               markersize=4, label="served, Larder Club"),
        Line2D([], [], color=plotting.ROUTE_COLORS[1], marker="o", linestyle="",
               markerfacecolor="white", markeredgewidth=1.5, markersize=6,
               label="postponed, regular"),
        Line2D([], [], color=plotting.ROUTE_COLORS[1], marker="s", linestyle="",
               markerfacecolor="white", markeredgewidth=1.5, markersize=6,
               label="postponed, Larder Club"),
        Line2D([], [], color=plotting.ROUTE_COLORS[1], marker="x", linestyle="",
               markersize=5, label="postponed for the second time")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    fig.subplots_adjust(bottom=0.2, wspace=0.08)
    plotting.save(fig, path)
    plt.close(fig)


def _plot_frontier(R, path):
    import matplotlib.pyplot as plt
    plotting.use_book_style()
    S = R["sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    xs = [s["served"] for s in S]
    ys = [s["routing"] for s in S]
    # markers only: no plan exists "between" two points of the sweep
    ax.scatter(xs, ys, s=26, color=plotting.ROUTE_COLORS[0], edgecolor="white",
               linewidth=0.7, zorder=3, label="one penalty $\\lambda$ for every order")
    shown = set()
    for s in S:
        key = (s["served"], round(s["routing"]))
        if key in shown or s["lambda"] in (5000,):
            continue
        shown.add(key)
        ax.annotate(f"\\${s['lambda']:,}", (s["served"], s["routing"]), xytext=(-4, 6),
                    textcoords="offset points", fontsize=7, color=plotting.INK_2,
                    ha="right")
    P = R["plans"]["policy"]
    ax.scatter([P["served"]], [P["routing"]], marker="D", s=34,
               color=plotting.ROUTE_COLORS[1], zorder=4, label="penalties from the policy")
    ax.set_xlabel("orders delivered")
    ax.set_ylabel("routing cost (\\$)")
    ax.grid(True, color=plotting.GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    plotting.save(fig, path)
    plt.close(fig)


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


if __name__ == "__main__":
    main()
