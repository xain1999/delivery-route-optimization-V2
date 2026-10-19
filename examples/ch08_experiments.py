"""Chapter 8: pickups and deliveries (Houston) and two depots (Chicago).

    python examples/ch08_experiments.py --book ../book      # about 5 minutes
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch06_experiments as ch06  # noqa: E402

from vrpbook import plotting
from vrpbook.instances import Instance, Site
from vrpbook.pdp import build, check_pdp, solve_pdp
from vrpbook.timewindows import TWData, solve_vrptw, validate_schedule

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "results" / "ch08"
M_PER_MI = 1609.344
# candidate sites for a second depot (approximate, industrial areas)
CANDIDATES = {"Niles": (42.0300, -87.7900),              # northwest suburbs
              "Irving Park": (41.9540, -87.7070),        # Northwest Side
              "Clybourn corridor": (41.9150, -87.6650)}  # North Branch industrial corridor
LIMIT = 30


def houston():
    with (DATA / "houston_restaurants.csv").open() as fh:
        rest = list(csv.DictReader(fh))
    with (DATA / "houston_lunch_orders.csv").open() as fh:
        orders = list(csv.DictReader(fh))
    with (DATA / "houston_couriers.csv").open() as fh:
        couriers = [(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(fh)]
    R = [(float(r["lat"]), float(r["lon"])) for r in rest]
    O = []
    for r in orders:
        h, m = map(int, r["ready_hhmm"].split(":"))
        O.append((int(r["restaurant_id"]), float(r["cust_lat"]), float(r["cust_lon"]),
                  h * 3600 + m * 60, int(r["bags"])))
    return couriers, O, R, rest


def run_pdp() -> dict:
    C, O, R, rest = houston()
    base = build(C, O, R)
    # shortest possible ride of each order: pick up, drive straight to the customer
    direct = [(base.service[p] + int(base.time[p, d])) / 60 for p, d in base.pairs]
    out = {"orders": len(O), "couriers": len(C), "restaurants": len(R),
           "direct_max": max(direct), "direct_mean": float(np.mean(direct)),
           "bags": sum(o[4] for o in O), "variants": []}
    tightest = math.ceil(max(direct))
    for ride, max_orders in [(tightest, 3), (30, 3), (45, 3), (60, 3), (30, 1)]:
        d = build(C, O, R, max_ride_min=ride)
        plan = solve_pdp(d, max_orders=max_orders, time_limit_s=LIMIT)
        row = {"ride": ride, "max_orders": max_orders, "status": plan.status}
        if plan.routes:
            chk = check_pdp(plan, d, 6, max_orders)
            row.update(chk)
            row["routes"] = plan.routes
        out["variants"].append(row)
        print(row if "routes" not in row else {k: v for k, v in row.items() if k != "routes"},
              flush=True)
    # worked example: two orders, one courier
    out["count_sequences"] = {n: math.factorial(2 * n) // 2 ** n for n in range(1, 7)}
    return out


def chicago_two_depots(site: tuple[float, float], name: str, baseline: bool) -> dict:
    data, inst = ch06.tuesday(2, n_vehicles=8)
    n = len(data.demands)
    # append Niles as an extra node (index n): a second depot
    depot_site = Site(n, f"{name} depot", name, *site)
    sites = inst.sites + [depot_site]
    from vrpbook import distance_matrix, travel_time_matrix
    dist = distance_matrix(sites)
    ttime = travel_time_matrix(dist, ch06.SPEED_MPH * 1.609344)
    weights = data.extra["Weight"][0] + [0]
    two = TWData(dist, ttime, data.service + [0], data.windows + [data.windows[0]],
                 data.demands + [0], 100, 8, {"Weight": (weights, 2400)})
    kw = dict(time_limit_s=LIMIT, driver_per_hour=ch06.DRIVER_PER_HOUR,
              vehicle_per_mile=ch06.VEHICLE_PER_MILE, fixed_per_van=ch06.FIXED_PER_VAN)
    res = {}

    def summarize(routes, d, depots):
        T = validate_schedule(routes, d, depots)
        vans = sum(1 for r in routes if len(r) > 2)
        miles = T["meters"] / M_PER_MI
        cost = (ch06.DRIVER_PER_HOUR * T["duration"] / 3600 + ch06.VEHICLE_PER_MILE * miles
                + ch06.FIXED_PER_VAN * vans)
        return {"vans": vans, "miles": miles, "paid_h": T["duration"] / 3600, "cost": cost,
                "wait_h": T["wait"] / 3600}

    if baseline:   # A: one depot, 8 vans (Chapter 6)
        one = TWData(data.dist, data.time, data.service, data.windows, data.demands, 100, 8,
                     data.extra)
        s = solve_vrptw(one, **kw)
        res["one_depot"] = {**summarize(s.routes, one, (0,)), "routes": s.routes}
    # B: two depots, four vans each, planned jointly
    starts = [0] * 4 + [n] * 4
    s = solve_vrptw(two, starts=starts, ends=starts, **kw)
    res["joint"] = {**summarize(s.routes, two, (0, n)), "routes": s.routes}
    res["joint"]["per_depot"] = [sum(1 for r in s.routes if len(r) > 2 and r[0] == dep)
                                 for dep in (0, n)]
    # C: assign each customer to the nearer depot by travel time, then route each
    assign = {i: (0 if two.time[0, i] <= two.time[n, i] else n) for i in range(1, n)}
    routes_c, ok = [], True
    for dep in (0, n):
        nodes = [dep] + [i for i in range(1, n) if assign[i] == dep]
        sub = TWData(dist[np.ix_(nodes, nodes)], ttime[np.ix_(nodes, nodes)],
                     [two.service[i] for i in nodes], [two.windows[i] for i in nodes],
                     [two.demands[i] for i in nodes], 100, 4,
                     {"Weight": ([weights[i] for i in nodes], 2400)})
        s = solve_vrptw(sub, **kw)
        if not s.routes:
            ok = False
            break
        routes_c += [[nodes[i] for i in r] for r in s.routes]
    if ok:
        res["assign_first"] = {**summarize(routes_c, two, (0, n)), "routes": routes_c}
    else:
        res["assign_first"] = {"status": "no plan"}
    res["assigned_to_niles"] = sum(1 for v in assign.values() if v == n)
    res["niles_index"] = n
    res["sites"] = [(s.lat, s.lon) for s in sites]
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {a: b for a, b in v.items() if a != "routes"}, flush=True)
    return res


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    P, D = R["pdp"], R["depots"]
    m = {"limit": LIMIT, "orders": P["orders"], "couriers": P["couriers"],
         "restaurants": P["restaurants"], "bags": P["bags"],
         "direct-max": f"{P['direct_max']:.1f}", "direct-mean": f"{P['direct_mean']:.0f}"}
    V = {(v["ride"], v["max_orders"]): v for v in P["variants"]}
    tight = min(v["ride"] for v in P["variants"])
    m["ride-tight"] = tight
    b30, s30 = V[(30, 3)], V[(30, 1)]
    m["batch30-hours"] = f"{b30['courier_hours']:.1f}"
    m["single30-hours"] = f"{s30['courier_hours']:.1f}"
    m["batching-saving"] = f"{100 * (1 - b30['courier_hours'] / s30['courier_hours']):.0f}"
    m["batch30-batched"] = b30["batched"]
    m["batch30-meanride"] = f"{b30['mean_ride_min']:.0f}"
    m["single30-meanride"] = f"{s30['mean_ride_min']:.0f}"
    for k, n in P["count_sequences"].items():
        m[f"seq{k}"] = f"{n:,}"
    one = D["one_depot"]
    m["one-cost"] = f"\\${one['cost']:,.0f}"
    m["one-miles"] = f"{one['miles']:.0f}"
    best = min(CANDIDATES, key=lambda c: D[c]["joint"]["cost"])
    m["best-site"] = best
    for c in CANDIDATES:
        key = c.split()[0].lower()
        J = D[c]["joint"]
        m[f"{key}-cost"] = f"\\${J['cost']:,.0f}"
        sav = one['cost'] - J['cost']
        m[f"{key}-saving"] = ("" if sav >= 0 else "$-$") + f"\\${abs(sav):,.0f}"
        m[f"{key}-pct"] = f"{max(0.0, 100 * (1 - J['cost'] / one['cost'])):.1f}"
        m[f"{key}-vans"] = J["vans"]
        m[f"{key}-split"] = "+".join(map(str, J["per_depot"]))
        AF = D[c]["assign_first"]
        m[f"{key}-af"] = f"\\${AF['cost']:,.0f}" if "cost" in AF else "no plan"
        if "cost" in AF:
            m[f"{key}-af-pct"] = f"{100 * (AF['cost'] / J['cost'] - 1):.0f}"
    m["best-saving-year"] = f"\\${(one['cost'] - D[best]['joint']['cost']) * 300:,.0f}"
    lines = ["% Generated by code/examples/ch08_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch08/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch08-results.tex", "\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for v in P["variants"]:
        label = f"{v['ride']} min" + (", one order at a time" if v["max_orders"] == 1 else "")
        if "couriers" not in v:
            rows.append(f"{label} & \\multicolumn{{4}}{{l}}{{no plan found}} \\\\")
            continue
        rows.append(f"{label} & {v['couriers']} & {v['courier_hours']:.1f} & "
                    f"{v['batched']} & {v['mean_ride_min']:.0f} \\\\")
    plotting.write_tex(gen / "ch08-pdp-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = [f"McKinley Park only & {one['vans']} & {one['miles']:.0f} & \\${one['cost']:,.0f} "
            f"& -- & -- \\\\", "\\midrule"]
    for c in CANDIDATES:
        J, AF = D[c]["joint"], D[c]["assign_first"]
        af = f"\\${AF['cost']:,.0f}" if "cost" in AF else "no plan found"
        rows.append(f"+ {c} & {J['vans']} ({'+'.join(map(str, J['per_depot']))}) & "
                    f"{J['miles']:.0f} & \\${J['cost']:,.0f} & "
                    f"{100 * (J['cost'] / one['cost'] - 1):+.1f}\\% & {af} \\\\")
    plotting.write_tex(gen / "ch08-depot-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    _plot_depots(D, best, book / "figures" / "ch08" / "two-depots.pdf")
    _plot_houston(R, book / "figures" / "ch08" / "houston.pdf")
    print("book assets written")


def _plot_depots(Dall, best, path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    D = Dall[best]
    sites = [Site(i, "", "", la, lo) for i, (la, lo) in enumerate(D["sites"])]
    inst = Instance("two", sites, [0] * len(sites), 8, 100)
    xs, ys = plotting.to_miles(inst)
    n = D["niles_index"]
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    for r in D["joint"]["routes"]:
        if len(r) <= 2:
            continue
        k = 0 if r[0] == 0 else 1
        ax.plot([xs[i] for i in r], [ys[i] for i in r], plotting.ROUTE_STYLES[k],
                color=plotting.ROUTE_COLORS[k], linewidth=1.3,
                label=None)
        ax.scatter([xs[i] for i in r[1:-1]], [ys[i] for i in r[1:-1]], s=22,
                   marker=plotting.ROUTE_MARKERS[k], color=plotting.ROUTE_COLORS[k],
                   edgecolor="white", linewidth=0.7, zorder=4)
    for idx, name in ((0, "McKinley Park"), (n, best)):
        ax.scatter([xs[idx]], [ys[idx]], marker="s", s=70, color=plotting.INK, zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(name, (xs[idx], ys[idx]), xytext=(6, -10), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=plotting.INK)
    # the other candidate sites, for reference
    d0 = Dall[best]["sites"][0]
    kx = 69.172 * math.cos(math.radians(d0[0]))
    for c, (la, lo) in CANDIDATES.items():
        if c == best:
            continue
        cx, cy = (lo - d0[1]) * kx, (la - d0[0]) * 68.707
        ax.scatter([cx], [cy], marker="s", s=50, facecolor="white", edgecolor=plotting.INK,
                   linewidth=1.2, zorder=5)
        ax.annotate(f"{c} (candidate)", (cx, cy), xytext=(6, -10), textcoords="offset points",
                    fontsize=7.5, color=plotting.INK_2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=plotting.ROUTE_COLORS[0], linestyle="-", marker="o",
                      label="Routes from McKinley Park"),
               Line2D([], [], color=plotting.ROUTE_COLORS[1], linestyle="--", marker="s",
                      label=f"Routes from {best}")]
    plotting._axes(ax)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              frameon=False)
    plotting.save(fig, path)
    plt.close(fig)


def _plot_houston(R, path):
    """Restaurants, customers and one courier's route with two batched orders."""
    import matplotlib.pyplot as plt
    plotting.use_book_style()
    C, O, Rr, rest = houston()
    v = next(x for x in R["pdp"]["variants"] if x["ride"] == 30 and x["max_orders"] == 3)
    lat0, lon0 = 29.7604, -95.3698          # downtown Houston as the origin
    kx = 69.172 * math.cos(math.radians(lat0))

    def xy(lat, lon):
        return (lon - lon0) * kx, (lat - lat0) * 68.707
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    rx = [xy(*p)[0] for p in Rr]
    ry = [xy(*p)[1] for p in Rr]
    ax.scatter([xy(o[1], o[2])[0] for o in O], [xy(o[1], o[2])[1] for o in O], s=14,
               color=plotting.ROUTE_COLORS[0], alpha=0.75, label="Customers", zorder=3)
    ax.scatter(rx, ry, marker="^", s=55, color=plotting.ROUTE_COLORS[1], edgecolor="white",
               label="Restaurants", zorder=4)
    # the busiest courier route
    route = max(v["routes"], key=len)
    pts = []
    for node in route[:-1]:
        if node <= len(C):
            pts.append(xy(*C[node - 1]))
        else:
            k = (node - len(C) - 1) // 2
            o = O[k]
            pts.append(xy(*Rr[o[0]]) if (node - len(C) - 1) % 2 == 0 else xy(o[1], o[2]))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=plotting.ROUTE_COLORS[2],
            linewidth=1.8, zorder=2, label="One courier's route")
    ax.scatter([pts[0][0]], [pts[0][1]], marker="o", s=40, color=plotting.INK, zorder=5,
               label="Courier at 11:00")
    ax.set_aspect("equal")
    ax.grid(True, color=plotting.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("miles east of downtown")
    ax.set_ylabel("miles north of downtown")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, labelspacing=1.1)
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
        R = {"pdp": run_pdp(), "depots": {}}
        for k, (name, site) in enumerate(CANDIDATES.items()):
            R["depots"][name] = chicago_two_depots(site, name, baseline=(k == 0))
        R["depots"]["one_depot"] = R["depots"]["Niles"].pop("one_depot")
        path.write_text(json.dumps(R, indent=1))
    if args.book:
        write_book_assets(args.book, R)


if __name__ == "__main__":
    main()
