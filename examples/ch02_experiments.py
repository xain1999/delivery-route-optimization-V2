"""Chapter 2: every number, table and figure in the chapter.

Run from the code/ directory:

    python examples/ch02_experiments.py --book ../book

Results go to results/ch02/results.json.  With --book, the script also
writes LaTeX macros and tables into <book>/generated/ and figures into
<book>/figures/ch02/, so the text can never drift from the code.

Time-limited searches depend on machine speed.  Numbers reported in the
book were produced on a single core of a 2026 cloud VM.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np

import ortools
from vrpbook import (chicago_ch02, distance_matrix, solve_cvrp,
                     travel_time_matrix, validate_routes, total_distance)
from vrpbook.distances import haversine_m, triangle_violations
from vrpbook.exact_tsp import solve_tsp_exact
from vrpbook import plotting

CIRCUITY = 1.3
SPEED_MPH = 15.0
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch02"

FIRST_STRATEGIES = ["PATH_CHEAPEST_ARC", "SAVINGS", "CHRISTOFIDES",
                    "PARALLEL_CHEAPEST_INSERTION", "LOCAL_CHEAPEST_INSERTION"]
METAHEURISTICS = ["GREEDY_DESCENT", "GUIDED_LOCAL_SEARCH",
                  "SIMULATED_ANNEALING", "TABU_SEARCH"]
TIME_LIMITS = [1, 3, 10]
SPAN_COEFS = [0, 10, 100]


METERS_PER_MILE = 1609.344


def mi(m: float) -> float:
    """Meters to miles (all costs are integer meters internally)."""
    return m / METERS_PER_MILE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--from-json", action="store_true",
                    help="skip solving, rebuild book assets from results.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    inst = chicago_ch02()
    dist = distance_matrix(inst.sites, circuity=CIRCUITY)
    ttime = travel_time_matrix(dist, SPEED_MPH * 1.609344)
    if args.from_json:
        R = json.loads((OUT / "results.json").read_text())
        bal_routes = {int(k): v for k, v in R["balance_routes"].items()}
        write_book_assets(args.book, inst, dist, R, R["cvrp"]["routes"], bal_routes)
        return
    R: dict = {"env": {"python": platform.python_version(),
                       "ortools": ortools.__version__,
                       "machine": platform.machine()}}

    # ---- instance facts ---------------------------------------------------
    offdiag = dist[~np.eye(len(dist), dtype=bool)]
    R["instance"] = {
        "customers": inst.n_customers, "vehicles": inst.n_vehicles,
        "capacity": inst.capacity, "total_demand": inst.total_demand,
        "min_vehicles": math.ceil(inst.total_demand / inst.capacity),
        "utilization_pct": 100 * inst.total_demand / (inst.n_vehicles * inst.capacity),
        "max_mi": mi(offdiag.max()), "mean_mi": mi(offdiag.mean()),
        "triangle_violations": triangle_violations(dist),
        "min_demand": min(inst.demands[1:]), "max_demand": max(inst.demands[1:]),
    }
    # straight-line vs circuity example: depot to Rogers Park (node 1)
    gc = haversine_m(inst.sites[0].lat, inst.sites[0].lon,
                     inst.sites[1].lat, inst.sites[1].lon)
    R["instance"]["depot_rp_gc_mi"] = mi(gc)
    R["instance"]["depot_rp_road_mi"] = mi(dist[0, 1])
    R["instance"]["depot_rp_min"] = ttime[0, 1] / 60

    # ---- one van, no capacity: TSP ------------------------------------------
    tsp = solve_cvrp(dist, [0] * len(dist), 1, 10**6, time_limit_s=5)
    tour, opt_len, rounds = solve_tsp_exact(dist)
    R["tsp"] = {"ortools_mi": mi(tsp.objective), "exact_mi": mi(opt_len),
                "cut_rounds": rounds, "route": tsp.routes[0],
                "gap_pct": 100 * (tsp.objective - opt_len) / opt_len,
                "hours": sum(int(ttime[a, b]) for a, b in
                             zip(tsp.routes[0], tsp.routes[0][1:])) / 3600}

    # ---- the fleet: CVRP ---------------------------------------------------
    best = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                      time_limit_s=10)
    stats = validate_routes(best.routes, dist, inst.demands, inst.capacity)
    assert total_distance(best.routes, dist) == best.objective
    R["cvrp"] = {"mi": mi(best.objective), "status": best.status,
                 "routes": best.routes,
                 "per_route": [{"stops": s.stops, "load": s.load,
                                "mi": mi(s.distance_m),
                                "hours": sum(int(ttime[a, b]) for a, b in
                                             zip(r, r[1:])) / 3600}
                               for s, r in zip(stats, best.routes)]}
    R["cvrp"]["vs_tsp_pct"] = 100 * (best.objective - opt_len) / opt_len
    lens = [p["mi"] for p in R["cvrp"]["per_route"]]
    R["cvrp"]["longest_mi"], R["cvrp"]["shortest_mi"] = max(lens), min(lens)

    # ---- first-solution strategies ----------------------------------------
    fs_rows = []
    for fs in FIRST_STRATEGIES:
        a = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                       first_solution=fs, metaheuristic="GREEDY_DESCENT",
                       time_limit_s=10, track_progress=True)
        b = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                       first_solution=fs, metaheuristic="GUIDED_LOCAL_SEARCH",
                       time_limit_s=10)
        validate_routes(a.routes, dist, inst.demands, inst.capacity)
        validate_routes(b.routes, dist, inst.demands, inst.capacity)
        fs_rows.append({"strategy": fs, "first_mi": mi(a.history[0][1]),
                        "descent_mi": mi(a.objective), "descent_s": a.wall_time_s,
                        "gls_mi": mi(b.objective)})
    R["first_solution"] = fs_rows

    # ---- best known value: three long runs from different starts -------------
    long_runs = {}
    for fs in ["PARALLEL_CHEAPEST_INSERTION", "CHRISTOFIDES", "SAVINGS"]:
        s = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                       first_solution=fs, time_limit_s=60)
        validate_routes(s.routes, dist, inst.demands, inst.capacity)
        long_runs[fs] = mi(s.objective)
    R["best_known"] = {"mi": min(long_runs.values()), "runs": long_runs}
    R["cvrp"]["gap_to_bks_pct"] = 100 * (R["cvrp"]["mi"] - R["best_known"]["mi"]) \
        / R["best_known"]["mi"]

    # ---- metaheuristics x time limits ---------------------------------------
    mh_rows = []
    for mh in METAHEURISTICS:
        row = {"metaheuristic": mh}
        for t in TIME_LIMITS:
            s = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                           metaheuristic=mh, time_limit_s=t)
            validate_routes(s.routes, dist, inst.demands, inst.capacity)
            row[f"t{t}"] = mi(s.objective)
            row[f"t{t}_wall"] = s.wall_time_s
        mh_rows.append(row)
    R["metaheuristics"] = mh_rows

    # ---- convergence trace (GLS) --------------------------------------------
    traces = {}
    for fs in ["PATH_CHEAPEST_ARC", "SAVINGS"]:
        s = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                       first_solution=fs, time_limit_s=10, track_progress=True)
        traces[fs] = [(t, mi(v)) for t, v in s.history]
    R["traces"] = traces

    # ---- balancing routes -------------------------------------------------
    bal = []
    bal_routes = {}
    for coef in SPAN_COEFS:
        s = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                       time_limit_s=10, span_cost_coefficient=coef)
        st = validate_routes(s.routes, dist, inst.demands, inst.capacity)
        L = [mi(x.distance_m) for x in st]
        bal.append({"coef": coef, "total_mi": mi(total_distance(s.routes, dist)),
                    "longest_mi": max(L), "shortest_mi": min(L),
                    "longest_h": max(L) / SPEED_MPH})
        bal_routes[coef] = s.routes
    R["balance"] = bal
    R["balance_routes"] = {str(k): v for k, v in bal_routes.items()}

    # ---- worked example: enumerate all tours of 4 customers -------------------
    W = [6, 13, 21, 7]  # Lincoln Park, The Loop, Hyde Park, Logan Square
    sub = inst.subset(W, name="worked-4", n_vehicles=1)
    dsub = distance_matrix(sub.sites, circuity=CIRCUITY)
    tours = {}
    for perm in itertools.permutations(range(1, 5)):
        key = min(perm, perm[::-1])  # a tour and its reverse are the same
        route = (0,) + perm + (0,)
        tours[key] = sum(int(dsub[a, b]) for a, b in zip(route, route[1:]))
    ranked = sorted(tours.items(), key=lambda kv: kv[1])
    ws = solve_cvrp(dsub, [0] * 5, 1, 10**6, time_limit_s=1)
    R["worked"] = {
        "customers": W, "names": [sub.sites[i].name for i in range(5)],
        "matrix_mi": [[round(mi(x), 1) for x in row] for row in dsub.tolist()],
        "tours": [{"order": [W[i - 1] for i in k], "mi": mi(v)} for k, v in ranked],
        "ortools_mi": mi(ws.objective),
    }

    (OUT / "results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("instance", "tsp")}, indent=1))
    print("CVRP", R["cvrp"]["mi"], R["cvrp"]["per_route"])
    print("FS", json.dumps(fs_rows, indent=1))
    print("MH", json.dumps(mh_rows, indent=1))
    print("BAL", json.dumps(bal, indent=1))
    print("WORKED", R["worked"]["tours"][:3], R["worked"]["ortools_mi"])

    if args.book:
        write_book_assets(args.book, inst, dist, R, best.routes, bal_routes)


# ---------------------------------------------------------------------------
def write_book_assets(book: Path, inst, dist, R, best_routes, bal_routes) -> None:
    gen = book / "generated"
    fig = book / "figures" / "ch02"
    gen.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    m: dict[str, str] = {}

    def f1(x): return f"{x:.1f}"

    I, T, C = R["instance"], R["tsp"], R["cvrp"]
    m.update({
        "customers": str(I["customers"]), "vehicles": str(I["vehicles"]),
        "capacity": str(I["capacity"]), "total-demand": str(I["total_demand"]),
        "min-vehicles": str(I["min_vehicles"]),
        "utilization": f"{I['utilization_pct']:.1f}",
        "max-mi": f1(I["max_mi"]), "mean-mi": f1(I["mean_mi"]),
        "min-demand": str(I["min_demand"]), "max-demand": str(I["max_demand"]),
        "tri-viol": str(I["triangle_violations"]),
        "rp-gc-mi": f1(I["depot_rp_gc_mi"]), "rp-road-mi": f1(I["depot_rp_road_mi"]),
        "rp-min": f"{I['depot_rp_min']:.0f}",
        "tsp-mi": f1(T["ortools_mi"]), "tsp-exact-mi": f1(T["exact_mi"]),
        "tsp-rounds": str(T["cut_rounds"]), "tsp-hours": f1(T["hours"]),
        "cvrp-mi": f1(C["mi"]), "cvrp-vs-tsp": f"{C['vs_tsp_pct']:.0f}",
        "cvrp-longest-mi": f1(C["longest_mi"]), "cvrp-shortest-mi": f1(C["shortest_mi"]),
        "bks-mi": f1(R["best_known"]["mi"]), "cvrp-gap-bks": f"{C['gap_to_bks_pct']:.1f}",
        "bal100-extra-mi": f1(R["balance"][-1]["total_mi"] - R["balance"][0]["total_mi"]),
        "bal100-pct-more": f"{100 * (R['balance'][-1]['total_mi'] / R['balance'][0]['total_mi'] - 1):.1f}",
        "ortools-version": R["env"]["ortools"], "python-version": R["env"]["python"],
        "circuity": f"{CIRCUITY}", "speed": f"{SPEED_MPH:.0f}",
        "worked-best-mi": f1(R["worked"]["tours"][0]["mi"]),
        "worked-worst-mi": f1(R["worked"]["tours"][-1]["mi"]),
        "worked-ortools-mi": f1(R["worked"]["ortools_mi"]),
        "worked-ntours": str(len(R["worked"]["tours"])),
        "worked-worst-pct": f"{100 * (R['worked']['tours'][-1]['mi'] / R['worked']['tours'][0]['mi'] - 1):.0f}",
    })
    for b in R["balance"]:
        c = b["coef"]
        m[f"bal{c}-total"] = f1(b["total_mi"])
        m[f"bal{c}-longest"] = f1(b["longest_mi"])
        m[f"bal{c}-shortest"] = f1(b["shortest_mi"])
    m["fs-best-first"] = min(R["first_solution"], key=lambda r: r["first_mi"])["strategy"]
    for r in R["first_solution"]:
        key = r["strategy"].lower().replace("_", "")
        m[f"fs-{key}-first"] = f1(r["first_mi"])
        m[f"fs-{key}-descent"] = f1(r["descent_mi"])
        m[f"fs-{key}-gls"] = f1(r["gls_mi"])
    for r in R["metaheuristics"]:
        key = r["metaheuristic"].lower().replace("_", "")
        for t in TIME_LIMITS:
            m[f"mh-{key}-t{t}"] = f1(r[f"t{t}"])
    ts, gls = m["mh-tabusearch-t10"], m["mh-guidedlocalsearch-t10"]
    m["mh-t10-phrase"] = (f"both reached {gls}~miles" if ts == gls else
                          f"tabu search reached {ts}~miles and guided local search {gls}~miles")

    lines = ["% Generated by code/examples/ch02_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch02/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch02-results.tex", "\n".join(lines) + "\n", encoding="utf-8")

    # ---- tables -------------------------------------------------------------
    # instance table (two side-by-side halves)
    rows = []
    n = len(inst.sites)
    half = math.ceil((n - 1) / 2)
    for k in range(1, half + 1):
        a = inst.sites[k]
        left = f"{a.id} & {a.name} & {inst.demands[a.id]}"
        if k + half < n:
            b = inst.sites[k + half]
            right = f"{b.id} & {b.name} & {inst.demands[b.id]}"
        else:
            right = "& &"
        rows.append(f"{left} && {right} \\\\")
    plotting.write_tex(gen / "ch02-instance-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # per-route table
    rows = []
    used = [(r, p) for r, p in zip(C["routes"], C["per_route"]) if len(r) > 2]
    for k, (r, p) in enumerate(used, 1):
        rows.append(f"Van {k} & {p['stops']} & {p['load']} & {p['mi']:.1f} & "
                    f"{p['hours']:.1f} & {', '.join(map(str, r[1:-1]))} \\\\")
    tot_stops = sum(p["stops"] for _, p in used)
    tot_load = sum(p["load"] for _, p in used)
    rows.append("\\midrule")
    rows.append(f"Total & {tot_stops} & {tot_load} & {C['mi']:.1f} & "
                f"{sum(p['hours'] for _, p in used):.1f} & \\\\")
    plotting.write_tex(gen / "ch02-route-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # first-solution table
    best_gls = min(r["gls_mi"] for r in R["first_solution"])
    rows = []
    for r in R["first_solution"]:
        gls = f"{r['gls_mi']:.1f}"
        if abs(r["gls_mi"] - best_gls) < 1e-9:
            gls = f"\\textbf{{{gls}}}"
        rows.append(f"\\py{{{r['strategy']}}} & {r['first_mi']:.1f} & "
                    f"{r['descent_mi']:.1f} & {gls} \\\\")
    plotting.write_tex(gen / "ch02-first-solution-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # metaheuristic table
    rows = []
    for r in R["metaheuristics"]:
        cells = " & ".join(f"{r[f't{t}']:.1f}" for t in TIME_LIMITS)
        rows.append(f"\\py{{{r['metaheuristic']}}} & {cells} \\\\")
    plotting.write_tex(gen / "ch02-metaheuristic-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # balance table
    rows = []
    for b in R["balance"]:
        rows.append(f"{b['coef']} & {b['total_mi']:.1f} & {b['longest_mi']:.1f} & "
                    f"{b['shortest_mi']:.1f} & {b['longest_mi'] - b['shortest_mi']:.1f} \\\\")
    plotting.write_tex(gen / "ch02-balance-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # worked example: matrix and tours
    W = R["worked"]
    labels = ["D"] + [str(i) for i in W["customers"]]
    rows = []
    for i, row in enumerate(W["matrix_mi"]):
        rows.append(f"\\textbf{{{labels[i]}}} & " + " & ".join(
            "--" if i == j else f"{x:.1f}" for j, x in enumerate(row)) + " \\\\")
    plotting.write_tex(gen / "ch02-worked-matrix.tex", "\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for k, t in enumerate(W["tours"], 1):
        seq = "D--" + "--".join(map(str, t["order"])) + "--D"
        cell = f"{t['mi']:.1f}"
        if k == 1:
            seq, cell = f"\\textbf{{{seq}}}", f"\\textbf{{{cell}}}"
        rows.append(f"{k} & {seq} & {cell} \\\\")
    plotting.write_tex(gen / "ch02-worked-tours.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # ---- figures --------------------------------------------------------------
    plotting.plot_instance(inst, fig / "instance.pdf")
    plotting.plot_routes(inst, [T["route"]], fig / "tsp.pdf",
                         route_mi=[T["ortools_mi"]], legend=False, label_ids=True)
    used_routes = [r for r in best_routes if len(r) > 2]
    plotting.plot_routes(inst, used_routes, fig / "cvrp.pdf",
                         route_mi=[p["mi"] for r, p in zip(C["routes"], C["per_route"])
                                   if len(r) > 2], label_ids=True, legend_cols=1)
    _plot_convergence(R["traces"], fig / "convergence.pdf")
    _plot_balance(inst, dist, bal_routes, fig / "balance.pdf")
    print("book assets written to", book)


def _plot_convergence(traces, path) -> None:
    import matplotlib.pyplot as plt
    plotting.use_book_style()
    figr, ax = plt.subplots(figsize=(5.2, 2.6))
    labels = {"PATH_CHEAPEST_ARC": "Path cheapest arc start",
              "SAVINGS": "Savings start"}
    for k, (fs, tr) in enumerate(traces.items()):
        t = [max(x[0], 1e-3) for x in tr]
        v = [x[1] for x in tr]
        # best-so-far step line
        best, bv = [], math.inf
        for x in v:
            bv = min(bv, x)
            best.append(bv)
        t.append(10.0)
        best.append(best[-1])
        ax.step(t, best, where="post", color=plotting.ROUTE_COLORS[k],
                linestyle=plotting.ROUTE_STYLES[k], linewidth=2, label=labels[fs])
        ax.annotate(f"{best[-1]:.1f} mi", (10, best[-1]), xytext=(4, 5 - 10 * k),
                    textcoords="offset points", va="center", fontsize=8,
                    color=plotting.INK_2)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 10)
    ax.set_xlabel("seconds since start (log scale)")
    ax.set_ylabel("best total distance (miles)")
    ax.grid(True, color=plotting.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    plotting.save(figr, path)
    plt.close(figr)


def _plot_balance(inst, dist, bal_routes, path) -> None:
    import matplotlib.pyplot as plt
    plotting.use_book_style()
    figr, axes = plt.subplots(1, 2, figsize=(5.4, 5.2), sharey=True)
    for ax, coef, title in [(axes[0], 0, "Distance only"),
                            (axes[1], 100, "With span cost (coefficient 100)")]:
        routes = [r for r in bal_routes[coef] if len(r) > 2]
        L = [sum(int(dist[a, b]) for a, b in zip(r, r[1:])) / METERS_PER_MILE for r in routes]
        plotting.plot_routes(inst, routes, None, title=title, ax=ax,
                             legend=True, route_mi=L)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1,
                  frameon=False, fontsize=7.5, handlelength=2.6)
    axes[1].set_ylabel("")
    plotting.save(figr, path)
    plt.close(figr)


if __name__ == "__main__":
    main()
