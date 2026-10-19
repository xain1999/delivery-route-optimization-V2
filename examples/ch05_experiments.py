"""Chapter 5: comparing TSP and CVRP formulations.

    python examples/ch05_experiments.py --book ../book      # about 20 minutes

TSP instances are random subsets of a pool of Chicago locations (the
Chapter 2 and Chapter 3 customers).  CVRP instances are random subsets of
Tuesday's stops from Chapter 3.  Seeds are fixed, so the instances are
the same on every machine. Solve times are not.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from vrpbook import (chicago_ch02, distance_matrix, load_csv, solve_cvrp,
                     validate_routes)
from vrpbook import plotting
from vrpbook.exact_tsp import solve_tsp_exact
from vrpbook.formulations import (cvrp_set_partitioning, cvrp_two_index, enumerate_routes,
                                  tsp_dfj, tsp_mtz, tsp_scf)
from vrpbook.instances import Instance

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch05"
TSP_SIZES = [10, 20, 30, 40, 50, 60]
TSP_LIMIT = 120
CVRP_SIZES = [8, 10, 12, 14]
CVRP_ROUTES = [3, 5]          # target average number of routes
M_PER_MI = 1609.344


def location_pool() -> Instance:
    a = chicago_ch02()
    b = load_csv(HERE / "data" / "chicago_tuesday.csv", 6, 100)
    seen, sites = set(), [a.sites[0]]
    for s in a.sites[1:] + b.sites[1:]:
        key = (round(s.lat, 4), round(s.lon, 4))
        if key not in seen:
            seen.add(key)
            sites.append(s)
    return Instance("pool", sites, [0] * len(sites), 1, 1)


def gap(opt: float, lb: float) -> float:
    return 100 * (opt - lb) / opt


def run_tsp() -> list[dict]:
    pool = location_pool()
    rng = np.random.default_rng(5)
    rows = []
    for n in TSP_SIZES:
        ids = sorted(rng.choice(range(1, len(pool.sites)), n, replace=False).tolist())
        d = distance_matrix(pool.subset(ids).sites)
        _, opt, _ = solve_tsp_exact(d)
        row = {"n": n, "opt_mi": opt / M_PER_MI}
        for name, f in [("dfj", tsp_dfj), ("scf", tsp_scf), ("mtz", tsp_mtz)]:
            lp = f(d, relax=True)
            ip = f(d, time_limit_s=TSP_LIMIT)
            row[name] = {"lp_gap": gap(opt, lp.objective), "seconds": ip.seconds,
                         "status": ip.status, "rounds": ip.rounds, "cuts": ip.cuts,
                         "final_gap": gap(ip.objective, ip.bound) if ip.objective > 0 else None,
                         "found_opt": abs(ip.objective - opt) < 0.5}
            print(n, name, row[name], flush=True)
        rows.append(row)
    return rows


def run_cvrp() -> list[dict]:
    inst = load_csv(HERE / "data" / "chicago_tuesday.csv", 6, 100)
    pool = [i for i in range(1, len(inst.sites)) if inst.demands[i] <= 30]
    rng = np.random.default_rng(2)
    rows = []
    for k in CVRP_ROUTES:
        for n in CVRP_SIZES:
            ids = sorted(rng.choice(pool, n, replace=False).tolist())
            sub = inst.subset(ids)
            d, q = distance_matrix(sub.sites), sub.demands
            Q, m = math.ceil(sum(q) / (k - 0.4)), k + 1
            t0 = time.perf_counter()
            R = enumerate_routes(d, q, Q)
            t_enum = time.perf_counter() - t0
            sp = cvrp_set_partitioning(R, len(d), m)
            validate_routes(sp.solution["routes"], d, q, Q)
            opt = sp.objective
            lp0 = cvrp_two_index(d, q, Q, m, relax=True, capacity_cuts=False)
            lp1 = cvrp_two_index(d, q, Q, m, relax=True)
            splp = cvrp_set_partitioning(R, len(d), m, relax=True)
            splp_k = cvrp_set_partitioning(R, len(d), m, relax=True,
                                           min_vehicles=math.ceil(sum(q) / Q))
            ip = cvrp_two_index(d, q, Q, m, time_limit_s=300)
            ot = solve_cvrp(d, q, m, Q, time_limit_s=2)
            validate_routes(ot.routes, d, q, Q)
            row = {"k": k, "n": n, "Q": Q, "routes_enumerated": len(R), "t_enum": t_enum,
                   "t_sp": sp.seconds, "opt_mi": opt / M_PER_MI, "vans": len(sp.solution["routes"]),
                   "gap_lp0": gap(opt, lp0.objective), "gap_rci": gap(opt, lp1.objective),
                   "rci_rounds": lp1.rounds, "gap_sp": gap(opt, splp.objective),
                   "gap_spk": gap(opt, splp_k.objective),
                   "ip_seconds": ip.seconds, "ip_rounds": ip.rounds, "ip_ok": abs(ip.objective - opt) < 0.5,
                   "ortools_gap": gap(ot.objective, opt) * -1}
            print(row, flush=True)
            rows.append(row)
    return rows


def worked_example() -> dict:
    """Six customers: every model, every number."""
    inst = load_csv(HERE / "data" / "chicago_tuesday.csv", 6, 100)
    ids = [4, 6, 18, 21, 26, 29]
    sub = inst.subset(ids)
    d, q = distance_matrix(sub.sites), sub.demands
    Q, m = 30, 4
    R = enumerate_routes(d, q, Q)
    sp = cvrp_set_partitioning(R, len(d), m)
    splp = cvrp_set_partitioning(R, len(d), m, relax=True)
    splp_k = cvrp_set_partitioning(R, len(d), m, relax=True,
                                   min_vehicles=math.ceil(sum(q) / Q))
    lp0 = cvrp_two_index(d, q, Q, m, relax=True, capacity_cuts=False)
    lp1 = cvrp_two_index(d, q, Q, m, relax=True)
    ip = cvrp_two_index(d, q, Q, m)
    return {"ids": ids, "names": [s.name for s in sub.sites], "q": q, "Q": Q, "m": m,
            "lat": [s.lat for s in sub.sites], "lon": [s.lon for s in sub.sites],
            "n_routes": len(R), "opt_mi": sp.objective / M_PER_MI,
            "routes": sp.solution["routes"],
            "lp0_mi": lp0.objective / M_PER_MI, "lp0_x": {f"{i}-{j}": v for (i, j), v in lp0.solution.items() if v > 1e-6},
            "lp1_mi": lp1.objective / M_PER_MI, "lp1_rounds": lp1.rounds, "lp1_cuts": lp1.cuts,
            "sp_lp_mi": splp.objective / M_PER_MI, "sp_lpk_mi": splp_k.objective / M_PER_MI,
            "sp_lp_vans": None, "ip_mi": ip.objective / M_PER_MI,
            "ip_rounds": ip.rounds, "ip_cuts": ip.cuts,
            "by_size": {str(s): sum(1 for r in R if len(r[0]) == s) for s in range(1, 7)}}


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
        R = {"worked": worked_example(), "cvrp": run_cvrp(), "tsp": run_tsp(),
             "tsp_limit": TSP_LIMIT}
        path.write_text(json.dumps(R, indent=1))
    if args.book:
        write_book_assets(args.book, R)


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    W = R["worked"]
    m = {"tsp-limit": R["tsp_limit"]}
    m.update({
        "we-Q": W["Q"], "we-total": sum(W["q"]), "we-nroutes": W["n_routes"],
        "we-opt": f"{W['opt_mi']:.2f}", "we-lp0": f"{W['lp0_mi']:.2f}",
        "we-lp1": f"{W['lp1_mi']:.2f}", "we-splp": f"{W['sp_lp_mi']:.2f}",
        "we-ip": f"{W['ip_mi']:.2f}", "we-splpk": f"{W['sp_lpk_mi']:.2f}",
        "we-splpk-gap": f"{gap(W['opt_mi'], W['sp_lpk_mi']):.1f}", "we-lp1-rounds": W["lp1_rounds"],
        "we-lp1-cuts": W["lp1_cuts"], "we-ip-rounds": W["ip_rounds"], "we-ip-cuts": W["ip_cuts"],
        "we-vans": len(W["routes"]),
        "we-lp0-gap": f"{gap(W['opt_mi'], W['lp0_mi']):.1f}",
        "we-lp1-gap": f"{gap(W['opt_mi'], W['lp1_mi']):.1f}",
        "we-splp-gap": f"{gap(W['opt_mi'], W['sp_lp_mi']):.1f}",
        "we-min-vans": math.ceil(sum(W["q"]) / W["Q"]),
    })
    for s, c in W["by_size"].items():
        m[f"we-size{s}"] = c
    T = R["tsp"]
    largest = T[-1]
    m["tsp-n-max"] = largest["n"]
    m["tsp-dfj-max-seconds"] = f"{max(r['dfj']['seconds'] for r in T):.1f}"
    m["tsp-dfj-max-lpgap"] = f"{max(r['dfj']['lp_gap'] for r in T):.2f}"
    unsolved = [r["n"] for r in T if r["mtz"]["status"] != "optimal"]
    m["tsp-mtz-first-unsolved"] = unsolved[0] if unsolved else "--"
    unsolved_scf = [r["n"] for r in T if r["scf"]["status"] != "optimal"]
    m["tsp-scf-first-unsolved"] = unsolved_scf[0] if unsolved_scf else "--"
    m["tsp-mtz-lpgap-min"] = f"{min(r['mtz']['lp_gap'] for r in T):.0f}"
    m["tsp-mtz-lpgap-max"] = f"{max(r['mtz']['lp_gap'] for r in T):.0f}"
    m["tsp-scf-lpgap-min"] = f"{min(r['scf']['lp_gap'] for r in T):.0f}"
    m["tsp-scf-lpgap-max"] = f"{max(r['scf']['lp_gap'] for r in T):.0f}"
    C = R["cvrp"]
    m["cvrp-lp0-min"] = f"{min(r['gap_lp0'] for r in C):.0f}"
    m["cvrp-lp0-max"] = f"{max(r['gap_lp0'] for r in C):.0f}"
    m["cvrp-rci-max"] = f"{max(r['gap_rci'] for r in C):.1f}"
    m["cvrp-sp-max"] = f"{max(r['gap_sp'] for r in C):.1f}"
    m["cvrp-spk-max"] = f"{max(r['gap_spk'] for r in C):.1f}"
    m["cvrp-ortools-max"] = f"{abs(max(r['ortools_gap'] for r in C)):.1f}"
    eps = 0.05
    m["cvrp-sp-better"] = sum(1 for r in C if r["gap_sp"] < r["gap_rci"] - eps)
    m["cvrp-sp-worse"] = sum(1 for r in C if r["gap_sp"] > r["gap_rci"] + eps)
    m["cvrp-sp-equal"] = len(C) - m["cvrp-sp-better"] - m["cvrp-sp-worse"]
    m["cvrp-ortools-nopt"] = sum(1 for r in C if r["ortools_gap"] < 1e-9)
    m["cvrp-count"] = len(C)
    m["cvrp-max-routes"] = f"{max(r['routes_enumerated'] for r in C):,}"
    m["cvrp-max-tenum"] = f"{max(r['t_enum'] for r in C):.1f}"
    m["cvrp-ip-max-seconds"] = f"{max(r['ip_seconds'] for r in C):.1f}"
    lines = ["% Generated by code/examples/ch05_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch05/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch05-results.tex", "\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for r in T:
        cells = [str(r["n"]), f"{r['opt_mi']:.1f}"]
        for f in ("dfj", "scf", "mtz"):
            cells.append(f"{r[f]['lp_gap']:.1f}")
        for f in ("dfj", "scf", "mtz"):
            x = r[f]
            if x["status"] == "optimal":
                cells.append(f"{x['seconds']:.1f}")
            elif x["final_gap"] is not None:
                cells.append(f"({x['final_gap']:.1f}\\%)")
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + " \\\\")
    plotting.write_tex(gen / "ch05-tsp-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for k in sorted({r["k"] for r in C}):
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{About {k} routes}}}} \\\\")
        for r in (r for r in C if r["k"] == k):
            rows.append(f"{r['n']} & {r['Q']} & {r['vans']} & \\num{{{r['routes_enumerated']}}} & "
                        f"{r['opt_mi']:.1f} & {r['gap_lp0']:.1f} & {r['gap_rci']:.1f} & "
                        f"{r['gap_sp']:.1f} & {r['gap_spk']:.1f} \\\\")
    plotting.write_tex(gen / "ch05-cvrp-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for i, (name, qi) in enumerate(zip(W["names"], W["q"])):
        if i == 0:
            continue
        rows.append(f"{i} & {name} & {qi} \\\\")
    plotting.write_tex(gen / "ch05-we-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    _plot_tsp(T, book / "figures" / "ch05" / "tsp-gaps.pdf")
    _plot_worked(W, book / "figures" / "ch05" / "worked.pdf")
    print("book assets written")


def _plot_tsp(T, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6))
    ns = [r["n"] for r in T]
    labels = {"dfj": "DFJ", "scf": "Single-commodity flow", "mtz": "MTZ"}
    for k, f in enumerate(("dfj", "scf", "mtz")):
        style = dict(color=plotting.ROUTE_COLORS[k], linestyle=plotting.ROUTE_STYLES[k],
                     marker=plotting.ROUTE_MARKERS[k], linewidth=2, markersize=5,
                     markeredgecolor="white", label=labels[f])
        axes[0].plot(ns, [r[f]["lp_gap"] for r in T], **style)
        solved = [(r["n"], r[f]["seconds"]) for r in T if r[f]["status"] == "optimal"]
        if solved:
            axes[1].plot(*zip(*solved), **style)
    axes[0].set_ylabel("LP gap to optimum (%)")
    axes[1].set_ylabel("seconds to prove optimality")
    axes[1].set_yscale("log")
    axes[1].axhline(T and 120, color=plotting.INK_2, linewidth=0.8, linestyle=":")
    axes[1].annotate("time limit", (ns[0], 120), xytext=(0, 3), textcoords="offset points",
                     fontsize=7, color=plotting.INK_2)
    for ax in axes:
        ax.set_xlabel("customers")
        ax.grid(True, color=plotting.GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.4)
    plotting.save(fig, path)
    plt.close(fig)


def _plot_worked(W, path: Path) -> None:
    import matplotlib.pyplot as plt
    plotting.use_book_style()
    inst = Instance("we", [type("S", (), {"lat": a, "lon": b})() for a, b in zip(W["lat"], W["lon"])],
                    W["q"], W["m"], W["Q"])
    xs, ys = plotting.to_miles(inst)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.0), sharey=True)
    ax = axes[0]
    for key, v in W["lp0_x"].items():
        i, j = map(int, key.split("-"))
        style = "-" if v > 0.99 else "--"
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], style, color=plotting.ROUTE_COLORS[0],
                linewidth=3.6 if v > 1.5 else 1.3, solid_capstyle="round")
        if abs(v - 1) > 1e-6 and abs(v - 2) > 1e-6:      # label fractional values
            ax.annotate(f"{v:g}", ((xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2), fontsize=7,
                        color=plotting.INK, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_title("LP relaxation, no capacity cuts", loc="left", color=plotting.INK)
    ax = axes[1]
    for k, r in enumerate(W["routes"]):
        ax.plot([xs[i] for i in r], [ys[i] for i in r], plotting.ROUTE_STYLES[k],
                color=plotting.ROUTE_COLORS[k], linewidth=1.6)
    ax.set_title("Optimal routes", loc="left", color=plotting.INK)
    for ax in axes:
        ax.scatter(xs[1:], ys[1:], s=30, color=plotting.INK_2, zorder=4, edgecolor="white")
        for i in range(1, len(xs)):
            ax.annotate(f"{i} ({W['q'][i]})", (xs[i], ys[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=plotting.INK)
        plotting._depot(ax, xs[0], ys[0])
        plotting._axes(ax)
    axes[1].set_ylabel("")
    plotting.save(fig, path)
    plt.close(fig)


if __name__ == "__main__":
    main()
