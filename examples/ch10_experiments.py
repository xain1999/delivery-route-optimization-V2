"""Chapter 10: construction heuristics, from scratch and in OR-Tools.

    python examples/ch10_experiments.py --book ../book      # about 10 minutes

Instances: the Chapter 2 instance (30 customers), Tuesday's cleaned orders
(Chapter 3, capacity only) and random Chicago instances of 100 to 2,000
customers (vrpbook.instances.chicago_synthetic).  The reference for each
instance is PyVRP, a state-of-the-art hybrid genetic search
(Chapter 12), run for a fixed time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from vrpbook import distance_matrix, load_csv, plotting, total_distance, validate_routes
from vrpbook.provenance import versions
from vrpbook import construction as C
from vrpbook.instances import chicago_ch02, chicago_synthetic

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch10"
M_PER_MI = 1609.344
SIZES = [100, 250, 500, 1000, 2000]
REF_SECONDS = float(os.environ.get("CH10_REF", 30))
ORTOOLS_MAX_N = 1000
ORTOOLS = ["PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION", "CHRISTOFIDES"]
WE_IDS = [16, 17, 18, 19, 26, 27]          # six customers south-west of the depot
WE_CAPACITY = 45


def xy_of(inst) -> np.ndarray:
    xs, ys = plotting.to_miles(inst)
    return np.column_stack([xs, ys])


def ours(inst, d) -> dict:
    q, Q = inst.demands, inst.capacity
    xy = xy_of(inst)
    return {"Nearest neighbor": lambda: C.nearest_neighbor(d, q, Q),
            "Savings (parallel)": lambda: C.clarke_wright(d, q, Q),
            "Savings (sequential)": lambda: C.clarke_wright_sequential(d, q, Q),
            "Sweep": lambda: C.sweep(d, q, Q, xy),
            "Cheapest insertion": lambda: C.insertion(d, q, Q, "cheapest"),
            "Regret insertion": lambda: C.insertion(d, q, Q, "regret")}


def reference(d, q, Q, seconds: float) -> float:
    import pyvrp
    from pyvrp.stop import MaxRuntime
    n = len(q)
    locs = [pyvrp.Location(x=0, y=0) for _ in range(n)]
    clients = [pyvrp.Client(location=i, delivery=[q[i]]) for i in range(1, n)]
    data = pyvrp.ProblemData(locs, clients, [pyvrp.Depot(location=0)],
                             [pyvrp.VehicleType(num_available=n, capacity=[Q])],
                             [np.asarray(d, dtype=np.int64)],
                             [np.zeros((n, n), dtype=np.int64)])
    res = pyvrp.solve(data, stop=MaxRuntime(seconds), seed=1)
    routes = [[0, *[clients[a.idx].location for a in r.schedule() if a.is_client()], 0]
              for r in res.best.routes()]
    validate_routes(routes, d, q, Q)
    return total_distance(routes, d)


def run_instance(name, inst) -> dict:
    t = time.perf_counter()
    d = distance_matrix(inst.sites)
    t_matrix = time.perf_counter() - t
    q, Q = inst.demands, inst.capacity
    row = {"name": name, "n": len(q) - 1, "matrix_s": t_matrix,
           "min_vans": math.ceil(sum(q) / Q),
           "lb": C.radial_lower_bound(d, q, Q), "methods": {}}
    for label, f in ours(inst, d).items():
        t = time.perf_counter()
        routes = f()
        el = time.perf_counter() - t
        validate_routes(routes, d, q, Q)
        row["methods"][label] = {"m": total_distance(routes, d), "routes": len(routes),
                                 "s": el, "kind": "ours"}
        if row["n"] == 250:
            row["methods"][label]["plan"] = routes
    if row["n"] <= ORTOOLS_MAX_N:
        vans = math.ceil(1.2 * row["min_vans"]) + 1
        for st in ORTOOLS:
            t = time.perf_counter()
            routes = C.ortools_first_solution(d, q, Q, vans, st)
            el = time.perf_counter() - t
            if routes:
                validate_routes(routes, d, q, Q)
                row["methods"]["OR-Tools " + st] = {"m": total_distance(routes, d),
                                                     "routes": len(routes), "s": el,
                                                     "kind": "ortools"}
    secs = REF_SECONDS * (2 if row["n"] >= 1000 else 1)
    row["ref"] = reference(d, q, Q, secs)
    row["ref_s"] = secs
    print(name, {k: (round(v["m"] / M_PER_MI, 1), round(v["s"], 2))
                 for k, v in row["methods"].items()}, "ref", round(row["ref"] / M_PER_MI, 1),
          "lb", round(row["lb"] / M_PER_MI, 1), flush=True)
    return row


def worked_example() -> dict:
    """Savings by hand: six customers, a 40-tote van, distances in miles."""
    base = chicago_ch02()
    keep = [0] + WE_IDS
    sub = base.subset(WE_IDS, n_vehicles=3)
    d_m = distance_matrix(sub.sites)
    d = np.round(d_m / M_PER_MI, 1)                  # tenths of a mile, as by hand
    q = sub.demands
    log: list = []
    routes = C.clarke_wright(d, q, WE_CAPACITY, log=log)
    return {"names": [base.sites[i].name for i in keep], "demands": q,
            "dist": d.tolist(), "log": [[s, i, j, w] for s, i, j, w in log],
            "routes": routes, "length": float(sum(d[a, b] for r in routes
                                                   for a, b in zip(r, r[1:]))),
            "star": float(2 * d[0, 1:].sum())}


def run() -> dict:
    R = {"instances": [], "worked": worked_example(), "env": versions()}
    R["instances"].append(run_instance("Chapter 2 (30)", chicago_ch02()))
    R["instances"].append(run_instance(
        "Tuesday (37)", load_csv(HERE / "data" / "chicago_tuesday.csv", 6, 100)))
    for n in SIZES:
        R["instances"].append(run_instance(f"Random {n:,}", chicago_synthetic(n)))
    return R


# ---------------------------------------------------------------- book assets
OUR_ORDER = ["Nearest neighbor", "Savings (parallel)", "Savings (sequential)", "Sweep",
             "Cheapest insertion", "Regret insertion"]


def gap(v, ref):
    return 100 * (v / ref - 1)


def pct(x: float) -> str:
    """Whole percent, without a minus sign on zero."""
    return f"{round(x):d}" if round(x) != 0 else "0"


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    I = R["instances"]
    m = {"ref-seconds": f"{REF_SECONDS:.0f}", "max-n": f"{max(SIZES):,}",
         "ortools-max-n": f"{ORTOOLS_MAX_N:,}"}

    # gaps table: rows = methods, columns = instances
    cols = [r["name"] for r in I]
    rows = []
    for label in OUR_ORDER + ["OR-Tools " + s for s in ORTOOLS]:
        cells = []
        for r in I:
            v = r["methods"].get(label)
            cells.append(pct(gap(v['m'], r['ref'])) if v else "--")
        name = label.replace("OR-Tools ", "")
        name = f"\\py{{{name}}}" if label.startswith("OR-Tools") else name
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if label == OUR_ORDER[-1]:
            rows.append("\\midrule")
    rows.append("\\midrule")
    rows.append("Lower bound & " + " & ".join(pct(gap(r['lb'], r['ref'])) for r in I)
                + " \\\\")
    plotting.write_tex(gen / "ch10-gap-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    plotting.write_tex(gen / "ch10-gap-head.tex", 
        " & ".join(c.replace("Chapter 2 (30)", "Ch.~2 (30)").replace("Random ", "")
                   for c in cols) + "\n", encoding="utf-8")

    rows = []
    for r in I:
        if r["n"] < 100:
            continue
        cells = [f"{r['methods'][lab]['s']:.2f}" for lab in OUR_ORDER]
        ot = r["methods"].get("OR-Tools SAVINGS")
        cells.append(f"{ot['s']:.2f}" if ot else "--")
        rows.append(f"{r['n']:,} & " + " & ".join(cells) + " \\\\")
    plotting.write_tex(gen / "ch10-time-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # headline numbers
    big = I[-1]
    cw = "Savings (parallel)"
    m["cw-gap-max"] = f"{max(gap(r['methods'][cw]['m'], r['ref']) for r in I):.0f}"
    m["cw-gap-min"] = pct(min(gap(r['methods'][cw]['m'], r['ref']) for r in I))
    m["big-cw-gap"] = pct(gap(big['methods'][cw]['m'], big['ref']))
    m["big-ref-s"] = f"{big['ref_s']:.0f}"
    mid = [r for r in I if r["n"] <= 500]
    m["mid-best-gap-min"] = pct(min(min(gap(r['methods'][k]['m'], r['ref']) for k in OUR_ORDER)
                                    for r in mid))
    m["nn-gap-min"] = f"{min(gap(r['methods']['Nearest neighbor']['m'], r['ref']) for r in I):.0f}"
    m["nn-gap-max"] = f"{max(gap(r['methods']['Nearest neighbor']['m'], r['ref']) for r in I):.0f}"
    m["big-cw-s"] = f"{big['methods'][cw]['s']:.1f}"
    m["big-nn-s"] = f"{big['methods']['Nearest neighbor']['s']:.2f}"
    m["big-regret-s"] = f"{big['methods']['Regret insertion']['s']:.1f}"
    m["big-matrix-s"] = f"{big['matrix_s']:.1f}"
    m["big-pairs"] = f"{big['n'] * (big['n'] - 1) // 2:,}"
    best_small = [min(OUR_ORDER, key=lambda k: r["methods"][k]["m"]) for r in I]
    m["cw-wins"] = sum(1 for b in best_small if b.startswith("Savings"))
    m["sweep-wins"] = sum(1 for b in best_small if b == "Sweep")
    m["n-instances"] = len(I)
    ch2 = I[0]
    m["ch2-sweep-gap"] = f"{gap(ch2['methods']['Sweep']['m'], ch2['ref']):.0f}"
    m["ch2-cw-gap"] = f"{gap(ch2['methods'][cw]['m'], ch2['ref']):.0f}"
    syn = [r for r in I if r["n"] >= 500]
    m["sweep-gap-large"] = f"{min(gap(r['methods']['Sweep']['m'], r['ref']) for r in syn):.0f}"
    m["lb-gap-max"] = f"{max(-gap(r['lb'], r['ref']) for r in I):.0f}"
    m["lb-gap-min"] = f"{min(-gap(r['lb'], r['ref']) for r in I):.0f}"
    m["regret-vs-cheapest"] = sum(1 for r in I if r["methods"]["Regret insertion"]["m"]
                                  < r["methods"]["Cheapest insertion"]["m"])
    ortools_cw = [r for r in I if "OR-Tools SAVINGS" in r["methods"]]
    m["ot-cw-gap-max"] = f"{max(gap(r['methods']['OR-Tools SAVINGS']['m'], r['ref']) for r in ortools_cw):.0f}"
    m["ot-pca-gap-min"] = f"{min(gap(r['methods']['OR-Tools PATH_CHEAPEST_ARC']['m'], r['ref']) for r in ortools_cw):.0f}"
    m["ot-pca-gap-max"] = f"{max(gap(r['methods']['OR-Tools PATH_CHEAPEST_ARC']['m'], r['ref']) for r in ortools_cw):.0f}"

    # worked example
    W = R["worked"]
    names = W["names"]
    codes = {"Pilsen": "PI", "Little Village": "LV", "Bridgeport": "BR",
             "Chinatown": "CT", "Back of the Yards": "BY", "Gage Park": "GP"}
    short = ["D"] + [codes[n] for n in names[1:]]
    m["we-capacity"] = WE_CAPACITY
    m["we-total"] = sum(W["demands"])
    m["we-length"] = f"{W['length']:.1f}"
    m["we-star"] = f"{W['star']:.1f}"
    m["we-saved"] = f"{W['star'] - W['length']:.1f}"
    m["we-routes"] = len(W["routes"])
    m["we-pairs"] = len(W["log"])
    rows = []
    for k, (s, i, j, why) in enumerate(W["log"], start=1):
        rows.append(f"{k} & {short[i]}--{short[j]} & {s:.1f} & {why} \\\\")
    plotting.write_tex(gen / "ch10-worked-savings.tex", "\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for i in range(len(names)):
        cells = [("--" if i == j else f"{W['dist'][i][j]:.1f}") for j in range(len(names))]
        rows.append(f"{short[i]} & {W['demands'][i] if i else ''} & " + " & ".join(cells) + " \\\\")
    plotting.write_tex(gen / "ch10-worked-matrix.tex", "\n".join(rows) + "\n", encoding="utf-8")
    plotting.write_tex(gen / "ch10-worked-head.tex", " & ".join(short) + "\n", encoding="utf-8")
    m["we-route-text"] = " and ".join(
        "depot--" + "--".join(names[i] for i in r[1:-1]) + "--depot" for r in W["routes"])
    lines = ["% Generated by code/examples/ch10_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch10/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch10-results.tex", "\n".join(lines) + "\n", encoding="utf-8")
    _plot_maps(R, book / "figures" / "ch10" / "maps.pdf")
    _plot_scaling(R, book / "figures" / "ch10" / "scaling.pdf")
    print("book assets written")


def _plot_maps(R, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    inst = chicago_synthetic(250)
    xs, ys = plotting.to_miles(inst)
    row = next(r for r in R["instances"] if r["n"] == 250)
    show = ["Nearest neighbor", "Sweep", "Savings (parallel)", "Regret insertion"]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 7.4), sharex=True, sharey=True)
    for ax, lab in zip(axes.flat, show):
        plan = row["methods"][lab]["plan"]
        for k, r in enumerate(plan):
            inner = r[1:-1]                 # trips to and from the depot drawn faint
            for a, b in ((r[0], r[1]), (r[-2], r[-1])):
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "-", linewidth=0.4,
                        color=plotting.GRID, zorder=1)
            ax.plot([xs[i] for i in inner], [ys[i] for i in inner], "-", linewidth=0.8,
                    color=plotting.ROUTE_COLORS[k % 3], zorder=2)
        ax.scatter(xs[1:], ys[1:], s=3, color=plotting.INK_2, zorder=3)
        plotting._depot(ax, xs[0], ys[0])
        plotting._axes(ax)
        g = gap(row["methods"][lab]["m"], row["ref"])
        ax.set_title(f"{lab}: +{g:.0f}\\%", loc="left", fontsize=8.5, color=plotting.INK)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supxlabel("miles east of depot", fontsize=9)
    fig.supylabel("miles north of depot", fontsize=9)
    handles = [Line2D([], [], color=plotting.ROUTE_COLORS[0], linewidth=0.8, label="route"),
               Line2D([], [], color=plotting.GRID, linewidth=0.8,
                      label="to and from the depot"),
               Line2D([], [], marker="o", color=plotting.INK_2, linestyle="", markersize=2.5,
                      label="customer"),
               Line2D([], [], marker="s", color=plotting.INK, linestyle="", markersize=6,
                      label="depot")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.045), fontsize=7.5)
    fig.subplots_adjust(bottom=0.08, hspace=0.12, wspace=0.08)
    plotting.save(fig, path)
    plt.close(fig)


def _plot_scaling(R, path):
    import matplotlib.pyplot as plt
    plotting.use_book_style()
    I = [r for r in R["instances"] if r["n"] >= 100]
    ns = [r["n"] for r in I]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    show = [("Nearest neighbor", 3), ("Savings (parallel)", 0), ("Sweep", 2),
            ("Regret insertion", 1)]
    for k, (lab, c) in enumerate(show):
        ax.plot(ns, [r["methods"][lab]["s"] for r in I], plotting.ROUTE_STYLES[k],
                marker=plotting.ROUTE_MARKERS[k], color=plotting.ROUTE_COLORS[c],
                linewidth=1.2, markersize=4, label=lab)
    ax.plot(ns, [r["matrix_s"] for r in I], ":", color=plotting.INK_2, linewidth=1.0,
            label="computing the distance matrix")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("customers")
    ax.set_ylabel("seconds")
    ax.grid(True, color=plotting.GRID, linewidth=0.5, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
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
