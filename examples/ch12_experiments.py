"""Chapter 12: metaheuristics, ours and PyVRP's, against OR-Tools.

    python examples/ch12_experiments.py --part compare       # about 15 minutes
    python examples/ch12_experiments.py --part tuning        # about 8 minutes
    python examples/ch12_experiments.py --book ../book --from-json

The two parts can run at the same time on a machine with two cores.
Instances and reference values are those of Chapter 10.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vrpbook import distance_matrix, load_csv, plotting, solve_cvrp, total_distance
from vrpbook import validate_routes
from vrpbook.instances import chicago_ch02, chicago_synthetic
from vrpbook.metaheuristics import alns, giant_tour, hgs, split
from vrpbook.provenance import versions
from vrpbook.pyvrp_adapter import solve_cvrp_pyvrp, solve_vrptw_pyvrp

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch12"
M_PER_MI = 1609.344
SECONDS = 30
BUDGETS = [5, 15, 60]
TEMPS = [0.005, 0.001, 0.0002]



def and_join(xs) -> str:
    """'a, b and c'."""
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

def instances():
    yield "Chapter 2 (30)", chicago_ch02()
    yield "Tuesday (37)", load_csv(HERE / "data" / "chicago_tuesday.csv", 6, 100)
    for n in (100, 250, 500, 1000):
        yield f"Random {n:,}", chicago_synthetic(n)


def xy_of(inst):
    xs, ys = plotting.to_miles(inst)
    return [list(p) for p in zip(xs, ys)]


def run_methods(inst, seconds, methods, histories=False) -> dict:
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    out = {}
    for name in methods:
        hist: list = []
        info = {}
        if name == "OR-Tools":
            vans = math.ceil(1.2 * math.ceil(sum(q) / Q)) + 1
            sol = solve_cvrp(d, q, vans, Q, first_solution="SAVINGS",
                             metaheuristic="GUIDED_LOCAL_SEARCH", time_limit_s=seconds,
                             track_progress=histories)
            routes = sol.used_routes
            hist = [(t, float(c)) for t, c in sol.history]
        elif name == "ALNS":
            routes = alns(d, q, Q, seconds, history=hist)
            info = {"iterations": hist[-1][2]}
            hist = [h[:2] for h in hist]
        elif name == "HGS (ours)":
            routes = hgs(d, q, Q, xy_of(inst), seconds, history=hist)
            info = {"iterations": hist[-1][2]}
            hist = [h[:2] for h in hist]
        else:
            routes, its = solve_cvrp_pyvrp(d, q, Q, seconds, seed=2)
            info = {"iterations": its}
        validate_routes(routes, d, q, Q)
        out[name] = {"m": total_distance(routes, d), **info}
        if histories:
            out[name]["history"] = hist
    return out


def part_compare() -> dict:
    R = {"instances": [], "env": versions()}
    for name, inst in instances():
        n = len(inst.demands) - 1
        secs = SECONDS * (2 if n >= 1000 else 1)
        res = run_methods(inst, secs, ["OR-Tools", "ALNS", "HGS (ours)", "PyVRP"],
                          histories=(n == 500))
        R["instances"].append({"name": name, "n": n, "seconds": secs, "methods": res})
        print(name, {k: round(v["m"] / M_PER_MI, 1) for k, v in res.items()}, flush=True)
    R["split"] = split_example()
    return R


def part_tuning() -> dict:
    inst = chicago_synthetic(500)
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    R = {"budgets": [], "temps": [], "env": versions()}
    for b in BUDGETS:
        res = run_methods(inst, b, ["OR-Tools", "ALNS", "PyVRP"])
        R["budgets"].append({"seconds": b, **{k: v["m"] for k, v in res.items()}})
        print("budget", b, {k: round(v["m"] / M_PER_MI, 1) for k, v in res.items()}, flush=True)
    for t in TEMPS:
        for polish in (False, True):
            r = alns(d, q, Q, SECONDS, start_worse=t, polish=polish)
            validate_routes(r, d, q, Q)
            R["temps"].append({"start_worse": t, "polish": polish, "m": total_distance(r, d)})
            print("temp", t, polish, round(total_distance(r, d) / M_PER_MI, 1), flush=True)
    import ch06_experiments as C6
    data, _ = C6.tuesday(2)
    routes, res = solve_vrptw_pyvrp(data, SECONDS, driver_per_hour=C6.DRIVER_PER_HOUR,
                                    vehicle_per_mile=C6.VEHICLE_PER_MILE,
                                    fixed_per_van=C6.FIXED_PER_VAN)
    R["vrptw"] = {"pyvrp": C6.summarize([r for r in routes if len(r) > 2], data),
                  "iterations": res.num_iterations,
                  "ortools": json.loads((HERE / "results" / "ch06" / "results.json")
                                        .read_text())["widths"]["2"]}
    R["vrptw"]["ortools"] = {k: v for k, v in R["vrptw"]["ortools"].items() if k != "routes"}
    print("vrptw", R["vrptw"]["pyvrp"]["cost"], R["vrptw"]["ortools"]["cost"], flush=True)
    return R


def split_example() -> dict:
    """Split by hand: the six customers of Chapter 10, in angular order."""
    base = chicago_ch02()
    ids = [16, 17, 18, 19, 26, 27]
    sub = base.subset(ids, n_vehicles=3)
    d = np.round(distance_matrix(sub.sites) / M_PER_MI, 1)
    D = d.tolist()
    q = sub.demands
    xy = xy_of(sub)
    tour = giant_tour([[0, c, 0] for c in range(1, 7)], xy)
    routes = split(tour, D, q, 45)
    arcs = []
    for i in range(6):
        load, cost = 0, 0.0
        for j in range(i, 6):
            c = tour[j]
            load += q[c]
            if load > 45:
                break
            cost = (D[0][c] if j == i else cost - D[tour[j - 1]][0] + D[tour[j - 1]][c]) + D[c][0]
            arcs.append([i, j + 1, round(cost, 1), load])
    V = [0.0] + [math.inf] * 6
    for i, j, c, _ in sorted(arcs):
        V[j] = min(V[j], V[i] + c)
    return {"tour": tour, "names": [base.sites[i].name for i in [0] + ids], "arcs": arcs,
            "V": V, "routes": routes, "demands": q}


# ---------------------------------------------------------------- book assets
METHODS = ["OR-Tools", "ALNS", "HGS (ours)", "PyVRP"]


def pct(x):
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def write_book_assets(book: Path, A: dict, B: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    refs = {r["name"]: r["ref"] for r in json.loads(
        (HERE / "results" / "ch10" / "results.json").read_text())["instances"]}
    m = {"seconds": SECONDS, "pyvrp-version": A["env"]["pyvrp"]}
    rows, wins = [], {k: 0 for k in METHODS}
    gaps = {k: [] for k in METHODS}
    extra500 = [b[k] for b in B["budgets"] for k in ("OR-Tools", "ALNS", "PyVRP")] + \
        [x["m"] for x in B["temps"]]
    for r in A["instances"]:
        best = min([refs[r["name"]]] + [v["m"] for v in r["methods"].values()]
                   + (extra500 if r["n"] == 500 else []))
        r["best"] = best
        cells = []
        top = min(r["methods"][k]["m"] for k in METHODS)
        for k in METHODS:
            g = 100 * (r["methods"][k]["m"] / best - 1)
            gaps[k].append(g)
            s = pct(g)
            if r["methods"][k]["m"] <= top + 1e-6:
                s = f"\\textbf{{{s}}}"
                wins[k] += 1
            cells.append(s)
        name = r["name"].replace("Chapter 2 (30)", "Ch.~2 (30)").replace("Random ", "")
        rows.append(f"{name} & {r['seconds']} & " + " & ".join(cells) + " \\\\")
    plotting.write_tex(gen / "ch12-compare-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    for k, key in zip(METHODS, ["ot", "alns", "hgs", "pyvrp"]):
        m[f"{key}-gap-max"] = pct(max(gaps[k]))
        m[f"{key}-gap-mean"] = pct(float(np.mean(gaps[k])))
        m[f"{key}-wins"] = wins[k]
    r500 = next(r for r in A["instances"] if r["n"] == 500)
    r1000 = next(r for r in A["instances"] if r["n"] == 1000)
    m["alns-its-500"] = f"{r500['methods']['ALNS']['iterations']:,}"
    m["hgs-its-500"] = f"{r500['methods']['HGS (ours)']['iterations']:,}"
    m["pyvrp-its-500"] = f"{r500['methods']['PyVRP']['iterations']:,}"
    m["pyvrp-its-1000"] = f"{r1000['methods']['PyVRP']['iterations']:,}"
    m["hgs-its-1000"] = f"{r1000['methods']['HGS (ours)']['iterations']:,}"
    m["n-instances"] = len(A["instances"])
    # budgets
    best500 = r500["best"]
    rows = []
    for b in B["budgets"]:
        rows.append(f"{b['seconds']} & " + " & ".join(
            pct(100 * (b[k] / best500 - 1)) for k in ("OR-Tools", "ALNS", "PyVRP")) + " \\\\")
    rows.insert(2, f"{SECONDS} & " + " & ".join(
        pct(100 * (r500["methods"][k]["m"] / best500 - 1)) for k in ("OR-Tools", "ALNS", "PyVRP"))
        + " \\\\")
    plotting.write_tex(gen / "ch12-budget-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for t in TEMPS:
        cells = [pct(100 * (next(x["m"] for x in B["temps"] if x["start_worse"] == t
                                 and x["polish"] == p) / best500 - 1)) for p in (False, True)]
        rows.append(f"{100 * t:g}\\% & " + " & ".join(cells) + " \\\\")
    plotting.write_tex(gen / "ch12-temp-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    tg = [100 * (x["m"] / best500 - 1) for x in B["temps"]]
    m["temp-gap-min"], m["temp-gap-max"] = pct(min(tg)), pct(max(tg))
    V = B["vrptw"]
    m["tw-pyvrp-cost"] = f"\\${V['pyvrp']['cost']:,.0f}"
    m["tw-ot-cost"] = f"\\${V['ortools']['cost']:,.0f}"
    m["tw-pyvrp-vans"] = V["pyvrp"]["vans"]
    m["tw-ot-vans"] = V["ortools"]["vans"]
    m["tw-saving"] = f"\\${V['ortools']['cost'] - V['pyvrp']['cost']:,.0f}"
    m["tw-saving-pct"] = f"{100 * (1 - V['pyvrp']['cost'] / V['ortools']['cost']):.1f}"
    m["tw-saving-year"] = f"\\${300 * (V['ortools']['cost'] - V['pyvrp']['cost']):,.0f}"
    m["tw-iterations"] = f"{V['iterations']:,}"
    # split example
    S = A["split"]
    codes = {"Pilsen": "PI", "Little Village": "LV", "Bridgeport": "BR", "Chinatown": "CT",
             "Back of the Yards": "BY", "Gage Park": "GP"}
    short = ["D"] + [codes[x] for x in S["names"][1:]]
    m["split-tour"] = ", ".join(short[c] for c in S["tour"])
    m["split-cost"] = f"{S['V'][-1]:.1f}"
    m["split-routes"] = and_join("D--" + "--".join(short[c] for c in r[1:-1]) + "--D"
                                  for r in S["routes"])
    m["split-arcs"] = len(S["arcs"])
    rows = []
    for i, j, c, load in S["arcs"]:
        seg = ", ".join(short[S["tour"][k]] for k in range(i, j))
        rows.append(f"$V_{i}\\to V_{j}$ & {seg} & {load} & {c:.1f} \\\\")
    plotting.write_tex(gen / "ch12-split-arcs.tex", "\n".join(rows) + "\n", encoding="utf-8")
    m["split-V"] = ", ".join(f"$V_{k} = {v:.1f}$" for k, v in enumerate(S["V"]) if k)
    lines = ["% Generated by code/examples/ch12_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch12/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch12-results.tex", "\n".join(lines) + "\n", encoding="utf-8")
    _plot_convergence(r500, book / "figures" / "ch12" / "convergence.pdf")
    print("book assets written")


def _plot_convergence(r, path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    best = r["best"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    show = [("OR-Tools", 3, "OR-Tools (guided local search)"), ("ALNS", 0, "our ALNS"),
            ("HGS (ours)", 1, "our hybrid genetic search")]
    for k, (name, c, label) in enumerate(show):
        h = r["methods"][name]["history"]
        if not h:
            continue
        t = [max(x[0], 0.05) for x in h] + [r["seconds"]]
        g = list(np.minimum.accumulate([100 * (x[1] / best - 1) for x in h]))  # best so far
        g = g + [g[-1]]
        ax.step(t, g, where="post", linestyle=plotting.ROUTE_STYLES[k],
                color=plotting.ROUTE_COLORS[c], linewidth=1.3, label=label)
    pv = 100 * (r["methods"]["PyVRP"]["m"] / best - 1)
    ax.scatter([r["seconds"]], [pv], marker="D", s=30, color=plotting.ROUTE_COLORS[2],
               zorder=5, label="PyVRP, final plan")
    ax.set_xscale("log")
    ax.set_xlabel("seconds")
    ax.set_ylabel("gap to the best known plan (\\%)")
    ax.grid(True, color=plotting.GRID, linewidth=0.5, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7.5)
    plotting.save(fig, path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--from-json", action="store_true")
    ap.add_argument("--part", choices=["compare", "tuning", "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pa, pb = OUT / "compare.json", OUT / "tuning.json"
    if not args.from_json:
        if args.part in ("compare", "all"):
            pa.write_text(json.dumps(part_compare(), indent=1))
        if args.part in ("tuning", "all"):
            pb.write_text(json.dumps(part_tuning(), indent=1))
    if args.book:
        write_book_assets(args.book, json.loads(pa.read_text()), json.loads(pb.read_text()))


if __name__ == "__main__":
    main()
