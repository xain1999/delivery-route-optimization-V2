"""Chapter 11: local search and fast move evaluation.

    python examples/ch10_experiments.py --book ../book      # first: the references
    python examples/ch11_experiments.py --book ../book      # about 10 minutes

Uses the instances and the PyVRP reference values of Chapter 10.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from vrpbook import distance_matrix, load_csv, plotting, solve_cvrp, total_distance
from vrpbook import validate_routes
from vrpbook import construction as C
from vrpbook.instances import chicago_ch02, chicago_synthetic
from vrpbook.localsearch import (OPERATORS, Plan, Segment, best_move_for, broken_pairs,
                                 local_search)

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch11"
M_PER_MI = 1609.344
ORTOOLS_MAX_N = 1000
ABLATION = [("relocate",), ("relocate", "swap"), ("relocate", "swap", "2opt"),
            ("relocate", "swap", "2opt", "2opt*"), OPERATORS]
KS = [5, 10, 20, 40, 100]
N_STARTS = 50



def and_join(xs) -> str:
    """'a, b and c'."""
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

def instances():
    yield "Chapter 2 (30)", chicago_ch02()
    yield "Tuesday (37)", load_csv(HERE / "data" / "chicago_tuesday.csv", 6, 100)
    for n in (100, 250, 500, 1000, 2000):
        yield f"Random {n:,}", chicago_synthetic(n)


def gap(v, ref):
    return 100 * (v / ref - 1)


def run_benchmark(refs) -> list[dict]:
    rows = []
    for name, inst in instances():
        d = distance_matrix(inst.sites)
        q, Q = inst.demands, inst.capacity
        ref = refs[name]
        row = {"name": name, "n": len(q) - 1, "ref": ref}
        for start, f in (("NN", C.nearest_neighbor), ("CW", C.clarke_wright)):
            t = time.perf_counter()
            r0 = f(d, q, Q)
            t_build = time.perf_counter() - t
            r, st = local_search(r0, d, q, Q)
            validate_routes(r, d, q, Q)
            row[start] = {"before": total_distance(r0, d), "after": total_distance(r, d),
                          "build_s": t_build, "ls_s": st["seconds"], "moves": st["moves"],
                          "evaluations": st["evaluations"], "by_op": st["by_op"]}
        if row["n"] <= ORTOOLS_MAX_N:
            vans = math.ceil(1.2 * math.ceil(sum(q) / Q)) + 1
            sol = solve_cvrp(d, q, vans, Q, first_solution="SAVINGS",
                             metaheuristic="GREEDY_DESCENT", time_limit_s=600)
            routes = sol.used_routes
            validate_routes(routes, d, q, Q)
            row["ortools"] = {"after": total_distance(routes, d), "s": sol.wall_time_s}
        print(name, {k: (round(gap(v["after"], ref), 1), round(v.get("ls_s", v.get("s") or 0), 2))
                     for k, v in row.items() if isinstance(v, dict)}, flush=True)
        rows.append(row)
    return rows


def run_ablation(ref500) -> dict:
    inst = chicago_synthetic(500)
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    out = {"ops": [], "k": [], "strategy": []}
    for start, f in (("NN", C.nearest_neighbor), ("CW", C.clarke_wright)):
        r0 = f(d, q, Q)
        for ops in ABLATION:
            r, st = local_search(r0, d, q, Q, ops=ops)
            out["ops"].append({"start": start, "ops": list(ops), "gap": gap(total_distance(r, d), ref500),
                               "s": st["seconds"]})
    r0 = C.nearest_neighbor(d, q, Q)
    for k in KS + [len(q) - 2]:
        r, st = local_search(r0, d, q, Q, k=k)
        out["k"].append({"k": k, "gap": gap(total_distance(r, d), ref500), "s": st["seconds"],
                         "evaluations": st["evaluations"]})
    for strategy in ("first", "best"):
        r, st = local_search(r0, d, q, Q, strategy=strategy)
        out["strategy"].append({"strategy": strategy, "gap": gap(total_distance(r, d), ref500),
                                "s": st["seconds"], "moves": st["moves"],
                                "evaluations": st["evaluations"]})
    print("ablation", out, flush=True)
    return out


def random_plan(q, Q, rng) -> list[list[int]]:
    perm = list(range(1, len(q)))
    rng.shuffle(perm)
    routes, cur, load = [], [0], 0
    for c in perm:
        if load + q[c] > Q:
            routes.append(cur + [0])
            cur, load = [0], 0
        cur.append(c)
        load += q[c]
    return routes + [cur + [0]]


def run_landscape(ref100) -> dict:
    inst = chicago_synthetic(100)
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    rng = random.Random(11)
    optima = []
    for s in range(N_STARTS):
        r0 = random_plan(q, Q, rng)
        r, _ = local_search(r0, d, q, Q, seed=s)
        optima.append({"start": total_distance(r0, d), "length": total_distance(r, d),
                       "routes": r})
    best = min(optima, key=lambda o: o["length"])
    for o in optima:
        o["dist_to_best"] = broken_pairs(o["routes"], best["routes"])
        o["gap"] = gap(o["length"], ref100)
        o["start_gap"] = gap(o["start"], ref100)
    distinct = len({round(o["length"]) for o in optima})
    print("landscape", distinct, min(o["gap"] for o in optima), flush=True)
    return {"optima": [{k: v for k, v in o.items() if k != "routes"} for o in optima],
            "distinct": distinct}


def run_segments() -> list[dict]:
    """Time to check every insertion position of a customer in a route,
    by rescheduling the route (naive) or by concatenating segments."""
    rng = np.random.default_rng(3)
    out = []
    for L in (10, 25, 50, 100):
        n = L + 2
        t = rng.integers(120, 900, size=(n, n)).tolist()
        service = [0] + [180] * (n - 1)
        windows = [(0, 10**6)] + [(int(a), int(a) + 7200)
                                  for a in rng.integers(0, 20000, size=n - 1)]
        route = [0] + list(range(1, L + 1)) + [0]
        c = n - 1                                   # the customer to insert
        segs = [Segment.of(i, service[i], *windows[i]) for i in range(n)]
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):                       # naive: rebuild every candidate route
            for p in range(1, L + 2):
                cand = route[:p] + [c] + route[p:]
                s = segs[cand[0]]
                for i in cand[1:]:
                    s = s.then(segs[i], t)
                _ = s.TW == 0
        naive = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):                       # prefix and suffix segments, O(1) each
            pre = [segs[route[0]]]
            for i in route[1:]:
                pre.append(pre[-1].then(segs[i], t))
            suf = [segs[route[-1]]]
            for i in reversed(route[:-1]):
                suf.append(segs[i].then(suf[-1], t))
            suf.reverse()
            for p in range(1, L + 2):
                s = pre[p - 1].then(segs[c], t).then(suf[p], t)
                _ = s.TW == 0
        fast = (time.perf_counter() - t0) / reps
        out.append({"L": L, "naive_ms": 1000 * naive, "fast_ms": 1000 * fast})
    print("segments", out, flush=True)
    return out


def worked_example() -> dict:
    """The nearest-neighbor plan of the six customers of Chapter 10."""
    base = chicago_ch02()
    ids = [16, 17, 18, 19, 26, 27]
    sub = base.subset(ids, n_vehicles=3)
    d = np.round(distance_matrix(sub.sites) / M_PER_MI, 1)
    q = sub.demands
    plan = C.nearest_neighbor(d, q, 45)
    P = Plan([list(r) for r in plan], q)
    D = d.tolist()
    moves = sorted((m for u in range(1, 7) for v in range(1, 7) if u != v
                    for m in [best_move_for(u, v, P, D, 45, OPERATORS)] if m),
                   key=lambda m: m[0])
    best = moves[0]
    after, _ = local_search(plan, d, q, 45, k=6)
    return {"names": [base.sites[i].name for i in [0] + ids], "plan": plan,
            "length": float(sum(d[a, b] for r in plan for a, b in zip(r, r[1:]))),
            "move": [best[0], list(best[1])], "dist": d.tolist(), "after": after,
            "after_length": float(sum(d[a, b] for r in after for a, b in zip(r, r[1:])))}


def run() -> dict:
    refs = {r["name"]: r["ref"] for r in
            json.loads((HERE / "results" / "ch10" / "results.json").read_text())["instances"]}
    R = {"bench": run_benchmark(refs), "worked": worked_example()}
    R["ablation"] = run_ablation(refs["Random 500"])
    R["landscape"] = run_landscape(refs["Random 100"])
    R["segments"] = run_segments()
    return R


# ---------------------------------------------------------------- book assets
def pct(x: float, digits: int = 1) -> str:
    s = f"{x:.{digits}f}"
    return s[1:] if s.startswith("-0") and float(s) == 0 else s


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    B = R["bench"]
    m = {}
    rows = []
    for r in B:
        ot = r.get("ortools")
        rows.append(
            f"{r['name'].replace('Chapter 2 (30)', 'Ch.~2 (30)').replace('Random ', '')} & "
            f"{pct(gap(r['NN']['before'], r['ref']), 0)} & {pct(gap(r['NN']['after'], r['ref']))} & "
            f"{pct(gap(r['CW']['before'], r['ref']), 0)} & {pct(gap(r['CW']['after'], r['ref']))} & "
            f"{(pct(gap(ot['after'], r['ref'])) if ot else '--')} & "
            f"{r['CW']['ls_s']:.2f} & {r['CW']['moves']:,} \\\\")
    plotting.write_tex(gen / "ch11-bench-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    cw_after = [gap(r["CW"]["after"], r["ref"]) for r in B]
    nn_after = [gap(r["NN"]["after"], r["ref"]) for r in B]
    m["cw-after-max"] = pct(max(cw_after))
    m["cw-after-min"] = pct(min(cw_after))
    m["nn-after-max"] = pct(max(nn_after))
    m["nn-after-min"] = pct(min(nn_after))
    big = B[-1]
    m["big-ls-s"] = f"{big['CW']['ls_s']:.1f}"
    m["big-nn-ls-s"] = f"{big['NN']['ls_s']:.1f}"
    m["big-nn-moves"] = f"{big['NN']['moves']:,}"
    m["big-nn-evals"] = f"{big['NN']['evaluations']:,}"
    m["big-cw-after"] = pct(gap(big["CW"]["after"], big["ref"]))
    ot = [r for r in B if "ortools" in r]
    m["ot-better"] = sum(1 for r in ot if r["ortools"]["after"] < r["CW"]["after"])
    m["ot-count"] = len(ot)
    m["ot-max-s"] = f"{max(r['ortools']['s'] for r in ot):.0f}"
    m["ot-max-n"] = f"{max(r['n'] for r in ot):,}"
    m["cw-at-ot-max-s"] = f"{max(r['CW']['ls_s'] for r in ot):.2f}"
    # ablation
    A = R["ablation"]
    rows = []
    labels = ["relocate", "+ swap", "+ 2-opt", "+ 2-opt*", "+ Or-opt"]
    for k, lab in enumerate(labels):
        nn = next(x for x in A["ops"] if x["start"] == "NN" and len(x["ops"]) == k + 1)
        cw = next(x for x in A["ops"] if x["start"] == "CW" and len(x["ops"]) == k + 1)
        rows.append(f"{lab} & {pct(nn['gap'])} & {nn['s']:.2f} & {pct(cw['gap'])} & {cw['s']:.2f} \\\\")
    plotting.write_tex(gen / "ch11-ops-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for x in A["k"]:
        lab = "all" if x["k"] > 200 else str(x["k"])
        rows.append(f"{lab} & {x['evaluations']:,} & {pct(x['gap'])} & {x['s']:.2f} \\\\")
    plotting.write_tex(gen / "ch11-k-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    k20 = next(x for x in A["k"] if x["k"] == 20)
    kall = A["k"][-1]
    m["k20-gap"] = pct(k20["gap"])
    m["kall-gap"] = pct(kall["gap"])
    m["k20-s"] = f"{k20['s']:.2f}"
    m["kall-s"] = f"{kall['s']:.1f}"
    m["k-speedup"] = f"{kall['s'] / k20['s']:.0f}"
    F, Bs = A["strategy"]
    m["first-gap"], m["best-gap"] = pct(F["gap"]), pct(Bs["gap"])
    m["first-s"], m["best-s"] = f"{F['s']:.2f}", f"{Bs['s']:.1f}"
    m["first-moves"], m["best-moves"] = f"{F['moves']:,}", f"{Bs['moves']:,}"
    m["first-evals"], m["best-evals"] = f"{F['evaluations']:,}", f"{Bs['evaluations']:,}"
    # landscape
    L = R["landscape"]
    gaps = [o["gap"] for o in L["optima"]]
    m["land-starts"] = len(gaps)
    m["land-distinct"] = L["distinct"]
    m["land-gap-min"], m["land-gap-max"] = pct(min(gaps)), pct(max(gaps))
    m["land-gap-mean"] = pct(float(np.mean(gaps)))
    m["land-start-gap"] = pct(float(np.mean([o["start_gap"] for o in L["optima"]])), 0)
    dd = [o["dist_to_best"] for o in L["optima"] if o["dist_to_best"] > 0]
    m["land-dist-mean"] = f"{100 * float(np.mean(dd)):.0f}"
    corr = float(np.corrcoef([o["dist_to_best"] for o in L["optima"]], gaps)[0, 1])
    m["land-corr"] = f"{corr:.2f}"
    # segments
    S = R["segments"]
    rows = [f"{s['L']} & {s['naive_ms']:.2f} & {s['fast_ms']:.2f} & "
            f"{s['naive_ms'] / s['fast_ms']:.0f} \\\\" for s in S]
    plotting.write_tex(gen / "ch11-seg-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    m["seg-speedup-max"] = f"{max(s['naive_ms'] / s['fast_ms'] for s in S):.0f}"
    # worked example
    W = R["worked"]
    names = W["names"]
    codes = {"Pilsen": "PI", "Little Village": "LV", "Bridgeport": "BR", "Chinatown": "CT",
             "Back of the Yards": "BY", "Gage Park": "GP"}
    short = ["D"] + [codes[x] for x in names[1:]]
    d = W["dist"]
    m["we-plan"] = and_join("D--" + "--".join(short[i] for i in r[1:-1]) + "--D"
                             for r in W["plan"])
    m["we-length"] = f"{W['length']:.1f}"
    delta, (kind, a, b) = W["move"][0], W["move"][1][:3]
    m["we-kind"] = kind
    r = next(rt for rt in W["plan"] if a in rt)
    na, nb = r[r.index(a) + 1], r[r.index(b) + 1]
    m["we-a"], m["we-b"], m["we-na"], m["we-nb"] = short[a], short[b], short[na], short[nb]
    m["we-dab"], m["we-dnanb"] = f"{d[a][b]:.1f}", f"{d[na][nb]:.1f}"
    m["we-dana"], m["we-dbnb"] = f"{d[a][na]:.1f}", f"{d[b][nb]:.1f}"
    m["we-delta"] = f"{delta:.1f}"
    m["we-after"] = and_join("D--" + "--".join(short[i] for i in rt[1:-1]) + "--D"
                              for rt in W["after"])
    m["we-after-length"] = f"{W['after_length']:.1f}"
    lines = ["% Generated by code/examples/ch11_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch11/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch11-results.tex", "\n".join(lines) + "\n", encoding="utf-8")
    _plot_landscape(R, book / "figures" / "ch11" / "landscape.pdf")
    print("book assets written")


def _plot_landscape(R, path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    O = R["landscape"]["optima"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.scatter([100 * o["dist_to_best"] for o in O], [o["gap"] for o in O], s=18,
               color=plotting.ROUTE_COLORS[0], edgecolor="white", linewidth=0.6,
               label="local optimum from a random start")
    ax.set_xlabel("distance to the best local optimum (\\% of links not shared)")
    ax.set_ylabel("gap to the reference (\\%)")
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
