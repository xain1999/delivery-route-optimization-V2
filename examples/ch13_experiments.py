"""Chapter 13: how good is my solution?  Bounds, exact solutions, fair tests.

    python examples/ch13_experiments.py --book ../book      # about 25 minutes

Uses the best plans found in Chapters 10 and 12 (run those first), the
benchmark instance E-n22-k4 of Christofides and Eilon (optimum 375), and
random Chicago instances.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


from vrpbook import distance_matrix, load_csv, plotting, total_distance, validate_routes
from vrpbook.bounds import exact_cvrp, lp_bound, qroute_bound, read_vrplib
from vrpbook.provenance import versions
from vrpbook.construction import radial_lower_bound
from vrpbook.instances import chicago_ch02, chicago_synthetic
from vrpbook.metaheuristics import alns
from vrpbook.pyvrp_adapter import solve_cvrp_pyvrp

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch13"
M_PER_MI = 1609.344
EXACT_LIMIT = 600
SEEDS = [1, 2, 3, 4, 5]
FAIR_SECONDS = 10


def instances():
    c, q, Q = read_vrplib(HERE / "data" / "benchmarks" / "E-n22-k4.vrp")
    yield "E-n22-k4", c, q, Q, 1.0
    for name, inst in (("Chapter 2 (30)", chicago_ch02()),
                       ("Tuesday (37)", load_csv(HERE / "data" / "chicago_tuesday.csv", 6, 100)),
                       ("Random 60", chicago_synthetic(60)),
                       ("Random 100", chicago_synthetic(100))):
        yield name, distance_matrix(inst.sites), inst.demands, inst.capacity, M_PER_MI


def known_best() -> dict:
    best = {}
    for r in json.loads((HERE / "results" / "ch10" / "results.json").read_text())["instances"]:
        best[r["name"]] = r["ref"]
    for r in json.loads((HERE / "results" / "ch12" / "compare.json").read_text())["instances"]:
        best[r["name"]] = min([best.get(r["name"], 1e18)] + [v["m"] for v in r["methods"].values()])
    return best


def run() -> dict:
    best = known_best()
    R = {"instances": [], "env": versions()}
    for name, c, q, Q, unit in instances():
        n = len(q) - 1
        routes, _ = solve_cvrp_pyvrp(c, q, Q, 10, seed=1)
        validate_routes(routes, c, q, Q)
        ub = min(best.get(name, 1e18), total_distance(routes, c))
        row = {"name": name, "n": n, "unit": unit, "ub": ub,
               "radial": radial_lower_bound(c, q, Q)}
        L = lp_bound(c, q, Q)
        row["lp"] = {k: v for k, v in L.items() if k != "cut_sets"}
        if n <= 40:
            row["cg"] = qroute_bound(c, q, Q)
        if n <= 60:
            E = exact_cvrp(c, q, Q, 2 * len(routes), root_cuts=L["cut_sets"],
                           time_limit_s=EXACT_LIMIT)
            row["exact"] = E
        print(name, {k: (v if not isinstance(v, dict) else
                         {kk: vv for kk, vv in v.items() if kk != "history"})
                     for k, v in row.items()}, flush=True)
        R["instances"].append(row)

    # fair comparison: several seeds on Random 100
    inst = chicago_synthetic(100)
    c, q, Q = distance_matrix(inst.sites), inst.demands, inst.capacity
    fair = {"ALNS": [], "PyVRP": []}
    for s in SEEDS:
        r = alns(c, q, Q, FAIR_SECONDS, seed=s)
        validate_routes(r, c, q, Q)
        fair["ALNS"].append(total_distance(r, c))
        r, _ = solve_cvrp_pyvrp(c, q, Q, FAIR_SECONDS, seed=s)
        validate_routes(r, c, q, Q)
        fair["PyVRP"].append(total_distance(r, c))
        print("fair", s, fair["ALNS"][-1], fair["PyVRP"][-1], flush=True)
    R["fair"] = fair
    return R


# ---------------------------------------------------------------- book assets
def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    m = {"exact-limit": EXACT_LIMIT, "fair-seconds": FAIR_SECONDS, "seeds": len(SEEDS)}
    rows = []
    for r in R["instances"]:
        u = r["unit"]
        fmt = (lambda v: f"{v / u:,.0f}") if u == 1 else (lambda v: f"{v / u:,.1f}")
        lbs = [r["radial"], r["lp"]["bound"]] + ([r["cg"]["bound"]] if "cg" in r else [])
        ex = r.get("exact", {})
        if ex.get("status") == "optimal":
            lbs.append(ex["value"])
            opt = fmt(ex["value"])
        elif ex.get("bound"):                     # branch and cut stopped: its bound
            lbs.append(ex["bound"])
            opt = f"$\\ge$ {fmt(ex['bound'])}"
        elif ex:
            opt = "--"
        else:
            opt = ""
        lb = max(lbs)
        gap = 100 * (r["ub"] / lb - 1)
        name = r["name"].replace("Chapter 2 (30)", "Ch.~2 (30)")
        rows.append(f"{name} & {fmt(r['ub'])} & {fmt(r['radial'])} & "
                    f"{fmt(r['cg']['bound']) if 'cg' in r else ''} & {fmt(r['lp']['bound'])} & "
                    f"{opt} & {gap:.1f} \\\\")
        key = {"E-n22-k4": "en22k4", "Chapter 2 (30)": "ch2", "Tuesday (37)": "tuesday",
               "Random 60": "random60", "Random 100": "random100"}[r["name"]]
        m[f"{key}-gap"] = f"{gap:.1f}"                        # (U - L) / L, as in the table
        m[f"{key}-below"] = f"{100 * (1 - lb / r['ub']):.1f}"    # (U - L) / U: bound below plan
        m[f"{key}-lp-below"] = f"{100 * (1 - r['lp']['bound'] / r['ub']):.1f}"
        m[f"{key}-ub"] = fmt(r["ub"])
        m[f"{key}-lp-gap"] = f"{100 * (r['ub'] / r['lp']['bound'] - 1):.1f}"
        m[f"{key}-lp-s"] = f"{r['lp']['seconds']:.1f}"
        m[f"{key}-lp-cuts"] = f"{r['lp']['cuts']:,}"
        m[f"{key}-lp-rounds"] = r["lp"]["rounds"]
        m[f"{key}-radial-gap"] = f"{100 * (r['ub'] / r['radial'] - 1):.0f}"
        if "cg" in r and r["cg"].get("converged"):     # only a converged master is a bound
            m[f"{key}-cg-gap"] = f"{100 * (r['ub'] / r['cg']['bound'] - 1):.1f}"
            m[f"{key}-cg-cols"] = f"{r['cg']['columns']:,}"
            m[f"{key}-cg-its"] = r["cg"]["iterations"]
            m[f"{key}-cg-s"] = f"{r['cg']['seconds']:.1f}"
        if ex:
            m[f"{key}-exact-status"] = ex["status"]
            m[f"{key}-exact-s"] = f"{ex['seconds']:.0f}"
            m[f"{key}-exact-rounds"] = ex.get("rounds", 0)
            if ex.get("bound"):
                m[f"{key}-bb-bound"] = fmt(ex["bound"])
            if ex.get("status") == "optimal":
                m[f"{key}-opt"] = fmt(ex["value"])
                m[f"{key}-ub-is-opt"] = "yes" if r["ub"] <= ex["value"] + 1e-6 * u else "no"
    plotting.write_tex(gen / "ch13-bounds-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    # fair comparison
    F = R["fair"]
    r100 = next(r for r in R["instances"] if r["name"] == "Random 100")
    best = min(min(F["ALNS"]), min(F["PyVRP"]), r100["ub"])
    rows = []
    for k in ("ALNS", "PyVRP"):
        g = [100 * (v / best - 1) for v in F[k]]
        rows.append(f"{k} & {min(g):.2f} & {statistics.mean(g):.2f} & {max(g):.2f} & "
                    f"{statistics.stdev(g):.2f} \\\\")
        m[f"fair-{k.lower()}-mean"] = f"{statistics.mean(g):.2f}"
        m[f"fair-{k.lower()}-min"] = f"{min(g):.2f}"
        m[f"fair-{k.lower()}-max"] = f"{max(g):.2f}"
    plotting.write_tex(gen / "ch13-fair-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    wins = sum(1 for a in F["ALNS"] for p in F["PyVRP"] if a < p - 1e-6)
    m["fair-alns-pair-wins"] = wins
    m["fair-pairs"] = len(F["ALNS"]) * len(F["PyVRP"])
    m["fair-wrong-pct"] = f"{100 * wins / m['fair-pairs']:.0f}"
    lines = ["% Generated by code/examples/ch13_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch13/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch13-results.tex", "\n".join(lines) + "\n", encoding="utf-8")
    _plot_lp(r100, book / "figures" / "ch13" / "lp-rounds.pdf")
    print("book assets written")


def _plot_lp(r, path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    h = r["lp"]["history"]
    ub = r["ub"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot([x[2] for x in h], [100 * (1 - x[1] / ub) for x in h], "-o", markersize=2.5,
            color=plotting.ROUTE_COLORS[0], linewidth=1.2,
            label="LP bound with capacity cuts")
    ax.axhline(100 * (1 - r["radial"] / ub), color=plotting.INK_2, linestyle=":",
               linewidth=1, label="radial bound (Chapter 10)")
    ax.set_xlabel("capacity cuts added")
    ax.set_ylabel("bound below best plan (\\%)")
    ax.set_ylim(bottom=0)
    ax.grid(True, color=plotting.GRID, linewidth=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
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
        path.write_text(json.dumps(R, indent=1, default=float))
    if args.book:
        write_book_assets(args.book, R)


if __name__ == "__main__":
    main()
