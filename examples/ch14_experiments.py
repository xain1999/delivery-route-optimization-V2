"""Chapter 14: re-optimizing during the day (Saguaro Courier, Phoenix).

    python examples/ch14_experiments.py --part normal     # about 8 minutes
    python examples/ch14_experiments.py --part busy       # about 15 minutes
    python examples/ch14_experiments.py --part warm       # warm starts only
    python examples/ch14_experiments.py --book ../book --from-json

Each simulated day uses the same order stream for every policy (common
random numbers), so that differences come from the policies alone.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from vrpbook import plotting
from vrpbook.dynamic import Car, Geometry, generate_day, plan, simulate

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "ch14"
DAYS = [1, 2, 3, 4, 5]
SECONDS = 0.5
NORMAL_CARS, BUSY_CARS, BUSY = 6, 4, 2.0
EPOCHS = [5, 10, 20]
BUDGETS = [0.1, 0.3, 1.0, 3.0]
WARM_HOURS = [11, 13, 15]
T95 = {4: 2.776}                     # Student t, 4 degrees of freedom


def run_days(policies, cars, busy=1.0, epoch_min=10, days=DAYS, log=None):
    out = {p: [] for p in policies}
    for d in days:
        for p in policies:
            res = simulate(generate_day(d, busy=busy), cars, p, seconds=SECONDS,
                           epoch_min=epoch_min)
            out[p].append({k: float(v) for k, v in res.items()})
            print(p, d, busy, epoch_min, {k: round(float(v), 1) for k, v in res.items()},
                  flush=True)
    return out


def part_normal() -> dict:
    R = {"policies": run_days(["now", "wait"], NORMAL_CARS)}
    R["epochs"] = {str(e): run_days(["wait"], NORMAL_CARS, epoch_min=e, days=DAYS[:3])["wait"]
                   for e in EPOCHS}
    return R


def part_busy() -> dict:
    return {"policies": run_days(["now", "wait", "accept"], BUSY_CARS, busy=BUSY),
            "warm": part_warm()}


def part_warm() -> dict:
    """Warm and cold starts at several moments of three busy days."""
    W: dict = {"moments": [], "runs": []}
    for day in DAYS[:3]:
        snap: dict = {"times": [h * 3600 for h in WARM_HOURS]}
        simulate(generate_day(day, busy=BUSY), BUSY_CARS, "wait", seconds=SECONDS, snapshot=snap)
        orders = generate_day(day, busy=BUSY)
        G = Geometry(orders)
        for mo in snap.get("moments", []):
            todo = [orders[i] for i in mo["todo"]]
            cars = [Car(k, f) for k, f in enumerate(mo["free"])]
            pen = {int(k): v for k, v in mo["penalty"].items()}
            start = {int(k): v for k, v in mo["last"].items()}
            known = {i for r in start.values() for i in r}
            W["moments"].append({"day": day, "now": mo["now"], "size": mo["size"],
                                 "new": sum(o.id not in known for o in todo)})
            for b in BUDGETS:
                _, cold = plan(mo["now"], cars, todo, G, penalty=pen, seconds=b)
                _, bare = plan(mo["now"], cars, todo, G, penalty=pen, seconds=b, initial=start,
                               insert_new=False)
                _, warm = plan(mo["now"], cars, todo, G, penalty=pen, seconds=b, initial=start)
                W["runs"].append({"day": day, "now": mo["now"], "seconds": b,
                                  "cold": cold, "bare": bare, "warm": warm})
                print("warm", day, mo["now"] // 3600, b, cold, bare, warm, flush=True)
    return W


# ---------------------------------------------------------------- book assets
def mean(xs):
    return statistics.mean(xs)


def paired_ci(a, b):
    """Mean of b - a and its 95% half-width over the paired days."""
    d = [y - x for x, y in zip(a, b)]
    return mean(d), T95[len(d) - 1] * statistics.stdev(d) / math.sqrt(len(d))


def write_book_assets(book: Path, A: dict, B: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    m = {"days": len(DAYS), "seconds": SECONDS, "normal-cars": NORMAL_CARS,
         "busy-cars": BUSY_CARS, "busy-factor": {2.0: "twice", 3.0: "three times"}.get(BUSY, f"{BUSY:g} times")}
    labels = {"now": "At once", "wait": "Wait", "accept": "Wait, decline"}

    def table(P, name):
        rows = []
        for p, runs in P.items():
            f = lambda k: mean([r[k] for r in runs])   # noqa: E731
            rows.append(f"{labels[p]} & {f('orders'):.0f} & {f('trips'):.1f} & "
                        f"{f('per_trip'):.1f} & {f('miles'):,.0f} & {f('driving_h'):.1f} & "
                        f"{f('mean_wait_min'):.0f} & {f('late'):.1f} & {f('late_min'):,.0f} & "
                        f"{f('rejected'):.1f} \\\\")
        plotting.write_tex(gen / f"ch14-{name}-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    table(A["policies"], "normal")
    table(B["policies"], "busy")
    N, Bp = A["policies"], B["policies"]
    for key, P, a, b in (("n", N, "now", "wait"), ("b", Bp, "now", "wait"),
                         ("ba", Bp, "wait", "accept")):
        for metric in ("miles", "late", "late_min", "mean_wait_min", "driving_h"):
            d, h = paired_ci([r[metric] for r in P[a]], [r[metric] for r in P[b]])
            m[f"{key}-{metric.replace('_', '')}-diff"] = f"{d:,.1f}"
            m[f"{key}-{metric.replace('_', '')}-ci"] = f"{h:,.1f}"
    for key, P in (("n", N), ("b", Bp)):
        for p in P:
            for metric in ("miles", "late", "late_min", "mean_wait_min", "orders", "rejected",
                           "per_trip", "driving_h"):
                m[f"{key}-{p}-{metric.replace('_', '')}"] = f"{mean([r[metric] for r in P[p]]):,.1f}"
    m["n-miles-pct"] = f"{100 * (1 - mean([r['miles'] for r in N['wait']]) / mean([r['miles'] for r in N['now']])):.0f}"
    m["b-latemin-pct"] = f"{100 * (1 - mean([r['late_min'] for r in Bp['accept']]) / mean([r['late_min'] for r in Bp['now']])):.0f}"
    m["b-reject-pct"] = f"{100 * mean([r['rejected'] / r['orders'] for r in Bp['accept']]):.0f}"
    rows = []
    for e in EPOCHS:
        runs = A["epochs"][str(e)]
        rows.append(f"{e} & {mean([r['miles'] for r in runs]):,.0f} & "
                    f"{mean([r['mean_wait_min'] for r in runs]):.0f} & "
                    f"{mean([r['late'] for r in runs]):.1f} \\\\")
    plotting.write_tex(gen / "ch14-epoch-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    W = B["warm"]
    best = {}
    for r in W["runs"]:
        k = (r["day"], r["now"])
        best[k] = min(best.get(k, float("inf")), r["cold"], r["bare"], r["warm"])
    excess = {(b, s): [] for b in BUDGETS for s in ("cold", "bare", "warm")}
    for r in W["runs"]:
        for s_ in ("cold", "bare", "warm"):
            excess[r["seconds"], s_].append(r[s_] - best[r["day"], r["now"]])
    rows = [f"{b:g} & \\${mean(excess[b, 'cold']):,.0f} & \\${mean(excess[b, 'bare']):,.0f} & "
            f"\\${mean(excess[b, 'warm']):,.0f} \\\\" for b in BUDGETS]
    plotting.write_tex(gen / "ch14-warm-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    Mo = W["moments"]
    m["warm-moments"] = len(Mo)
    m["warm-days"] = len({x["day"] for x in Mo})
    m["warm-size-min"], m["warm-size-max"] = min(x["size"] for x in Mo), max(x["size"] for x in Mo)
    m["warm-new-mean"] = f"{mean([x['new'] for x in Mo]):.0f}"
    m["warm-hours"] = ", ".join(f"{h}:00" for h in WARM_HOURS)
    pairs = [(r["warm"], r["cold"]) for r in W["runs"]]
    m["warm-beats-cold"] = sum(w < c - 0.5 for w, c in pairs)
    m["warm-ties-cold"] = sum(abs(w - c) <= 0.5 for w, c in pairs)
    m["warm-runs"] = len(pairs)
    for b in BUDGETS:
        t = f"{b:g}".replace(".", "")
        for s_ in ("cold", "bare", "warm"):
            m[f"warm-{s_}-{t}"] = f"\\${mean(excess[b, s_]):,.0f}"
    lines = ["% Generated by code/examples/ch14_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch14/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch14-results.tex", "\n".join(lines) + "\n", encoding="utf-8")
    _plot_day(book / "figures" / "ch14" / "day.pdf")
    print("book assets written")


def _plot_day(path):
    """Orders waiting at the hub and cars on the road through one normal day."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for k, (p, label) in enumerate((("now", "dispatch at once"),
                                    ("wait", "plan all cars, wait while safe"))):
        log: list = []
        simulate(generate_day(DAYS[0]), NORMAL_CARS, p, seconds=SECONDS, log=log)
        ax.step([t / 3600 for t, _, _ in log], [w for _, w, _ in log], where="post",
                linestyle=plotting.ROUTE_STYLES[k], color=plotting.ROUTE_COLORS[k],
                linewidth=1.3, label=f"orders at the hub: {label}")
    ax.set_xlabel("time of day (hours)")
    ax.set_ylabel("orders at the hub")
    ax.grid(True, color=plotting.GRID, linewidth=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    plotting.save(fig, path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--from-json", action="store_true")
    ap.add_argument("--part", choices=["normal", "busy", "warm", "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pa, pb = OUT / "normal.json", OUT / "busy.json"
    if not args.from_json:
        if args.part in ("normal", "all"):
            pa.write_text(json.dumps(part_normal(), indent=1))
        if args.part in ("busy", "all"):
            pb.write_text(json.dumps(part_busy(), indent=1))
        if args.part == "warm":                 # redo only the warm-start comparison
            B = json.loads(pb.read_text())
            B["warm"] = part_warm()
            pb.write_text(json.dumps(B, indent=1))
    if args.book:
        write_book_assets(args.book, json.loads(pa.read_text()), json.loads(pb.read_text()))


if __name__ == "__main__":
    main()
