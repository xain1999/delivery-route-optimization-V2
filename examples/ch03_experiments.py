"""Chapter 3: from Tuesday's order export to a clean routing instance.

    python examples/ch03_experiments.py --book ../book

Writes data/chicago_tuesday.csv (the cleaned instance, used in later
chapters), results/ch03/results.json and, with --book, the chapter's
tables, figures and numbers.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from vrpbook import distance_matrix, solve_cvrp, validate_routes
from vrpbook import plotting
from vrpbook.addresses import parse
from vrpbook.geocode import CachedGeocoder, ChicagoGridGeocoder, GeocoderChain
from vrpbook.orders import Rules, clean_orders, load_raw, to_instance, write_instance_csv

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "results" / "ch03"
METERS_PER_MILE = 1609.344
CUTOFF = datetime(2026, 10, 12, 23, 59, 59)
TIME_LIMIT = 10
PAYLOAD_SWEEP = [2200, 2300, 2400, 2500, 3000]

ACTIONS = {
    "duplicate_row": "dropped", "not_paid": "dropped", "after_cutoff": "next day",
    "outside_area": "review", "po_box": "review", "no_house_number": "review",
    "unparseable": "review", "empty_order": "review", "implausible_weight": "review",
    "not_geocoded": "review", "weight_estimated": "kept", "street_corrected": "kept",
    "split_delivery": "kept", "merged": "merged",
}
LABELS = {
    "duplicate_row": "Exact duplicate row", "not_paid": "Order cancelled",
    "after_cutoff": "Placed after cut-off", "outside_area": "Outside service area",
    "po_box": "PO box", "no_house_number": "No house number",
    "unparseable": "Address cannot be parsed", "empty_order": "Zero totes",
    "implausible_weight": "Implausible weight", "not_geocoded": "Could not be located",
    "weight_estimated": "Weight missing, estimated", "street_corrected": "Street name corrected",
    "split_delivery": "Larger than one van, split", "merged": "Second order, same customer",
}


def mi(m: float) -> float:
    return m / METERS_PER_MILE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- cleaning ----------------------------------------------------------
    rows = load_raw(DATA / "chicago_tuesday_orders_raw.csv")
    geocoder = GeocoderChain(CachedGeocoder(DATA / "geocode_cache.json"),
                             ChicagoGridGeocoder())
    fleet = json.loads((DATA / "fleet_tuesday.json").read_text())
    vt = fleet["vehicle_types"]
    n_vehicles = sum(v["count"] for v in vt)
    totes_cap, payload = vt[0]["capacity_totes"], vt[0]["payload_lb"]
    rules = Rules(cutoff=CUTOFF, vehicle_totes=totes_cap)
    res = clean_orders(rows, rules, geocoder)
    inst, weights = to_instance(res, fleet["depot"], n_vehicles, totes_cap)
    write_instance_csv(inst, weights, DATA / "chicago_tuesday.csv",
                       units=[""] + [st.unit for st in res.stops])

    R: dict = {"rows_in": res.rows_in, "stops": len(res.stops),
               "issues": [vars(i) for i in res.issues]}
    counts = res.counts()
    R["counts"] = {f"{sev}:{code}": n for (sev, code), n in counts.items()}
    R["orders_planned"] = len({o for s in res.stops for o in s.order_ids})
    R["customers_planned"] = len({s.customer_id for s in res.stops})
    R["review"] = len({i.order_id for i in res.review_queue()})
    R["dropped"] = sum(n for (sev, code), n in counts.items()
                       if code in ("duplicate_row", "not_paid"))
    R["deferred"] = len(res.deferred)
    R["totes"] = sum(s.totes for s in res.stops)
    R["weight"] = sum(s.weight_lb for s in res.stops)
    R["sources"] = {src: sum(1 for s in res.stops if s.geocode_source == src)
                    for src in sorted({s.geocode_source for s in res.stops})}
    R["fleet"] = {"vehicles": n_vehicles, "totes": totes_cap, "payload": payload}

    # ---- worked example: grid geocoding by hand ------------------------------
    addr = parse("2400 N Ashland Ave")
    g = ChicagoGridGeocoder().geocode(addr)
    R["worked"] = {"lat": g.lat, "lon": g.lon}

    # ---- capacity in two dimensions ----------------------------------------
    dist = distance_matrix(inst.sites)
    lb_totes = math.ceil(R["totes"] / totes_cap)
    lb_weight = math.ceil(R["weight"] / payload)
    R["bounds"] = {"totes": lb_totes, "weight": lb_weight}

    def route_loads(routes):
        return [{"totes": sum(inst.demands[i] for i in r[1:-1]),
                 "weight": sum(weights[i] for i in r[1:-1]),
                 "mi": mi(sum(int(dist[a, b]) for a, b in zip(r, r[1:])))}
                for r in routes if len(r) > 2]

    one = solve_cvrp(dist, inst.demands, n_vehicles, totes_cap, time_limit_s=TIME_LIMIT)
    validate_routes(one.routes, dist, inst.demands, totes_cap)
    two = solve_cvrp(dist, inst.demands, n_vehicles, totes_cap, time_limit_s=TIME_LIMIT,
                     extra_capacities={"Weight": (weights, payload)})
    validate_routes(two.routes, dist, inst.demands, totes_cap)
    validate_routes(two.routes, dist, weights, payload)        # weight checked too
    R["one_dim"] = {"mi": mi(one.objective), "routes": one.routes,
                    "loads": route_loads(one.routes)}
    R["two_dim"] = {"mi": mi(two.objective), "routes": two.routes,
                    "loads": route_loads(two.routes)}
    R["one_dim"]["overloaded"] = [l for l in R["one_dim"]["loads"] if l["weight"] > payload]

    sweep = []
    for P in PAYLOAD_SWEEP:
        s = solve_cvrp(dist, inst.demands, n_vehicles, totes_cap, time_limit_s=TIME_LIMIT,
                       extra_capacities={"Weight": (weights, P)})
        ok = s.objective >= 0 and bool(s.routes)
        if ok:
            validate_routes(s.routes, dist, weights, P)
        sweep.append({"payload": P, "bound": max(lb_totes, math.ceil(R["weight"] / P)),
                      "util": 100 * R["weight"] / (n_vehicles * P),
                      "status": s.status, "mi": mi(s.objective) if ok else None})
    R["sweep"] = sweep
    (OUT / "results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R.items() if k not in ("issues",)}, indent=1)[:3000])

    if args.book:
        write_book_assets(args.book, inst, res, R)


def write_book_assets(book: Path, inst, res, R) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    F = R["fleet"]
    one, two = R["one_dim"], R["two_dim"]
    m = {
        "rows-in": R["rows_in"], "stops": R["stops"], "orders-planned": R["orders_planned"],
        "customers-planned": R["customers_planned"], "review": R["review"],
        "dropped": R["dropped"], "deferred": R["deferred"],
        "totes": R["totes"], "weight": f"{R['weight']:,}",
        "src-grid": R["sources"].get("chicago-grid", 0), "src-cache": R["sources"].get("cache", 0),
        "vehicles": F["vehicles"], "cap-totes": F["totes"], "payload": f"{F['payload']:,}",
        "lb-totes": R["bounds"]["totes"], "lb-weight": R["bounds"]["weight"],
        "util-totes": f"{100 * R['totes'] / (F['vehicles'] * F['totes']):.1f}",
        "util-weight": f"{100 * R['weight'] / (F['vehicles'] * F['payload']):.1f}",
        "one-mi": f"{one['mi']:.1f}", "two-mi": f"{two['mi']:.1f}",
        "two-pct": f"{100 * (two['mi'] / one['mi'] - 1):.1f}",
        "n-overloaded": len(one["overloaded"]),
        "max-overload": f"{max((l['weight'] for l in one['overloaded']), default=0):,}",
        "worked-lat": f"{R['worked']['lat']:.4f}", "worked-lon": f"{R['worked']['lon']:.4f}",
        "time-limit": TIME_LIMIT,
    }
    for s in R["sweep"]:
        m[f"sweep{s['payload']}-bound"] = s["bound"]
        m[f"sweep{s['payload']}-util"] = f"{s['util']:.1f}"
        m[f"sweep{s['payload']}-mi"] = f"{s['mi']:.1f}" if s["mi"] else "--"
    lines = ["% Generated by code/examples/ch03_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch03/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch03-results.tex", "\n".join(lines) + "\n", encoding="utf-8")

    # issue table, in pipeline order
    order = list(ACTIONS)
    counts = {}
    for i in res.issues:
        counts[i.code] = counts.get(i.code, 0) + 1
    rows = []
    for code in order:
        if code in counts:
            sev = next(i.severity for i in res.issues if i.code == code)
            tt = code.replace("_", "\\_")
            rows.append(f"{LABELS[code]} & \\texttt{{{tt}}} & "
                        f"{sev} & {ACTIONS[code]} & {counts[code]} \\\\")
    plotting.write_tex(gen / "ch03-issue-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # review queue
    rows = []
    for i in res.review_queue():
        msg = i.message.replace("_", "\\_").replace("'", "")
        tt = i.code.replace("_", "\\_")
        rows.append(f"{i.order_id} & \\texttt{{{tt}}} & {msg} \\\\")
    plotting.write_tex(gen / "ch03-review-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # load table: per van, both plans
    rows = []
    P = F["payload"]
    for k in range(max(len(one["loads"]), len(two["loads"]))):
        cells = [f"Van {k + 1}"]
        for plan in (one, two):
            if k < len(plan["loads"]):
                l = plan["loads"][k]
                w = f"{l['weight']:,}"
                if l["weight"] > P:
                    w = f"\\textbf{{{w}}}"
                cells += [str(l["totes"]), w, f"{l['mi']:.1f}"]
            else:
                cells += ["", "", ""]
        rows.append(" & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    rows.append(f"Total & {R['totes']} & {R['weight']:,} & {one['mi']:.1f} & "
                f"{R['totes']} & {R['weight']:,} & {two['mi']:.1f} \\\\")
    plotting.write_tex(gen / "ch03-load-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    # payload sweep table
    rows = []
    for s in R["sweep"]:
        if s["mi"]:
            result = f"{s['mi']:.1f}"
        elif s["bound"] > F["vehicles"]:
            result = "impossible (bound)"
        else:
            result = f"none found in {TIME_LIMIT}\\,s"
        rows.append(f"\\num{{{s['payload']}}} & {s['util']:.1f} & {s['bound']} & {result} \\\\")
    plotting.write_tex(gen / "ch03-sweep-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    _plot_stops(inst, res, book / "figures" / "ch03" / "stops.pdf")
    _plot_loads(R, book / "figures" / "ch03" / "loads.pdf")
    print("book assets written to", book)


def _plot_stops(inst, res, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plotting.use_book_style()
    xs, ys = plotting.to_miles(inst)
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    styles = {"chicago-grid": ("o", plotting.ROUTE_COLORS[0], "Grid geocoder"),
              "cache": ("D", plotting.ROUTE_COLORS[1], "Cache (verified by driver GPS)")}
    for src, (mk, col, lab) in styles.items():
        idx = [k + 1 for k, s in enumerate(res.stops) if s.geocode_source == src]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], marker=mk,
                   s=[12 + 3 * inst.demands[i] for i in idx], color=col, label=lab,
                   edgecolor="white", linewidth=0.8, zorder=4)
    for k, s in enumerate(res.stops):
        if s.part:
            ax.annotate("split office order", (xs[k + 1], ys[k + 1]), xytext=(8, -2),
                        textcoords="offset points", fontsize=7, color=plotting.INK_2)
            break
    plotting._depot(ax, xs[0], ys[0])
    plotting._axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False,
              labelspacing=1.2, markerscale=1.0)
    plotting.save(fig, path)
    plt.close(fig)


def _plot_loads(R: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    plotting.use_book_style()
    F = R["fleet"]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
    for ax, key, title in [(axes[0], "one_dim", "Totes only"),
                           (axes[1], "two_dim", "Totes and weight")]:
        loads = R[key]["loads"]
        x = np.arange(len(loads))
        t = [100 * l["totes"] / F["totes"] for l in loads]
        w = [100 * l["weight"] / F["payload"] for l in loads]
        ax.bar(x - 0.2, t, width=0.38, color=plotting.ROUTE_COLORS[0], label="Totes",
               edgecolor="white", linewidth=1.0)
        ax.bar(x + 0.2, w, width=0.38, color=plotting.ROUTE_COLORS[1], label="Weight",
               edgecolor="white", linewidth=1.0, hatch="////")
        ax.axhline(100, color=plotting.INK, linewidth=0.9, linestyle="--")
        ax.set_xticks(x, [f"Van {k + 1}" for k in x], fontsize=7)
        ax.set_title(title, loc="left", color=plotting.INK)
        ax.set_ylim(0, 125)
        ax.grid(True, axis="y", color=plotting.GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("% of van capacity")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.05, -0.16), ncol=2, frameon=False)
    plotting.save(fig, path)
    plt.close(fig)


if __name__ == "__main__":
    main()
