"""Chapter 4: road networks and travel times.

    python tools/fetch_chicago_network.py                    # once, about 32 MB
    python examples/ch04_experiments.py --book ../book       # about 10 minutes

Loads Chicago's street network from OpenStreetMap (load_osm), calibrates the circuity factor of
Chapter 2 against it, measures asymmetry, triangle-inequality violations
and the cost of computing matrices, and plans Tuesday's deliveries (two-hour
slots, Chapter 6) with six travel-time models, each judged by driving its
plan through time-dependent road travel times.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from vrpbook import distance_matrix, load_csv, plotting, travel_time_matrix
from vrpbook.instances import Site
from vrpbook.roads import (M_PER_MI, PEAK_FACTOR, MatrixCache, asymmetry, congestion, extend,
                           gc_matrix,
                           load_osm, metric_closure, node_distances, road_matrices,
                           slot_matrix, snap, td_drive)
from vrpbook.distances import triangle_violations
from vrpbook.timewindows import (HOUR, TWData, best_departure, load_preferences,
                                 service_seconds, slot_windows, solve_vrptw, validate_schedule)

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "results" / "ch04"
SEED = 404
N_CALIB = 400                    # random addresses for calibration
BANDS = [(0.5, 2), (2, 5), (5, 10), (10, 30)]       # straight-line miles
SIZES = [50, 200, 1000]
NX_SIZES = [50, 200]
LIMIT = 30
DEPOT_HOURS = (int(6.5 * HOUR), int(15.5 * HOUR))
SHIFT = (7 * HOUR, 15 * HOUR)
MAX_VANS = 8
FIXED_PER_VAN, DRIVER_PER_HOUR, VEHICLE_PER_MILE = 50.0, 30.81, 0.76   # Chapters 1 and 6
CITY = (41.66, 42.02, -87.84, -87.53)       # lat, lat, lon, lon: the city, roughly


def random_addresses(G, n: int, rng) -> list[Site]:
    """Random points near random street intersections inside the city (not
    on expressways): a cheap stand-in for a sample of delivery addresses."""
    lat0, lat1, lon0, lon1 = CITY
    nodes = [u for u, d in G.nodes(data=True) if lat0 < d["y"] < lat1 and lon0 < d["x"] < lon1
             and any(k != "expressway" for *_, k in G.out_edges(u, data="kind"))]
    pick = rng.choice(len(nodes), n, replace=False)
    out = []
    for k, p in enumerate(pick):
        d = G.nodes[nodes[p]]
        r, a = 250 * np.sqrt(rng.random()), 2 * np.pi * rng.random()    # within 250 m
        out.append(Site(k, f"address {k}", "", d["y"] + r * np.sin(a) / 111_000,
                        d["x"] + r * np.cos(a) / (111_000 * np.cos(np.radians(d["y"])))))
    return out


def pairs(n):
    i, j = np.where(~np.eye(n, dtype=bool))
    return i, j


# ------------------------------------------------------------------ run
def run() -> dict:
    rng = np.random.default_rng(SEED)
    R: dict = {}
    t = time.perf_counter()
    G = load_osm()
    R["load_s"] = time.perf_counter() - t
    R["nodes"], R["edges"], R["dropped"] = G.number_of_nodes(), G.number_of_edges(), G.graph["dropped_nodes"]
    kinds = {}
    for kind in ("expressway", "arterial", "local"):
        E = [d for *_, d in G.edges(data=True) if d["kind"] == kind]
        miles = sum(d["length"] for d in E) / M_PER_MI
        hours = sum(d["travel_time"] for d in E) / 3600
        kinds[kind] = {"links": len(E), "miles": miles,
                       "free_mph": miles / hours,          # length-weighted, delays included
                       "peak_factor": PEAK_FACTOR[kind],
                       "classes": sorted({d["highway"] for d in E})}
    R["kinds"] = kinds
    R["signals"] = sum(1 for _, c in G.nodes(data="control") if c == "traffic_signals")
    R["stops"] = sum(1 for _, c in G.nodes(data="control") if c == "stop")

    # ---- Tuesday's sites ---------------------------------------------------
    inst = load_csv(DATA / "chicago_tuesday.csv", MAX_VANS, 100)
    S = inst.sites
    n = len(S)
    # how much of the city is nearer an expressway-only node than an arterial one?
    lat0, lat1, lon0, lon1 = CITY
    grid = [Site(0, "", "", la, lo) for la in np.arange(lat0, lat1, 0.0036)
            for lo in np.arange(lon0, lon1, 0.0048)]                 # about every 0.25 mi
    near, d_any = snap(G, grid, kinds=("arterial", "local", "expressway"))
    on_land = d_any < 800
    expressway_only = {u for u in G if all(k == "expressway" for *_, k in G.out_edges(u, data="kind"))}
    R["snap_grid"] = int(on_land.sum())
    R["snap_expressway"] = int(sum(u in expressway_only for u in near[on_land]))
    L = road_matrices(G, S, "length")
    F = road_matrices(G, S, "travel_time")
    P = road_matrices(G, S, "peak_time")
    R["access_mi"] = (F.access_m / M_PER_MI).tolist()
    # worked example: the depot to customer 1, in Rogers Park
    gc01 = gc_matrix(S[:2])[0, 1]
    R["we"] = {"name": S[1].name, "gc_mi": gc01 / M_PER_MI, "road_mi": L.meters[0, 1] / M_PER_MI,
               "fast_mi": F.meters[0, 1] / M_PER_MI, "free_min": F.seconds[0, 1] / 60,
               "peak_min": P.seconds[0, 1] / 60, "back_free_min": F.seconds[1, 0] / 60,
               "back_peak_min": P.seconds[1, 0] / 60, "gc_m": gc01}
    gc = gc_matrix(S)
    i, j = pairs(n)
    ok = gc[i, j] > 0.5 * M_PER_MI
    i, j = i[ok], j[ok]
    R["tuesday_pairs"] = int(ok.sum())
    R["tue_circuity_model"] = float(np.median(L.meters[i, j] / gc[i, j]))
    # the network's own circuity between Tuesday's nodes: no access legs, no shortcut
    tn_m, tn_gc = node_distances(G, L.nodes)
    ti, tj = pairs(n)
    tk = (L.nodes[ti] != L.nodes[tj]) & (tn_gc[ti, tj] > 0.5 * M_PER_MI)
    ti, tj = ti[tk], tj[tk]
    R["tue_circuity"] = float(np.median(tn_m[ti, tj] / tn_gc[ti, tj]))
    R["fastest_longer_pct"] = float(100 * np.median(F.meters[i, j] / L.meters[i, j] - 1))
    R["fastest_longer_p90"] = float(100 * np.percentile(F.meters[i, j] / L.meters[i, j] - 1, 90))
    R["asym"] = {name: {"median": float(np.median(asymmetry(M.seconds))),
                        "p90": float(np.percentile(asymmetry(M.seconds), 90)),
                        "over10": float(np.mean(asymmetry(M.seconds) > 0.10))}
                 for name, M in (("free", F), ("peak", P))}
    R["asym"]["length"] = {"median": float(np.median(asymmetry(L.meters))),
                           "p90": float(np.percentile(asymmetry(L.meters), 90)),
                           "over10": float(np.mean(asymmetry(L.meters) > 0.10))}
    R["peak_ratio"] = float(np.median(P.seconds[i, j] / F.seconds[i, j]))
    R["peak_extra_max_min"] = float((P.seconds - F.seconds).max() / 60)
    tri = {}
    for name, M in (("shortest m", L.meters), ("fastest m", F.meters), ("fastest s", F.seconds),
                    ("peak s", P.seconds)):
        worst = 0
        for k in range(n):
            worst = max(worst, int((M - (M[:, k][:, None] + M[k, :][None, :])).max()))
        tri[name] = {"count": triangle_violations(M), "worst": worst,
                     "triples": n * (n - 1) * (n - 2)}
    tri["closure_fix"] = float((F.seconds - metric_closure(F.seconds)).max())
    R["triangle"] = tri

    # ---- calibration sample -------------------------------------------------
    A = random_addresses(G, N_CALIB, rng)
    LA = road_matrices(G, A, "length")
    LA_net = road_matrices(G, A, "length", shortcut=False)     # door to door, no shortcut
    FA = road_matrices(G, A, "travel_time")
    PA = road_matrices(G, A, "peak_time")
    gA = gc_matrix(A)
    a, b = pairs(len(A))
    keep = gA[a, b] > 0.5 * M_PER_MI
    a, b = a[keep], b[keep]
    w_day = float(np.mean(congestion(np.arange(SHIFT[0], SHIFT[1], 60))))
    R["w_day"] = w_day
    gmi = gA[a, b] / M_PER_MI
    circ_model = LA.meters[a, b] / gA[a, b]          # capped at 1.3 by the shortcut
    circ_noshort = LA_net.meters[a, b] / gA[a, b]
    day_s = FA.seconds[a, b] + w_day * (PA.seconds[a, b] - FA.seconds[a, b])
    R["calib_pairs"] = int(len(a))
    # the calibration itself: the network between the addresses' nodes, where no
    # circuity factor is assumed anywhere
    net_m, net_gc = node_distances(G, LA.nodes)
    na, nb = pairs(len(A))
    nk = (LA.nodes[na] != LA.nodes[nb]) & (net_gc[na, nb] > 0.5 * M_PER_MI)
    na, nb = na[nk], nb[nk]
    nmi = net_gc[na, nb] / M_PER_MI
    circ = net_m[na, nb] / net_gc[na, nb]
    R["net_pairs"] = int(len(na))
    R["alpha"] = float(np.median(circ))
    R["alpha_mean"] = float(np.mean(circ))
    R["alpha_iqr"] = [float(np.percentile(circ, 25)), float(np.percentile(circ, 75))]
    R["model_capped"] = float(np.mean(circ_model > 1.2999))
    R["model_short_capped"] = float(np.mean(circ_model[gmi < 2] > 1.2999))
    bands = []
    for lo, hi in BANDS:
        m = (gmi >= lo) & (gmi < hi)
        k = (nmi >= lo) & (nmi < hi)
        bands.append({"lo": lo, "hi": hi, "pairs": int(k.sum()),
                      "circ": float(np.median(circ[k])),
                      "q1": float(np.percentile(circ[k], 25)), "q3": float(np.percentile(circ[k], 75)),
                      "noshort": float(np.median(circ_noshort[m])),
                      "model": float(np.median(circ_model[m])),
                      "free_mph": float(np.median(FA.meters[a, b][m] / M_PER_MI / (FA.seconds[a, b][m] / 3600))),
                      "peak_mph": float(np.median(PA.meters[a, b][m] / M_PER_MI / (PA.seconds[a, b][m] / 3600))),
                      "gc_mph_day": float(np.median(gmi[m] / (day_s[m] / 3600)))})
    R["bands"] = bands
    # straight-line time models fitted to the day-average road times
    alpha = R["alpha"]
    v_prop = float(np.sum((alpha * gA[a, b]) ** 2) / np.sum(alpha * gA[a, b] * day_s))   # m/s
    X = np.column_stack([np.ones(len(a)), gA[a, b]])
    t0, slope = np.linalg.lstsq(X, day_s, rcond=None)[0]
    R["fit"] = {"v_prop_mph": v_prop * 3600 / M_PER_MI, "t0_min": t0 / 60,
                "affine_mph": 3600 / slope / M_PER_MI}
    R["scatter"] = {"gc_mi": nmi[::25].tolist(), "circ": circ[::25].tolist(),
                    "model_gc_mi": gmi[::25].tolist(), "model_circ": circ_model[::25].tolist()}
    # prediction error on Tuesday's pairs, against day-average road times
    truth = F.seconds + w_day * (P.seconds - F.seconds)
    v15 = 15 * M_PER_MI / 3600
    models = {"rule": 1.3 * gc / v15, "alpha": alpha * gc / v_prop, "affine": t0 + slope * gc}
    i, j = pairs(n)
    ok = gc[i, j] > 0.5 * M_PER_MI
    i, j = i[ok], j[ok]
    we = R["we"]
    we["day_min"] = (F.seconds[0, 1] + w_day * (P.seconds[0, 1] - F.seconds[0, 1])) / 60
    we.update({f"{k}_min": float(M[0, 1] / 60) for k, M in models.items()})
    R["model_err"] = {}
    for name, M in models.items():
        e = (M[i, j] - truth[i, j]) / truth[i, j]
        R["model_err"][name] = {"med_abs": float(np.median(np.abs(e))),
                                "p90_abs": float(np.percentile(np.abs(e), 90)),
                                "bias": float(np.median(e))}
    # distance errors on Tuesday's nodes, against the network alone
    R["dist_err"] = {name: float(np.median(np.abs(c * tn_gc[ti, tj] - tn_m[ti, tj]) / tn_m[ti, tj]))
                     for name, c in (("rule", 1.3), ("alpha", alpha))}

    # ---- the cost of computing matrices -------------------------------------
    R["timing"] = timing(G, rng)

    # ---- plans ---------------------------------------------------------------
    R["plans"] = plans(inst, L, F, P, w_day, alpha, v_prop, t0, slope)
    R["speed_curve"] = speed_curve(FA, PA, gA, a, b)
    return R


def timing(G, rng) -> list[dict]:
    rows = []
    W = G.graph  # noqa: F841 (to_csr caches here)
    for n in SIZES:
        S = random_addresses(G, n, rng)
        row = {"n": n}
        t = time.perf_counter()
        gc_matrix(S)
        row["gc_s"] = time.perf_counter() - t
        t = time.perf_counter()
        M = road_matrices(G, S)
        row["scipy_s"] = time.perf_counter() - t
        if n in NX_SIZES:
            nodes, _ = snap(G, S)
            t = time.perf_counter()
            for u in nodes:
                nx.single_source_dijkstra_path_length(G, u, weight="travel_time")
            row["nx_s"] = time.perf_counter() - t
        if n == SIZES[-1]:
            base = road_matrices(G, S[:-1])
            t = time.perf_counter()
            extend(G, base, S[:-1], S[-1])
            row["extend_s"] = time.perf_counter() - t
            cache = MatrixCache(OUT / "cache", "osm-chicago-v1")
            for f in cache.folder.glob("*.npz"):
                f.unlink()
            t = time.perf_counter()
            cache.get(G, S)
            row["miss_s"] = time.perf_counter() - t
            t = time.perf_counter()
            cache.get(G, S)
            row["hit_s"] = time.perf_counter() - t
            row["mb"] = sum(f.stat().st_size for f in cache.folder.glob("*.npz")) / 1e6
        rows.append(row)
        print("timing", {k: round(v, 3) for k, v in row.items()}, flush=True)
        del M
    return rows


def speed_curve(FA, PA, gA, a, b) -> dict:
    """Median door-to-door road speed against straight-line distance."""
    gmi = gA[a, b] / M_PER_MI
    edges = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    out = {"mid": [], "free": [], "peak": []}
    for lo, hi in zip(edges, edges[1:]):
        m = (gmi >= lo) & (gmi < hi)
        if m.sum() < 30:
            continue
        out["mid"].append(float((lo + hi) / 2))
        for k, M in (("free", FA), ("peak", PA)):
            out[k].append(float(np.median(M.meters[a, b][m] / M_PER_MI / (M.seconds[a, b][m] / 3600))))
    return out


def plans(inst, L, F, P, w_day, alpha, v_prop, t0, slope) -> dict:
    S = inst.sites
    n = len(S)
    with (DATA / "chicago_tuesday.csv").open() as fh:
        weights = [int(r["weight_lb"]) for r in csv.DictReader(fh)]
    pref = load_preferences(DATA / "chicago_tuesday_slots.csv")
    service = [0] + [service_seconds(q) for q in inst.demands[1:]]
    windows = [DEPOT_HOURS] + slot_windows([pref[i] for i in range(1, n)], 2)
    gc = gc_matrix(S)
    same = gc < 1
    rule_m = distance_matrix(S)
    straight_t = np.where(same, 0, np.rint(t0 + slope * gc)).astype(np.int64)
    models = {
        "rule": (rule_m, travel_time_matrix(rule_m, 15 * 1.609344)),
        "straight": (np.rint(alpha * gc).astype(np.int64), straight_t),
        "free": (F.meters, F.seconds),
        "peak": (P.meters, P.seconds),
        "day": (F.meters, np.rint(F.seconds + w_day * (P.seconds - F.seconds)).astype(np.int64)),
        "slot": (F.meters, slot_matrix(F.seconds, P.seconds, windows, service,
                                       depot_leave_s=SHIFT[0] - 1800)),
    }
    out = {}
    for name, (dist, ttime) in models.items():
        data = TWData(dist, ttime, service, windows, inst.demands, 100, MAX_VANS,
                      {"Weight": (weights, 2400)})
        sol = solve_vrptw(data, objective="cost", time_limit_s=LIMIT,
                          driver_per_hour=DRIVER_PER_HOUR, vehicle_per_mile=VEHICLE_PER_MILE,
                          fixed_per_van=FIXED_PER_VAN)
        T = validate_schedule(sol.routes, data)
        used = [r for r in sol.routes if len(r) > 2]
        planned_h = T["duration"] / HOUR
        planned_cost = (DRIVER_PER_HOUR * planned_h + VEHICLE_PER_MILE * T["meters"] / M_PER_MI
                        + FIXED_PER_VAN * len(used))
        # reality: road distances, time-dependent road travel times, same departures
        real_h, late, late_s, max_late, miles, finish = 0.0, 0, 0.0, 0.0, 0.0, 0.0
        for r in used:
            dep = best_departure(r, data)
            back, stops = td_drive(r, dep, F.seconds, P.seconds, service, windows)
            real_h += (back - dep) / HOUR
            finish = max(finish, back)
            miles += sum(F.meters[x, y] for x, y in zip(r, r[1:])) / M_PER_MI
            for _, _, ls in stops:
                if ls > 0:
                    late += 1
                    late_s += ls
                    max_late = max(max_late, ls)
        out[name] = {"vans": len(used), "planned_h": planned_h, "planned_cost": planned_cost,
                     "real_h": real_h, "miles": miles,
                     "real_cost": DRIVER_PER_HOUR * real_h + VEHICLE_PER_MILE * miles
                     + FIXED_PER_VAN * len(used),
                     "late": late, "late_min": late_s / 60, "max_late_min": max_late / 60,
                     "last_back": finish, "routes": sol.routes}
        print(name, {k: (round(v, 1) if isinstance(v, float) else v)
                     for k, v in out[name].items() if k != "routes"}, flush=True)
    return out


# ---------------------------------------------------------------- book assets
NAMES = {"rule": "Straight line, Chapter 2 rule (1.3, 15 mph)",
         "straight": "Straight line, affine (calibrated)",
         "free": "Road, free flow", "peak": "Road, peak all day",
         "day": "Road, average of the day", "slot": "Road, timed by slot"}


def write_book_assets(book: Path, R: dict) -> None:
    gen = book / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    fig = book / "figures" / "ch04"
    fig.mkdir(parents=True, exist_ok=True)
    K = R["kinds"]
    classes = {"expressway": "motorway, trunk", "arterial": "primary, secondary",
               "local": "tertiary, residential, other"}
    rows = [f"{k.capitalize()} & {classes[k]} & {v['links']:,} & {v['miles']:,.0f} & "
            f"{v['free_mph']:.0f} & {v['peak_factor']:.2f} \\\\" for k, v in K.items()]
    plotting.write_tex(gen / "ch04-network-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for bnd in R["bands"]:
        lab = f"{bnd['lo']:g}--{bnd['hi']:g}" if bnd["hi"] < 30 else f"over {bnd['lo']:g}"
        rows.append(f"{lab} & {bnd['pairs']:,} & {bnd['circ']:.2f} & {bnd['q1']:.2f}--{bnd['q3']:.2f} & "
                    f"{bnd['noshort']:.2f} & {bnd['model']:.2f} & "
                    f"{bnd['free_mph']:.0f} & {bnd['peak_mph']:.0f} \\\\")
    plotting.write_tex(gen / "ch04-circuity-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    E = R["model_err"]
    lab = {"rule": "Chapter 2 rule: $1.3\\,d$ at 15 mph",
           "alpha": f"Calibrated: ${R['alpha']:.2f}\\,d$ at {R['fit']['v_prop_mph']:.1f} mph",
           "affine": f"Affine: {R['fit']['t0_min']:.1f} min $+\\ d$ at {R['fit']['affine_mph']:.1f} mph"}
    rows = [f"{lab[k]} & {100 * v['bias']:+.0f}\\%".replace("-", "$-$") + f" & {100 * v['med_abs']:.0f}\\% & "
            f"{100 * v['p90_abs']:.0f}\\% \\\\" for k, v in E.items()]
    plotting.write_tex(gen / "ch04-models-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for t in R["timing"]:
        nx_s = f"{t['nx_s']:.1f}" if "nx_s" in t else "--"
        rows.append(f"{t['n']:,} & {t['n'] ** 2:,} & {t['gc_s']:.3f} & {nx_s} & {t['scipy_s']:.1f} \\\\")
    plotting.write_tex(gen / "ch04-timing-table.tex", "\n".join(rows) + "\n", encoding="utf-8")
    Pl = R["plans"]
    rows = []
    for k, v in Pl.items():
        if k == "free":
            rows.append("\\addlinespace")
        rows.append(f"{NAMES[k]} & {v['vans']} & {v['planned_h']:.1f} & {v['real_h']:.1f} & "
                    f"\\${v['real_cost']:,.0f} & {v['late']} & {v['late_min']:.0f} \\\\")
    plotting.write_tex(gen / "ch04-plans-table.tex", "\n".join(rows) + "\n", encoding="utf-8")

    T, tri = R["timing"], R["triangle"]
    big = T[-1]
    ar, ex, lo = K["arterial"], K["expressway"], K["local"]
    best = min(Pl, key=lambda k: (Pl[k]["late"] > 0, Pl[k]["real_cost"]))
    m = {"nodes": f"{R['nodes']:,}", "edges": f"{R['edges']:,}", "dropped": R["dropped"],
         "load-s": f"{R['load_s']:.1f}",
         "art-links": f"{ar['links']:,}", "exp-links": f"{ex['links']:,}",
         "art-mi": f"{ar['miles']:,.0f}", "exp-mi": f"{ex['miles']:,.0f}",
         "art-mph": f"{ar['free_mph']:.0f}", "exp-mph": f"{ex['free_mph']:.0f}",
         "loc-mi": f"{lo['miles']:,.0f}", "loc-mph": f"{lo['free_mph']:.0f}",
         "art-peak": f"{ar['peak_factor']:.1f}", "exp-peak": f"{ex['peak_factor']:.1f}",
         "loc-peak": f"{lo['peak_factor']:.2f}",
         "signals": f"{R['signals']:,}", "stops": f"{R['stops']:,}",
         "tot-mi": f"{ar['miles'] + ex['miles'] + lo['miles']:,.0f}",
         "access-med": f"{np.median(R['access_mi']):.2f}", "access-max": f"{max(R['access_mi']):.2f}",
         "snap-exp": R["snap_expressway"], "snap-grid": f"{R['snap_grid']:,}",
         "snap-exp-pct": f"{100 * R['snap_expressway'] / R['snap_grid']:.0f}",
         "tue-circ": f"{R['tue_circuity']:.2f}", "tue-pairs": f"{R['tuesday_pairs']:,}",
         "tue-circ-model": f"{R['tue_circuity_model']:.2f}",
         "net-pairs": f"{R['net_pairs']:,}",
         "model-capped": f"{100 * R['model_capped']:.0f}",
         "model-short-capped": f"{100 * R['model_short_capped']:.0f}",
         "noshort-short": f"{R['bands'][0]['noshort']:.2f}",
         "fast-longer": f"{R['fastest_longer_pct']:.0f}", "fast-longer-p90": f"{R['fastest_longer_p90']:.0f}",
         "asym-free-med": f"{100 * R['asym']['free']['median']:.1f}",
         "asym-free-p90": f"{100 * R['asym']['free']['p90']:.0f}",
         "asym-peak-med": f"{100 * R['asym']['peak']['median']:.1f}",
         "asym-peak-p90": f"{100 * R['asym']['peak']['p90']:.0f}",
         "asym-peak-over10": f"{100 * R['asym']['peak']['over10']:.0f}",
         "asym-free-over10": f"{100 * R['asym']['free']['over10']:.0f}",
         "asym-len-med": f"{100 * R['asym']['length']['median']:.1f}",
         "peak-ratio": f"{R['peak_ratio']:.2f}", "peak-extra-max": f"{R['peak_extra_max_min']:.0f}",
         "tri-short": tri["shortest m"]["count"], "tri-short-worst": tri["shortest m"]["worst"],
         "tri-fast": f"{tri['fastest m']['count']:,}",
         "tri-fast-worst": f"{tri['fastest m']['worst'] / M_PER_MI:.1f}",
         "tri-time": tri["fastest s"]["count"], "tri-time-worst": tri["fastest s"]["worst"],
         "tri-triples": f"{tri['fastest m']['triples']:,}",
         "tri-fast-pct": f"{100 * tri['fastest m']['count'] / tri['fastest m']['triples']:.1f}",
         "closure-fix": f"{tri['closure_fix']:.0f}",
         "calib-n": N_CALIB, "calib-pairs": f"{R['calib_pairs']:,}",
         "alpha": f"{R['alpha']:.2f}", "alpha-mean": f"{R['alpha_mean']:.2f}",
         "alpha-q1": f"{R['alpha_iqr'][0]:.2f}", "alpha-q3": f"{R['alpha_iqr'][1]:.2f}",
         "circ-short": f"{R['bands'][0]['circ']:.2f}", "circ-long": f"{R['bands'][-1]['circ']:.2f}",
         "w-day": f"{R['w_day']:.2f}",
         "v-prop": f"{R['fit']['v_prop_mph']:.1f}", "t0": f"{R['fit']['t0_min']:.1f}",
         "v-affine": f"{R['fit']['affine_mph']:.1f}",
         "err-rule": f"{100 * E['rule']['med_abs']:.0f}", "err-rule-bias": f"{100 * E['rule']['bias']:+.0f}",
         "err-alpha": f"{100 * E['alpha']['med_abs']:.0f}", "err-affine": f"{100 * E['affine']['med_abs']:.0f}",
         "err-alpha-p90": f"{100 * E['alpha']['p90_abs']:.0f}",
         "err-affine-p90": f"{100 * E['affine']['p90_abs']:.0f}",
         "dist-err-rule": f"{100 * R['dist_err']['rule']:.0f}",
         "dist-err-alpha": f"{100 * R['dist_err']['alpha']:.0f}",
         "big-n": f"{big['n']:,}", "big-pairs": f"{big['n'] ** 2:,}",
         "big-scipy": f"{big['scipy_s']:.1f}", "big-gc": f"{big['gc_s']:.2f}",
         "extend-s": f"{big['extend_s']:.2f}", "miss-s": f"{big['miss_s']:.1f}",
         "hit-s": f"{big['hit_s']:.3f}", "cache-mb": f"{big['mb']:.1f}",
         "nx-200": f"{T[1]['nx_s']:.1f}", "scipy-200": f"{T[1]['scipy_s']:.1f}",
         "nx-ratio": f"{T[1]['nx_s'] / T[1]['scipy_s']:.0f}",
         "limit": LIMIT, "best-plan": NAMES[best]}
    we = R["we"]
    m.update({"we-name": we["name"].title(), "we-gc": f"{we['gc_mi']:.1f}",
              "we-road": f"{we['road_mi']:.1f}", "we-fast": f"{we['fast_mi']:.1f}",
              "we-circ": f"{we['road_mi'] / we['gc_mi']:.2f}",
              "we-free": f"{we['free_min']:.0f}", "we-peak": f"{we['peak_min']:.0f}",
              "we-back-free": f"{we['back_free_min']:.0f}", "we-back-peak": f"{we['back_peak_min']:.0f}",
              "we-day": f"{we['day_min']:.0f}", "we-rule": f"{we['rule_min']:.0f}",
              "we-alpha": f"{we['alpha_min']:.0f}", "we-affine": f"{we['affine_min']:.0f}",
              "we-free-mph": f"{we['fast_mi'] / (we['free_min'] / 60):.0f}",
              "we-peak-mph": f"{we['fast_mi'] / (we['peak_min'] / 60):.0f}"})
    for k, v in Pl.items():
        m[f"{k}-late"] = v["late"]
        m[f"{k}-late-min"] = f"{v['late_min']:.0f}"
        m[f"{k}-max-late"] = f"{v['max_late_min']:.0f}"
        m[f"{k}-cost"] = f"\\${v['real_cost']:,.0f}"
        m[f"{k}-vans"] = v["vans"]
        m[f"{k}-planned-h"] = f"{v['planned_h']:.1f}"
        m[f"{k}-real-h"] = f"{v['real_h']:.1f}"
    m["rule-hours-gap"] = f"{Pl['rule']['planned_h'] - Pl['rule']['real_h']:.1f}"
    m["slot-vs-peak"] = f"\\${Pl['peak']['real_cost'] - Pl['slot']['real_cost']:,.0f}"
    m["slot-vs-peak-pct"] = f"{100 * (1 - Pl['slot']['real_cost'] / Pl['peak']['real_cost']):.0f}"
    lines = ["% Generated by code/examples/ch04_experiments.py -- do not edit."]
    lines += [f"\\SetResult{{ch04/{k}}}{{{v}}}" for k, v in sorted(m.items())]
    plotting.write_tex(gen / "ch04-results.tex", "\n".join(lines) + "\n", encoding="utf-8")
    figures(R, fig)
    print("book assets written")


# ---------------------------------------------------------------- figures
def figures(R: dict, fig: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    plotting.use_book_style()
    G = load_osm()
    inst = load_csv(DATA / "chicago_tuesday.csv", MAX_VANS, 100)
    d0 = inst.sites[0]
    kx, ky = 69.172 * np.cos(np.radians(d0.lat)), 68.707

    def xy(lat, lon):
        return (np.asarray(lon) - d0.lon) * kx, (np.asarray(lat) - d0.lat) * ky

    sx, sy = xy([s.lat for s in inst.sites], [s.lon for s in inst.sites])
    box = (sx.min() - 2, sx.max() + 2, sy.min() - 2, sy.max() + 2)

    def segments(kind):
        seg = []
        for u, v, k in G.edges(data="kind"):
            if k != kind:
                continue
            (x1, y1), (x2, y2) = (xy(G.nodes[w]["y"], G.nodes[w]["x"]) for w in (u, v))
            if box[0] < x1 < box[1] and box[2] < y1 < box[3]:
                seg.append([(x1, y1), (x2, y2)])
        return seg

    loc, art, exp = segments("local"), segments("arterial"), segments("expressway")

    # 1. the network
    f, ax = plt.subplots(figsize=(5.2, 5.6))
    ax.add_collection(LineCollection(loc, colors="#e4e2dc", linewidths=0.2, label="local street"))
    ax.add_collection(LineCollection(art, colors="#9d9b94", linewidths=0.45, label="arterial"))
    ax.add_collection(LineCollection(exp, colors=plotting.ROUTE_COLORS[1], linewidths=1.1,
                                     label="expressway"))
    ax.scatter(sx[1:], sy[1:], s=14, color=plotting.ROUTE_COLORS[0], zorder=4,
               edgecolor="white", linewidth=0.6, label="Tuesday's customers")
    plotting._depot(ax, sx[0], sy[0])
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    plotting._axes(ax)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
    plotting.save(f, fig / "network.pdf")
    plt.close(f)

    # 2. circuity
    f, ax = plt.subplots(figsize=(5.2, 2.9))
    sc = R["scatter"]
    ax.scatter(sc["gc_mi"], sc["circ"], s=3, color=plotting.ROUTE_COLORS[0], alpha=0.35,
               linewidth=0, label="pairs of network nodes")
    mids = [(b["lo"] + min(b["hi"], 20)) / 2 for b in R["bands"]]
    ax.plot(mids, [b["circ"] for b in R["bands"]], "-o", color=plotting.INK, markersize=3.5,
            linewidth=1.2, label="median in each distance band")
    ax.plot(mids, [b["model"] for b in R["bands"]], ":^", color=plotting.INK_2, markersize=3.5,
            linewidth=1.0, label="door-to-door model (capped at 1.3)")
    ax.axhline(1.3, color=plotting.ROUTE_COLORS[1], linestyle="--", linewidth=1,
               label="Chapter 2's assumption, 1.3")
    ax.set_xlabel("straight-line distance (miles)")
    ax.set_ylabel("road / straight line")
    ax.set_ylim(0.95, 1.85)
    ax.grid(True, color=plotting.GRID, linewidth=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    plotting.save(f, fig / "circuity.pdf")
    plt.close(f)

    # 3. profile and speed
    f, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    h = np.linspace(5, 20, 301)
    a1.plot(h, congestion(h * 3600), color=plotting.INK, linewidth=1.2)
    a1.axvspan(SHIFT[0] / 3600, SHIFT[1] / 3600, color=plotting.GRID, alpha=0.6, linewidth=0)
    a1.text(11, 0.08, "delivery slots", ha="center", fontsize=7, color=plotting.INK_2)
    a1.set_xlabel("hour of the day")
    a1.set_ylabel("share of peak delay")
    a1.set_xticks([6, 9, 12, 15, 18])
    a1.set_ylim(0, 1.08)
    curve = R["speed_curve"]
    a2.plot(curve["mid"], curve["free"], "-o", markersize=3, color=plotting.ROUTE_COLORS[0],
            linewidth=1.2, label="free flow")
    a2.plot(curve["mid"], curve["peak"], "--s", markersize=3, color=plotting.ROUTE_COLORS[1],
            linewidth=1.2, label="peak")
    a2.set_xlabel("straight-line miles")
    a2.set_ylabel("road mph, door to door")
    a2.legend(loc="lower right", frameon=False)
    for ax in (a1, a2):
        ax.grid(True, color=plotting.GRID, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    f.tight_layout(w_pad=1.5)
    plotting.save(f, fig / "speed.pdf")
    plt.close(f)

    # 4. one route on the roads
    routes = [r for r in R["plans"]["slot"]["routes"] if len(r) > 2]
    r = max(routes, key=lambda r: gc_matrix([inst.sites[k] for k in r]).max())
    nodes, _ = snap(G, inst.sites)
    f, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.add_collection(LineCollection(art, colors="#d8d6cf", linewidths=0.3))
    ax.add_collection(LineCollection(exp, colors="#e9b9a3", linewidths=0.9))
    for k, (u, v) in enumerate(zip(r, r[1:])):
        path = nx.shortest_path(G, nodes[u], nodes[v], weight="travel_time")
        px, py = xy([G.nodes[w]["y"] for w in path], [G.nodes[w]["x"] for w in path])
        px = np.concatenate([[sx[u]], px, [sx[v]]])
        py = np.concatenate([[sy[u]], py, [sy[v]]])
        ax.plot(px, py, color=plotting.ROUTE_COLORS[0], linewidth=1.5,
                label="road route" if k == 0 else None)
        ax.plot([sx[u], sx[v]], [sy[u], sy[v]], color=plotting.INK_2, linewidth=0.8,
                linestyle=":", label="straight line" if k == 0 else None)
    for pos, k in enumerate(r[1:-1], start=1):
        ax.scatter([sx[k]], [sy[k]], s=22, color=plotting.ROUTE_COLORS[0], zorder=5,
                   edgecolor="white", linewidth=0.7)
        ax.annotate(str(pos), (sx[k], sy[k]), xytext=(4, 3), textcoords="offset points",
                    fontsize=7, color=plotting.INK)
    plotting._depot(ax, sx[0], sy[0])
    xs = [sx[k] for k in r]
    ys = [sy[k] for k in r]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    plotting._axes(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
    plotting.save(f, fig / "route.pdf")
    plt.close(f)


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
        R = json.loads(path.read_text())
    if args.book:
        write_book_assets(args.book, R)


if __name__ == "__main__":
    main()
