"""Construction heuristics for the CVRP, from scratch (Chapter 10).

Every function takes a distance matrix (node 0 is the depot), the demands
and the vehicle capacity, and returns routes [0, i, j, ..., 0].  The
number of routes is not fixed: a new route opens when nothing fits.
"""
from __future__ import annotations

import math

import numpy as np


class InfeasibleInstance(ValueError):
    """Data that no plan can satisfy."""


def check_inputs(dist, demands, capacity) -> None:
    """Fail at once, with a reason, on inputs the constructions assume
    away: a square matrix with one row per node, no demand at the depot,
    a positive capacity, and every customer fitting in one vehicle.
    Without the last check, a customer too large for any vehicle would
    make some constructions open empty routes forever."""
    n = len(demands)
    if np.shape(dist) != (n, n):
        raise ValueError(f"distance matrix is {np.shape(dist)}, expected {(n, n)}")
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, not {capacity}")
    if demands[0] != 0:
        raise ValueError("node 0 is the depot and must have no demand")
    bad = [i for i in range(1, n) if not 0 <= demands[i] <= capacity]
    if bad:
        raise InfeasibleInstance(f"customers {bad[:10]} have demands outside "
                                 f"0..{capacity}, so no vehicle can serve them")


# <<nearest_neighbor>>
def nearest_neighbor(dist: np.ndarray, demands, capacity: int) -> list[list[int]]:
    """From the current stop, drive to the closest customer that still fits.
    When none fits, return to the depot and start a new route."""
    check_inputs(dist, demands, capacity)
    n = len(demands)
    q = np.asarray(demands)
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    routes = []
    while unvisited.any():
        route, here, load = [0], 0, 0
        while True:
            fits = unvisited & (load + q <= capacity)
            if not fits.any():
                break
            d = np.where(fits, dist[here], np.inf)
            here = int(np.argmin(d))
            route.append(here)
            load += q[here]
            unvisited[here] = False
        routes.append(route + [0])
    return routes
# <<end_nearest_neighbor>>


# <<savings>>
def savings_list(dist: np.ndarray, lam: float = 1.0) -> list[tuple[float, int, int]]:
    """All savings s_ij = d_0i + d_0j - lam * d_ij, largest first."""
    n = len(dist)
    i, j = np.triu_indices(n - 1, k=1)
    i, j = i + 1, j + 1
    s = dist[0, i] + dist[0, j] - lam * dist[i, j]
    order = np.argsort(-s, kind="stable")
    return [(float(s[k]), int(i[k]), int(j[k])) for k in order if s[k] > 0]


def clarke_wright(dist: np.ndarray, demands, capacity: int, lam: float = 1.0,
                  log: list | None = None) -> list[list[int]]:
    """Parallel savings: start with one route per customer and merge two
    routes end to end whenever the pair (i, j) with the largest saving
    sits at the ends of two different routes and the merged load fits."""
    check_inputs(dist, demands, capacity)
    n = len(demands)
    route = {i: [i] for i in range(1, n)}          # route id -> customers
    of = list(range(n))                             # customer -> route id
    load = {i: demands[i] for i in range(1, n)}
    for s, i, j in savings_list(dist, lam):
        ri, rj = of[i], of[j]
        if ri == rj:
            why = "same route"
        elif load[ri] + load[rj] > capacity:
            why = "capacity"
        elif i not in (route[ri][0], route[ri][-1]) or j not in (route[rj][0], route[rj][-1]):
            why = "not at an end"
        else:
            a, b = route[ri], route[rj]
            if a[-1] != i:
                a.reverse()                         # make i the last stop of a
            if b[0] != j:
                b.reverse()                         # and j the first stop of b
            if len(a) < len(b):                     # relabel the shorter route
                b[:0] = a
                keep, gone = rj, ri
            else:
                a.extend(b)
                keep, gone = ri, rj
            for c in route[gone]:
                of[c] = keep
            load[keep] += load.pop(gone)
            del route[gone]
            why = "merge"
        if log is not None:
            log.append((s, i, j, why))
    return [[0, *r, 0] for r in route.values()]
# <<end_savings>>


def clarke_wright_sequential(dist: np.ndarray, demands, capacity: int,
                             lam: float = 1.0) -> list[list[int]]:
    """Sequential savings: grow one route at a time, always at one of its two
    ends, with the unrouted customer of largest saving that fits."""
    check_inputs(dist, demands, capacity)
    n = len(demands)
    q = np.asarray(demands)
    free = np.ones(n, dtype=bool)
    free[0] = False
    routes = []
    while free.any():
        # seed: the free customer farthest from the depot
        seed = int(np.argmax(np.where(free, dist[0], -1)))
        r, load = [seed], q[seed]
        free[seed] = False
        while True:
            ok = free & (load + q <= capacity)
            if not ok.any():
                break
            s_first = dist[0, r[0]] + dist[0] - lam * dist[r[0]]
            s_last = dist[0, r[-1]] + dist[0] - lam * dist[r[-1]]
            s_first = np.where(ok, s_first, -np.inf)
            s_last = np.where(ok, s_last, -np.inf)
            jf, jl = int(np.argmax(s_first)), int(np.argmax(s_last))
            if max(s_first[jf], s_last[jl]) <= 0:
                break
            if s_first[jf] >= s_last[jl]:
                r.insert(0, jf)
                j = jf
            else:
                r.append(jl)
                j = jl
            load += q[j]
            free[j] = False
        routes.append([0, *r, 0])
    return routes


# <<sweep>>
def polar_angles(xy: np.ndarray) -> np.ndarray:
    """Angle of each customer around the depot (row 0 of xy)."""
    d = xy - xy[0]
    return np.arctan2(d[:, 1], d[:, 0])


def sweep(dist: np.ndarray, demands, capacity: int, xy: np.ndarray,
          starts: int = 12) -> list[list[int]]:
    """Cluster first, route second: sweep a ray around the depot and cut a
    new cluster whenever the next customer does not fit. Route each cluster
    by cheapest insertion.  Try several starting angles, keep the best."""
    check_inputs(dist, demands, capacity)
    n = len(demands)
    order = 1 + np.argsort(polar_angles(xy)[1:], kind="stable")
    best, best_len = None, math.inf
    for k in range(starts):
        rot = np.roll(order, -round(k * (n - 1) / starts))
        clusters, cur, load = [], [], 0
        for c in rot:
            if load + demands[c] > capacity:
                clusters.append(cur)
                cur, load = [], 0
            cur.append(int(c))
            load += demands[c]
        clusters.append(cur)
        routes = [tsp_cheapest_insertion(dist, cl) for cl in clusters]
        length = sum(route_length(dist, r) for r in routes)
        if length < best_len:
            best, best_len = routes, length
    return best
# <<end_sweep>>


def tsp_cheapest_insertion(dist: np.ndarray, customers: list[int]) -> list[int]:
    """One route through the depot and the given customers."""
    route, rest = [0, 0], list(customers)
    while rest:
        best = None
        for c in rest:
            for p in range(len(route) - 1):
                a, b = route[p], route[p + 1]
                delta = dist[a, c] + dist[c, b] - dist[a, b]
                if best is None or delta < best[0]:
                    best = (delta, c, p + 1)
        _, c, p = best
        route.insert(p, c)
        rest.remove(c)
    return route


def route_length(dist: np.ndarray, route: list[int]) -> float:
    return float(sum(dist[a, b] for a, b in zip(route, route[1:])))


# <<insertion>>
def insertion(dist: np.ndarray, demands, capacity: int, rule: str = "regret",
              n_routes: int | None = None, seeds: bool = True) -> list[list[int]]:
    """Parallel insertion into n_routes routes (default: the fewest that the
    total demand allows), opening a new route only when a customer fits
    nowhere.  With seeds, each route starts with one customer, chosen far
    from the depot and from the other seeds.  rule picks the next customer:
      "cheapest": the one with the cheapest insertion anywhere,
      "farthest": the one whose cheapest insertion is most expensive,
      "regret":   the one with the largest regret, the extra cost of its
                  second-best route over its best (regret-2)."""
    check_inputs(dist, demands, capacity)
    n = len(demands)
    q = np.asarray(demands, dtype=float)
    m = n_routes or math.ceil(q.sum() / capacity)
    routes = [[0, 0] for _ in range(m)]
    loads = [0.0] * m
    free = np.ones(n, dtype=bool)
    free[0] = False
    if seeds:                     # spread the routes out: one far-apart seed each
        near = np.asarray(dist[0], dtype=float).copy()
        near[0] = -1
        for r in range(m):
            c = int(np.argmax(np.where(free, near, -1)))
            routes[r] = [0, c, 0]
            loads[r] = q[c]
            free[c] = False
            near = np.minimum(near, dist[c])        # distance to depot or nearest seed
    # cost[c, r]: cheapest insertion of c into route r, pos[c, r]: where
    cost = np.full((n, m), np.inf)
    pos = np.zeros((n, m), dtype=int)

    def refresh(r: int) -> None:                    # only route r has changed
        rt = np.asarray(routes[r])
        a, b = rt[:-1], rt[1:]
        delta = dist[:, a] + dist[:, b] - dist[a, b]         # (n, positions)
        k = np.argmin(delta, axis=1)
        cost[:, r] = np.where(loads[r] + q <= capacity, delta[np.arange(n), k], np.inf)
        pos[:, r] = k + 1

    for r in range(m):
        refresh(r)
    while free.any():
        c_idx = np.flatnonzero(free)
        C = cost[c_idx]
        best = C.min(axis=1)
        if np.isinf(best).any():                    # someone fits nowhere: new route
            routes.append([0, 0])
            loads.append(0.0)
            cost = np.hstack([cost, np.full((n, 1), np.inf)])
            pos = np.hstack([pos, np.zeros((n, 1), dtype=int)])
            m += 1
            refresh(m - 1)
            continue
        if rule == "cheapest":
            pick = int(np.argmin(best))
        elif rule == "farthest":
            pick = int(np.argmax(best))
        else:
            second = np.partition(C, 1, axis=1)[:, 1] if C.shape[1] > 1 else best
            regret = np.where(np.isinf(second), 1e18, second - best)
            pick = int(np.lexsort((best, -regret))[0])  # max regret, then cheapest
        c = int(c_idx[pick])
        r = int(np.argmin(cost[c]))
        routes[r].insert(int(pos[c, r]), c)
        loads[r] += q[c]
        free[c] = False
        refresh(r)
    return [r for r in routes if len(r) > 2]
# <<end_insertion>>


# <<radial_bound>>
def radial_lower_bound(dist: np.ndarray, demands, capacity: int) -> float:
    """No plan can be shorter than (1/Q) * sum_i q_i * (d_0i + d_i0), which is
    (2/Q) * sum_i q_i * d_0i on symmetric distances (Chapter 10).  Needs the
    triangle inequality."""
    q = np.asarray(demands, dtype=float)
    d = np.asarray(dist, dtype=float)
    return float(q @ (d[0, :] + d[:, 0])) / capacity
# <<end_radial_bound>>


def ortools_first_solution(dist: np.ndarray, demands, capacity: int, n_vehicles: int,
                           strategy: str) -> list[list[int]]:
    """OR-Tools' first solution alone, with no local search afterwards."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(demands)
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    d = np.asarray(dist, dtype=np.int64)
    cb = routing.RegisterTransitCallback(
        lambda f, t: int(d[manager.IndexToNode(f), manager.IndexToNode(t)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    dem = routing.RegisterUnaryTransitCallback(lambda f: int(demands[manager.IndexToNode(f)]))
    routing.AddDimensionWithVehicleCapacity(dem, 0, [capacity] * n_vehicles, True, "Load")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, strategy)
    params.solution_limit = 1                       # stop at the first solution
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return []
    routes = []
    for v in range(n_vehicles):
        idx, r = routing.Start(v), []
        while not routing.IsEnd(idx):
            r.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        if len(r) > 1:
            routes.append(r + [0])
    return routes
