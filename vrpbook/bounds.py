"""Lower bounds and exact solutions for the CVRP (Chapter 13).

* lp_bound: the two-index LP with rounded capacity inequalities found by
  heuristic separation on fractional solutions,
* qroute_bound: column generation over q-routes with 2-cycle elimination,
* exact_cvrp: branch and cut with SCIP, integer separation, root cuts.
"""
from __future__ import annotations

import itertools
import math
import time
from functools import reduce
from pathlib import Path

import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp


def read_vrplib(path: str | Path):
    """A CVRPLIB file with EUC_2D coordinates: distances rounded to the
    nearest integer, as in the literature.  Returns (c, demands, capacity)."""
    lines = Path(path).read_text().splitlines()
    cap, coords, dem, sec = 0, [], [], None
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0].startswith("CAPACITY"):
            cap = int(t[-1])
        elif t[0] in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"):
            sec = t[0]
        elif sec == "NODE_COORD_SECTION":
            coords.append((float(t[1]), float(t[2])))
        elif sec == "DEMAND_SECTION":
            dem.append(int(t[1]))
    xy = np.array(coords)
    c = np.rint(np.linalg.norm(xy[:, None] - xy[None, :], axis=2)).astype(int)
    return c, dem, cap


# <<separation>>
def capacity_cuts(x: dict, n: int, q: list[int], Q: int, max_cuts: int = 50,
                  eps: float = 1e-6, hints=()) -> list[tuple[float, frozenset]]:
    """Heuristic separation of rounded capacity inequalities on a fractional
    solution.  Since every customer has degree 2, the inequality
    x(delta(S)) >= 2 r(S) is equivalent to x(E(S)) <= |S| - r(S), with
    r(S) = ceil(q(S) / Q). We look for sets S that violate the second form."""
    W = np.zeros((n, n))
    for (i, j), v in x.items():
        if i and v > eps:
            W[i, j] = W[j, i] = v
    found: dict[frozenset, float] = {}

    def check(S, inside):
        viol = inside - (len(S) - math.ceil(sum(q[i] for i in S) / Q))
        if viol > eps:
            found[frozenset(S)] = max(found.get(frozenset(S), 0), viol)

    # 1. connected components of the support graph, at several thresholds
    for t in (eps, 0.3, 0.5, 0.7):
        G = nx.Graph()
        G.add_nodes_from(range(1, n))
        G.add_edges_from((i, j) for i in range(1, n) for j in range(i + 1, n) if W[i, j] >= t)
        for comp in nx.connected_components(G):
            S = sorted(comp)
            check(S, W[np.ix_(S, S)].sum() / 2)
    # 2. grow a set greedily from every customer, by the strongest link
    for seed in range(1, n):
        S, inS, inside = [seed], np.zeros(n, bool), 0.0
        inS[seed] = True
        link = W[seed].copy()
        for _ in range(n - 2):
            cand = np.where(inS, -1, link)
            cand[0] = -1
            j = int(np.argmax(cand))
            if cand[j] <= eps:
                break
            inside += link[j]
            S.append(j)
            inS[j] = True
            link += W[j]
            check(S, inside)
    # 3. sets suggested by the caller, e.g. groups of routes of a good plan
    for S in hints:
        S = sorted(S)
        check(S, W[np.ix_(S, S)].sum() / 2)
    best = sorted(found.items(), key=lambda kv: -kv[1])[:max_cuts]
    return [(v, S) for S, v in best]
# <<end_separation>>


def _edges(n):
    return list(itertools.combinations(range(n), 2))


# <<lp_bound>>
def lp_bound(c: np.ndarray, q: list[int], Q: int, max_rounds: int = 300,
             time_limit_s: float = 600.0, hints=(), lp_solver: str = "CLP") -> dict:
    """Two-index LP relaxation strengthened by rounded capacity inequalities."""
    n = len(c)
    s = pywraplp.Solver.CreateSolver(lp_solver)
    x = {(i, j): s.NumVar(0, 2 if i == 0 else 1, "") for (i, j) in _edges(n)}
    inc = {k: [] for k in range(n)}
    for (i, j), v in x.items():
        inc[i].append(v)
        inc[j].append(v)
    for i in range(1, n):
        s.Add(sum(inc[i]) == 2)                              # each customer: degree 2
    s.Add(sum(inc[0]) >= 2 * math.ceil(sum(q) / Q))         # enough routes
    s.Minimize(sum(int(c[e]) * v for e, v in x.items()))
    t0 = time.perf_counter()
    history, cuts, cut_sets = [], 0, []
    z = 0.0
    for rnd in range(max_rounds):
        if s.Solve() != pywraplp.Solver.OPTIMAL:     # numerical trouble: keep the last bound
            break
        z = s.Objective().Value()
        history.append((time.perf_counter() - t0, z, cuts))
        val = {e: v.solution_value() for e, v in x.items()}
        new = capacity_cuts(val, n, q, Q, hints=hints)
        if not new or time.perf_counter() - t0 > time_limit_s:
            break
        for _, S in new:
            S = sorted(S)
            cut_sets.append(S)
            s.Add(sum(x[i, j] for i, j in itertools.combinations(S, 2))
                  <= len(S) - math.ceil(sum(q[k] for k in S) / Q))
            cuts += 1
    return {"bound": z, "rounds": rnd + 1, "cuts": cuts, "history": history,
            "cut_sets": cut_sets,
            "seconds": time.perf_counter() - t0}
# <<end_lp_bound>>


# <<qroutes>>
def qroute_bound(c: np.ndarray, q: list[int], Q: int, max_iter: int = 500,
                 cols_per_iter: int = 30) -> dict:
    """Column generation over q-routes: walks from the depot whose total
    demand fits in a van, which may revisit a customer but never return to
    the one they just left (2-cycle elimination).  Every route is a
    q-route, so the master LP value is a valid lower bound once pricing
    finds no column of negative reduced cost.  Before that it is only the
    value of a restricted master, which can be far above the optimum, so
    "bound" is None unless "converged" is True."""
    n = len(c)
    g = reduce(math.gcd, [d for d in q if d] + [Q])
    qq, QQ = [d // g for d in q], Q // g                  # demands in units of g
    k_min = math.ceil(sum(q) / Q)
    cols = [[0, i, 0] for i in range(1, n)]               # one route per customer
    t0 = time.perf_counter()
    z, converged = math.inf, False
    for it in range(1, max_iter + 1):
        s = pywraplp.Solver.CreateSolver("GLOP")
        lam = [s.NumVar(0, s.infinity(), "") for _ in cols]
        cover = []
        visits = [np.bincount(r[1:-1], minlength=n) for r in cols]
        for i in range(1, n):
            cover.append(s.Add(sum(int(v[i]) * l for v, l in zip(visits, lam) if v[i]) >= 1))
        fleet = s.Add(sum(lam) >= k_min)
        s.Minimize(sum(_route_cost(r, c) * l for r, l in zip(cols, lam)))
        if s.Solve() != pywraplp.Solver.OPTIMAL:          # no trustworthy duals
            break
        z = s.Objective().Value()
        pi = np.array([0.0] + [ct.dual_value() for ct in cover])
        sigma = fleet.dual_value()
        new = _price(c, qq, QQ, pi, sigma, cols_per_iter)
        if not new:                                       # pricing proves optimality
            converged = True
            break
        cols.extend(new)
    return {"bound": z if converged else None, "converged": converged,
            "master_value": z, "iterations": it, "columns": len(cols),
            "seconds": time.perf_counter() - t0}


def _route_cost(r, c):
    return float(sum(c[a, b] for a, b in zip(r, r[1:])))


def _price(c, q, Q, pi, sigma, k):
    """Labels f[l, j, r]: cheapest reduced-cost walk from the depot to j with
    load l, r = 0 is the best label, r = 1 the best one whose predecessor
    differs from that of r = 0 (which allows 2-cycle elimination)."""
    n = len(c)
    cr = c.astype(float) - pi[None, :]                    # pay pi_j on arriving at j
    INF = math.inf
    f = np.full((Q + 1, n, 2), INF)
    pred = np.full((Q + 1, n, 2, 2), -1, dtype=int)       # (node, rank) of predecessor
    for l in range(1, Q + 1):
        for j in range(1, n):
            L = l - q[j]
            if L < 0:
                continue
            if L == 0:
                cands = [(cr[0, j], 0, 0)]
            else:
                vals = f[L, :, :] + cr[:, j][:, None]       # (n, 2)
                cands = []
                for i in range(1, n):
                    if i == j:
                        continue
                    for r in (0, 1):
                        if vals[i, r] < INF and pred[L, i, r, 0] != j:   # no j -> i -> j
                            cands.append((vals[i, r], i, r))
                            break
            cands.sort()
            if cands:
                f[l, j, 0] = cands[0][0]
                pred[l, j, 0] = cands[0][1:]
                for v, i, r in cands[1:]:
                    if i != cands[0][1]:
                        f[l, j, 1] = v
                        pred[l, j, 1] = (i, r)
                        break
    out = []
    for l in range(1, Q + 1):
        for j in range(1, n):
            for r in (0, 1):
                red = f[l, j, r] + c[j, 0] - sigma
                if red < -1e-6:
                    out.append((red, l, j, r))
    out.sort()
    routes, seen = [], set()
    for red, l, j, r in out:
        path, L, node, rank = [], l, j, r
        while node != 0:
            path.append(node)
            pn, pr = pred[L, node, rank]
            L -= q[node]
            node, rank = pn, pr
        route = [0, *path[::-1], 0]
        key = tuple(route)
        if key not in seen:
            seen.add(key)
            routes.append(route)
        if len(routes) >= k:
            break
    return routes
# <<end_qroutes>>


# <<exact>>
def exact_cvrp(c: np.ndarray, q: list[int], Q: int, m: int, root_cuts=(),
               time_limit_s: float = 600.0) -> dict:
    """Two-index branch and cut with SCIP: the root cuts found by lp_bound
    are added up front, and capacity cuts violated by each integer solution
    are added until none is (the loop of Chapter 5)."""
    n = len(c)
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): s.IntVar(0, 2 if i == 0 else 1, "") for (i, j) in _edges(n)}
    inc = {k: [] for k in range(n)}
    for (i, j), v in x.items():
        inc[i].append(v)
        inc[j].append(v)
    for i in range(1, n):
        s.Add(sum(inc[i]) == 2)
    s.Add(sum(inc[0]) >= 2 * math.ceil(sum(q) / Q))
    s.Add(sum(inc[0]) <= 2 * m)

    def add_cut(S):
        S = sorted(S)
        s.Add(sum(x[i, j] for i, j in itertools.combinations(S, 2))
              <= len(S) - math.ceil(sum(q[k] for k in S) / Q))

    for S in root_cuts:
        add_cut(S)
    s.Minimize(sum(int(c[e]) * v for e, v in x.items()))
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, 0.0)
    t0 = time.perf_counter()
    rounds = 0
    while True:
        left = time_limit_s - (time.perf_counter() - t0)
        if left <= 0:
            return {"status": "time limit", "rounds": rounds, "seconds": time_limit_s}
        s.SetTimeLimit(int(left * 1000))
        status = s.Solve(params)
        if status != pywraplp.Solver.OPTIMAL:
            return {"status": "time limit", "rounds": rounds,
                    "seconds": time.perf_counter() - t0, "bound": s.Objective().BestBound()}
        rounds += 1
        G = nx.Graph()
        G.add_nodes_from(range(1, n))
        G.add_edges_from(e for e, v in x.items() if e[0] and v.solution_value() > 0.5)
        bad = [comp for comp in nx.connected_components(G)
               if not any(x[min(0, k), k].solution_value() > 0.5 for k in comp)
               or sum(q[k] for k in comp) > Q]
        if not bad:
            break
        for S in bad:
            add_cut(S)
    return {"status": "optimal", "value": s.Objective().Value(), "rounds": rounds,
            "seconds": time.perf_counter() - t0}
# <<end_exact>>
