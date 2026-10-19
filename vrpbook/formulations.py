"""Integer programming models of the TSP and the CVRP (Chapter 5).

All models are built with OR-Tools' linear solver wrapper, which ships
with the open-source solvers SCIP (MIP, Apache 2.0 license) and GLOP (LP),
so no extra installation is needed.  Costs are integer matrices with node
0 the depot.

These implementations favor clarity over speed.  They are meant for
small instances (up to a few dozen customers) and for computing bounds.
Chapter 13 develops faster exact methods.
"""
from __future__ import annotations

import itertools
import math
import time
import time as time_mod
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp


@dataclass
class ModelResult:
    objective: float           # optimal (or best) value, LP value if relaxed
    seconds: float
    bound: float               # best lower bound proven (= objective if optimal)
    status: str
    rounds: int = 0            # cutting-plane rounds, if any
    cuts: int = 0              # constraints added by separation
    solution: dict = field(default_factory=dict)


def _solver(relax: bool, time_limit_s: float | None):
    s = pywraplp.Solver.CreateSolver("GLOP" if relax else "SCIP")
    s.SuppressOutput()
    if time_limit_s is not None:
        s.SetTimeLimit(int(time_limit_s * 1000))
    return s


def _solve(s) -> int:
    """Solve to proven optimality (the wrapper's default stops at 0.01%)."""
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, 0.0)
    return s.Solve(params)


def _values(s) -> tuple[float, float]:
    """Objective value and best proven lower bound (equal for an LP)."""
    obj = s.Objective().Value()
    return obj, (s.Objective().BestBound() if s.IsMip() else obj)


def _res(s, t0: float, status: int):
    obj, bound = _values(s)
    return obj, time.perf_counter() - t0, bound, _status(status)


def _status(code: int) -> str:
    return {pywraplp.Solver.OPTIMAL: "optimal", pywraplp.Solver.FEASIBLE: "feasible",
            pywraplp.Solver.INFEASIBLE: "infeasible",
            pywraplp.Solver.NOT_SOLVED: "not solved"}.get(code, str(code))


# =========================================================================
# TSP: three directed formulations
# =========================================================================

# <<tsp_assignment>>
def _tsp_assignment(c: np.ndarray, relax: bool, time_limit_s: float | None):
    """Arc variables x[i, j] and the degree (assignment) constraints."""
    n = len(c)
    s = _solver(relax, time_limit_s)
    ub = 1.0
    x = {(i, j): (s.NumVar(0, ub, f"x{i}_{j}") if relax else s.BoolVar(f"x{i}_{j}"))
         for i in range(n) for j in range(n) if i != j}
    for i in range(n):
        s.Add(sum(x[i, j] for j in range(n) if j != i) == 1)   # leave i once
        s.Add(sum(x[j, i] for j in range(n) if j != i) == 1)   # enter i once
    s.Minimize(sum(int(c[i, j]) * v for (i, j), v in x.items()))
    return s, x
# <<end_tsp_assignment>>


# <<tsp_mtz>>
def tsp_mtz(c: np.ndarray, relax: bool = False,
            time_limit_s: float | None = None) -> ModelResult:
    """Miller-Tucker-Zemlin: order variables u_i forbid subtours."""
    n = len(c)
    s, x = _tsp_assignment(c, relax, time_limit_s)
    u = {i: s.NumVar(1, n - 1, f"u{i}") for i in range(1, n)}
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s.Add(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
    t0 = time.perf_counter()
    status = _solve(s)
    return ModelResult(*_res(s, t0, status),
                       solution={a: v.solution_value() for a, v in x.items()})
# <<end_tsp_mtz>>


# <<tsp_scf>>
def tsp_scf(c: np.ndarray, relax: bool = False,
            time_limit_s: float | None = None) -> ModelResult:
    """Single-commodity flow (Gavish-Graves): the depot ships one unit to
    every other node, and flow may use only arcs of the tour."""
    n = len(c)
    s, x = _tsp_assignment(c, relax, time_limit_s)
    f = {a: s.NumVar(0, n - 1, f"f{a[0]}_{a[1]}") for a in x}
    s.Add(sum(f[0, j] for j in range(1, n)) == n - 1)
    for i in range(1, n):
        inflow = sum(f[j, i] for j in range(n) if j != i)
        outflow = sum(f[i, j] for j in range(n) if j != i)
        s.Add(inflow - outflow == 1)            # node i keeps one unit
    for a in x:
        s.Add(f[a] <= (n - 1) * x[a])
    t0 = time.perf_counter()
    status = _solve(s)
    return ModelResult(*_res(s, t0, status),
                       solution={a: v.solution_value() for a, v in x.items()})
# <<end_tsp_scf>>


# <<tsp_dfj>>
def tsp_dfj(c: np.ndarray, relax: bool = False, time_limit_s: float | None = None,
            max_rounds: int = 500) -> ModelResult:
    """Dantzig-Fulkerson-Johnson: subtour elimination constraints
    x(delta_in(S)) >= 1 for every S not containing the depot, added only
    when violated (a cutting-plane loop)."""
    n = len(c)
    s, x = _tsp_assignment(c, relax, time_limit_s)
    t0 = time.perf_counter()
    cuts = 0
    for rnd in range(1, max_rounds + 1):
        status = _solve(s)
        val = {a: v.solution_value() for a, v in x.items()}
        violated = _violated_subtours(n, val, fractional=relax)
        if not violated:
            break
        for S in violated:
            s.Add(sum(x[i, j] for i in range(n) for j in S if i not in S) >= 1)
            cuts += 1
    return ModelResult(*_res(s, t0, status),
                       rnd, cuts, val)


def _violated_subtours(n: int, val: dict, fractional: bool, eps: float = 1e-6):
    if not fractional:
        # integer solution: every cycle that misses the depot is a subtour
        G = nx.DiGraph((i, j) for (i, j), v in val.items() if v > 0.5)
        return [set(comp) for comp in nx.strongly_connected_components(G) if 0 not in comp]
    # fractional: a min cut from the depot to t below 1 gives a violated S
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (i, j), v in val.items():
        if v > eps:
            G.add_edge(i, j, capacity=v)
    found = []
    for t in range(1, n):
        cut, (_, sink_side) = nx.minimum_cut(G, 0, t)
        if cut < 1 - eps and set(sink_side) not in found:
            found.append(set(sink_side))
    return found
# <<end_tsp_dfj>>


# =========================================================================
# CVRP: two-index vehicle flow with rounded capacity inequalities
# =========================================================================

# <<cvrp_two_index>>
def cvrp_two_index(c: np.ndarray, q: list[int], Q: int, m: int,
                   relax: bool = False, capacity_cuts: bool = True,
                   time_limit_s: float | None = None,
                   max_rounds: int = 500) -> ModelResult:
    """Undirected two-index model.  x_e counts how often edge e is used:
    0/1 between customers, 0/1/2 between depot and customer (2 = a route
    with a single customer).  Rounded capacity inequalities
        x(delta(S)) >= 2 * ceil(q(S) / Q)
    are added by separation."""
    n = len(c)
    s = _solver(relax, time_limit_s)
    E = list(itertools.combinations(range(n), 2))
    x = {}
    for (i, j) in E:
        ub = 2 if i == 0 else 1
        x[i, j] = s.NumVar(0, ub, f"x{i}_{j}") if relax else s.IntVar(0, ub, f"x{i}_{j}")
    def delta(S):                      # edges with exactly one end in S
        return [x[e] for e in E if (e[0] in S) != (e[1] in S)]
    for i in range(1, n):
        s.Add(sum(delta({i})) == 2)                     # visit each customer
    k_min = math.ceil(sum(q) / Q)
    s.Add(sum(delta({0})) >= 2 * k_min)                 # enough vans leave
    s.Add(sum(delta({0})) <= 2 * m)                     # fleet size
    s.Minimize(sum(int(c[e]) * v for e, v in x.items()))
    t0 = time.perf_counter()
    cuts = 0
    for rnd in range(1, max_rounds + 1):
        status = _solve(s)
        val = {e: v.solution_value() for e, v in x.items()}
        if not capacity_cuts:
            break
        violated = _violated_capacity(n, q, Q, val, fractional=relax)
        if not violated:
            break
        for S in violated:
            s.Add(sum(delta(S)) >= 2 * math.ceil(sum(q[i] for i in S) / Q))
            cuts += 1
    return ModelResult(*_res(s, t0, status),
                       rnd, cuts, val)
# <<end_cvrp_two_index>>


def _violated_capacity(n, q, Q, val, fractional, eps=1e-6):
    customers = range(1, n)
    if not fractional:
        # integer: each connected group of customers is one route's
        # customers (or a subtour), check it directly
        G = nx.Graph()
        G.add_nodes_from(customers)
        G.add_edges_from(e for e, v in val.items() if v > 0.5 and e[0] != 0)
        candidates = [set(cc) for cc in nx.connected_components(G)]
    else:
        # fractional: exact separation by enumeration (small n only)
        if n - 1 > 14:
            raise ValueError("exact fractional separation limited to 14 customers")
        candidates = (set(S) for r in range(1, n) for S in itertools.combinations(customers, r))
    out = []
    for S in candidates:
        lhs = sum(v for e, v in val.items() if (e[0] in S) != (e[1] in S))
        rhs = 2 * math.ceil(sum(q[i] for i in S) / Q)
        if lhs < rhs - eps:
            out.append(S)
    if fractional and out:                       # add the most violated few
        out.sort(key=lambda S: sum(v for e, v in val.items() if (e[0] in S) != (e[1] in S))
                 - 2 * math.ceil(sum(q[i] for i in S) / Q))
        out = out[:20]
    return out


# =========================================================================
# CVRP: set partitioning over all feasible routes
# =========================================================================

# <<enumerate_routes>>
def enumerate_routes(c: np.ndarray, q: list[int], Q: int) -> list[tuple[frozenset, int, list[int]]]:
    """Every capacity-feasible set of customers, with its cheapest route.

    Uses the Held-Karp dynamic program: best[S, j] is the length of the
    shortest path that leaves the depot, visits exactly the customers in
    S and ends at j.  Exponential in n. Fine for about 15 customers.
    """
    customers = list(range(1, len(c)))
    best: dict[tuple[frozenset, int], tuple[int, int | None]] = {}
    for j in customers:
        if q[j] <= Q:
            best[frozenset([j]), j] = (int(c[0, j]), None)
    by_size = [[frozenset([j]) for j in customers if q[j] <= Q]]
    for size in range(2, len(customers) + 1):
        layer = set()
        for S in by_size[-1]:
            load = sum(q[i] for i in S)
            for k in customers:
                if k in S or load + q[k] > Q:
                    continue
                T = S | {k}
                for j in S:
                    cand = best[S, j][0] + int(c[j, k])
                    if (T, k) not in best or cand < best[T, k][0]:
                        best[T, k] = (cand, j)
                layer.add(T)
        if not layer:
            break
        by_size.append(list(layer))
    routes = []
    for layer in by_size:
        for S in layer:
            cost, last = min((best[S, j][0] + int(c[j, 0]), j) for j in S)
            order, T, j = [], S, last          # walk back through predecessors
            while j is not None:
                order.append(j)
                prev = best[T, j][1]
                T, j = T - {j}, prev
            routes.append((S, cost, [0] + order[::-1] + [0]))
    return routes
# <<end_enumerate_routes>>


# <<set_partitioning>>
def cvrp_set_partitioning(routes, n: int, m: int, relax: bool = False,
                          min_vehicles: int = 0,
                          time_limit_s: float | None = None) -> ModelResult:
    """Choose at most m routes covering every customer exactly once.
    min_vehicles: optional valid lower bound on the number of routes,
    such as ceil(total demand / capacity). It tightens the LP."""
    s = _solver(relax, time_limit_s)
    lam = [s.NumVar(0, 1, f"r{k}") if relax else s.BoolVar(f"r{k}") for k in range(len(routes))]
    for i in range(1, n):
        s.Add(sum(lam[k] for k, (S, _, _) in enumerate(routes) if i in S) == 1)
    s.Add(sum(lam) <= m)
    if min_vehicles:
        s.Add(sum(lam) >= min_vehicles)
    s.Minimize(sum(cost * lam[k] for k, (_, cost, _) in enumerate(routes)))
    t0 = time.perf_counter()
    status = _solve(s)
    chosen = [routes[k][2] for k, v in enumerate(lam) if v.solution_value() > 0.5]
    return ModelResult(*_res(s, t0, status),
                       solution={"routes": chosen,
                                 "fractional": sum(1 for v in lam
                                                   if 1e-6 < v.solution_value() < 1 - 1e-6)})
# <<end_set_partitioning>>


# =========================================================================
# VRPTW: two-index model with big-M time and load constraints
# =========================================================================

# <<vrptw_mip>>
def vrptw_mip(dist, time, service, windows, q, Q, m, relax=False,
              time_limit_s=None) -> ModelResult:
    """Directed two-index VRPTW minimizing distance.  Start-of-service
    times T_i and loads L_i are linked to the arcs by big-M constraints,
    which also rule out subtours (Chapter 6)."""
    n = len(dist)
    s = _solver(relax, time_limit_s)
    A = [(i, j) for i in range(n) for j in range(n) if i != j]
    x = {a: s.NumVar(0, 1, f"x{a}") if relax else s.BoolVar(f"x{a}") for a in A}
    T = {i: s.NumVar(windows[i][0], windows[i][1], f"T{i}") for i in range(1, n)}
    L = {i: s.NumVar(q[i], Q, f"L{i}") for i in range(1, n)}
    a0, b0 = windows[0]
    for i in range(1, n):
        s.Add(sum(x[i, j] for j in range(n) if j != i) == 1)
        s.Add(sum(x[j, i] for j in range(n) if j != i) == 1)
    s.Add(sum(x[0, j] for j in range(1, n)) <= m)
    for (i, j) in A:
        if i == 0:      # leave the depot no earlier than it opens
            s.Add(T[j] >= a0 + int(time[0, j]) * x[0, j])
        elif j == 0:    # return before the depot closes
            M = max(0, windows[i][1] + service[i] + int(time[i, 0]) - b0)
            s.Add(T[i] + service[i] + int(time[i, 0]) <= b0 + M * (1 - x[i, 0]))
        else:
            M = max(0, windows[i][1] + service[i] + int(time[i, j]) - windows[j][0])
            s.Add(T[j] >= T[i] + service[i] + int(time[i, j]) - M * (1 - x[i, j]))
            s.Add(L[j] >= L[i] + q[j] - Q * (1 - x[i, j]))
    s.Minimize(sum(int(dist[a]) * v for a, v in x.items()))
    t0 = time_mod.perf_counter()
    status = _solve(s)
    return ModelResult(*_res(s, t0, status),
                       solution={a: v.solution_value() for a, v in x.items()})
# <<end_vrptw_mip>>
