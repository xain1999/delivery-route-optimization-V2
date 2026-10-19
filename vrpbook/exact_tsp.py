"""A small exact TSP solver used to certify optimality on small cases.

Symmetric Dantzig-Fulkerson-Johnson model solved with HiGHS (through
SciPy), adding violated subtour-elimination constraints one round at a
time.  Chapter 13 develops this properly. Here it only serves to check
how far a heuristic tour is from the optimum.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for v in range(n):
        groups.setdefault(find(v), []).append(v)
    return list(groups.values())


def solve_tsp_exact(dist: np.ndarray, max_rounds: int = 200) -> tuple[list[int], int, int]:
    """Return (tour starting and ending at 0, length, number of cut rounds)."""
    n = len(dist)
    E = list(itertools.combinations(range(n), 2))
    c = np.array([dist[i, j] for i, j in E], dtype=float)
    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []
    # degree constraints: sum of incident edges = 2
    for v in range(n):
        a = np.zeros(len(E))
        for k, (i, j) in enumerate(E):
            if v in (i, j):
                a[k] = 1
        rows.append(a)
        lo.append(2)
        hi.append(2)
    for rnd in range(1, max_rounds + 1):
        A = lil_matrix((len(rows), len(E)))
        for r, a in enumerate(rows):
            for k in np.flatnonzero(a):
                A[r, k] = a[k]
        res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi),
                   integrality=np.ones(len(E)), bounds=Bounds(0, 1),
                   options={"mip_rel_gap": 0})     # prove optimality exactly
        if res.x is None:
            raise RuntimeError(res.message)
        chosen = [E[k] for k in np.flatnonzero(res.x > 0.5)]
        comps = _components(n, chosen)
        if len(comps) == 1:
            break
        for S in comps:  # x(E(S)) <= |S| - 1
            Sset = set(S)
            a = np.array([1.0 if (i in Sset and j in Sset) else 0.0 for i, j in E])
            rows.append(a)
            lo.append(-np.inf)
            hi.append(len(S) - 1)
    else:
        raise RuntimeError("cut loop did not converge")
    # turn the edge set into a tour
    adj: dict[int, list[int]] = {v: [] for v in range(n)}
    for i, j in chosen:
        adj[i].append(j)
        adj[j].append(i)
    tour, prev, cur = [0], None, 0
    while True:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        tour.append(nxt)
        if nxt == 0:
            break
        prev, cur = cur, nxt
    length = sum(int(dist[a, b]) for a, b in zip(tour, tour[1:]))
    return tour, length, rnd
