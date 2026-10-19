"""Local search for the CVRP, from scratch (Chapter 11).

A plan is a list of routes [0, i, j, ..., 0].  Every move is evaluated in
constant time from the few arcs it changes, and only between customers
that are close to each other (a granular neighborhood).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

OPERATORS = ("relocate", "swap", "2opt", "2opt*", "oropt")


# <<neighbors>>
def neighbor_lists(dist: np.ndarray, k: int) -> list[list[int]]:
    """The k nearest customers of every customer (the depot is not one)."""
    n = len(dist)
    d = np.asarray(dist, dtype=float).copy()
    d[:, 0] = np.inf
    np.fill_diagonal(d, np.inf)
    k = min(k, n - 2)
    near = np.argpartition(d, k, axis=1)[:, :k]
    order = np.take_along_axis(d, near, axis=1).argsort(axis=1)
    return [[] if i == 0 else [int(v) for v in np.take_along_axis(near, order, 1)[i]]
            for i in range(n)]
# <<end_neighbors>>


@dataclass
class Plan:
    """Routes plus the bookkeeping that makes every check O(1)."""
    routes: list[list[int]]
    demands: list[int]
    route_of: list[int] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)
    load: list[int] = field(default_factory=list)
    prefix: list[list[int]] = field(default_factory=list)   # load up to each position

    def __post_init__(self):
        n = len(self.demands)
        self.route_of, self.pos = [0] * n, [0] * n
        self.load, self.prefix = [0] * len(self.routes), [[] for _ in self.routes]
        for r in range(len(self.routes)):
            self.refresh(r)

    def refresh(self, r: int) -> None:
        route, q, acc, pre = self.routes[r], self.demands, 0, []
        for k, c in enumerate(route):
            acc += q[c]
            pre.append(acc)
            if c:
                self.route_of[c], self.pos[c] = r, k
        self.prefix[r], self.load[r] = pre, acc


# <<evaluate>>
def best_move_for(u: int, v: int, P: Plan, D, Q: int, ops) -> tuple[float, tuple] | None:
    """The best improving move that brings customer u next to customer v."""
    r1, r2 = P.route_of[u], P.route_of[v]
    R1, R2 = P.routes[r1], P.routes[r2]
    i, j = P.pos[u], P.pos[v]
    pu, nu, pv, nv = R1[i - 1], R1[i + 1], R2[j - 1], R2[j + 1]
    q = P.demands
    best = None

    def consider(delta, move):
        nonlocal best
        if delta < -1e-9 and (best is None or delta < best[0]):
            best = (delta, move)

    same = r1 == r2
    if "relocate" in ops and (same or P.load[r2] + q[u] <= Q):
        gain = D[pu][u] + D[u][nu] - D[pu][nu]               # remove u
        if v != pu:                                           # insert after v
            consider(D[v][u] + D[u][nv] - D[v][nv] - gain, ("relocate", u, v, "after"))
        if v != nu:                                           # insert before v
            consider(D[pv][u] + D[u][v] - D[pv][v] - gain, ("relocate", u, v, "before"))
    if "swap" in ops and (not same or abs(i - j) > 1):
        if same or (P.load[r1] - q[u] + q[v] <= Q and P.load[r2] - q[v] + q[u] <= Q):
            delta = (D[pu][v] + D[v][nu] - D[pu][u] - D[u][nu]
                     + D[pv][u] + D[u][nv] - D[pv][v] - D[v][nv])
            consider(delta, ("swap", u, v))
    if "2opt" in ops and same and abs(i - j) > 1:             # reverse between them
        a, b = (u, v) if i < j else (v, u)
        na, nb = R1[P.pos[a] + 1], R1[P.pos[b] + 1]
        consider(D[a][b] + D[na][nb] - D[a][na] - D[b][nb], ("2opt", a, b))
    if "2opt*" in ops and not same:                           # exchange route tails
        pre1, pre2, L1, L2 = P.prefix[r1][i], P.prefix[r2][j], P.load[r1], P.load[r2]
        if pre1 + pre2 <= Q and (L1 - pre1) + (L2 - pre2) <= Q:   # u-v joined
            consider(D[u][v] + D[nu][nv] - D[u][nu] - D[v][nv], ("2opt*", u, v, "rev"))
        if pre1 + (L2 - pre2) <= Q and pre2 + (L1 - pre1) <= Q:   # u-nv, v-nu
            consider(D[u][nv] + D[v][nu] - D[u][nu] - D[v][nv], ("2opt*", u, v, "tail"))
    if "oropt" in ops:                                        # move u's segment after v
        for L in (2, 3):
            if i + L > len(R1) - 1:
                break
            seg = R1[i:i + L]
            if v in seg or v == pu:
                continue
            sq = sum(q[c] for c in seg)
            if not same and P.load[r2] + sq > Q:
                continue
            b = R1[i + L]
            gain = D[pu][seg[0]] + D[seg[-1]][b] - D[pu][b]
            fwd = D[v][seg[0]] + D[seg[-1]][nv] - D[v][nv]
            rev = D[v][seg[-1]] + D[seg[0]][nv] - D[v][nv]
            consider(fwd - gain, ("oropt", u, v, L, False))
            consider(rev - gain, ("oropt", u, v, L, True))
    return best
# <<end_evaluate>>


def apply_move(move: tuple, P: Plan) -> None:
    kind, u, v = move[0], move[1], move[2]
    r1, r2 = P.route_of[u], P.route_of[v]
    R1, R2 = P.routes[r1], P.routes[r2]
    if kind == "relocate":
        R1.remove(u)
        k = R2.index(v)
        R2.insert(k + 1 if move[3] == "after" else k, u)
    elif kind == "swap":
        i, j = P.pos[u], P.pos[v]
        R1[i], R2[j] = v, u
    elif kind == "2opt":
        i, j = P.pos[u], P.pos[v]
        R1[i + 1:j + 1] = R1[i + 1:j + 1][::-1]
    elif kind == "2opt*":
        i, j = P.pos[u], P.pos[v]
        if move[3] == "tail":
            P.routes[r1], P.routes[r2] = R1[:i + 1] + R2[j + 1:], R2[:j + 1] + R1[i + 1:]
        else:
            P.routes[r1] = R1[:i + 1] + R2[1:j + 1][::-1] + [0]
            P.routes[r2] = [0] + R1[i + 1:-1][::-1] + R2[j + 1:]
    elif kind == "oropt":
        L, reverse = move[3], move[4]
        i = P.pos[u]
        seg = R1[i:i + L]
        del R1[i:i + L]
        k = R2.index(v)
        R2[k + 1:k + 1] = seg[::-1] if reverse else seg
    P.refresh(r1)
    if r2 != r1:
        P.refresh(r2)


# <<local_search>>
def local_search(routes: list[list[int]], dist: np.ndarray, demands, capacity: int,
                 k: int = 20, ops=OPERATORS, strategy: str = "first",
                 seed: int = 0, max_seconds: float = 1e9,
                 near=None, D=None, focus=None) -> tuple[list[list[int]], dict]:
    """Improve a plan until no move between neighbors improves it.

    strategy "first": apply the first improving move found for each
    customer, "best": scan every customer, apply the single best move.
    near and D (the matrix as nested lists) can be passed in when the
    search runs many times on one instance, and focus restricts the
    search to moves of the given customers (Chapter 12)."""
    if D is None:
        D = np.asarray(dist, dtype=float).tolist()       # list access is faster
    if near is None:
        near = neighbor_lists(dist, k)
    P = Plan([list(r) for r in routes], list(demands))
    rng = random.Random(seed)
    order = list(focus) if focus is not None else [c for r in routes for c in r[1:-1]]
    stats = {"moves": 0, "evaluations": 0, "passes": 0, "by_op": {}}
    t0 = time.perf_counter()
    improved = True
    while improved and time.perf_counter() - t0 < max_seconds:
        improved = False
        stats["passes"] += 1
        rng.shuffle(order)
        best_all = None
        for u in order:
            for v in near[u]:
                stats["evaluations"] += 1
                m = best_move_for(u, v, P, D, capacity, ops)
                if m is None:
                    continue
                if strategy == "first":
                    apply_move(m[1], P)
                    stats["moves"] += 1
                    stats["by_op"][m[1][0]] = stats["by_op"].get(m[1][0], 0) + 1
                    improved = True
                    break                                 # u has moved on
                if best_all is None or m[0] < best_all[0]:
                    best_all = m
        if strategy == "best" and best_all is not None:
            apply_move(best_all[1], P)
            stats["moves"] += 1
            stats["by_op"][best_all[1][0]] = stats["by_op"].get(best_all[1][0], 0) + 1
            improved = True
    stats["seconds"] = time.perf_counter() - t0
    return [r for r in P.routes if len(r) > 2], stats
# <<end_local_search>>


def broken_pairs(a: list[list[int]], b: list[list[int]]) -> float:
    """Share of customer adjacencies of plan a that plan b does not have."""
    def pairs(plan):
        return {frozenset(e) for r in plan for e in zip(r, r[1:])}
    pa, pb = pairs(a), pairs(b)
    return len(pa - pb) / len(pa)


# ---------------------------------------------------------------- time windows
# <<segments>>
@dataclass(frozen=True)
class Segment:
    """Summary of a sequence of visits for time-window checks: the minimum
    duration D, the time warp TW (lateness that would be needed), and the
    earliest E and latest L start of the first visit that achieve them."""
    first: int
    last: int
    D: float
    TW: float
    E: float
    L: float

    @staticmethod
    def of(i: int, service: float, a: float, b: float) -> "Segment":
        return Segment(i, i, service, 0.0, a, b)

    def then(self, other: "Segment", t) -> "Segment":
        """This segment followed by other, t is the travel-time matrix."""
        delta = self.D - self.TW + t[self.last][other.first]
        wait = max(other.E - delta - self.L, 0.0)
        warp = max(self.E + delta - other.L, 0.0)
        return Segment(self.first, other.last,
                       self.D + other.D + t[self.last][other.first] + wait,
                       self.TW + other.TW + warp,
                       max(other.E - delta, self.E) - wait,
                       min(other.L - delta, self.L) + warp)
# <<end_segments>>


def route_segment(route: list[int], service, windows, t) -> Segment:
    s = Segment.of(route[0], service[route[0]], *windows[route[0]])
    for c in route[1:]:
        s = s.then(Segment.of(c, service[c], *windows[c]), t)
    return s
