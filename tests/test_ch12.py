"""Tests for Chapter 12: split, crossover, ALNS, HGS and the PyVRP adapter."""
import itertools
import random

import numpy as np

from vrpbook import distance_matrix, total_distance, validate_routes
from vrpbook import construction as C
from vrpbook.instances import chicago_synthetic
from vrpbook.metaheuristics import alns, hgs, ox_crossover, split
from vrpbook.pyvrp_adapter import solve_cvrp_pyvrp


def _cost(routes, D):
    return sum(D[a][b] for r in routes for a, b in zip(r, r[1:]))


def test_split_is_optimal_for_its_tour():
    rng = np.random.default_rng(7)
    n = 8
    xy = rng.uniform(0, 10, size=(n + 1, 2))
    D = np.linalg.norm(xy[:, None] - xy[None, :], axis=2).tolist()
    q = [0] + list(rng.integers(1, 6, size=n))
    tour = list(rng.permutation(range(1, n + 1)))
    best = np.inf
    for cuts in itertools.product([0, 1], repeat=n - 1):     # every way to cut the tour
        routes, cur = [], [tour[0]]
        for c, cut in zip(tour[1:], cuts):
            if cut:
                routes.append(cur)
                cur = []
            cur.append(c)
        routes.append(cur)
        if all(sum(q[c] for c in r) <= 10 for r in routes):
            best = min(best, _cost([[0, *r, 0] for r in routes], D))
    assert abs(_cost(split(tour, D, q, 10), D) - best) < 1e-9


def test_order_crossover_gives_a_permutation():
    rng = random.Random(3)
    p1 = list(range(1, 21))
    p2 = p1[:]
    rng.shuffle(p2)
    for _ in range(50):
        child = ox_crossover(p1, p2, rng)
        assert sorted(child) == p1


def test_metaheuristics_improve_on_savings_and_stay_feasible():
    inst = chicago_synthetic(40, seed=8)
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    start = total_distance(C.clarke_wright(d, q, Q), d)
    xy = [[s.lon, s.lat] for s in inst.sites]
    for routes in (alns(d, q, Q, 2.0), hgs(d, q, Q, xy, 2.0)):
        validate_routes(routes, d, q, Q)
        assert total_distance(routes, d) <= start


def test_pyvrp_adapter_returns_a_valid_plan():
    inst = chicago_synthetic(30, seed=1)
    d = distance_matrix(inst.sites)
    routes, iterations = solve_cvrp_pyvrp(d, inst.demands, inst.capacity, 1.0)
    validate_routes(routes, d, inst.demands, inst.capacity)
    assert iterations > 0


def test_tiny_instances_finish_within_their_time():
    import time
    rng = np.random.default_rng(3)
    for n in (0, 1, 2, 3):
        xy = rng.random((n + 1, 2)) * 1000
        D = np.rint(np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))).astype(int)
        q = [0] + [1] * n
        for method in (lambda: alns(D, q, 2, 0.05), lambda: hgs(D, q, 2, xy, 0.05)):
            t = time.perf_counter()
            routes = method()
            assert time.perf_counter() - t < 1.0
            validate_routes(routes, D, q, 2)
