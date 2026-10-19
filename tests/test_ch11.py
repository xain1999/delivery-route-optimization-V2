"""Tests for Chapter 11: local search and segment concatenation."""
import copy
import random

import numpy as np

from vrpbook import distance_matrix, total_distance, validate_routes
from vrpbook import construction as C
from vrpbook.instances import chicago_synthetic
from vrpbook.localsearch import (OPERATORS, Plan, Segment, apply_move, best_move_for,
                                 local_search, route_segment)
from vrpbook.timewindows import TWData, best_departure, schedule


def test_every_move_changes_the_length_by_its_delta():
    inst = chicago_synthetic(60, seed=5)
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    D = np.asarray(d, dtype=float).tolist()
    plan = C.nearest_neighbor(d, q, Q)
    rng = random.Random(1)
    checked = set()
    for _ in range(3000):
        u, v = rng.sample(range(1, len(q)), 2)
        P = Plan([list(r) for r in plan], q)
        for op in OPERATORS:
            m = best_move_for(u, v, P, D, Q, (op,))
            if m is None:
                continue
            P2 = copy.deepcopy(P)
            before = total_distance(P2.routes, d)
            apply_move(m[1], P2)
            routes = [r for r in P2.routes if len(r) > 2]
            validate_routes(routes, d, q, Q)
            assert abs(total_distance(routes, d) - before - m[0]) < 1e-6 * before + 2
            checked.add(op)
    assert checked == set(OPERATORS)


def test_local_search_reaches_a_local_optimum():
    inst = chicago_synthetic(80, seed=2)
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    start = C.nearest_neighbor(d, q, Q)
    routes, stats = local_search(start, d, q, Q, k=79)
    validate_routes(routes, d, q, Q)
    assert total_distance(routes, d) < total_distance(start, d)
    P = Plan([list(r) for r in routes], q)
    D = np.asarray(d, dtype=float).tolist()
    assert all(best_move_for(u, v, P, D, Q, OPERATORS) is None
               for u in range(1, len(q)) for v in range(1, len(q)) if u != v)


def test_segments_match_the_schedule_of_chapter_6():
    rng = np.random.default_rng(4)
    n = 9
    t = rng.integers(300, 1500, size=(n, n))
    t = (t + t.T) // 2
    np.fill_diagonal(t, 0)
    service = [0] + [300] * (n - 1)
    for _ in range(200):
        starts = rng.integers(7 * 3600, 13 * 3600, size=n)
        windows = [(6 * 3600, 18 * 3600)] + [(int(a), int(a) + int(rng.integers(1800, 7200)))
                                             for a in starts[1:]]
        data = TWData(np.zeros((n, n)), t, service, windows, [0] * n, 99, 1)
        route = [0, *rng.permutation(range(1, n)).tolist(), 0]
        seg = route_segment(route, service, windows, t)
        vis = schedule(route, data, depart=best_departure(route, data))
        feasible = all(v.start <= windows[v.node][1] for v in vis)
        assert (seg.TW == 0) == feasible
        if feasible:
            assert seg.D == vis[-1].arrival - vis[0].departure


def test_segment_concatenation_is_associative():
    t = [[0, 5, 9], [5, 0, 4], [9, 4, 0]]
    a, b, c = (Segment.of(0, 0, 0, 100), Segment.of(1, 3, 20, 30), Segment.of(2, 2, 10, 15))
    left = a.then(b, t).then(c, t)
    right = a.then(b.then(c, t), t)
    assert left == right
