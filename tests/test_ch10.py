"""Tests for Chapter 10: construction heuristics."""
import numpy as np

from vrpbook import distance_matrix, total_distance, validate_routes
from vrpbook import construction as C
from vrpbook.instances import chicago_synthetic


def test_savings_merges_customers_in_the_same_direction():
    # depot at 0, customers at 5 and 6 on one side and at -5 on the other
    pos = np.array([0, 5, 6, -5])
    d = np.abs(pos[:, None] - pos[None, :]).astype(float)
    routes = C.clarke_wright(d, [0, 1, 1, 1], capacity=10)
    assert sorted(map(sorted, routes)) == [[0, 0, 1, 2], [0, 0, 3]]


def test_savings_respects_capacity_and_logs_decisions():
    pos = np.array([0, 5, 6])
    d = np.abs(pos[:, None] - pos[None, :]).astype(float)
    log = []
    routes = C.clarke_wright(d, [0, 6, 6], capacity=10, log=log)
    assert len(routes) == 2 and log == [(10.0, 1, 2, "capacity")]


def test_every_heuristic_is_feasible_and_above_the_bound():
    inst = chicago_synthetic(60, seed=3)
    d = distance_matrix(inst.sites)
    q, Q = inst.demands, inst.capacity
    xs = np.array([s.lon for s in inst.sites])
    ys = np.array([s.lat for s in inst.sites])
    xy = np.column_stack([xs, ys])
    lb = C.radial_lower_bound(d, q, Q)
    plans = [C.nearest_neighbor(d, q, Q), C.clarke_wright(d, q, Q),
             C.clarke_wright_sequential(d, q, Q), C.sweep(d, q, Q, xy),
             C.insertion(d, q, Q, "cheapest"), C.insertion(d, q, Q, "regret")]
    for routes in plans:
        validate_routes(routes, d, q, Q)
        assert total_distance(routes, d) >= lb - len(q)     # integer rounding


def test_radial_bound_is_tight_for_one_customer_per_route():
    # every customer fills a van: the optimum is the sum of round trips
    pos = np.array([0, 3, -4, 7])
    d = np.abs(pos[:, None] - pos[None, :]).astype(float)
    assert C.radial_lower_bound(d, [0, 10, 10, 10], 10) == 2 * (3 + 4 + 7)


def test_constructions_refuse_a_customer_larger_than_a_vehicle():
    import pytest
    d = np.array([[0, 5], [5, 0]])
    for build in (C.nearest_neighbor, C.clarke_wright, C.clarke_wright_sequential,
                  C.insertion):
        with pytest.raises(C.InfeasibleInstance):
            build(d, [0, 2], 1)
    with pytest.raises(C.InfeasibleInstance):
        C.sweep(d, [0, 2], 1, np.array([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError):
        C.nearest_neighbor(d, [0, 1], 0)                 # no capacity at all
