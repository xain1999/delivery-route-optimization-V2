"""Tests for the Chapter 5 formulations.  Run with:  pytest -q"""
import itertools
import pytest

from vrpbook import chicago_ch02, distance_matrix
from vrpbook.exact_tsp import solve_tsp_exact
from vrpbook.formulations import (cvrp_set_partitioning, cvrp_two_index, enumerate_routes,
                                  tsp_dfj, tsp_mtz, tsp_scf)


@pytest.fixture(scope="module")
def small():
    inst = chicago_ch02()
    sub = inst.subset([1, 5, 9, 13, 17, 21, 25])
    return distance_matrix(sub.sites), sub.demands


def test_three_tsp_models_agree(small):
    d, _ = small
    opt = solve_tsp_exact(d)[1]
    for f in (tsp_dfj, tsp_mtz, tsp_scf):
        assert f(d).objective == pytest.approx(opt)


def test_lp_relaxations_are_lower_bounds_and_ordered(small):
    d, _ = small
    opt = solve_tsp_exact(d)[1]
    dfj, scf, mtz = (f(d, relax=True).objective for f in (tsp_dfj, tsp_scf, tsp_mtz))
    assert max(dfj, scf, mtz) <= opt + 1e-6
    assert dfj >= mtz - 1e-6 and dfj >= scf - 1e-6


def test_enumerated_routes_are_optimal_tours(small):
    d, q = small
    Q = 40
    for S, cost, order in enumerate_routes(d, q, Q):
        assert sum(q[i] for i in S) <= Q
        assert order[0] == order[-1] == 0 and set(order[1:-1]) == set(S)
        best = min(sum(int(d[a, b]) for a, b in zip((0,) + p, p + (0,)))
                   for p in itertools.permutations(S))
        assert cost == best


def test_two_index_matches_set_partitioning(small):
    d, q = small
    Q, m = 40, 4
    sp = cvrp_set_partitioning(enumerate_routes(d, q, Q), len(d), m)
    ip = cvrp_two_index(d, q, Q, m)
    assert ip.objective == pytest.approx(sp.objective)
    lp = cvrp_two_index(d, q, Q, m, relax=True)
    assert lp.objective <= sp.objective + 1e-6
