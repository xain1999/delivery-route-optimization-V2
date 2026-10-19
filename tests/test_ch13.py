"""Tests for Chapter 13: bounds and exact solutions on a classic benchmark."""
import math
from pathlib import Path

import numpy as np

from vrpbook.bounds import capacity_cuts, exact_cvrp, lp_bound, qroute_bound, read_vrplib
from vrpbook.construction import radial_lower_bound

E22 = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "E-n22-k4.vrp"


def test_e_n22_k4_is_solved_to_its_known_optimum():
    c, q, Q = read_vrplib(E22)
    lp = lp_bound(c, q, Q)
    assert abs(lp["bound"] - 375) < 1e-6          # the LP bound is already tight here
    ex = exact_cvrp(c, q, Q, 4, root_cuts=lp["cut_sets"], time_limit_s=120)
    assert ex["status"] == "optimal" and round(ex["value"]) == 375


def test_bounds_are_ordered():
    c, q, Q = read_vrplib(E22)
    cg = qroute_bound(c, q, Q)["bound"]
    assert radial_lower_bound(c, q, Q) <= cg + 1e-6 <= 375 + 1e-6


def test_unfinished_column_generation_is_not_reported_as_a_bound():
    # two customers, one van of capacity 2: the optimum is 0-1-2-0 = 21
    c = np.array([[0, 10, 10], [10, 0, 1], [10, 1, 0]])
    early = qroute_bound(c, [0, 1, 1], 2, max_iter=1)
    assert not early["converged"] and early["bound"] is None
    assert early["master_value"] == 40          # restricted master: above the optimum
    done = qroute_bound(c, [0, 1, 1], 2)
    assert done["converged"] and abs(done["bound"] - 21) < 1e-6


def test_capacity_cuts_never_cut_off_a_feasible_plan():
    c, q, Q = read_vrplib(E22)
    lp = lp_bound(c, q, Q)
    # a feasible integer plan: one route per customer is always feasible
    plan_edges = {(0, i): 2 for i in range(1, len(q))}
    for S in lp["cut_sets"]:
        inside = sum(v for (i, j), v in plan_edges.items() if i in S and j in S)
        assert inside <= len(S) - math.ceil(sum(q[k] for k in S) / Q)
    # and the separation finds nothing to cut on that plan's x
    assert capacity_cuts({e: float(v) for e, v in plan_edges.items()}, len(q), q, Q) == []
