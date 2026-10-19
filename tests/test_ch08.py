"""Tests for Chapter 8: pickup and delivery."""
import itertools
import math

from vrpbook.pdp import build, check_pdp, solve_pdp


def tiny():
    rest = [(29.75, -95.37), (29.76, -95.39)]
    couriers = [(29.75, -95.38)]
    t = 11 * 3600 + 900
    orders = [(0, 29.74, -95.36, t, 1), (1, 29.77, -95.40, t, 2)]
    return build(couriers, orders, rest, max_ride_min=40)


def test_pairs_precedence_and_same_courier():
    d = tiny()
    plan = solve_pdp(d, time_limit_s=2)
    chk = check_pdp(plan, d, 6, 3)
    assert chk["violations"] == []
    route = next(r for r in plan.routes if len(r) > 2)
    for p, q in d.pairs:
        assert route.index(p) < route.index(q)


def test_one_order_at_a_time_means_no_batching():
    d = tiny()
    plan = solve_pdp(d, max_orders=1, time_limit_s=2)
    assert check_pdp(plan, d, 6, 1)["batched"] == 0


def test_count_of_precedence_feasible_sequences():
    for n in range(1, 5):
        nodes = [(k, s) for k in range(n) for s in "PD"]
        ok = sum(1 for perm in itertools.permutations(nodes)
                 if all(perm.index((k, "P")) < perm.index((k, "D")) for k in range(n)))
        assert ok == math.factorial(2 * n) // 2 ** n
