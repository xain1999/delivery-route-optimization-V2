"""Tests for Chapter 9: optional orders, penalties and orienteering."""
from itertools import permutations

import numpy as np

from vrpbook.optional import (PenaltyPolicy, orienteering_mip, removal_savings,
                              solve_orienteering)
from vrpbook.timewindows import HOUR, TWData, solve_vrptw, validate_schedule


def test_policy_penalty():
    pol = PenaltyPolicy()
    p = pol.penalty(400, "regular", False)
    assert abs(p - (15 + 0.25 * 0.30 * 400 + 0.08 * 1000)) < 1e-9
    assert abs(pol.penalty(400, "regular", True) - 3 * p) < 1e-9
    assert pol.penalty(400, "club", False) > p


def test_removal_savings_on_a_line():
    pos = np.array([0.0, 2.0, 3.0, 6.0])
    cost = np.abs(pos[:, None] - pos[None, :])
    s = removal_savings([0, 1, 2, 3, 0], cost)
    assert s == {1: 0.0, 2: 0.0, 3: 6.0}


def _line_data():
    pos = np.array([0, 1, 2, 3])
    d = 8000 * np.abs(pos[:, None] - pos[None, :])
    t = 600 * np.abs(pos[:, None] - pos[None, :])
    windows = [(0, 10 * HOUR)] * 4
    return TWData(d, t, [0, 300, 300, 300], windows, [0, 8, 8, 8], 16, 1)


def test_capacity_forces_the_cheapest_drop():
    data = _line_data()
    # equal penalties: dropping the far customer saves the most driving
    sol = solve_vrptw(data, objective="distance", time_limit_s=1,
                      penalties=[None, 10**6, 10**6, 10**6])
    assert validate_schedule(sol.routes, data, allow_dropped=True)["dropped"] == [3]
    # make the far customer precious: someone else goes instead
    sol = solve_vrptw(data, objective="distance", time_limit_s=1,
                      penalties=[None, 10**6, 10**6, 10**8])
    assert 3 not in validate_schedule(sol.routes, data, allow_dropped=True)["dropped"]


def _brute_force(time, service, prize, demand, budget, capacity):
    n, best = len(prize), 0.0
    for k in range(1, n):
        for perm in permutations(range(1, n), k):
            r = [0, *perm, 0]
            dur = sum(service[i] + time[i, j] for i, j in zip(r, r[1:]))
            if dur <= budget and sum(demand[i] for i in perm) <= capacity:
                best = max(best, sum(prize[i] for i in perm))
    return best


def test_orienteering_mip_and_ortools_are_optimal():
    rng = np.random.default_rng(9)
    xy = rng.uniform(0, 6000, size=(7, 2))
    time = np.rint(np.linalg.norm(xy[:, None] - xy[None, :], axis=2) / 6).astype(int)
    service = [0] + [300] * 6
    prize = [0.0] + list(rng.integers(10, 60, size=6).astype(float))
    demand = [0] + list(rng.integers(1, 5, size=6))
    budget, cap = 4000, 10
    opt = _brute_force(time, service, prize, demand, budget, cap)
    ex = orienteering_mip(time, service, prize, demand, budget, cap)
    assert ex["status"] == "optimal" and abs(ex["prize"] - opt) < 1e-6
    ot = solve_orienteering(time, service, prize, demand, budget, cap, time_limit_s=2)
    assert abs(ot["prize"] - opt) < 1e-6 and ot["duration"] <= budget


def test_soft_window_allows_a_priced_late_arrival():
    # two customers 30 minutes apart, both with a window that closes at 8:10.
    # serving both from one van means one of them is late
    d = np.array([[0, 1000, 1000], [1000, 0, 1000], [1000, 1000, 0]])
    t = np.array([[0, 600, 600], [600, 0, 1800], [600, 1800, 0]])
    windows = [(7 * HOUR, 12 * HOUR), (8 * HOUR, 8 * HOUR + 600), (8 * HOUR, 8 * HOUR + 600)]
    data = TWData(d, t, [0, 0, 0], windows, [0, 1, 1], 10, 1)
    hard = solve_vrptw(data, objective="distance", time_limit_s=1)
    assert hard.routes == []                       # impossible with hard windows
    soft = solve_vrptw(data, objective="distance", time_limit_s=1,
                       late_per_min=1.0, max_late_min=30)
    T = validate_schedule(soft.routes, data, max_late=30 * 60)
    assert T["n_late"] == 1 and T["late"] == 1200  # 20 minutes late


def test_orienteering_is_prize_first_then_time():
    # one visit fits, $1.01 far away beats $1.00 next door (the review's case)
    t = np.array([[0, 1, 1001], [1, 0, 1001], [1001, 1001, 0]])
    r = solve_orienteering(t, [0, 0, 0], [0, 1.00, 1.01], [0, 1, 1], 3000, 1, time_limit_s=1)
    assert r["route"] == [0, 2, 0] and r["prize"] == 1.01
