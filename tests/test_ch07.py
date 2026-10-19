"""Tests for Chapter 7: fleets, shifts, breaks, compatibility."""
import numpy as np

from vrpbook.fleet import BreakRule, VehicleType, check_plan, expand, solve_rich, true_cost
from vrpbook.timewindows import HOUR, TWData

H = lambda h: int(h * HOUR)  # noqa: E731


def toy():
    pos = np.array([0, 1, 2, 3, 4])
    t = 900 * np.abs(pos[:, None] - pos[None, :])
    d = 8000 * np.abs(pos[:, None] - pos[None, :])
    windows = [(H(6), H(18))] + [(H(8), H(16))] * 4
    return TWData(d, t, [0, 600, 600, 600, 600], windows, [0, 10, 10, 10, 10], 100, 2)


FLEET = [VehicleType("Van", 1, 30, 1000, 1.0, 0, 0, (H(6), H(18))),
         VehicleType("Fridge", 1, 30, 1000, 2.0, 0, 0, (H(6), H(18)), refrigerated=True)]


def test_cold_orders_go_in_the_fridge():
    data = toy()
    cold = [False, False, True, False, True]
    plan = solve_rich(data, expand(FLEET), [0, 1, 1, 1, 1], cold=cold, time_limit_s=2)
    chk = check_plan(plan, data, [0, 1, 1, 1, 1], cold)
    assert chk["violations"] == []
    fridge_route = plan.routes[1]
    assert 2 in fridge_route and 4 in fridge_route


def test_break_is_scheduled_in_its_window():
    data = toy()
    rule = BreakRule(minutes=30, earliest_after_h=3, latest_after_h=5, min_shift_h=7.5)
    plan = solve_rich(data, expand(FLEET), [0, 1, 1, 1, 1], breaks=rule, time_limit_s=2)
    chk = check_plan(plan, data, [0, 1, 1, 1, 1], None, rule)
    assert chk["violations"] == []
    for (r, b) in zip(plan.routes, plan.breaks):
        if len(r) > 2:
            assert H(9) <= b[0] <= H(11)


def test_salaried_time_is_not_charged():
    data = toy()
    big = [VehicleType("Van", 1, 100, 1000, 1.0, 0, 0, (H(6), H(18)))]
    plan = solve_rich(data, big, [0, 1, 1, 1, 1], time_limit_s=2)
    cost = true_cost(plan, data)
    assert cost["total"] == cost["vehicle"]            # no rental, no hourly pay
    assert abs(cost["miles"] - 8 * 8000 / 1609.344) < 1e-6
