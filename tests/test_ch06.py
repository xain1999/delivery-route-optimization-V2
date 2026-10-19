"""Tests for Chapter 6: time windows.  Run with:  pytest -q"""
import numpy as np
import pytest

from vrpbook.formulations import vrptw_mip
from vrpbook.timewindows import (HOUR, TWData, best_departure, schedule, service_seconds,
                                 slot_windows, solve_vrptw, validate_schedule)


def toy() -> TWData:
    # depot 0 and three customers on a line, 10 minutes apart
    pos = np.array([0, 1, 2, 3])
    t = 600 * np.abs(pos[:, None] - pos[None, :])
    d = 5000 * np.abs(pos[:, None] - pos[None, :])
    windows = [(6 * HOUR, 16 * HOUR), (9 * HOUR, 10 * HOUR),
               (9 * HOUR, 11 * HOUR), (8 * HOUR, 12 * HOUR)]
    return TWData(d, t, [0, 300, 300, 300], windows, [0, 1, 1, 1], 10, 2)


def test_service_time_model():
    assert service_seconds(0) == 180
    assert service_seconds(20) == 180 + 20 * 24


def test_slots_tile_the_day():
    assert slot_windows([7, 8, 9, 14], 2) == [(7 * HOUR, 9 * HOUR), (7 * HOUR, 9 * HOUR),
                                              (9 * HOUR, 11 * HOUR), (13 * HOUR, 15 * HOUR)]
    assert slot_windows([10], 8) == [(7 * HOUR, 15 * HOUR)]


def test_best_departure_removes_waiting():
    data = toy()
    route = [0, 1, 2, 3, 0]
    early = schedule(route, data, depart=6 * HOUR)
    assert early[1].wait == 9 * HOUR - (6 * HOUR + 600)          # waits for 9:00
    best = best_departure(route, data)
    assert best == 9 * HOUR - 600                                  # arrive exactly at 9:00
    vis = schedule(route, data)
    assert sum(v.wait for v in vis) == 0


def test_solver_respects_windows():
    data = toy()
    sol = solve_vrptw(data, objective="distance", time_limit_s=2)
    totals = validate_schedule(sol.routes, data)
    assert totals["meters"] == 30000                              # one out-and-back trip


def test_mip_matches_solver_on_toy():
    data = toy()
    ip = vrptw_mip(data.dist, data.time, data.service, data.windows, data.demands, 10, 2)
    assert ip.objective == pytest.approx(30000)


def test_tight_windows_force_order():
    data = toy()
    # customer 3 must be served before 8:30 and customer 1 after 9:00
    data.windows[3] = (8 * HOUR, 8 * HOUR + 1800)
    sol = solve_vrptw(data, objective="distance", time_limit_s=2)
    validate_schedule(sol.routes, data)
    route = next(r for r in sol.routes if len(r) > 2)
    assert route.index(3) < route.index(1)
