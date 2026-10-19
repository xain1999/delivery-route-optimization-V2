"""Tests for the Chapter 2 code.  Run with:  pytest -q"""
import math

import numpy as np
import pytest

from vrpbook import (chicago_ch02, distance_matrix, haversine_m, solve_cvrp,
                     total_distance, validate_routes)
from vrpbook.distances import triangle_violations
from vrpbook.validate import InvalidPlan
from vrpbook.exact_tsp import solve_tsp_exact


@pytest.fixture(scope="module")
def data():
    inst = chicago_ch02()
    return inst, distance_matrix(inst.sites)


def test_haversine_known_value():
    # One degree of latitude along a meridian is about 111.2 km.
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-4)


def test_instance_shape(data):
    inst, dist = data
    assert inst.n_customers == 30
    assert inst.total_demand == 358
    assert dist.shape == (31, 31)
    assert np.all(dist == dist.T) and np.all(np.diag(dist) == 0)


def test_triangle_inequality(data):
    _, dist = data
    assert triangle_violations(dist) == 0


def test_cvrp_solution_is_feasible(data):
    inst, dist = data
    sol = solve_cvrp(dist, inst.demands, inst.n_vehicles, inst.capacity,
                     time_limit_s=2)
    stats = validate_routes(sol.routes, dist, inst.demands, inst.capacity)
    assert sum(s.load for s in stats) == inst.total_demand
    assert total_distance(sol.routes, dist) == sol.objective


def test_validator_catches_overload(data):
    inst, dist = data
    one_route = [[0] + list(range(1, 31)) + [0]]
    with pytest.raises(InvalidPlan, match="carries"):
        validate_routes(one_route, dist, inst.demands, inst.capacity)


def test_validator_catches_missing_customer(data):
    inst, dist = data
    with pytest.raises(InvalidPlan, match="never visited"):
        validate_routes([[0, 1, 0]], dist, inst.demands, inst.capacity)


def test_exact_tsp_matches_brute_force(data):
    inst, _ = data
    sub = inst.subset([6, 13, 21, 7])
    d = distance_matrix(sub.sites)
    import itertools
    def length_of(perm):
        route = (0,) + perm + (0,)
        return sum(int(d[a, b]) for a, b in zip(route, route[1:]))

    brute = min(length_of(p) for p in itertools.permutations(range(1, 5)))
    _, length, _ = solve_tsp_exact(d)
    assert length == brute
    assert math.factorial(4) // 2 == 12


def test_cost_model_per_mile():
    from vrpbook.costs import CostModel
    m = CostModel(driver_cost_per_hour=30.0, vehicle_cost_per_mile=0.75,
                  service_minutes_per_stop=6.0, average_speed_mph=15.0)
    assert m.cost_per_mile_driven == pytest.approx(0.75 + 2.0)
    assert m.route_hours(15.0, 10) == pytest.approx(1.0 + 1.0)
    driver, vehicle = m.route_cost(15.0, 10)
    assert driver == pytest.approx(60.0) and vehicle == pytest.approx(11.25)


def test_validator_rejects_invalid_indices_and_extra_routes(data):
    inst, dist = data
    n = len(inst.demands)
    good = [[0, *range(1, n), 0]]
    with pytest.raises(InvalidPlan, match="invalid stops"):
        validate_routes([[0, *range(1, n), -1, 0]], dist, inst.demands, 10_000)
    with pytest.raises(InvalidPlan, match="routes for"):
        validate_routes([[0, 1, 0], [0, *range(2, n), 0]], dist, inst.demands, 10_000,
                        n_vehicles=1)
    validate_routes(good, dist, inst.demands, 10_000, n_vehicles=1)


def test_the_printed_first_program_runs_on_its_own(tmp_path):
    """Listing 2.4 prints examples/ch02_first_route.py from line 8 (the book's
    firstline=8): that part alone, imports included, must run."""
    import re
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    tex = (root.parent / "book" / "chapters" / "ch02-first-optimizer.tex")
    first = 8
    if tex.exists():                       # in the book's sources, read the real value
        first = int(re.search(r"lstinputlisting\[firstline=(\d+),", tex.read_text()).group(1))
    lines = (root / "examples" / "ch02_first_route.py").read_text().splitlines()
    program = tmp_path / "listing.py"
    program.write_text("\n".join(lines[first - 1:]) + "\n")
    out = subprocess.run([sys.executable, str(program)], capture_output=True, text=True,
                         timeout=120, cwd=root)
    assert out.returncode == 0, out.stderr
    assert "Length:" in out.stdout
