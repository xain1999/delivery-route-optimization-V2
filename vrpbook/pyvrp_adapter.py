"""Solving the book's problems with PyVRP (Chapter 12).

PyVRP works in integers, so costs are scaled to thousandths of a cent,
the same units as the OR-Tools models of Chapter 6.
"""
from __future__ import annotations

import numpy as np
import pyvrp
from pyvrp.stop import MaxRuntime

from .timewindows import TWData

UNITS = 100_000                     # cost units per dollar
M_PER_MI = 1609.344


def _routes(result, clients) -> list[list[int]]:
    """PyVRP routes as node lists [0, i, j, ..., 0]. Client k is node k + 1."""
    return [[0, *[clients[a.idx].location for a in r.schedule() if a.is_client()], 0]
            for r in result.best.routes()]


def solve_cvrp_pyvrp(dist, demands, capacity: int, seconds: float, seed: int = 1):
    n = len(demands)
    locs = [pyvrp.Location(x=0, y=0) for _ in range(n)]
    clients = [pyvrp.Client(location=i, delivery=[int(demands[i])]) for i in range(1, n)]
    data = pyvrp.ProblemData(locs, clients, [pyvrp.Depot(location=0)],
                             [pyvrp.VehicleType(num_available=n, capacity=[capacity])],
                             [np.asarray(dist, dtype=np.int64)],
                             [np.zeros((n, n), dtype=np.int64)])
    res = pyvrp.solve(data, stop=MaxRuntime(seconds), seed=seed)
    return _routes(res, clients), res.num_iterations


# <<pyvrp_vrptw>>
def solve_vrptw_pyvrp(data: TWData, seconds: float, *, driver_per_hour: float,
                      vehicle_per_mile: float, fixed_per_van: float, seed: int = 1):
    """The time-window problem of Chapter 6 with PyVRP's modeling interface."""
    m = pyvrp.Model()
    n = len(data.demands)
    weights, payload = data.extra["Weight"] if data.extra else ([0] * n, 0)
    locs = [m.add_location(x=0, y=0) for _ in range(n)]
    a0, b0 = data.windows[0]
    depot = m.add_depot(locs[0], tw_early=a0, tw_late=b0)
    m.add_vehicle_type(
        num_available=data.n_vehicles,
        capacity=[data.capacity, payload],
        tw_early=a0, tw_late=b0,
        fixed_cost=round(fixed_per_van * UNITS),
        unit_distance_cost=round(vehicle_per_mile * UNITS / M_PER_MI),   # per meter
        unit_duration_cost=round(driver_per_hour * UNITS / 3600),        # per second paid
        start_depot=depot, end_depot=depot)
    for i in range(1, n):
        a, b = data.windows[i]
        m.add_client(locs[i], delivery=[int(data.demands[i]), int(weights[i])],
                     service_duration=int(data.service[i]), tw_early=a, tw_late=b)
    for i in range(n):
        for j in range(n):
            if i != j:
                m.add_edge(locs[i], locs[j], distance=int(data.dist[i, j]),
                           duration=int(data.time[i, j]))
    res = m.solve(stop=MaxRuntime(seconds), seed=seed, display=False)
    clients = m.data().clients()
    return _routes(res, clients), res
# <<end_pyvrp_vrptw>>
