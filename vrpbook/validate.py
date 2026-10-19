"""Independent checking of routing solutions.

A validator should share as little code as possible with the solver it
checks.  This one uses only the instance data and the list of routes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RouteStats:
    vehicle: int
    stops: int
    load: int
    distance_m: int


# <<validate>>
class InvalidPlan(ValueError):
    """A plan that breaks a rule of the problem."""


def validate_routes(routes: list[list[int]], dist: np.ndarray, demands: list[int],
                    capacity: int, n_vehicles: int | None = None) -> list[RouteStats]:
    """Raise InvalidPlan if the routes are not a feasible CVRP solution.

    Checks: every route starts and ends at the depot, every stop is a valid
    customer index, every customer is visited exactly once, no route exceeds
    the vehicle capacity and, if given, no more than n_vehicles routes are
    used.  Explicit exceptions, not assert, so that `python -O` cannot
    switch the checks off.  Returns per-route statistics recomputed from
    scratch."""
    def require(ok: bool, msg: str) -> None:
        if not ok:
            raise InvalidPlan(msg)
    n = len(demands)
    seen: list[int] = []
    stats = []
    used = [r for r in routes if len(r) > 2]
    if n_vehicles is not None:
        require(len(used) <= n_vehicles,
                f"{len(used)} routes for {n_vehicles} vehicles")
    for v, route in enumerate(routes):
        require(len(route) >= 2 and route[0] == 0 and route[-1] == 0,
                f"route {v} must start and end at the depot")
        customers = route[1:-1]
        bad = [i for i in customers
               if not (isinstance(i, (int, np.integer)) and 1 <= i < n)]
        require(not bad, f"route {v} has invalid stops {bad}")
        load = sum(demands[i] for i in customers)
        require(load <= capacity, f"route {v} carries {load} > {capacity}")
        length = sum(int(dist[a, b]) for a, b in zip(route, route[1:]))
        seen.extend(customers)
        stats.append(RouteStats(v, len(customers), load, length))
    missing = set(range(1, n)) - set(seen)
    require(not missing, f"customers never visited: {sorted(missing)}")
    require(len(seen) == len(set(seen)), "some customer is visited twice")
    return stats
# <<end_validate>>


def total_distance(routes: list[list[int]], dist: np.ndarray) -> int:
    return sum(int(dist[a, b]) for r in routes for a, b in zip(r, r[1:]))
