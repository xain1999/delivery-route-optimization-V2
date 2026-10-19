"""Solve capacitated vehicle routing problems with Google OR-Tools.

This module is introduced in Chapter 2.  It deliberately keeps the
OR-Tools "plumbing" in one place so that later chapters can add time
windows, shifts and optional stops without rewriting it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

FIRST_SOLUTION = routing_enums_pb2.FirstSolutionStrategy
METAHEURISTIC = routing_enums_pb2.LocalSearchMetaheuristic


@dataclass
class Solution:
    """A solver result in plain Python types (independent of OR-Tools)."""

    routes: list[list[int]]          # one list per vehicle: [0, i, j, ..., 0]
    objective: int                   # value reported by the solver
    status: str
    wall_time_s: float
    history: list[tuple[float, int]] = field(default_factory=list)

    @property
    def used_routes(self) -> list[list[int]]:
        return [r for r in self.routes if len(r) > 2]


def solve_cvrp(
    dist: np.ndarray,
    demands: list[int],
    n_vehicles: int,
    capacity: int,
    *,
    first_solution: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str = "GUIDED_LOCAL_SEARCH",
    time_limit_s: float = 10.0,
    span_cost_coefficient: int = 0,
    track_progress: bool = False,
    extra_capacities: dict[str, tuple[list[int], int]] | None = None,
) -> Solution:
    """Minimize total distance subject to vehicle capacity.

    dist: square integer matrix, node 0 is the depot.
    span_cost_coefficient: if > 0, also penalize the longest route,
        which spreads work more evenly across drivers.
    extra_capacities: further capacity dimensions, e.g.
        {"Weight": (weights, payload)}. Each must hold on every route.
    """
    # <<model>>
    n = len(dist)
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)  # depot = node 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(dist[i, j])

    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    # <<end_model>>

    # <<capacity>>
    def demand_cb(from_index: int) -> int:
        return int(demands[manager.IndexToNode(from_index)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,                        # no slack: load changes only at customers
        [capacity] * n_vehicles,  # one capacity per vehicle
        True,                     # start every van's load count at zero
        "Load",
    )
    # <<end_capacity>>

    # <<extra_dims>>
    for dim_name, (dim_demands, dim_capacity) in (extra_capacities or {}).items():
        def dim_cb(from_index: int, _q=dim_demands) -> int:
            return int(_q[manager.IndexToNode(from_index)])

        idx = routing.RegisterUnaryTransitCallback(dim_cb)
        routing.AddDimensionWithVehicleCapacity(
            idx, 0, [dim_capacity] * n_vehicles, True, dim_name)
    # <<end_extra_dims>>

    # <<span>>
    if span_cost_coefficient > 0:
        routing.AddDimension(transit, 0, int(dist.sum()), True, "Distance")
        routing.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(
            span_cost_coefficient)
    # <<end_span>>

    history: list[tuple[float, int]] = []
    t0 = time.perf_counter()
    if track_progress:
        def on_solution() -> None:
            history.append((time.perf_counter() - t0, routing.CostVar().Value()))
        routing.AddAtSolutionCallback(on_solution)

    # <<search>>
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(FIRST_SOLUTION, first_solution)
    params.local_search_metaheuristic = getattr(METAHEURISTIC, metaheuristic)
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    assignment = routing.SolveWithParameters(params)
    # <<end_search>>
    wall = time.perf_counter() - t0
    status = _status_name(routing.status())
    if assignment is None:
        return Solution([], -1, status, wall, history)

    # <<extract>>
    routes = []
    for v in range(n_vehicles):
        index = routing.Start(v)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = assignment.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    # <<end_extract>>
    return Solution(routes, assignment.ObjectiveValue(), status, wall, history)


def _status_name(code: int) -> str:
    try:
        return routing_enums_pb2.RoutingSearchStatus.Value.Name(code)
    except ValueError:
        return str(code)
