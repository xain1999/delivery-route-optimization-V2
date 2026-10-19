"""Time windows, service times and schedules (Chapter 6).

Times are integer seconds after midnight.  A schedule is computed from
a route and the data alone, so it doubles as an independent validator of
any solver's output.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .ortools_solver import Solution, _status_name

HOUR = 3600


def hhmm(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:d}:{(t % 3600) // 60:02d}"


# <<service_time>>
def service_seconds(totes: int, fixed_min: float = 3.0, per_tote_min: float = 0.4) -> int:
    """Time at a stop: park and walk to the door, then carry the totes."""
    return round(60 * (fixed_min + per_tote_min * totes))
# <<end_service_time>>


# <<slots>>
def slot_windows(preferred_hour: list[int], width_h: int, day=(7, 15)) -> list[tuple[int, int]]:
    """Map each customer's preferred hour to the slot of the given width
    that contains it.  Slots tile the delivery day: with width 2 they are
    7-9, 9-11, 11-13 and 13-15."""
    start, end = day
    out = []
    for h in preferred_hour:
        k = (h - start) // width_h
        a = start + k * width_h
        out.append((a * HOUR, min(a + width_h, end) * HOUR))
    return out
# <<end_slots>>


def load_preferences(path: str | Path) -> dict[int, int]:
    with Path(path).open(encoding="utf-8") as fh:
        return {int(r["id"]): int(r["preferred_hour"]) for r in csv.DictReader(fh)}


@dataclass
class TWData:
    """Everything a time-window solver needs, in seconds and meters."""
    dist: np.ndarray          # meters
    time: np.ndarray          # travel seconds
    service: list[int]        # seconds at each node (0 at the depot)
    windows: list[tuple[int, int]]   # [a_i, b_i], windows[0] = depot opening hours
    demands: list[int]
    capacity: int
    n_vehicles: int
    extra: dict | None = None  # further capacities, as in solve_cvrp


# <<schedule>>
@dataclass
class Visit:
    node: int
    arrival: int
    start: int        # max(arrival, window opens)
    departure: int    # start + service

    @property
    def wait(self) -> int:
        return self.start - self.arrival


def schedule(route: list[int], data: TWData, depart: int | None = None) -> list[Visit]:
    """Schedule of a route that starts every visit as early as possible
    after leaving the depot at `depart` (default: best_departure)."""
    if depart is None:
        depart = best_departure(route, data)
    visits = [Visit(route[0], depart, depart, depart)]
    t = depart
    for i, j in zip(route, route[1:]):
        arrive = t + int(data.time[i, j])
        start = max(arrive, data.windows[j][0])
        t = start + data.service[j]
        visits.append(Visit(j, arrive, start, t))
    return visits


def best_departure(route: list[int], data: TWData) -> int:
    """Depot departure that minimizes the route's duration.

    Start from the earliest schedule (leave when the depot opens).
    Delaying the departure by d removes up to d of waiting, but stop p
    then starts max(0, d - W_p) later, where W_p is the waiting before
    and at p.  The forward time slack F = min_p (W_p + b_p - S_p) is the
    largest feasible delay, and delaying beyond the total waiting W saves
    nothing, so the best delay is min(F, W).
    """
    vis = schedule(route, data, depart=data.windows[route[0]][0])
    waited, slack = 0, float("inf")
    for v in vis[1:]:
        waited += v.wait
        slack = min(slack, waited + data.windows[v.node][1] - v.start)
    return vis[0].departure + int(max(0, min(slack, waited)))
# <<end_schedule>>


def validate_schedule(routes: list[list[int]], data: TWData,
                      depots: tuple[int, ...] = (0,), allow_dropped: bool = False,
                      max_late: int = 0) -> dict:
    """Check windows, capacity and coverage. Return totals in seconds.

    With allow_dropped, customers may be missing (Chapter 9) but none may
    be visited twice, totals["dropped"] lists the missing ones.  With
    max_late (seconds), service may start that long after a window closes,
    totals["late"] sums the lateness and totals["n_late"] counts late stops."""
    seen = []
    totals = {"drive": 0, "service": 0, "wait": 0, "duration": 0, "meters": 0,
              "late": 0, "n_late": 0}
    for r in routes:
        if len(r) <= 2:
            continue
        vis = schedule(r, data, depart=_first_departure(r, data))
        for v in vis[1:]:
            a, b = data.windows[v.node]
            assert v.start <= b + max_late, (
                f"node {v.node} starts at {hhmm(v.start)} after {hhmm(b)}")
            if v.start > b:
                totals["late"] += v.start - b
                totals["n_late"] += 1
        assert sum(data.demands[i] for i in r[1:-1]) <= data.capacity
        seen += r[1:-1]
        totals["drive"] += sum(int(data.time[i, j]) for i, j in zip(r, r[1:]))
        totals["service"] += sum(data.service[i] for i in r[1:-1])
        totals["wait"] += sum(v.wait for v in vis[1:])
        totals["duration"] += vis[-1].arrival - vis[0].departure
        totals["meters"] += sum(int(data.dist[i, j]) for i, j in zip(r, r[1:]))
    expected = [i for i in range(len(data.demands)) if i not in depots]
    if allow_dropped:
        assert len(seen) == len(set(seen)) and set(seen) <= set(expected), "coverage"
    else:
        assert sorted(seen) == expected, "coverage"
    totals["dropped"] = sorted(set(expected) - set(seen))
    return totals


def _first_departure(route, data):
    return best_departure(route, data)


# <<precheck>>
def precheck(data: TWData) -> dict:
    """Necessary conditions that catch most impossible inputs early."""
    a0, b0 = data.windows[0]
    unreachable, cannot_return = [], []
    n = len(data.demands)
    for i in range(1, n):
        a, b = data.windows[i]
        earliest = max(a, a0 + int(data.time[0, i]))
        if earliest > b:
            unreachable.append(i)
        elif earliest + data.service[i] + int(data.time[i, 0]) > b0:
            cannot_return.append(i)
    # arcs that no feasible route can use: leaving i as early as possible,
    # j's window has already closed
    blocked = sum(1 for i in range(1, n) for j in range(1, n) if i != j and
                  data.windows[i][0] + data.service[i] + int(data.time[i, j]) > data.windows[j][1])
    return {"unreachable": unreachable, "cannot_return": cannot_return,
            "blocked_arcs": blocked, "arcs": (n - 1) * (n - 2)}
# <<end_precheck>>


def solve_vrptw(data: TWData, objective: str = "cost", time_limit_s: float = 10.0,
                driver_per_hour: float = 30.81, vehicle_per_mile: float = 0.76,
                fixed_per_van: float = 0.0, max_wait_s: int | None = None,
                first_solution: str = "PARALLEL_CHEAPEST_INSERTION",
                starts: list[int] | None = None, ends: list[int] | None = None,
                penalties: list[float | None] | None = None,
                initial_routes: list[list[int]] | None = None,
                min_served: int | None = None, late_per_min: float = 0.0,
                max_late_min: int = 0) -> Solution:
    """Vehicle routing with time windows in OR-Tools.

    objective: "distance" (meters), "time" (travel seconds) or "cost"
    (vehicle cost per mile plus driver cost for every paid second, i.e.
    for the whole duration of each route, including waiting).
    max_wait_s: longest wait allowed at any one stop (slack of the time
    dimension). By default the whole working day.
    penalties: cost of leaving each node unserved (dollars with the "cost"
    objective, objective units otherwise). None for a node that must be
    served.  Without penalties every customer must be served (Chapter 9).
    initial_routes: a plan to start the search from (customer lists per
    vehicle, depots excluded), instead of building a first solution.
    min_served: serve at least this many customers (with penalties).
    late_per_min, max_late_min: soft windows. Service may start up to
    max_late_min after a window closes, at late_per_min dollars a minute.
    """
    n = len(data.demands)
    hard_windows = data.windows
    if max_late_min > 0:                     # soft windows: widen here, charge below
        fixed = set(starts or [0]) | set(ends or [])
        data = replace(data, windows=[w if i in fixed else (w[0], w[1] + 60 * max_late_min)
                                      for i, w in enumerate(data.windows)])
    if starts is None:                       # one depot, node 0
        manager = pywrapcp.RoutingIndexManager(n, data.n_vehicles, 0)
    else:                                    # each vehicle has its own depot
        manager = pywrapcp.RoutingIndexManager(n, data.n_vehicles, starts, ends or starts)
    depot_nodes = set(starts or [0]) | set(ends or [])
    routing = pywrapcp.RoutingModel(manager)

    # <<time_dimension>>
    def time_cb(from_index: int, to_index: int) -> int:
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return data.service[i] + int(data.time[i, j])   # serve i, then drive

    time_idx = routing.RegisterTransitCallback(time_cb)
    a0, b0 = data.windows[0]
    if max_wait_s is None:                 # waiting is limited only by the day
        max_wait_s = b0 - a0
    routing.AddDimension(time_idx, max_wait_s, b0, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(n):
        if node in depot_nodes:
            continue
        a, b = data.windows[node]
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(a, b)
    for v in range(data.n_vehicles):
        time_dim.CumulVar(routing.Start(v)).SetRange(a0, b0)
        time_dim.CumulVar(routing.End(v)).SetRange(a0, b0)
    # <<end_time_dimension>>

    # <<objective>>
    if objective == "distance":
        cost_idx = routing.RegisterTransitCallback(
            lambda f, t: int(data.dist[manager.IndexToNode(f), manager.IndexToNode(t)]))
        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)
    elif objective == "time":
        routing.SetArcCostEvaluatorOfAllVehicles(time_idx)
    else:  # "cost", in thousandths of a cent
        per_meter = vehicle_per_mile * 100_000 / 1609.344
        cost_idx = routing.RegisterTransitCallback(
            lambda f, t: round(per_meter * data.dist[manager.IndexToNode(f),
                                                     manager.IndexToNode(t)]))
        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)
        # every second between leaving and returning to the depot is paid
        time_dim.SetSpanCostCoefficientForAllVehicles(round(driver_per_hour * 100_000 / 3600))
        routing.SetFixedCostOfAllVehicles(round(fixed_per_van * 100_000))
    # <<end_objective>>

    # <<soft_windows>>
    if max_late_min > 0:          # the windows above were widened by max_late_min
        scale = 100_000 if objective == "cost" else 1
        per_s = round(late_per_min * scale / 60)             # cost of a second late
        for node in range(n):
            if node not in depot_nodes:
                time_dim.SetCumulVarSoftUpperBound(manager.NodeToIndex(node),
                                                   hard_windows[node][1], per_s)
    # <<end_soft_windows>>

    def demand_cb(from_index: int) -> int:
        return int(data.demands[manager.IndexToNode(from_index)])
    routing.AddDimensionWithVehicleCapacity(routing.RegisterUnaryTransitCallback(demand_cb),
                                            0, [data.capacity] * data.n_vehicles, True, "Load")
    for name, (dem, cap) in (data.extra or {}).items():
        idx = routing.RegisterUnaryTransitCallback(
            lambda f, _d=dem: int(_d[manager.IndexToNode(f)]))
        routing.AddDimensionWithVehicleCapacity(idx, 0, [cap] * data.n_vehicles, True, name)

    # <<disjunctions>>
    if penalties is not None:
        scale = 100_000 if objective == "cost" else 1     # same units as the costs
        for node, p in enumerate(penalties):
            if node in depot_nodes or p is None:
                continue                                  # None: must be served
            routing.AddDisjunction([manager.NodeToIndex(node)], round(p * scale))
    # <<end_disjunctions>>
    # <<min_served>>
    if min_served is not None:
        solver = routing.solver()
        active = [routing.ActiveVar(manager.NodeToIndex(i))
                  for i in range(n) if i not in depot_nodes]
        solver.Add(solver.Sum(active) >= min_served)
    # <<end_min_served>>

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    if initial_routes is not None:
        routing.CloseModelWithParameters(params)
        start = routing.ReadAssignmentFromRoutes(initial_routes, True)
        sol = routing.SolveFromAssignmentWithParameters(start, params)
    else:
        sol = routing.SolveWithParameters(params)
    status = _status_name(routing.status())
    if sol is None:
        return Solution([], -1, status, time_limit_s)
    routes = []
    for v in range(data.n_vehicles):
        idx, r = routing.Start(v), []
        while not routing.IsEnd(idx):
            r.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        routes.append(r + [manager.IndexToNode(idx)])
    return Solution(routes, sol.ObjectiveValue(), status, time_limit_s)
