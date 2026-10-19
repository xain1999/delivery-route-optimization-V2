"""From a request to a validated plan (Chapter 15).

Every stop is optional at a penalty equal to its priority (Chapter 9), and
a stop left out comes back with the reason, found by the same checks as
Chapter 6's precheck.  Optional stops make an empty plan feasible, but they
do not make a time-limited search return one: if the solver finds nothing,
even on a second and longer try, or if its plan fails validate_plan, the
service publishes no routes, every stop unassigned, fallback=True and a
warning that says why.  All arithmetic uses whole seconds and meters.
Rounding happens only when the response is written.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .models import PlanRequest, PlanResult, Route, Unassigned, Visit

log = logging.getLogger("routing_service")

UNITS = 100_000                       # cost units per dollar
RETRY_MIN_S = 2.0                     # the second try, if the first finds nothing
M_PER_MI = 1609.344
CIRCUITY = 1.3


def to_s(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 3600 + int(m) * 60


def to_hhmm(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def matrices(req: PlanRequest):
    """Haversine distance times a circuity factor (Chapter 2), vectorized,
    a road-network matrix (Chapter 4) can replace it without other changes."""
    pts = [(req.depot.lat, req.depot.lon)] + [(s.location.lat, s.location.lon) for s in req.stops]
    lat = np.radians([p[0] for p in pts])
    lon = np.radians([p[1] for p in pts])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = CIRCUITY * 2 * 6_371_000 * np.arcsin(np.sqrt(a))
    dist = np.rint(dist).astype(np.int64)                 # meters, as in Chapter 2
    secs = np.rint(dist / (req.speed_mph * M_PER_MI / 3600)).astype(np.int64)
    return dist, secs


# <<precheck>>
def precheck(req: PlanRequest, secs) -> dict[str, str]:
    """Reasons why a stop cannot be served by any vehicle, before solving."""
    biggest = max(v.capacity for v in req.vehicles)
    reasons = {}
    for k, s in enumerate(req.stops, start=1):
        if s.demand > biggest:
            reasons[s.id] = f"demand {s.demand} exceeds the largest vehicle ({biggest})"
            continue
        if s.window_end:
            ok = False
            for v in req.vehicles:
                arrive = max(to_s(v.shift_start) + secs[0, k],
                             to_s(s.window_start) if s.window_start else 0)
                back = arrive + 60 * s.service_min + secs[k, 0]
                if arrive <= to_s(s.window_end) and back <= to_s(v.shift_end):
                    ok = True
                    break
            if not ok:
                reasons[s.id] = "no vehicle can reach it within its window and shift"
    return reasons
# <<end_precheck>>


class NoPlan(RuntimeError):
    """The solver found no plan within its time."""


class InvalidPlan(ValueError):
    """A plan that breaks a rule of the request."""


@dataclass
class Exact:
    """A route in whole seconds and meters, before any rounding."""
    v: int                    # index of the vehicle in the request
    nodes: list[int]          # stops, numbered from 1 as in the matrices
    start: int
    arrive: list[int]
    begin: list[int]
    leave: list[int]
    back: int
    meters: int
    load: int


# <<solve>>
def solve(req: PlanRequest) -> PlanResult:
    t0 = time.perf_counter()
    dist, secs = matrices(req)
    service = [0] + [round(60 * s.service_min) for s in req.stops]
    warnings: list[str] = []
    try:
        routes, status = optimize(req, dist, secs, service)
        validate_plan(req, routes, dist, secs, service)
    except (NoPlan, InvalidPlan) as exc:       # never publish a plan we cannot vouch for
        log.error("no plan published: %s", exc)
        routes, status = [], type(exc).__name__
        warnings.append(f"no routes: {exc}")
    return publish(req, routes, precheck(req, secs), dist, service, status, warnings,
                   time.perf_counter() - t0)
# <<end_solve>>


def optimize(req: PlanRequest, dist, secs, service) -> tuple[list[Exact], str]:
    n, V = len(req.stops) + 1, len(req.vehicles)
    manager = pywrapcp.RoutingIndexManager(n, V, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit(f, t):
        i, j = manager.IndexToNode(f), manager.IndexToNode(t)
        return service[i] + int(secs[i, j])
    t_idx = routing.RegisterTransitCallback(transit)
    day = 24 * 3600
    routing.AddDimension(t_idx, day, day, False, "Time")
    T = routing.GetDimensionOrDie("Time")
    keep = [transit]
    for v, veh in enumerate(req.vehicles):           # each vehicle its own costs
        per_m = veh.cost_per_mile * UNITS / M_PER_MI
        cb = (lambda f, t, c=per_m: round(c * dist[manager.IndexToNode(f),
                                                     manager.IndexToNode(t)]))
        keep.append(cb)
        routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(cb), v)
        T.SetSpanCostCoefficientForVehicle(round(veh.cost_per_hour * UNITS / 3600), v)
        routing.SetFixedCostOfVehicle(round(veh.fixed_cost * UNITS), v)
        T.CumulVar(routing.Start(v)).SetRange(to_s(veh.shift_start), to_s(veh.shift_end))
        T.CumulVar(routing.End(v)).SetRange(to_s(veh.shift_start), to_s(veh.shift_end))
    for k, s in enumerate(req.stops, start=1):
        idx = manager.NodeToIndex(k)
        a = to_s(s.window_start) if s.window_start else 0
        b = to_s(s.window_end) if s.window_end else day
        T.CumulVar(idx).SetRange(a, b)
        routing.AddDisjunction([idx], round(s.priority * UNITS))
    dem = routing.RegisterUnaryTransitCallback(
        lambda f: 0 if manager.IndexToNode(f) == 0 else req.stops[manager.IndexToNode(f) - 1].demand)
    keep.append(dem)
    routing.AddDimensionWithVehicleCapacity(dem, 0, [v.capacity for v in req.vehicles],
                                            True, "Load")
    for v in range(V):                               # leave as late as possible
        routing.AddVariableMinimizedByFinalizer(T.CumulVar(routing.End(v)))
        routing.AddVariableMaximizedByFinalizer(T.CumulVar(routing.Start(v)))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromMilliseconds(max(1, round(req.time_limit_s * 1000)))
    sol = routing.SolveWithParameters(params)
    if sol is None:                                  # no first solution in time: one more try
        params.time_limit.FromMilliseconds(round(1000 * max(RETRY_MIN_S, req.time_limit_s)))
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
        sol = routing.SolveWithParameters(params)
    status = routing_enums_pb2.RoutingSearchStatus.Value.Name(routing.status())
    if sol is None:
        raise NoPlan(f"the solver found no plan in time ({status}). "
                     "Retry with a longer time_limit_s")
    routes = []
    for v in range(V):
        idx = sol.Value(routing.NextVar(routing.Start(v)))
        nodes = []
        while not routing.IsEnd(idx):
            nodes.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        if nodes:
            start = sol.Min(T.CumulVar(routing.Start(v)))
            routes.append(schedule(req, v, nodes, start, dist, secs, service))
    return routes, status


def schedule(req, v, nodes, start, dist, secs, service) -> Exact:
    """Recompute the schedule from the data alone: earliest start at every stop."""
    t, prev, arrive, begin, leave, load = int(start), 0, [], [], [], 0
    for k in nodes:
        s = req.stops[k - 1]
        a = t + int(secs[prev, k])
        b = max(a, to_s(s.window_start) if s.window_start else 0)
        t = b + service[k]
        arrive.append(a)
        begin.append(b)
        leave.append(t)
        load += s.demand
        prev = k
    meters = int(sum(dist[a, b] for a, b in zip([0] + nodes, nodes + [0])))
    return Exact(v, list(nodes), int(start), arrive, begin, leave,
                 t + int(secs[prev, 0]), meters, load)


# <<validate_plan>>
def validate_plan(req: PlanRequest, routes: list[Exact], dist, secs, service) -> None:
    """Check a plan against the request before it is published, from the
    matrices and whole seconds: every stop at most once and only known
    stops, each vehicle at most once, capacity, shift, windows, and times
    and meters that add up.  Raises InvalidPlan."""
    def require(ok, msg):
        if not ok:
            raise InvalidPlan(msg)
    n = len(req.stops)
    visited = [k for r in routes for k in r.nodes]
    require(all(1 <= k <= n for k in visited), "a route visits an unknown stop")
    require(len(visited) == len(set(visited)), "a stop is visited twice")
    used = [r.v for r in routes]
    require(len(used) == len(set(used))
            and all(0 <= v < len(req.vehicles) for v in used),
            "a vehicle is used twice, or does not exist")
    for r in routes:
        veh = req.vehicles[r.v]
        load = sum(req.stops[k - 1].demand for k in r.nodes)
        require(load == r.load <= veh.capacity,
                f"vehicle {veh.id}: load {r.load} above capacity {veh.capacity}")
        require(to_s(veh.shift_start) <= r.start and r.back <= to_s(veh.shift_end),
                f"vehicle {veh.id}: route outside its shift")
        t, prev = r.start, 0
        for k, a, b, d in zip(r.nodes, r.arrive, r.begin, r.leave, strict=True):
            s = req.stops[k - 1]
            require(a == t + secs[prev, k] and b >= a and d == b + service[k],
                    f"stop {s.id}: times do not add up")
            require(s.window_start is None or b >= to_s(s.window_start),
                    f"stop {s.id}: served before its window opens")
            require(s.window_end is None or b <= to_s(s.window_end),
                    f"stop {s.id}: served after its window closes")
            t, prev = d, k
        require(r.back == t + secs[prev, 0],
                f"vehicle {veh.id}: return time does not add up")
        legs = zip([0] + r.nodes, r.nodes + [0])
        require(r.meters == sum(dist[a, b] for a, b in legs),
                f"vehicle {veh.id}: distance does not add up")
# <<end_validate_plan>>


def route_cost(veh, r: Exact) -> float:
    return (veh.fixed_cost + r.meters / M_PER_MI * veh.cost_per_mile
            + (r.back - r.start) / 3600 * veh.cost_per_hour)


def publish(req, routes: list[Exact], reasons, dist, service, status, warnings,
            seconds) -> PlanResult:
    """The response: exact totals first, rounded only for display."""
    served = {k for r in routes for k in r.nodes}
    out = []
    for r in routes:
        veh = req.vehicles[r.v]
        visits = [Visit(stop_id=req.stops[k - 1].id, arrival=to_hhmm(a), start=to_hhmm(b),
                        departure=to_hhmm(d))
                  for k, a, b, d in zip(r.nodes, r.arrive, r.begin, r.leave)]
        out.append(Route(vehicle_id=veh.id, depart=to_hhmm(r.start), back=to_hhmm(r.back),
                         visits=visits, miles=round(r.meters / M_PER_MI, 2), load=r.load,
                         hours=round((r.back - r.start) / 3600, 3),
                         cost=round(route_cost(veh, r), 2)))
    default = ("not planned: no plan was published" if not routes else
               "left out: serving it costs more than its priority, or no room")
    unassigned = [Unassigned(stop_id=s.id, reason=reasons.get(s.id, default))
                  for k, s in enumerate(req.stops, start=1) if k not in served]
    return PlanResult(
        routes=out, unassigned=unassigned,
        miles=round(sum(r.meters for r in routes) / M_PER_MI, 2),
        paid_hours=round(sum(r.back - r.start for r in routes) / 3600, 2),
        cost=round(sum(route_cost(req.vehicles[r.v], r) for r in routes), 2),
        solver_status=status, fallback=bool(warnings), warnings=warnings,
        solve_seconds=round(seconds, 3))
