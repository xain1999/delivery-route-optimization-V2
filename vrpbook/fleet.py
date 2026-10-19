"""Heterogeneous fleets, shifts, breaks and compatibility (Chapter 7).

A plan is built for a list of individual vehicles, each with its own
capacity, costs, shift and driver pay rule.  The model extends the time
window model of Chapter 6.
"""
from __future__ import annotations

from dataclasses import dataclass

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .ortools_solver import _status_name
from .timewindows import HOUR, TWData

MINUTE = 60
UNIT = 100_000          # objective unit: 1 dollar = 100,000


# <<vehicle_type>>
@dataclass(frozen=True)
class VehicleType:
    name: str
    count: int
    totes: int
    payload_lb: int
    per_mile: float           # dollars per mile driven
    fixed: float              # dollars per day if used (rental fee)
    hourly: float             # dollars per paid hour, 0 for salaried drivers
    shift: tuple[int, int]    # earliest start, latest end (seconds)
    max_hours: float | None = None   # longest paid day, if limited
    refrigerated: bool = False
    overtime_h: float = 0.0          # salaried: hours allowed past the shift end
    overtime_rate: float = 0.0       # dollars per overtime hour
# <<end_vehicle_type>>


@dataclass(frozen=True)
class BreakRule:
    """A meal break of `minutes`, for shifts of at least `min_shift_h`
    hours, starting between `earliest_after_h` and `latest_after_h` hours
    after the shift starts.  The defaults follow Illinois' One Day Rest in
    Seven Act (20 minutes, beginning no later than 5 hours into a shift of
    7.5 hours or more). The earliest start is company policy."""
    minutes: int = 20
    earliest_after_h: float = 2.5
    latest_after_h: float = 5.0
    min_shift_h: float = 7.5


@dataclass
class RichPlan:
    routes: list[list[int]]
    vehicles: list[VehicleType]
    starts: list[list[int]]           # start of service at each node of each route
    breaks: list[tuple[int, int] | None]   # (start, end) of each vehicle's break
    objective: float
    status: str


def expand(types: list[VehicleType]) -> list[VehicleType]:
    return [t for t in types for _ in range(t.count)]


def solve_rich(data: TWData, vehicles: list[VehicleType], weights: list[int],
               cold: list[bool] | None = None, breaks: BreakRule | None = None,
               price_salaried_time: float = 0.0, time_limit_s: float = 30.0,
               reload_nodes: list[int] = (), reload_cost: float = 10.0,
               first_solution: str = "PARALLEL_CHEAPEST_INSERTION") -> RichPlan:
    """price_salaried_time > 0 charges salaried drivers' route time at that
    hourly rate (the Chapter 6 assumption), 0 treats it as already paid.
    reload_nodes: optional visits to the depot that empty the van
    (see add_reload_nodes), a van may use any number of them."""
    n, V = len(data.demands), len(vehicles)
    manager = pywrapcp.RoutingIndexManager(n, V, 0)
    routing = pywrapcp.RoutingModel(manager)
    keep = []           # Python callbacks must outlive the solve
    reload = set(reload_nodes)
    per_reload = round(reload_cost * UNIT)       # handling cost of a reload stop

    # <<per_vehicle_costs>>
    # vehicle cost per mile, one callback per distinct rate
    rate_cb = {}
    for v, vt in enumerate(vehicles):
        if vt.per_mile not in rate_cb:
            per_meter = vt.per_mile * UNIT / 1609.344
            def cb(f, t, k=per_meter):
                i, j = manager.IndexToNode(f), manager.IndexToNode(t)
                return round(k * data.dist[i, j]) + (per_reload if i in reload else 0)
            keep.append(cb)
            rate_cb[vt.per_mile] = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfVehicle(rate_cb[vt.per_mile], v)
        routing.SetFixedCostOfVehicle(round(vt.fixed * UNIT), v)
    # <<end_per_vehicle_costs>>

    # <<capacities>>
    # capacities differ by vehicle, a reload node unloads the van
    for name, dem, caps in (("Totes", data.demands, [vt.totes for vt in vehicles]),
                            ("Weight", weights, [vt.payload_lb for vt in vehicles])):
        unload = -max(caps)
        dem = [unload if i in reload else int(q) for i, q in enumerate(dem)]
        cb = (lambda f, d=dem: d[manager.IndexToNode(f)])
        keep.append(cb)
        idx = routing.RegisterUnaryTransitCallback(cb)
        # slack lets the load drop to anywhere between 0 and the capacity at
        # a reload, elsewhere it can only make the van look fuller
        slack = max(caps) if reload else 0
        routing.AddDimensionWithVehicleCapacity(idx, slack, caps, True, name)
    for r in reload:
        routing.AddDisjunction([manager.NodeToIndex(r)], 0)   # optional, free
    # <<end_capacities>>

    # <<shifts>>
    def time_cb(f: int, t: int) -> int:
        i, j = manager.IndexToNode(f), manager.IndexToNode(t)
        return data.service[i] + int(data.time[i, j])
    time_idx = routing.RegisterTransitCallback(time_cb)
    day_end = max(vt.shift[1] + int(vt.overtime_h * HOUR) for vt in vehicles)
    routing.AddDimension(time_idx, day_end, day_end, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, n):
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(*data.windows[node])
    for v, vt in enumerate(vehicles):
        start, end = vt.shift
        time_dim.CumulVar(routing.Start(v)).SetRange(start, end)
        time_dim.CumulVar(routing.End(v)).SetRange(start, end + int(vt.overtime_h * HOUR))
        if vt.overtime_h:          # each second past the shift end costs extra
            time_dim.SetCumulVarSoftUpperBound(routing.End(v), end,
                                               round(vt.overtime_rate * UNIT / HOUR))
        if vt.max_hours is not None:
            time_dim.SetSpanUpperBoundForVehicle(int(vt.max_hours * HOUR), v)
        rate = vt.hourly or price_salaried_time
        if rate:
            time_dim.SetSpanCostCoefficientForVehicle(round(rate * UNIT / HOUR), v)
    # <<end_shifts>>

    # <<compatibility>>
    if cold is not None:
        fridge = [v for v, vt in enumerate(vehicles) if vt.refrigerated]
        for node in range(1, n):
            if cold[node]:
                # the vehicle serving this node must be refrigerated
                routing.VehicleVar(manager.NodeToIndex(node)).SetValues(fridge)
    # <<end_compatibility>>

    # <<breaks>>
    solver = routing.solver()
    break_vars: dict[int, object] = {}
    if breaks is not None:
        visit_times = [data.service[manager.IndexToNode(i)]
                       for i in range(routing.Size())]
        for v, vt in enumerate(vehicles):
            start, end = vt.shift
            longest = min(end - start, int((vt.max_hours or 24) * HOUR))
            if longest < breaks.min_shift_h * HOUR:
                continue                          # short shifts need no break
            brk = solver.FixedDurationIntervalVar(
                start + int(breaks.earliest_after_h * HOUR),        # may start in this range
                start + int(breaks.latest_after_h * HOUR),
                breaks.minutes * MINUTE, False, f"break-{v}")
            time_dim.SetBreakIntervalsOfVehicle([brk], v, visit_times)
            break_vars[v] = brk
    # <<end_breaks>>

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy,
                                             first_solution)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    sol = routing.SolveWithParameters(params)
    status = _status_name(routing.status())
    if sol is None:
        return RichPlan([], vehicles, [], [], -1, status)

    routes, starts, brks = [], [], []
    for v in range(V):
        idx, r, s = routing.Start(v), [], []
        while True:
            r.append(manager.IndexToNode(idx))
            s.append(sol.Min(time_dim.CumulVar(idx)))
            if routing.IsEnd(idx):
                break
            idx = sol.Value(routing.NextVar(idx))
        routes.append(r)
        starts.append(s)
        if v in break_vars and len(r) > 2:
            b = sol.StartValue(break_vars[v])
            brks.append((b, b + breaks.minutes * MINUTE))
        else:
            brks.append(None)
    return RichPlan(routes, vehicles, starts, brks, sol.ObjectiveValue() / UNIT, status)


# <<check_plan>>
def check_plan(plan: RichPlan, data: TWData, weights: list[int],
               cold: list[bool] | None = None, breaks: BreakRule | None = None,
               reload_nodes: list[int] = ()) -> dict:
    """Independent check of a plan against every rule of this chapter.
    Returns a list of violations (empty if the plan is executable) and
    the totals needed to cost it."""
    bad, seen = [], []
    tot = {"miles": 0.0, "drive": 0, "service": 0, "duration": 0, "vans": 0}
    for v, (r, s, vt) in enumerate(zip(plan.routes, plan.starts, plan.vehicles)):
        if len(r) <= 2:
            continue
        tot["vans"] += 1
        reload = set(reload_nodes)
        custs = [i for i in r[1:-1] if i not in reload]
        seen += custs
        trips, trip = [], []                     # split the route at reloads
        for i in r[1:-1]:
            if i in reload:
                trips.append(trip)
                trip = []
            else:
                trip.append(i)
        trips.append(trip)
        tot["trips"] = tot.get("trips", 0) + sum(1 for t in trips if t)
        for trip in trips:
            if sum(data.demands[i] for i in trip) > vt.totes:
                bad.append(f"vehicle {v}: totes over capacity")
            if sum(weights[i] for i in trip) > vt.payload_lb:
                bad.append(f"vehicle {v}: weight over payload")
        if cold is not None and not vt.refrigerated and any(cold[i] for i in custs):
            bad.append(f"vehicle {v}: refrigerated order in a dry van")
        if not (vt.shift[0] <= s[0] and s[-1] <= vt.shift[1] + vt.overtime_h * HOUR):
            bad.append(f"vehicle {v}: outside its shift")
        span = s[-1] - s[0]
        if vt.max_hours is not None and span > vt.max_hours * HOUR:
            bad.append(f"vehicle {v}: longer than {vt.max_hours} h")
        brk = plan.breaks[v]
        for k, (i, j) in enumerate(zip(r, r[1:])):
            if j != 0 and j not in reload and \
                    not (data.windows[j][0] <= s[k + 1] <= data.windows[j][1]):
                bad.append(f"vehicle {v}: node {j} outside its window")
            ready = s[k] + data.service[i] + int(data.time[i, j])
            if brk and s[k] + data.service[i] <= brk[0] < s[k + 1]:
                ready += brk[1] - brk[0]          # break taken on this leg
            if s[k + 1] < ready:
                bad.append(f"vehicle {v}: cannot reach node {j} by {s[k + 1]}")
            if brk and s[k] < brk[1] and brk[0] < s[k] + data.service[i] and i != 0:
                bad.append(f"vehicle {v}: break overlaps service at node {i}")
        if breaks and vt.shift[1] - vt.shift[0] >= breaks.min_shift_h * HOUR and \
                (vt.max_hours or 24) >= breaks.min_shift_h:
            if brk is None or not (vt.shift[0] + breaks.earliest_after_h * HOUR <= brk[0]
                                   <= vt.shift[0] + breaks.latest_after_h * HOUR):
                bad.append(f"vehicle {v}: no meal break in time")
        tot["miles"] += sum(int(data.dist[i, j]) for i, j in zip(r, r[1:])) / 1609.344
        tot["drive"] += sum(int(data.time[i, j]) for i, j in zip(r, r[1:]))
        tot["service"] += sum(data.service[i] for i in custs)
        tot["reload"] = tot.get("reload", 0) + sum(data.service[i] for i in r if i in reload)
        tot["duration"] += span
    expected = [i for i in range(1, len(data.demands)) if i not in set(reload_nodes)]
    if sorted(seen) != expected:
        bad.append("not every customer is served exactly once")
    return {"violations": bad, **tot}
# <<end_check_plan>>


def true_cost(plan: RichPlan, data: TWData, reload_nodes: list[int] = (),
              reload_cost: float = 10.0) -> dict:
    """Cost under the company's actual pay rules (salaried time is sunk)."""
    miles = drive_cost = fixed = hourly = overtime = handling = 0.0
    hours = {"salaried": 0.0, "hourly": 0.0, "overtime": 0.0}
    for r, s, vt in zip(plan.routes, plan.starts, plan.vehicles):
        if len(r) <= 2:
            continue
        m = sum(int(data.dist[i, j]) for i, j in zip(r, r[1:])) / 1609.344
        miles += m
        drive_cost += vt.per_mile * m
        fixed += vt.fixed
        handling += reload_cost * sum(1 for i in r if i in set(reload_nodes))
        h = (s[-1] - s[0]) / HOUR
        hours["hourly" if vt.hourly else "salaried"] += h
        hourly += vt.hourly * h
        over = max(0, s[-1] - vt.shift[1]) / HOUR
        hours["overtime"] += over
        overtime += vt.overtime_rate * over
    return {"total": drive_cost + fixed + hourly + overtime + handling, "vehicle": drive_cost,
            "rental": fixed, "agency": hourly, "overtime": overtime, "reloads": handling,
            "miles": miles, **{f"{k}_h": v for k, v in hours.items()}}


# <<reload_nodes>>
def add_reload_nodes(data: TWData, weights: list[int], cold: list[bool] | None,
                     count: int, loading_s: int = 15 * MINUTE):
    """Append `count` copies of the depot as optional 'reload' stops.

    A van that visits one returns to the depot, spends `loading_s` loading
    and leaves empty-counted, so it can serve a second wave of orders.
    Returns the extended data, weights, cold flags and the reload node ids.
    """
    import numpy as np
    n = len(data.demands)
    idx = list(range(n)) + [0] * count           # reloads sit at the depot
    ext = lambda m: m[np.ix_(idx, idx)]           # noqa: E731
    reloads = list(range(n, n + count))
    data2 = TWData(ext(data.dist), ext(data.time),
                   data.service + [loading_s] * count,
                   data.windows + [data.windows[0]] * count,
                   data.demands + [0] * count, data.capacity, data.n_vehicles, data.extra)
    cold2 = None if cold is None else cold + [False] * count
    return data2, weights + [0] * count, cold2, reloads
# <<end_reload_nodes>>
