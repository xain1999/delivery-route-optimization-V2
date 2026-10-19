"""Re-optimizing during the day: a same-day courier simulator (Chapter 14).

Saguaro Courier (fictional) delivers prescriptions in Phoenix from one
hub.  Orders arrive during the day and must be delivered within a
promised time.  Every few minutes a planner re-optimizes the orders at the
hub that are not yet loaded. Cars on the road keep their routes (frozen).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .distances import haversine_m

MIN = 60
HUB = (33.4560, -112.0450)
AREAS = {"Downtown Phoenix": (33.4484, -112.0740), "Tempe": (33.4255, -111.9400),
         "Scottsdale": (33.4942, -111.9261), "Arcadia": (33.4970, -111.9800),
         "Maryvale": (33.4950, -112.1650), "Paradise Valley": (33.5310, -111.9426),
         "South Phoenix": (33.3900, -112.0700), "Encanto": (33.4750, -112.0900)}


@dataclass
class Order:
    id: int
    lat: float
    lon: float
    release: int                      # seconds after midnight, when it reaches the hub
    deadline: int
    status: str = "new"               # new, pending, rejected, loaded, delivered
    delivered: int | None = None


def generate_day(seed: int, promise_min: int = 180, busy: float = 1.0,
                 rates=((8, 10, 10), (10, 13, 18), (13, 16, 14))) -> list[Order]:
    """Orders for one day: a Poisson process whose rate (orders per hour)
    changes by period. Half the customers spread over the metro area and
    half around its main neighborhoods."""
    rng = np.random.default_rng(seed)
    orders, centers = [], list(AREAS.values())
    for start, end, rate in rates:
        t = start * 3600.0
        while True:
            t += rng.exponential(3600.0 / (rate * busy))
            if t >= end * 3600:
                break
            if rng.random() < 0.5:
                lat, lon = rng.uniform(33.37, 33.56), rng.uniform(-112.18, -111.90)
            else:
                c = centers[rng.integers(len(centers))]
                lat, lon = c[0] + rng.normal(0, 0.012), c[1] + rng.normal(0, 0.014)
            orders.append(Order(len(orders), float(lat), float(lon), int(t),
                                int(t) + promise_min * MIN))
    return orders


@dataclass
class Car:
    id: int
    free_at: int                      # when it is next at the hub, empty
    trips: list = field(default_factory=list)
    miles: float = 0.0


class Geometry:
    def __init__(self, orders, speed_mph=28.0, circuity=1.3):
        pts = [HUB] + [(o.lat, o.lon) for o in orders]
        n = len(pts)
        self.dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                self.dist[i, j] = self.dist[j, i] = circuity * haversine_m(*pts[i], *pts[j])
        self.time = np.rint(self.dist / (speed_mph * 1609.344 / 3600)).astype(int)

    def node(self, order_id: int) -> int:
        return order_id + 1


# <<plan>>
def plan(now: int, cars: list[Car], todo: list[Order], G: Geometry, *,
         penalty: dict[int, float], service_s: int = 4 * MIN, capacity: int = 20,
         late_per_min: float = 1.0, per_mile: float = 0.76, seconds: float = 1.0,
         initial: dict[int, list[int]] | None = None,
         insert_new: bool = True) -> tuple[dict[int, list[int]], float]:
    """One trip per car, starting at the hub when the car is free: which of
    the orders at the hub should each car take, in which order?  Lateness is
    allowed at a price. An order may be left for a later trip, at its penalty."""
    n = len(todo) + 1
    nodes = [0] + [G.node(o.id) for o in todo]
    V = len(cars)
    manager = pywrapcp.RoutingIndexManager(n, V, 0)
    routing = pywrapcp.RoutingModel(manager)
    keep = []
    scale = 100_000                                   # cost units per dollar

    def time_cb(f, t):
        i, j = manager.IndexToNode(f), manager.IndexToNode(t)
        return (service_s if i else 0) + int(G.time[nodes[i], nodes[j]])
    def cost_cb(f, t):
        i, j = manager.IndexToNode(f), manager.IndexToNode(t)
        return round(per_mile * scale * G.dist[nodes[i], nodes[j]] / 1609.344)
    keep += [time_cb, cost_cb]
    t_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(cost_cb))
    horizon = 30 * 3600
    routing.AddDimension(t_idx, horizon, horizon, False, "Time")
    T = routing.GetDimensionOrDie("Time")
    for v, car in enumerate(cars):
        T.CumulVar(routing.Start(v)).SetRange(max(now, car.free_at), horizon)
    for k, o in enumerate(todo, start=1):
        idx = manager.NodeToIndex(k)
        T.SetCumulVarSoftUpperBound(idx, o.deadline, round(late_per_min * scale / 60))
        routing.AddDisjunction([idx], round(penalty[o.id] * scale))
    one = routing.RegisterUnaryTransitCallback(lambda f: 1 if manager.IndexToNode(f) else 0)
    routing.AddDimension(one, 0, capacity, True, "Load")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromMilliseconds(int(seconds * 1000))
    pos = {o.id: k for k, o in enumerate(todo, start=1)}
    sol = None
    if initial:                                        # warm start from the last plan
        start = [[pos[i] for i in initial.get(car.id, []) if i in pos] for car in cars]
        if insert_new:                                 # new orders: cheapest insertion
            placed = {k for r in start for k in r}
            for k in range(1, n):
                if k in placed:
                    continue
                best = None
                for r in start:
                    if len(r) >= capacity:
                        continue
                    seq = [0] + r + [0]
                    for p in range(len(seq) - 1):
                        a, b = nodes[seq[p]], nodes[seq[p + 1]]
                        d = G.dist[a, nodes[k]] + G.dist[nodes[k], b] - G.dist[a, b]
                        if best is None or d < best[0]:
                            best = (d, r, p)
                if best:
                    best[1].insert(best[2], k)
        routing.CloseModelWithParameters(params)
        assignment = routing.ReadAssignmentFromRoutes(start, True)
        if assignment is not None:
            sol = routing.SolveFromAssignmentWithParameters(assignment, params)
    if sol is None:
        sol = routing.SolveWithParameters(params)
    if sol is None:          # the budget ran out before a first plan: allow more time
        params.time_limit.FromMilliseconds(int(5000 * max(seconds, 1.0)))
        sol = routing.SolveWithParameters(params)
    out = {}
    for v, car in enumerate(cars):
        idx, r = sol.Value(routing.NextVar(routing.Start(v))), []
        while not routing.IsEnd(idx):
            r.append(todo[manager.IndexToNode(idx) - 1].id)
            idx = sol.Value(routing.NextVar(idx))
        out[car.id] = r
    return out, sol.ObjectiveValue() / scale
# <<end_plan>>


def trip_times(start: int, route: list[int], orders, G: Geometry, service_s=4 * MIN):
    """Delivery time of each order of a trip that leaves the hub at start,
    and the time the car is back."""
    t, prev, when = start, 0, []
    for i in route:
        t += int(G.time[prev, G.node(i)])
        when.append(t)
        t += service_s
        prev = G.node(i)
    return when, t + int(G.time[prev, 0])


# <<latest_departure>>
def latest_departure(now: int, route: list[int], orders, G: Geometry) -> int:
    """The latest time a car can leave with this route and still deliver every
    order by its deadline (or now, if some order would be late anyway)."""
    when, _ = trip_times(now, route, orders, G)
    slack = min(orders[i].deadline - w for i, w in zip(route, when))
    return now + max(0, slack)
# <<end_latest_departure>>


# <<simulate>>
def simulate(orders: list[Order], n_cars: int, policy: str, *, epoch_min: int = 10,
             end_h: float = 20.0, seconds: float = 1.0, value: float = 25.0,
             warm: bool = True, log: list | None = None,
             snapshot: dict | None = None) -> dict:
    """Run one day.  policy:
      "now":    the planner sees only the cars at the hub, and they leave at once,
      "wait":   it waits at the hub while every order it would carry can
                still wait one more epoch, so that later orders can join,
      "accept": as "wait", and a new order the planner cannot fit is
                declined at once (worth `value`), instead of being delivered late."""
    G = Geometry(orders)
    cars = [Car(v, 8 * 3600) for v in range(n_cars)]
    last: dict[int, list[int]] = {}
    for now in range(8 * 3600, int(end_h * 3600) + 1, epoch_min * MIN):
        for o in orders:                                  # orders reach the hub
            if o.status == "new" and o.release <= now:
                o.status = "arrived"
        todo = [o for o in orders if o.status in ("arrived", "pending")]
        if not todo:
            continue
        pen = {o.id: (value if (policy == "accept" and o.status == "arrived") else 10_000.0)
               for o in todo}
        # "now" plans only for the cars at the hub, the others plan for the
        # cars on the road too, so that orders can wait for a better trip
        fleet = [c for c in cars if c.free_at <= now] if policy == "now" else cars
        if not fleet:
            continue
        routes, _ = plan(now, fleet, todo, G, penalty=pen, seconds=seconds,
                         initial=last if warm else None)
        if snapshot is not None:     # the planner's inputs, for experiments
            info = dict(size=len(todo), now=now, todo=[o.id for o in todo],
                        free=[c.free_at for c in fleet], penalty=pen, last=dict(last))
            if now in snapshot.get("times", ()):
                snapshot.setdefault("moments", []).append(info)
            if len(todo) > snapshot.get("size", 0):
                snapshot.update(info)
        planned = {i for r in routes.values() for i in r}
        for o in todo:
            if o.status == "arrived":
                o.status = "rejected" if (policy == "accept" and o.id not in planned) \
                    else "pending"
        last = routes
        final = now >= end_h * 3600
        for car in cars:
            r = routes.get(car.id, [])
            if not r or car.free_at > now:
                continue
            if policy != "now" and not final and \
                    latest_departure(now, r, orders, G) >= now + epoch_min * MIN:
                continue                                  # it can wait for more orders
            when, back = trip_times(now, r, orders, G)
            for i, w in zip(r, when):
                orders[i].status, orders[i].delivered = "delivered", w
            car.trips.append((now, back, list(r)))
            car.miles += (sum(G.dist[a, b] for a, b in zip([0] + [G.node(i) for i in r],
                                                            [G.node(i) for i in r] + [0]))
                          / 1609.344)
            car.free_at = back
            last.pop(car.id, None)
        if log is not None:
            log.append((now, len(todo), sum(1 for c in cars if c.free_at > now)))
    done = [o for o in orders if o.status == "delivered"]
    late = [o for o in done if o.delivered > o.deadline]
    return {"orders": len(orders), "delivered": len(done), "late": len(late),
            "rejected": sum(o.status == "rejected" for o in orders),
            "unserved": sum(o.status in ("arrived", "pending", "new") for o in orders),
            "late_min": sum((o.delivered - o.deadline) / MIN for o in late),
            "max_late_min": max([(o.delivered - o.deadline) / MIN for o in late], default=0),
            "mean_wait_min": float(np.mean([(o.delivered - o.release) / MIN for o in done])),
            "miles": sum(c.miles for c in cars), "trips": sum(len(c.trips) for c in cars),
            "driving_h": sum(b - a for c in cars for a, b, _ in c.trips) / 3600,
            "per_trip": len(done) / max(1, sum(len(c.trips) for c in cars)),
            "last_back_h": max([c.trips[-1][1] for c in cars if c.trips], default=0) / 3600}
# <<end_simulate>>
