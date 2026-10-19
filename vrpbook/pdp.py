"""Pickup and delivery with time windows (Chapter 8).

Each order is a pair of stops: a pickup at a restaurant and a delivery to
a customer.  Couriers start wherever they are and end their route at the
last delivery (open routes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .distances import haversine_m
from .ortools_solver import _status_name

MIN = 60


@dataclass
class PDPData:
    """Node 0 is a dummy end (routes are open). Nodes 1..V are courier
    starts. Then each order contributes a pickup and a delivery node."""
    dist: np.ndarray                 # meters
    time: np.ndarray                 # seconds
    service: list[int]
    windows: list[tuple[int, int]]
    pairs: list[tuple[int, int]]     # (pickup node, delivery node) per order
    bags: list[int]                  # per order
    starts: list[int]                # start node of each courier
    max_ride: int                    # seconds from pickup to delivery


def build(couriers, orders, restaurants, *, speed_mph=20.0, circuity=1.3,
          start_time=11 * 3600, promise_min=45, hold_min=20, max_ride_min=30,
          pickup_service_min=4, delivery_service_min=3) -> PDPData:
    """couriers: [(lat, lon)]. Orders: [(restaurant_id, lat, lon, ready_s, bags)]."""
    points = [None] + list(couriers)
    windows = [(start_time, start_time + 6 * 3600)] * (1 + len(couriers))
    service = [0] * (1 + len(couriers))
    pairs, bags = [], []
    for r, lat, lon, ready, b in orders:
        p = len(points)
        points.append(restaurants[r])
        windows.append((ready, ready + hold_min * MIN))          # food waits at most this
        service.append(pickup_service_min * MIN)
        points.append((lat, lon))
        windows.append((ready, ready + promise_min * MIN))       # promised delivery time
        service.append(delivery_service_min * MIN)
        pairs.append((p, p + 1))
        bags.append(b)
    n = len(points)
    dist = np.zeros((n, n), dtype=np.int64)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                dist[i, j] = round(circuity * haversine_m(*points[i], *points[j]))
    dist[:, 0] = 0                                               # open routes end anywhere
    time = np.rint(dist / (speed_mph * 1609.344 / 3600)).astype(np.int64)
    return PDPData(dist, time, service, windows, pairs, bags,
                   list(range(1, 1 + len(couriers))), max_ride_min * MIN)


@dataclass
class PDPPlan:
    routes: list[list[int]]
    starts: list[list[int]]          # start of service at each node of each route
    status: str
    objective: int


def solve_pdp(data: PDPData, *, bag_capacity: int = 6, max_orders: int = 3,
              courier_fixed_s: int = 15 * MIN, time_limit_s: float = 20.0) -> PDPPlan:
    V = len(data.starts)
    n = len(data.service)
    manager = pywrapcp.RoutingIndexManager(n, V, data.starts, [0] * V)
    routing = pywrapcp.RoutingModel(manager)
    keep = []

    def time_cb(f, t):
        i, j = manager.IndexToNode(f), manager.IndexToNode(t)
        return data.service[i] + int(data.time[i, j])
    keep.append(time_cb)
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(time_idx)
    routing.SetFixedCostOfAllVehicles(courier_fixed_s)
    horizon = max(b for _, b in data.windows)
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetSpanCostCoefficientForAllVehicles(1)          # courier time
    for node in range(V + 1, n):
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(*data.windows[node])
    for v in range(V):
        time_dim.CumulVar(routing.Start(v)).SetRange(*data.windows[data.starts[v]])

    # load: bags and number of orders on board
    load_bags, load_orders = [0] * n, [0] * n
    for (p, d), b in zip(data.pairs, data.bags):
        load_bags[p], load_bags[d] = b, -b
        load_orders[p], load_orders[d] = 1, -1
    for name, load, cap in (("Bags", load_bags, bag_capacity),
                            ("Orders", load_orders, max_orders)):
        cb = (lambda f, q=load: q[manager.IndexToNode(f)])
        keep.append(cb)
        routing.AddDimension(routing.RegisterUnaryTransitCallback(cb), 0, cap, True, name)

    # <<pairs>>
    solver = routing.solver()
    for p, d in data.pairs:
        pi, di = manager.NodeToIndex(p), manager.NodeToIndex(d)
        routing.AddPickupAndDelivery(pi, di)                       # both or neither
        solver.Add(routing.VehicleVar(pi) == routing.VehicleVar(di))     # same courier
        solver.Add(time_dim.CumulVar(pi) <= time_dim.CumulVar(di))      # pick up first
        solver.Add(time_dim.CumulVar(di) - time_dim.CumulVar(pi)
                   <= data.max_ride)                                     # food stays fresh
    # <<end_pairs>>

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    sol = routing.SolveWithParameters(params)
    status = _status_name(routing.status())
    if sol is None:
        return PDPPlan([], [], status, -1)
    routes, starts = [], []
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
    return PDPPlan(routes, starts, status, sol.ObjectiveValue())


def check_pdp(plan: PDPPlan, data: PDPData, bag_capacity: int, max_orders: int) -> dict:
    """Independent check. Returns violations and summary statistics."""
    # <<check_pairs>>
    bad = []
    where = {}                                   # node -> (route, position)
    for v, r in enumerate(plan.routes):
        for k, node in enumerate(r[1:-1], start=1):
            where[node] = (v, k)
    rides, active = [], 0
    spans: dict[int, list[tuple[int, int, int]]] = {}       # route -> (start, end, order)
    for o, (p, d) in enumerate(data.pairs):
        if p not in where or d not in where:
            bad.append(f"order {o} not served")
            continue
        (vp, kp), (vd, kd) = where[p], where[d]
        if vp != vd:
            bad.append(f"order {o}: picked up and delivered by different couriers")
            continue
        if kp > kd:
            bad.append(f"order {o}: delivered before pickup")
        tp, td = plan.starts[vp][kp], plan.starts[vd][kd]
        rides.append(td - tp)
        if td - tp > data.max_ride:
            bad.append(f"order {o}: ride {(td - tp) // 60} min")
        spans.setdefault(vp, []).append((kp, kd, o))
    # an order is batched if another order is in the car at the same time
    batched = sum(1 for sp in spans.values() for (a1, b1, o1) in sp
                  if any(o2 != o1 and max(a1, a2) < min(b1, b2) for (a2, b2, o2) in sp))
    # <<end_check_pairs>>
    for v, (r, s) in enumerate(zip(plan.routes, plan.starts)):
        if len(r) <= 2:
            continue
        active += 1
        bags = orders = 0
        for k, (i, j) in enumerate(zip(r, r[1:])):
            if j != 0:
                a, b = data.windows[j]
                if not a <= s[k + 1] <= b:
                    bad.append(f"courier {v}: node {j} outside its window")
                if s[k + 1] < s[k] + data.service[i] + int(data.time[i, j]):
                    bad.append(f"courier {v}: cannot reach node {j} in time")
        for node in r[1:-1]:
            o = next(k for k, pd in enumerate(data.pairs) if node in pd)
            sign = 1 if node == data.pairs[o][0] else -1
            bags += sign * data.bags[o]
            orders += sign
            if bags > bag_capacity or orders > max_orders:
                bad.append(f"courier {v}: car over capacity")
    # active time: from leaving the start (no waiting there) to the last delivery
    hours = sum((s[-1] - (s[1] - int(data.time[r[0], r[1]]))) / 3600
                for r, s in zip(plan.routes, plan.starts) if len(r) > 2)
    return {"violations": bad, "couriers": active, "batched": batched,
            "mean_ride_min": float(np.mean(rides)) / 60 if rides else 0.0,
            "max_ride_min": max(rides) / 60 if rides else 0.0, "courier_hours": hours,
            "orders": len(data.pairs)}
