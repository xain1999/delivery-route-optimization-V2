"""Orders that may be left out: penalties, prizes and service levels (Chapter 9)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.linear_solver import pywraplp
from ortools.util.optional_boolean_pb2 import BOOL_TRUE

from .timewindows import TWData, solve_vrptw


# <<policy>>
@dataclass
class PenaltyPolicy:
    """What it costs the business to postpone one order to tomorrow."""
    voucher: float = 15.0                 # credit given to every postponed customer
    cancel_prob: float = 0.25             # share of postponed orders then cancelled
    margin: float = 0.30                  # gross margin on the basket
    churn_prob: dict = field(default_factory=lambda: {"club": 0.10, "regular": 0.08})
    future_margin: dict = field(default_factory=lambda: {"club": 3000.0, "regular": 1000.0})
    repeat_factor: float = 3.0            # an order postponed twice hurts much more

    def penalty(self, value: float, tier: str, postponed: bool) -> float:
        p = (self.voucher                                   # we pay for sure
             + self.cancel_prob * self.margin * value       # sale lost today
             + self.churn_prob[tier] * self.future_margin[tier])   # customer lost
        return p * (self.repeat_factor if postponed else 1.0)
# <<end_policy>>


# <<portfolio>>
STRATEGIES = ("SAVINGS", "PARALLEL_CHEAPEST_INSERTION", "GLOBAL_CHEAPEST_ARC",
              "LOCAL_CHEAPEST_INSERTION")


def solve_with_penalties(data: TWData, penalties: list[float | None],
                         strategies=STRATEGIES, time_limit_s: float = 30.0, **cost):
    """With optional orders the first solution largely decides which orders
    are dropped, so start from several and keep the cheapest plan."""
    runs = {}
    for st in strategies:
        runs[st] = solve_vrptw(data, objective="cost", time_limit_s=time_limit_s,
                               first_solution=st, penalties=penalties, **cost)
    best = min((s for s in runs.values() if s.routes), key=lambda s: s.objective)
    return best, {st: s.objective / 100_000 for st, s in runs.items() if s.routes}
# <<end_portfolio>>


def removal_savings(route: list[int], cost: np.ndarray) -> dict[int, float]:
    """Cost saved by skipping each stop of a route, the others unchanged:
    c(i-1, i) + c(i, i+1) - c(i-1, i+1)."""
    return {route[k]: cost[route[k - 1], route[k]] + cost[route[k], route[k + 1]]
            - cost[route[k - 1], route[k + 1]] for k in range(1, len(route) - 1)}


# <<orienteering_mip>>
def orienteering_mip(time: np.ndarray, service: list[int], prize: list[float],
                     demand: list[int], budget: int, capacity: int,
                     time_limit_s: float = 600.0) -> dict:
    """One vehicle, node 0 the depot: choose and sequence stops to collect
    the largest total prize within a time budget.  Exact, by cutting planes:
    solve, find subtours, cut them off, solve again (as in Chapter 5)."""
    n = len(prize)
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): s.BoolVar(f"x{i}_{j}") for i in range(n) for j in range(n) if i != j}
    y = [s.BoolVar(f"y{i}") for i in range(n)]
    s.Add(y[0] == 1)
    for i in range(n):
        s.Add(sum(x[i, j] for j in range(n) if j != i) == y[i])     # leave once if visited
        s.Add(sum(x[j, i] for j in range(n) if j != i) == y[i])     # enter once if visited
    # a single route, so its duration is a sum over the arcs it uses
    s.Add(sum((service[i] + int(time[i, j])) * x[i, j] for (i, j) in x) <= budget)
    s.Add(sum(demand[i] * y[i] for i in range(n)) <= capacity)
    s.Maximize(sum(prize[i] * y[i] for i in range(1, n)))
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, 0.0)
    rounds = 0
    while True:
        s.SetTimeLimit(int(time_limit_s * 1000))
        if s.Solve(params) != pywraplp.Solver.OPTIMAL:
            return {"status": "not solved", "rounds": rounds}
        succ = {i: j for (i, j), v in x.items() if v.solution_value() > 0.5}
        tours, seen = [], set()
        for start in succ:
            if start in seen:
                continue
            tour, k = [], start
            while k not in seen:
                seen.add(k)
                tour.append(k)
                k = succ[k]
            tours.append(tour)
        subtours = [t for t in tours if 0 not in t]
        if not subtours:
            break
        rounds += 1
        for S in subtours:            # generalized subtour elimination cuts
            inner = sum(x[i, j] for i in S for j in S if i != j)
            for k in S:
                s.Add(inner <= sum(y[i] for i in S) - y[k])
    route, node = [0], 0
    while True:
        node = succ[node]
        route.append(node)
        if node == 0:
            break
    return {"status": "optimal", "prize": s.Objective().Value(), "route": route,
            "rounds": rounds}
# <<end_orienteering_mip>>


# <<orienteering_ortools>>
def solve_orienteering(time: np.ndarray, service: list[int], prize: list[float],
                       demand: list[int], budget: int, capacity: int,
                       time_limit_s: float = 10.0, swap_moves: bool = True,
                       metaheuristic: str = "TABU_SEARCH") -> dict:
    """The same problem in OR-Tools: every stop is optional, skipping it
    costs its prize, and travel time only breaks ties between equal prizes.
    Strictly so: prizes count in whole cents, and a cent is priced above
    budget + 1 seconds, more than any feasible route can drive."""
    n = len(prize)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(f, t):
        i, j = manager.IndexToNode(f), manager.IndexToNode(t)
        return service[i] + int(time[i, j])
    cb = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)               # seconds
    routing.AddDimension(cb, 0, budget, True, "Time")          # hard time budget
    dem = routing.RegisterUnaryTransitCallback(lambda f: demand[manager.IndexToNode(f)])
    routing.AddDimension(dem, 0, capacity, True, "Load")
    per_cent = budget + 1     # a cent of prize outweighs any feasible amount of driving
    for i in range(1, n):
        penalty = round(prize[i] * 100) * per_cent
        routing.AddDisjunction([manager.NodeToIndex(i)], penalty)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic)
    if swap_moves:            # replace a served stop by one or more unserved ones
        params.local_search_operators.use_extended_swap_active = BOOL_TRUE
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("no solution within the time limit")
    route, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    route.append(0)
    return {"route": route, "prize": sum(prize[i] for i in route[1:-1]),
            "duration": sum(time_cb(manager.NodeToIndex(i), manager.NodeToIndex(j))
                            for i, j in zip(route, route[1:]))}
# <<end_orienteering_ortools>>
