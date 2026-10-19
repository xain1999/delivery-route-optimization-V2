"""Chapter 2, Listing 2.1: the smallest useful route optimizer.

One van visits all 30 customers and returns to the depot (a traveling
salesman problem).  Capacity is ignored for now. Section 2.5 adds it.

Run:  python examples/ch02_first_route.py
"""
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from vrpbook import chicago_ch02, distance_matrix

# 1. Data: sites and an integer distance matrix (meters).
inst = chicago_ch02()
dist = distance_matrix(inst.sites, circuity=1.3)

# 2. Model: one vehicle, starting and ending at node 0 (the depot).
manager = pywrapcp.RoutingIndexManager(len(dist), 1, 0)
routing = pywrapcp.RoutingModel(manager)


def distance_cb(from_index, to_index):
    """Cost of driving from one stop to the next, in meters."""
    i = manager.IndexToNode(from_index)
    j = manager.IndexToNode(to_index)
    return int(dist[i, j])


transit = routing.RegisterTransitCallback(distance_cb)
routing.SetArcCostEvaluatorOfAllVehicles(transit)

# 3. Search: build a first route greedily, then improve it for 5 seconds.
params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
params.time_limit.FromSeconds(5)
solution = routing.SolveWithParameters(params)

# 4. Read the route back as a list of node numbers.
index, route = routing.Start(0), []
while not routing.IsEnd(index):
    route.append(manager.IndexToNode(index))
    index = solution.Value(routing.NextVar(index))
route.append(manager.IndexToNode(index))

print("Route:", " -> ".join(map(str, route)))
print(f"Length: {solution.ObjectiveValue() / 1609.344:.1f} miles")
