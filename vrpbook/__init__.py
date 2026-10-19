"""vrpbook: companion code for *Delivery Route Optimization with Python*."""

from .distances import distance_matrix, haversine_m, travel_time_matrix
from .instances import Instance, Site, chicago_ch02, load_csv
from .ortools_solver import Solution, solve_cvrp
from .validate import RouteStats, total_distance, validate_routes

__version__ = "0.1.0"

__all__ = [
    "Instance", "Site", "Solution", "RouteStats", "chicago_ch02", "load_csv",
    "distance_matrix", "haversine_m", "travel_time_matrix", "solve_cvrp",
    "validate_routes", "total_distance",
]
