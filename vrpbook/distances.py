"""Travel distances and times between sites.

All distances are integer meters and all times integer seconds. Convert
to miles or kilometers only for display.

Chapter 2 uses great-circle (haversine) distances inflated by a
*circuity factor* that accounts, on average, for the fact that roads are
not straight.  Chapter 4 replaces this with shortest paths on the real
road network.
"""
from __future__ import annotations

import math

import numpy as np

from .instances import Site

EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius in meters (IUGG)


# <<haversine>>
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))
# <<end_haversine>>


# <<distance_matrix>>
def distance_matrix(sites: list[Site], circuity: float = 1.3) -> np.ndarray:
    """Estimated road distance in whole meters between every pair of sites.

    circuity: ratio of road distance to straight-line distance.  Values
    of 1.2-1.4 are typical in cities. Calibrate it on a sample of real
    trips before relying on it (see Chapter 4).
    """
    n = len(sites)
    d = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            gc = haversine_m(sites[i].lat, sites[i].lon, sites[j].lat, sites[j].lon)
            d[i, j] = d[j, i] = round(circuity * gc)
    return d
# <<end_distance_matrix>>


def travel_time_matrix(dist_m: np.ndarray, speed_kmh: float = 25.0) -> np.ndarray:
    """Travel times in whole seconds, assuming a constant average speed."""
    return np.rint(dist_m / (speed_kmh / 3.6)).astype(np.int64)


def triangle_violations(d: np.ndarray, tol: int = 1) -> int:
    """Count triples (i, j, k) with d[i, k] > d[i, j] + d[j, k] + tol.

    Rounding to integers can create violations of at most a meter or
    two, hence the tolerance.
    """
    n = len(d)
    count = 0
    for j in range(n):
        # d[i,j] + d[j,k] for all i, k at once
        via = d[:, j][:, None] + d[j, :][None, :]
        count += int(np.sum(d > via + tol))
    return count
