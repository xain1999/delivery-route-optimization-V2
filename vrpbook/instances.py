"""Problem instances: sites, demands and fleets.

The book's running example is a fictional grocery-delivery start-up in
Chicago.  Coordinates are approximate neighborhood centers. They are good
enough for planning at the level of Chapter 2 and are replaced by real
road-network data in Chapter 4.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Site:
    """A location that a vehicle can visit (the depot or a customer)."""

    id: int
    name: str
    district: str
    lat: float
    lon: float


# <<instance>>
@dataclass
class Instance:
    """A capacitated vehicle routing instance.

    Node 0 is always the depot. Nodes 1..n are customers.
    """

    name: str
    sites: list[Site]
    demands: list[int]          # demands[0] == 0 for the depot
    n_vehicles: int
    capacity: int               # identical vehicles in Chapter 2
    meta: dict = field(default_factory=dict)

    @property
    def n_customers(self) -> int:
        return len(self.sites) - 1

    @property
    def total_demand(self) -> int:
        return sum(self.demands)
    # <<end_instance>>

    def subset(self, customer_ids: list[int], name: str | None = None,
               n_vehicles: int | None = None) -> "Instance":
        """Return a smaller instance with the depot and chosen customers."""
        keep = [0] + list(customer_ids)
        return Instance(
            name=name or f"{self.name}-subset",
            sites=[self.sites[i] for i in keep],
            demands=[self.demands[i] for i in keep],
            n_vehicles=n_vehicles or self.n_vehicles,
            capacity=self.capacity,
            meta={"original_ids": keep},
        )


# <<load_csv>>
def load_csv(path: str | Path, n_vehicles: int, capacity: int,
             name: str | None = None) -> Instance:
    """Read sites and demands from a CSV file with columns
    id, name, district, lat, lon, demand (row 0 must be the depot)."""
    path = Path(path)
    sites, demands = [], []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sites.append(Site(int(row["id"]), row["name"], row["district"],
                              float(row["lat"]), float(row["lon"])))
            demands.append(int(row["demand"]))
    if demands[0] != 0:
        raise ValueError("Row 0 must be the depot, with zero demand.")
    if [s.id for s in sites] != list(range(len(sites))):
        raise ValueError("Site ids must be 0, 1, ..., n in order.")
    return Instance(name or path.stem, sites, demands, n_vehicles, capacity)
# <<end_load_csv>>


def chicago_ch02() -> Instance:
    """The Chapter 2 instance: 30 customers, 4 vans of 100 totes."""
    return load_csv(DATA_DIR / "chicago_ch02.csv", n_vehicles=4,
                    capacity=100, name="chicago-30")


# <<synthetic>>
def chicago_synthetic(n: int, seed: int = 0, capacity: int = 100,
                      min_demand: int = 3, max_demand: int = 25) -> Instance:
    """n random customers in Chicago, half spread over the city and half
    around the neighborhood centers of Chapter 2, west of the lakeshore.
    Used in Part III to see how algorithms scale."""
    import numpy as np
    rng = np.random.default_rng(seed)
    base = chicago_ch02()
    centers = [(s.lat, s.lon) for s in base.sites[1:]]

    def shore(lat: float) -> float:          # rough longitude of the lakeshore
        return float(np.interp(lat, [41.65, 41.88, 42.02], [-87.53, -87.61, -87.66]))

    sites = [base.sites[0]]
    while len(sites) <= n:
        if rng.random() < 0.5:
            lat, lon = rng.uniform(41.65, 42.02), rng.uniform(-87.85, -87.53)
        else:
            c = centers[rng.integers(len(centers))]
            lat, lon = c[0] + rng.normal(0, 0.008), c[1] + rng.normal(0, 0.010)
        if lon < shore(lat) - 0.003 and 41.64 < lat < 42.03 and lon > -87.86:
            sites.append(Site(len(sites), f"C{len(sites)}", "", float(lat), float(lon)))
    demands = [0] + [int(x) for x in rng.integers(min_demand, max_demand + 1, size=n)]
    vans = -(-sum(demands) // capacity)
    return Instance(f"chicago-syn-{n}", sites, demands, vans, capacity)
# <<end_synthetic>>
