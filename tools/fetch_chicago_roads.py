"""Download Chicago's drivable street network and build travel matrices.

Run this once on a computer with internet access (it needs OpenStreetMap's
Overpass API, a few minutes and about 1 GB of memory):

    pip install osmnx
    python tools/fetch_chicago_roads.py

It writes, under data/osm/:
    chicago_drive.graphml          the street network
    chicago_ch02_matrix.npz        meters and seconds between the
    chicago_tuesday_matrix.npz     sites of Chapters 2 and 3

Map data (c) OpenStreetMap contributors, Open Database License.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import osmnx as ox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vrpbook import load_csv                       # noqa: E402
from vrpbook.roads import road_matrices            # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = DATA / "osm"
# west, south, east, north: the city plus a margin for routes along its edge
BBOX = (-87.95, 41.62, -87.50, 42.05)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("downloading the drivable network (several minutes)...")
    # <<osmnx>>
    G = ox.graph_from_bbox(BBOX, network_type="drive")   # every street a car may use
    G = ox.routing.add_edge_speeds(G)          # km/h from maxspeed tags or road class
    G = ox.routing.add_edge_travel_times(G)    # seconds per edge, free flow
    ox.save_graphml(G, OUT / "chicago_drive.graphml")
    # <<end_osmnx>>
    print(f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    for name in ("chicago_ch02", "chicago_tuesday"):
        sites = load_csv(DATA / f"{name}.csv", 1, 1).sites
        M = road_matrices(G, sites)             # the same function as for load_osm()'s graph
        np.savez_compressed(OUT / f"{name}_matrix.npz", nodes=M.nodes, meters=M.meters,
                            seconds=M.seconds)
        print(f"{name}: {len(sites)} sites")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
