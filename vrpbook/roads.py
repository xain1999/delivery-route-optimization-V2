"""Road networks, shortest paths and travel-time matrices (Chapter 4).

Graphs follow the conventions of OSMnx, so a street network downloaded
with tools/fetch_chicago_roads.py, or the snapshot the book uses
(tools/fetch_chicago_network.py, loaded by load_osm), work with the same
functions:

    node attributes   x (longitude), y (latitude)
    edge attributes   length (meters), travel_time (seconds, free flow),
                      load_osm adds peak_time (seconds) and kind

All matrices are integer meters and seconds, as in Chapter 2.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .distances import EARTH_RADIUS_M
from .instances import Site

M_PER_MI = 1609.344
OSM_FILE = Path(__file__).resolve().parent.parent / "data" / "osm" / "Chicago_Network.graphml"


# <<load_osm>>
KIND = {"motorway": "expressway", "motorway_link": "expressway", "trunk": "expressway",
        "trunk_link": "expressway", "primary": "arterial", "primary_link": "arterial",
        "secondary": "arterial", "secondary_link": "arterial"}   # all others: local
DEFAULT_MPH = {"expressway": 55, "arterial": 30, "local": 25}  # Chicago's usual limits
ENTRY_DELAY_S = {"traffic_signals": 20.0, "stop": 8.0}          # entering such a node
PEAK_FACTOR = {"expressway": 2.2, "arterial": 1.5, "local": 1.15}  # assumed, see text


def load_osm(path: Path = OSM_FILE) -> nx.DiGraph:
    """Chicago's drivable streets from an OSMnx GraphML file (OpenStreetMap).

    Free-flow time is length at the posted speed limit (or the usual limit
    for the road's kind) plus a delay for entering an intersection with a
    signal or a stop sign.  Peak time multiplies it by an assumed factor
    for the road's kind: OpenStreetMap has no traffic.  Of parallel edges
    the fastest is kept."""
    raw = nx.read_graphml(path)
    G = nx.DiGraph(name="OpenStreetMap Chicago")
    for n, d in raw.nodes(data=True):
        G.add_node(int(n), x=float(d["x"]), y=float(d["y"]), control=d.get("highway"))
    for u, v, d in raw.edges(data=True):
        u, v = int(u), int(v)
        highway = _first(d["highway"])
        kind = KIND.get(highway, "local")
        mph = _mph(d.get("maxspeed")) or DEFAULT_MPH[kind]
        free = (float(d["length"]) / (mph * M_PER_MI / 3600)
                + ENTRY_DELAY_S.get(G.nodes[v]["control"], 0.0))
        if G.has_edge(u, v) and G[u][v]["travel_time"] <= free:
            continue
        G.add_edge(u, v, length=float(d["length"]), travel_time=free,
                   peak_time=free * PEAK_FACTOR[kind], kind=kind, highway=highway)
    # keep the part of the network where every node can reach every other
    core = max(nx.strongly_connected_components(G), key=len)
    G.graph["dropped_nodes"] = G.number_of_nodes() - len(core)
    return G.subgraph(core).copy()


def _first(value: str) -> str:
    """OSMnx writes a list of tags as "['primary', 'secondary']": keep the first."""
    return value.strip("[]").split(",")[0].strip(" '\"")


def _mph(value: str | None) -> float | None:
    """'30 mph', or a list of them: the lowest limit, or None if unknown."""
    if not value:
        return None
    limits = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mph", value)]
    return min(limits) if limits else None
# <<end_load_osm>>


def _xy(lat, lon, lat0: float) -> np.ndarray:
    """Meters east and north on a local equirectangular projection."""
    k = np.pi / 180 * EARTH_RADIUS_M
    return np.column_stack([np.asarray(lon) * k * np.cos(np.radians(lat0)),
                            np.asarray(lat) * k])


def gc_matrix(a: list[Site], b: list[Site] | None = None) -> np.ndarray:
    """Great-circle meters between two lists of sites (haversine)."""
    b = a if b is None else b
    p1, l1 = np.radians([s.lat for s in a])[:, None], np.radians([s.lon for s in a])[:, None]
    p2, l2 = np.radians([s.lat for s in b])[None, :], np.radians([s.lon for s in b])[None, :]
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# <<snap>>
def snap(G: nx.DiGraph, sites: list[Site], kinds=("arterial", "local")) -> tuple[np.ndarray, np.ndarray]:
    """Nearest node to each site among nodes on roads of the given kinds
    (any node if edges carry no kind, as in OSMnx graphs).  Returns the
    nodes and the straight-line distances to them in meters."""
    ok = sorted({n for u, v, k in G.edges(data="kind") if k is None or k in kinds
                 for n in (u, v)})
    lat0 = float(np.mean([s.lat for s in sites]))
    tree = cKDTree(_xy([G.nodes[n]["y"] for n in ok], [G.nodes[n]["x"] for n in ok], lat0))
    dist, idx = tree.query(_xy([s.lat for s in sites], [s.lon for s in sites], lat0))
    return np.array(ok)[idx], dist
# <<end_snap>>


def to_csr(G: nx.DiGraph, weight: str) -> tuple[csr_matrix, csr_matrix, dict]:
    """Sparse matrices of `weight` and of length, keeping the edge with the
    smallest weight between any two nodes (OSMnx graphs have parallel
    edges).  Cached on the graph, since building them takes a while."""
    cache = G.graph.setdefault("_csr", {})
    if weight not in cache:
        index = {u: k for k, u in enumerate(G.nodes)}
        best: dict = {}
        for u, v, d in G.edges(data=True):
            if (u, v) not in best or d[weight] < best[u, v][0]:
                best[u, v] = (d[weight], d["length"])
        r = [index[u] for u, _ in best]
        c = [index[v] for _, v in best]
        w = [max(x[0], 1e-6) for x in best.values()]     # scipy drops explicit zeros
        n = len(index)
        cache[weight] = (csr_matrix((w, (r, c)), shape=(n, n)),
                         csr_matrix(([x[1] for x in best.values()], (r, c)), shape=(n, n)),
                         index)
    return cache[weight]


# <<matrices>>
@dataclass
class RoadMatrices:
    meters: np.ndarray      # length of the route the weight prefers
    seconds: np.ndarray     # its travel time (or its length, if weight="length")
    nodes: np.ndarray       # graph node each site was snapped to
    access_m: np.ndarray    # straight line from each site to its node


def road_matrices(G: nx.DiGraph, sites: list[Site], weight: str = "travel_time",
                  local_mph: float = 12.0, local_circuity: float = 1.3,
                  kinds=("arterial", "local"), shortcut: bool = True) -> RoadMatrices:
    """Door-to-door trips between all sites: local streets from the site
    to its node, the best network route between nodes, local streets to
    the destination, or, if shortcut, local streets all the way when that
    is better.  A regional network has no local streets, so they are
    modeled as in Chapter 2: straight line times a circuity factor, at
    local_mph.  This is a hybrid model, not a measurement: with the
    shortcut, no trip is ever longer than local_circuity times the
    straight line."""
    nodes, access = snap(G, sites, kinds)
    W, L, index = to_csr(G, weight)
    k = np.array([index[u] for u in nodes])
    net_w, net_m = _network(W, L, k, k)
    return _assemble(sites, sites, net_w, net_m, access, access, nodes, weight,
                     local_mph, local_circuity, shortcut)


def _network(W, L, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weight and length of the best routes from each source to each
    destination: one Dijkstra per source, then walk the shortest-path
    tree back from all destinations at once to add up the lengths."""
    best, pred = dijkstra(W, indices=src, return_predecessors=True)
    assert np.isfinite(best[:, dst]).all(), "some site cannot be reached"
    length = np.zeros((len(src), len(dst)))
    for i, s in enumerate(src):
        cur, acc = dst.copy(), length[i]
        on = cur != s
        while on.any():
            p = pred[i, cur[on]]
            acc[on] += np.asarray(L[p, cur[on]]).ravel()
            cur[on] = p
            on = cur != s
    return best[:, dst], length


def _assemble(A, B, net_w, net_m, acc_a, acc_b, nodes, weight, local_mph, circuity,
              shortcut=True):
    v = local_mph * M_PER_MI / 3600                      # meters per second
    via_m = circuity * acc_a[:, None] + net_m + circuity * acc_b[None, :]
    direct_m = circuity * gc_matrix(A, B)
    if weight == "length":
        via_w, direct_w = via_m, direct_m
    else:
        via_w = circuity * (acc_a[:, None] + acc_b[None, :]) / v + net_w
        direct_w = direct_m / v
    local = (direct_w < via_w) & shortcut
    m = np.where(local, direct_m, via_m)
    w = np.where(local, direct_w, via_w)
    return RoadMatrices(np.rint(m).astype(np.int64), np.rint(w).astype(np.int64),
                        nodes, acc_a)
# <<end_matrices>>


def node_distances(G: nx.DiGraph, nodes, weight: str = "length") -> tuple[np.ndarray, np.ndarray]:
    """Network-only distances between graph nodes, with no access legs and
    no local-street model: (meters of the best route, straight-line meters
    between the nodes).  Their ratio is the network's own circuity."""
    W, L, index = to_csr(G, weight)
    k = np.array([index[u] for u in nodes])
    _, net_m = _network(W, L, k, k)
    pts = [Site(0, "", "", G.nodes[u]["y"], G.nodes[u]["x"]) for u in nodes]
    return net_m, gc_matrix(pts)


# <<extend>>
def extend(G: nx.DiGraph, M: RoadMatrices, sites: list[Site], new: Site,
           weight: str = "travel_time", local_mph: float = 12.0,
           local_circuity: float = 1.3, kinds=("arterial", "local"),
           shortcut: bool = True) -> RoadMatrices:
    """Add one site to existing matrices with two Dijkstra runs: forward
    from the new site for its row, backward (on the reversed graph) for
    its column."""
    node, acc = snap(G, [new], kinds)
    W, L, index = to_csr(G, weight)
    k_all = np.array([index[u] for u in np.append(M.nodes, node)])
    k_new = k_all[-1:]
    acc_all = np.append(M.access_m, acc)
    every = sites + [new]
    row_w, row_m = _network(W, L, k_new, k_all)
    col_w, col_m = _network(W.T.tocsr(), L.T.tocsr(), k_new, k_all)
    row = _assemble([new], every, row_w, row_m, acc, acc_all, node, weight,
                    local_mph, local_circuity, shortcut)
    col = _assemble(every, [new], col_w.T, col_m.T, acc_all, acc, node, weight,
                    local_mph, local_circuity, shortcut)
    n = len(sites)
    meters = np.zeros((n + 1, n + 1), dtype=np.int64)
    seconds = np.zeros_like(meters)
    for out, old, r, c in ((meters, M.meters, row.meters, col.meters),
                           (seconds, M.seconds, row.seconds, col.seconds)):
        out[:n, :n] = old
        out[n, :] = r[0]
        out[:, n] = c[:, 0]
        out[n, n] = 0
    return RoadMatrices(meters, seconds, np.append(M.nodes, node), acc_all)
# <<end_extend>>


# <<profile>>
# Share of the peak delay present at each hour of a weekday: an assumption,
# shaped like typical urban traffic, to replace with your own GPS data.
PROFILE_H = [0, 6, 7, 9, 10, 15, 16, 18, 20, 24]
PROFILE_W = [0, 0.1, 1, 1, 0.4, 0.4, 1, 1, 0.2, 0]


def congestion(t_s) -> np.ndarray:
    """Share of the peak delay at t_s seconds after midnight."""
    return np.interp(np.asarray(t_s) / 3600, PROFILE_H, PROFILE_W)


def td_time(free, peak, depart_s):
    """Travel time when leaving at depart_s: free flow plus the share of
    the extra peak time present at that moment."""
    return free + congestion(depart_s) * (peak - free)
# <<end_profile>>


# <<slot_matrix>>
def slot_matrix(free: np.ndarray, peak: np.ndarray, windows, service,
                depot_leave_s: int) -> np.ndarray:
    """A static matrix that approximates time-dependent travel times: the
    trip from i is timed when a van most likely leaves i, at the middle
    of i's window plus its service time."""
    leave = np.array([(a + b) / 2 + s for (a, b), s in zip(windows, service)])
    leave[0] = depot_leave_s
    return np.rint(td_time(free, peak, leave[:, None])).astype(np.int64)
# <<end_slot_matrix>>


# <<td_drive>>
def td_drive(route: list[int], depart: int, free, peak, service, windows):
    """Drive a route (depot at both ends) with time-dependent travel times.
    Returns the time back at the depot and, for each stop, (node, start of
    service, seconds late)."""
    t, stops = float(depart), []
    for i, j in zip(route, route[1:]):
        t += td_time(free[i, j], peak[i, j], t)
        if j == route[-1]:
            break
        a, b = windows[j]
        start = max(t, a)
        stops.append((j, start, max(0.0, start - b)))
        t = start + service[j]
    return t, stops
# <<end_td_drive>>


# <<closure>>
def metric_closure(T: np.ndarray) -> np.ndarray:
    """Shortcut every trip that is faster by way of another site: a van
    can always drive past j without stopping (Floyd-Warshall, O(n^3))."""
    T = T.copy()
    for j in range(len(T)):
        T = np.minimum(T, T[:, j][:, None] + T[j, :][None, :])
    return T
# <<end_closure>>


def asymmetry(T: np.ndarray) -> np.ndarray:
    """|t_ij - t_ji| relative to their mean, for every pair i < j."""
    i, j = np.triu_indices(len(T), 1)
    mean = (T[i, j] + T[j, i]) / 2
    return np.abs(T[i, j] - T[j, i]) / np.maximum(mean, 1)


# <<cache>>
class MatrixCache:
    """Road matrices on disk, keyed by everything that changes them: the
    network's version and contents, every argument of road_matrices, and
    the sites' coordinates rounded to about a meter."""

    def __init__(self, folder: Path, network_version: str):
        self.folder, self.version = Path(folder), network_version
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def key(self, G: nx.DiGraph, sites: list[Site], weight: str, **kw) -> str:
        args = inspect.signature(road_matrices).bind(G, sites, weight, **kw)
        args.apply_defaults()               # an omitted default = the same default
        params = {k: v for k, v in args.arguments.items() if k not in ("G", "sites")}
        coords = [(round(s.lat, 5), round(s.lon, 5)) for s in sites]
        blob = json.dumps([self.version, graph_fingerprint(G), params, coords])
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    def get(self, G: nx.DiGraph, sites: list[Site], weight: str = "travel_time",
            **kw) -> RoadMatrices:
        path = self.folder / f"{self.key(G, sites, weight, **kw)}.npz"
        if path.exists():
            self.hits += 1
            z = np.load(path)
            return RoadMatrices(z["meters"], z["seconds"], z["nodes"], z["access_m"])
        self.misses += 1
        M = road_matrices(G, sites, weight, **kw)
        np.savez_compressed(path, meters=M.meters, seconds=M.seconds,
                            nodes=M.nodes, access_m=M.access_m)
        return M
# <<end_cache>>


def graph_fingerprint(G: nx.DiGraph) -> str:
    """A hash of the nodes, edges and their attributes (not of G.graph,
    where to_csr keeps its cache).  Edits to speeds, lengths or the
    topology change it."""
    h = hashlib.sha256()
    for u, d in sorted(G.nodes(data=True), key=lambda e: repr(e[0])):
        h.update(repr((u, sorted(d.items()))).encode())
    for u, v, d in sorted(G.edges(data=True), key=lambda e: (repr(e[0]), repr(e[1]))):
        h.update(repr((u, v, sorted(d.items()))).encode())
    return h.hexdigest()[:20]


# <<osrm>>
def osrm_table(sites: list[Site], base_url: str = "http://localhost:5000",
               profile: str = "driving", client=None) -> tuple[np.ndarray, np.ndarray]:
    """Meters and seconds between all sites from an OSRM server's table
    service (Appendix A shows how to run one)."""
    import httpx
    coords = ";".join(f"{s.lon:.6f},{s.lat:.6f}" for s in sites)   # longitude first
    client = client or httpx.Client(timeout=120)
    r = client.get(f"{base_url}/table/v1/{profile}/{coords}",
                   params={"annotations": "distance,duration"})
    r.raise_for_status()
    body = r.json()
    if body.get("code") != "Ok":
        raise RuntimeError(f"OSRM: {body.get('code')} {body.get('message', '')}")
    if any(x is None for row in body["durations"] for x in row):
        raise RuntimeError("OSRM: some sites cannot reach each other")
    return (np.rint(np.array(body["distances"], float)).astype(np.int64),
            np.rint(np.array(body["durations"], float)).astype(np.int64))
# <<end_osrm>>
