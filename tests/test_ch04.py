"""Tests for Chapter 4: road networks and travel-time matrices."""

import httpx
import networkx as nx
import numpy as np
import pytest

from vrpbook.distances import triangle_violations
from vrpbook.instances import Site
from vrpbook.roads import (OSM_FILE, PROFILE_H, PROFILE_W, MatrixCache, extend, load_osm,
                           metric_closure, osrm_table, road_matrices, snap,
                           td_time)

M_PER_DEG = 111_195.0


def grid(n=8, block_m=400.0, one_way_row=3):
    """A small street grid around Chicago's latitude: 30 mph streets, one
    of them one-way eastbound, with a 50 mph diagonal shortcut."""
    G = nx.DiGraph()
    lat0, lon0 = 41.85, -87.65
    k = np.cos(np.radians(lat0))
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, y=lat0 + i * block_m / M_PER_DEG,
                       x=lon0 + j * block_m / (M_PER_DEG * k))
    mps = 30 * 1609.344 / 3600
    for i in range(n):
        for j in range(n):
            for di, dj in ((0, 1), (1, 0)):
                if i + di >= n or j + dj >= n:
                    continue
                a, b = i * n + j, (i + di) * n + j + dj
                d = dict(length=block_m, travel_time=block_m / mps, kind="arterial")
                G.add_edge(a, b, **d)
                if not (di == 0 and i == one_way_row):
                    G.add_edge(b, a, **d)
    for t in range(n - 1):
        a, b = t * n + t, (t + 1) * n + t + 1
        L = block_m * 2 ** 0.5
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, length=L, travel_time=L / (50 * 1609.344 / 3600), kind="expressway")
    return G


def sites_on(G, cells, n=8):
    """Sites exactly at the given (row, column) intersections."""
    nodes = [i * n + j for i, j in cells]
    return [Site(k, str(u), "", G.nodes[u]["y"], G.nodes[u]["x"]) for k, u in enumerate(nodes)]


def test_matrices_match_networkx_shortest_paths():
    G = grid()
    S = sites_on(G, [(0, 0), (7, 7), (3, 6), (6, 1), (2, 2)])
    M = road_matrices(G, S, "travel_time", local_mph=1e-3)   # no shortcuts on local streets
    for i, a in enumerate(S):
        for j, b in enumerate(S):
            t = nx.shortest_path_length(G, int(a.name), int(b.name), weight="travel_time")
            assert abs(M.seconds[i, j] - t) <= 1
    # the diagonal expressway makes the fastest path longer than the shortest one
    L = road_matrices(G, S, "length", local_mph=1e-3)
    assert (M.meters >= L.meters - 1).all()


def test_one_way_street_makes_the_matrix_asymmetric():
    G = grid()
    S = sites_on(G, [(3, 1), (3, 5)])
    M = road_matrices(G, S, "travel_time", local_mph=1e-3)
    assert M.seconds[0, 1] < M.seconds[1, 0]       # eastbound is direct, westbound detours


def test_snap_skips_expressway_only_nodes():
    G = grid()
    ramp = 999
    G.add_edge(ramp, 0, length=10, travel_time=1, kind="expressway")
    G.add_edge(0, ramp, length=10, travel_time=1, kind="expressway")
    G.nodes[ramp].update(y=41.85 - 0.0001, x=-87.65)
    s = [Site(0, "near the ramp", "", 41.85 - 0.0001, -87.65)]
    assert snap(G, s, kinds=("arterial", "expressway"))[0][0] == ramp
    assert snap(G, s)[0][0] == 0


def test_extend_equals_full_recomputation():
    G = grid()
    S = sites_on(G, [(0, 0), (7, 7), (3, 6), (6, 1), (2, 5)])
    full = road_matrices(G, S)
    part = extend(G, road_matrices(G, S[:-1]), S[:-1], S[-1])
    assert np.abs(full.seconds - part.seconds).max() <= 1
    assert np.abs(full.meters - part.meters).max() <= 1


def test_metric_closure_removes_triangle_violations():
    T = np.array([[0, 10, 50], [10, 0, 10], [50, 10, 0]])
    assert triangle_violations(T, tol=0) == 2
    C = metric_closure(T)
    assert triangle_violations(C, tol=0) == 0 and C[0, 2] == 20


def test_profile_keeps_fifo():
    """Leaving later never means arriving earlier, as long as the extra
    peak time is below 1 / (steepest slope of the profile)."""
    slopes = np.abs(np.diff(PROFILE_W) / np.diff(PROFILE_H))       # per hour
    max_extra = 3600 / slopes.max()
    free, peak = 600.0, 600.0 + 0.95 * max_extra
    t = np.arange(0, 24 * 3600, 30.0)
    arrive = t + td_time(free, peak, t)
    assert (np.diff(arrive) > 0).all()


def test_cache_returns_the_same_matrices(tmp_path):
    G = grid()
    S = sites_on(G, [(0, 0), (7, 7), (3, 6)])
    cache = MatrixCache(tmp_path, "grid-v1")
    a = cache.get(G, S)
    b = cache.get(G, S)
    assert (cache.hits, cache.misses) == (1, 1)
    assert (a.seconds == b.seconds).all()
    assert cache.key(G, S, "travel_time") != MatrixCache(tmp_path, "grid-v2").key(G, S, "travel_time")
    # an omitted default and the same value given explicitly share an entry
    assert cache.key(G, S, "travel_time") == cache.key(G, S, "travel_time", local_mph=12.0)


@pytest.mark.parametrize("change", [dict(local_mph=60.0), dict(local_circuity=1.6),
                                    dict(kinds=("arterial", "expressway")),
                                    dict(shortcut=False)])
def test_cache_key_covers_every_argument(tmp_path, change):
    G = grid()
    S = sites_on(G, [(0, 0), (7, 7), (3, 6)])
    cache = MatrixCache(tmp_path, "grid-v1")
    assert cache.key(G, S, "travel_time") != cache.key(G, S, "travel_time", **change)


def test_cache_does_not_return_matrices_for_other_settings(tmp_path):
    # the review's case: cached at 1 mph, asked again at 60 mph
    G = grid()
    S = [Site(0, "a", "", 41.8500, -87.6500), Site(1, "b", "", 41.8520, -87.6480)]
    cache = MatrixCache(tmp_path, "grid-v1")
    slow = cache.get(G, S, local_mph=1.0)
    fast = cache.get(G, S, local_mph=60.0)
    assert (fast.seconds == road_matrices(G, S, local_mph=60.0).seconds).all()
    assert fast.seconds[0, 1] < slow.seconds[0, 1]


def test_cache_notices_an_edited_network(tmp_path):
    G = grid()
    S = sites_on(G, [(0, 0), (7, 7)])
    cache = MatrixCache(tmp_path, "grid-v1")
    before = cache.key(G, S, "travel_time")
    for *_, d in G.edges(data=True):
        d["travel_time"] *= 2                      # a slower speed profile, same version label
    assert cache.key(G, S, "travel_time") != before


def test_without_the_shortcut_trips_are_never_shorter():
    G = grid()
    S = [Site(0, "a", "", 41.8510, -87.6490), Site(1, "b", "", 41.8530, -87.6470),
         Site(2, "c", "", 41.8700, -87.6200)]
    on = road_matrices(G, S, "length")
    off = road_matrices(G, S, "length", shortcut=False)
    assert (off.meters >= on.meters).all()
    # with the shortcut no trip exceeds 1.3 times the straight line, by construction
    from vrpbook.roads import gc_matrix
    gc = gc_matrix(S)
    assert (on.meters <= 1.3 * gc + 1).all()


def test_osrm_client_parses_the_table_service():
    S = [Site(0, "a", "", 41.83, -87.68), Site(1, "b", "", 41.88, -87.63)]

    def handler(request):
        assert request.url.path.startswith("/table/v1/driving/-87.680000,41.830000;")
        assert request.url.params["annotations"] == "distance,duration"
        return httpx.Response(200, json={"code": "Ok", "durations": [[0, 801.4], [779.6, 0]],
                                         "distances": [[0, 7703.2], [7655.0, 0]]})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    m, s = osrm_table(S, client=client)
    assert s.tolist() == [[0, 801], [780, 0]] and m[0, 1] == 7703


def test_load_osm_reads_speed_limits_and_intersection_control(tmp_path):
    G = nx.MultiDiGraph()
    G.add_node(1, x=-87.68, y=41.83)
    G.add_node(2, x=-87.67, y=41.83, highway="traffic_signals")
    G.add_node(3, x=-87.66, y=41.83)
    G.add_edge(1, 2, length=804.672, highway="secondary", maxspeed="['30 mph', '35 mph']")
    G.add_edge(2, 1, length=804.672, highway="secondary")
    G.add_edge(2, 3, length=804.672, highway="['motorway', 'motorway_link']")
    G.add_edge(3, 2, length=804.672, highway="motorway", maxspeed="55 mph")
    G.add_edge(1, 2, length=900.0, highway="residential")       # a slower parallel edge
    for _, _, d in G.edges(data=True):
        d.update({k: str(v) for k, v in d.items()})
    path = tmp_path / "tiny.graphml"
    nx.write_graphml(G, path)
    H = load_osm(path)
    assert abs(H[1][2]["travel_time"] - (60 + 20)) < 1e-6      # half a mile at 30 mph + signal
    assert H[2][3]["kind"] == "expressway" and abs(H[2][3]["travel_time"] - 32.727) < 1e-3
    assert H[1][2]["peak_time"] == pytest.approx(1.5 * H[1][2]["travel_time"])
    assert H.number_of_edges() == 4                             # parallel edge dropped


@pytest.mark.skipif(not OSM_FILE.exists(), reason="run tools/fetch_chicago_network.py first")
def test_osm_network_is_in_chicago_and_routes_agree_with_networkx():
    G = load_osm()
    lat = [d["y"] for _, d in G.nodes(data=True)]
    assert 41.6 < np.median(lat) < 42.1
    S = [Site(0, "depot", "", 41.83, -87.68), Site(1, "Rogers Park", "", 42.0105, -87.667)]
    M = road_matrices(G, S, "travel_time", local_mph=1e-3)
    u, v = M.nodes
    t = nx.shortest_path_length(G, u, v, weight="travel_time")
    acc = 1.3 * M.access_m / (1e-3 * 1609.344 / 3600)
    assert abs(M.seconds[0, 1] - (acc[0] + t + acc[1])) <= 1
    assert all(d["peak_time"] >= d["travel_time"] for *_, d in G.edges(data=True))
