# Delivery Route Optimization with Python — companion code

[![tests](https://github.com/xain1999/Delivery-Route-Optimization-with-Python/actions/workflows/tests.yml/badge.svg)](https://github.com/xain1999/Delivery-Route-Optimization-with-Python/actions/workflows/tests.yml)

Companion code for the book **_Delivery Route Optimization with Python: From a
First Working Planner to Production Systems_** by Mohammed Islam Hadjoudj.

The book takes a delivery start-up from its first optimized routes to a
production routing service: clean data, road-network travel times, time
windows, shifts and breaks, mixed fleets, pickups and deliveries, optional
orders, construction heuristics, local search, ALNS and hybrid genetic search
written from scratch and compared with PyVRP, lower bounds and exact methods,
re-optimization during the day, a FastAPI routing service, and a complete case
study.

Every listing in the book is extracted from this repository, and every number,
table and figure in the book is produced by a script in `examples/`. The tag
`v1.0.0` is the code as printed in the first edition.

## Install

Python 3.11 or later.

```bash
git clone https://github.com/xain1999/Delivery-Route-Optimization-with-Python.git
cd Delivery-Route-Optimization-with-Python
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev,service]"
pytest -q                                             # a few minutes
```

Extras: `dev` (pytest, Hypothesis, httpx) for the tests, `service` (FastAPI,
Uvicorn, Pydantic) for the routing service of Chapter 15 and the case study of
Chapter 16, whose tests are skipped without it, and `roads` (pyproj, OSMnx) for
the road networks of Chapter 4.

To reproduce the book's numbers with the exact library versions that produced
them (Python 3.13, OR-Tools 9.15, PyVRP 0.14.0), install the lock file first:

```bash
pip install -r requirements-lock.txt
pip install -e ".[dev,service,roads]" --no-deps
```

Each saved result in `results/` records the versions that produced it.

## A first route in ten seconds

```bash
python examples/ch02_first_route.py
```

```python
from vrpbook import chicago_ch02, distance_matrix, solve_cvrp, validate_routes

inst = chicago_ch02()
dist = distance_matrix(inst.sites, circuity=1.3)          # integer meters
sol = solve_cvrp(dist, inst.demands, n_vehicles=4, capacity=100, time_limit_s=10)
validate_routes(sol.routes, dist, inst.demands, 100)      # independent check
print(sol.routes)
```

## Layout

| Path | Contents |
|---|---|
| `vrpbook/` | The library, built up chapter by chapter |
| `routing_service/` | The FastAPI routing service of Chapter 15 |
| `case_study/` | The Hearthline Home case study of Chapter 16 |
| `examples/` | One script per chapter. Every result in the book comes from them |
| `tools/` | Data builders and downloaders |
| `data/` | Instances: Chicago (Lakeshore Larder), Houston, Atlanta, a CVRPLIB benchmark |
| `results/` | Saved results of the experiment scripts (redraw with `--from-json`) |
| `tests/` | Unit, property-based and regression tests |

## The code, chapter by chapter

| Chapter | Module(s) | Script | Time |
|---|---|---|---|
| 1 Delivery routing in practice | `costs.py` | `ch01_costs.py` | seconds |
| 2 Your first route optimizer | `distances.py`, `instances.py`, `ortools_solver.py`, `validate.py` | `ch02_first_route.py`, `ch02_experiments.py` | 6 min |
| 3 Preparing the data | `orders.py`, `addresses.py`, `geocode.py` | `ch03_experiments.py` | 1 min |
| 4 Road networks and travel times | `roads.py` | `ch04_experiments.py` | 10 min |
| 5 The TSP and the CVRP | `formulations.py`, `exact_tsp.py` | `ch05_experiments.py` | 15 min |
| 6 Time windows | `timewindows.py` | `ch06_experiments.py` | 15 min |
| 7 Shifts, breaks, fleets | `fleet.py` | `ch07_experiments.py` | 3 min |
| 8 Pickups and multiple depots | `pdp.py` | `ch08_experiments.py` | 5 min |
| 9 Optional orders | `optional.py` | `ch09_experiments.py` | 20 min |
| 10 Construction heuristics | `construction.py` | `ch10_experiments.py` | 5 min |
| 11 Local search | `localsearch.py` | `ch11_experiments.py` | 2 min |
| 12 ALNS and hybrid genetic search | `metaheuristics.py`, `pyvrp_adapter.py` | `ch12_experiments.py` | 25 min |
| 13 How good is my solution? | `bounds.py` | `ch13_experiments.py` | 25 min |
| 14 Re-optimizing during the day | `dynamic.py` | `ch14_experiments.py` | 15 min |
| 15 A routing service | `routing_service/` | `ch15_experiments.py` | 1 min |
| 16 Case study | `case_study/hearthline.py` | `ch16_experiments.py` | 6 min |

Run a script from the repository root, for example
`python examples/ch10_experiments.py`. Results are written to `results/chNN/`.
`--from-json` redraws from saved results without solving again. (`--book PATH`
writes the book's LaTeX tables and figures into a copy of the book source. You do not need it to run the code.)

A few scripts need something first:

- Chapter 4: `python tools/fetch_chicago_network.py` downloads the OpenStreetMap street network (32 MB).
- Chapters 11, 12 and 13 reuse the instances and results of Chapter 10. Chapter 13 also uses Chapter 12's.
- Chapter 16: `python tools/make_hearthline.py` rebuilds the raw order export (already included).

Searches with a time limit find slightly different plans on a faster or
slower computer, even with the same seed, so your numbers may differ a little
from the printed ones. The conclusions should not. Where an exact rerun
matters, stop on a count of iterations or solutions instead of seconds.

## The routing service

```bash
uvicorn routing_service.app:app --reload        # http://localhost:8000/docs
# or
docker build -t routing-service .
docker run -p 8000:8000 -e ROUTING_API_KEY=change-me routing-service
curl -X POST localhost:8000/plans/solve -H "Content-Type: application/json" \
     -H "X-API-Key: change-me" -d @data/requests/tuesday.json
```

Without `ROUTING_API_KEY` the service runs in development mode: it accepts
any caller and logs a warning. Settings (`ROUTING_MAX_WORKERS`,
`ROUTING_MAX_PENDING`, `ROUTING_MAX_TIME_LIMIT_S`, `ROUTING_SYNC_LIMIT_S`,
`ROUTING_JOB_TTL_S`, ...) are in `routing_service/config.py`. It is a
single-process demonstration of the pattern (jobs live in memory), not a
complete production system. Chapter 15 says what to change for that.

## Data and licenses

- The code is released under the MIT license (see `LICENSE`).
- The companies, customers and orders are fictional. Neighborhoods and
  coordinates are approximate real places in Chicago, Houston, Phoenix and Atlanta.
- `data/benchmarks/E-n22-k4.vrp` (Christofides and Eilon, 1969) is from CVRPLIB.
- The Chicago street network of Chapter 4 is OpenStreetMap data,
  © OpenStreetMap contributors, available under the
  [Open Database License](https://www.openstreetmap.org/copyright). It is not
  included here. `tools/fetch_chicago_network.py` downloads a pinned OSMnx
  extract (about 32 MB) into `data/osm/` and checks its SHA-256.
  `tools/fetch_chicago_roads.py` downloads a fresh extract with OSMnx.

## Citing

If this code helps your work, please cite the book (see `CITATION.cff`).
