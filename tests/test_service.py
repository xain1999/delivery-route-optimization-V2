"""Tests for the routing service of Chapter 15: unit, property-based, regression."""
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason='needs pip install -e ".[service]"')

from fastapi.testclient import TestClient  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from routing_service.app import app  # noqa: E402
from routing_service.samples import tuesday_request  # noqa: E402
from routing_service import app as app_module  # noqa: E402
from routing_service import config, solver  # noqa: E402
from routing_service.models import PlanRequest  # noqa: E402
from routing_service.solver import Exact, InvalidPlan, NoPlan, to_s  # noqa: E402

client = TestClient(app)
GOLDEN = Path(__file__).resolve().parent / "golden" / "tuesday_service.json"


def small_request(**over):
    req = {"depot": {"lat": 41.83, "lon": -87.68},
           "stops": [{"id": f"s{k}", "location": {"lat": 41.83 + 0.01 * k, "lon": -87.68},
                      "demand": 3} for k in range(1, 6)],
           "vehicles": [{"id": "v1", "capacity": 10, "shift_start": "08:00",
                         "shift_end": "17:00"},
                        {"id": "v2", "capacity": 10, "shift_start": "08:00",
                         "shift_end": "17:00"}],
           "time_limit_s": 0.5}
    req.update(over)
    return req


# <<check_plan>>
def check_plan(req: dict, plan: dict) -> None:
    """Invariants every plan must satisfy, whatever the request."""
    stops = {s["id"]: s for s in req["stops"]}
    vehicles = {v["id"]: v for v in req["vehicles"]}
    seen = [v["stop_id"] for r in plan["routes"] for v in r["visits"]]
    seen += [u["stop_id"] for u in plan["unassigned"]]
    assert sorted(seen) == sorted(stops)                     # each stop exactly once
    for r in plan["routes"]:
        veh = vehicles[r["vehicle_id"]]
        assert sum(stops[v["stop_id"]]["demand"] for v in r["visits"]) <= veh["capacity"]
        assert to_s(veh["shift_start"]) <= to_s(r["depart"])
        assert to_s(r["back"]) <= to_s(veh["shift_end"]) + 60          # rounding to minutes
        for v in r["visits"]:
            s = stops[v["stop_id"]]
            if s.get("window_end"):
                assert to_s(v["start"]) <= to_s(s["window_end"])
            if s.get("window_start"):
                assert to_s(v["start"]) >= to_s(s["window_start"])
    # the money reconciles: route costs, rounded to cents, add up to the total
    gap = abs(sum(r["cost"] for r in plan["routes"]) - plan["cost"])
    assert gap <= 0.01 * (len(plan["routes"]) + 1)
# <<end_check_plan>>


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_a_small_plan_is_valid():
    req = small_request()
    r = client.post("/plans/solve", json=req)
    assert r.status_code == 200
    check_plan(req, r.json())


def test_bad_requests_are_rejected_with_a_reason():
    req = small_request()
    req["stops"][0]["window_start"], req["stops"][0]["window_end"] = "12:00", "11:00"
    r = client.post("/plans/solve", json=req)
    assert r.status_code == 422 and "window ends before it starts" in r.text
    req = small_request()
    req["stops"][1]["id"] = req["stops"][0]["id"]
    assert client.post("/plans/solve", json=req).status_code == 422


def test_impossible_stops_come_back_with_a_reason():
    req = small_request()
    req["stops"][0].update(window_start="06:00", window_end="06:05")
    req["stops"][1]["demand"] = 50
    plan = client.post("/plans/solve", json=req).json()
    reasons = {u["stop_id"]: u["reason"] for u in plan["unassigned"]}
    assert "window" in reasons["s1"] and "exceeds" in reasons["s2"]


def test_background_jobs():
    r = client.post("/plans", json=small_request())
    assert r.status_code == 202
    job_id = r.json()["id"]
    for _ in range(100):
        job = client.get(f"/plans/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "done"
    check_plan(small_request(), job["result"])
    assert client.get("/plans/nope").status_code == 404


# <<property>>
stop = st.fixed_dictionaries({
    "lat": st.floats(41.70, 42.00), "lon": st.floats(-87.80, -87.55),
    "demand": st.integers(0, 12), "open": st.integers(7, 15), "width": st.integers(1, 4)})


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(stop, min_size=1, max_size=12), st.integers(1, 3))
def test_any_request_gives_a_valid_plan(raw, n_vehicles):
    req = small_request(
        stops=[{"id": f"s{k}", "location": {"lat": s["lat"], "lon": s["lon"]},
                "demand": s["demand"], "window_start": f"{s['open']:02d}:00",
                "window_end": f"{min(s['open'] + s['width'], 23):02d}:00"}
               for k, s in enumerate(raw)],
        vehicles=[{"id": f"v{k}", "capacity": 20, "shift_start": "07:00",
                   "shift_end": "18:00"} for k in range(n_vehicles)],
        time_limit_s=0.2)
    r = client.post("/plans/solve", json=req)
    assert r.status_code == 200
    check_plan(req, r.json())
# <<end_property>>


# <<regression>>
def test_tuesday_is_no_worse_than_the_golden_plan():
    """A change to the code must not make a known day more expensive."""
    golden = json.loads(GOLDEN.read_text())
    req = tuesday_request(time_limit_s=5)
    plan = client.post("/plans/solve", json=req).json()
    check_plan(req, plan)
    assert not plan["unassigned"]
    assert plan["cost"] <= golden["cost"] * 1.02          # 2% tolerance: a heuristic varies
# <<end_regression>>


# ---- the review's reproductions -------------------------------------------
def test_duplicate_vehicle_ids_are_rejected():
    req = small_request()
    req["vehicles"][1]["id"] = "v1"
    r = client.post("/plans/solve", json=req)
    assert r.status_code == 422 and "vehicle ids must be unique" in r.text


def test_money_is_computed_before_rounding():
    # a stop at the depot with 30 seconds of service, at $60 an hour: $0.50, not $0
    req = small_request(
        stops=[{"id": "s1", "location": {"lat": 41.83, "lon": -87.68}, "service_min": 0.5}],
        vehicles=[{"id": "v1", "capacity": 10, "shift_start": "08:00", "shift_end": "17:00",
                   "cost_per_hour": 60, "cost_per_mile": 0}])
    plan = client.post("/plans/solve", json=req).json()
    assert plan["cost"] == 0.5 and plan["routes"][0]["cost"] == 0.5


def test_a_tiny_time_limit_is_refused():
    assert client.post("/plans/solve", json=small_request(time_limit_s=1e-6)).status_code == 422


def test_no_solution_gives_an_honest_empty_plan(monkeypatch):
    def nothing(*args):
        raise NoPlan("the solver found no plan in time")
    monkeypatch.setattr(solver, "optimize", nothing)
    req = small_request()
    plan = solver.solve(PlanRequest(**req)).model_dump()
    assert plan["fallback"] and not plan["routes"] and plan["warnings"]
    assert len(plan["unassigned"]) == len(req["stops"])
    check_plan(req, plan)


def test_an_invalid_plan_is_never_published(monkeypatch):
    req = PlanRequest(**small_request())
    dist, secs = solver.matrices(req)
    service = [0] + [300] * len(req.stops)
    good = solver.schedule(req, 0, [1, 2], 8 * 3600, dist, secs, service)
    solver.validate_plan(req, [good], dist, secs, service)            # passes
    for broken in (solver.schedule(req, 0, [1, 1], 8 * 3600, dist, secs, service),   # twice
                   solver.schedule(req, 0, [1, 2, 3, 4], 8 * 3600, dist, secs, service),  # 12 > 10
                   solver.schedule(req, 0, [1], 6 * 3600, dist, secs, service)):   # before shift
        with pytest.raises(InvalidPlan):
            solver.validate_plan(req, [broken], dist, secs, service)
    late = Exact(**{**vars(good), "back": good.back + 1})               # does not add up
    with pytest.raises(InvalidPlan):
        solver.validate_plan(req, [late], dist, secs, service)
    monkeypatch.setattr(solver, "optimize", lambda *a: ([late], "ROUTING_SUCCESS"))
    plan = solver.solve(req)
    assert plan.fallback and not plan.routes


def test_admission_control_and_job_states(monkeypatch):
    monkeypatch.setattr(app_module, "pending", __import__("threading").BoundedSemaphore(1))
    r = client.post("/plans", json=small_request(time_limit_s=1))
    assert r.status_code == 202 and r.json()["status"] in ("queued", "running")
    busy = client.post("/plans", json=small_request())             # the one slot is taken
    assert busy.status_code == 429 and busy.headers["Retry-After"]
    job_id = r.json()["id"]
    for _ in range(100):
        if client.get(f"/plans/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.1)
    # finished jobs expire
    monkeypatch.setattr(config, "JOB_TTL_S", 0.0)
    time.sleep(0.01)
    assert client.get(f"/plans/{job_id}").status_code == 404
