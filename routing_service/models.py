"""Request and response models of the routing service (Chapter 15).

Times are "HH:MM" strings on the day of the plan. Distances in miles,
money in dollars, rounded for display (totals are computed before
rounding).  Pydantic checks types and simple ranges before the planner
sees anything.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

# <<models>>
HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"            # "08:30"


class Location(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Stop(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    location: Location
    demand: int = Field(ge=0, default=1)              # totes, parcels, bags...
    service_min: float = Field(ge=0, le=240, default=5)
    window_start: str | None = Field(default=None, pattern=HHMM)
    window_end: str | None = Field(default=None, pattern=HHMM)
    priority: float = Field(ge=0, default=100.0)      # dollars lost if not served

    @model_validator(mode="after")
    def window_is_ordered(self):
        if (self.window_start and self.window_end
                and self.window_start >= self.window_end):
            raise ValueError(f"stop {self.id}: window ends before it starts")
        return self


class Vehicle(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    capacity: int = Field(gt=0)
    shift_start: str = Field(pattern=HHMM)
    shift_end: str = Field(pattern=HHMM)
    cost_per_hour: float = Field(ge=0, default=30.81)
    cost_per_mile: float = Field(ge=0, default=0.76)
    fixed_cost: float = Field(ge=0, default=0.0)

    @model_validator(mode="after")
    def shift_is_ordered(self):
        if self.shift_start >= self.shift_end:
            raise ValueError(f"vehicle {self.id}: shift ends before it starts")
        return self


class PlanRequest(BaseModel):
    depot: Location
    stops: list[Stop] = Field(min_length=1, max_length=2000)
    vehicles: list[Vehicle] = Field(min_length=1, max_length=200)
    time_limit_s: float = Field(ge=0.1, le=300, default=10)
    speed_mph: float = Field(gt=0, le=80, default=15)

    @field_validator("stops", "vehicles")
    @classmethod
    def ids_are_unique(cls, items, info):
        ids = [x.id for x in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{info.field_name[:-1]} ids must be unique")
        return items


class Visit(BaseModel):
    stop_id: str
    arrival: str
    start: str
    departure: str


class Route(BaseModel):
    vehicle_id: str
    depart: str
    back: str
    visits: list[Visit]
    miles: float
    load: int
    hours: float                                      # paid: back minus depart
    cost: float


class Unassigned(BaseModel):
    stop_id: str
    reason: str


class PlanResult(BaseModel):
    routes: list[Route]
    unassigned: list[Unassigned]
    miles: float
    paid_hours: float
    cost: float                                       # from exact times and meters
    solver_status: str
    solve_seconds: float
    fallback: bool = False                            # True: no routes published
    warnings: list[str] = []


class Job(BaseModel):
    id: str
    status: str                                       # queued, running, done, failed
    created: float                                    # Unix time
    finished: float | None = None                     # kept for JOB_TTL_S after this
    result: PlanResult | None = None
    error: str | None = None
# <<end_models>>
