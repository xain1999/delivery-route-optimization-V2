"""A simple cost model for delivery routes (Chapter 1).

The model splits the cost of a route into a part proportional to the
time the driver is paid for and a part proportional to the distance the
vehicle drives.  It is deliberately simple. Chapter 7 refines it.
"""
from __future__ import annotations

from dataclasses import dataclass

METERS_PER_MILE = 1609.344


# <<cost_model>>
@dataclass(frozen=True)
class CostModel:
    """Cost parameters for one vehicle type.  All money in US dollars."""

    driver_cost_per_hour: float     # wage plus payroll taxes and benefits
    vehicle_cost_per_mile: float    # fuel, maintenance, tires, depreciation
    service_minutes_per_stop: float = 5.0
    average_speed_mph: float = 15.0

    def route_hours(self, miles: float, stops: int) -> float:
        """Paid hours for one route: driving plus time at customers."""
        return miles / self.average_speed_mph + stops * self.service_minutes_per_stop / 60

    def route_cost(self, miles: float, stops: int) -> tuple[float, float]:
        """Return (driver cost, vehicle cost) of one route."""
        driver = self.driver_cost_per_hour * self.route_hours(miles, stops)
        vehicle = self.vehicle_cost_per_mile * miles
        return driver, vehicle

    @property
    def cost_per_mile_driven(self) -> float:
        """Marginal cost of one extra mile: vehicle cost plus driver time."""
        return self.vehicle_cost_per_mile + self.driver_cost_per_hour / self.average_speed_mph
# <<end_cost_model>>
