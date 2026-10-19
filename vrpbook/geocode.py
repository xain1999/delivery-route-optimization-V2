"""Geocoding: turning addresses into coordinates (Chapter 3).

Three geocoders share one interface and are combined in a chain:

* CachedGeocoder: addresses already resolved (and preferably verified by
  a driver's GPS at delivery).  Always consulted first.
* ChicagoGridGeocoder: an offline geocoder based on Chicago's address
  grid.  Exact enough for planning on grid streets, deterministic, free.
* NominatimGeocoder: OpenStreetMap's public geocoder, through geopy.
  Optional. Respect its usage policy (at most one request per second).
"""
from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .addresses import ParsedAddress
from .distances import EARTH_RADIUS_M

METERS_PER_MILE = 1609.344


@dataclass(frozen=True)
class GeocodeResult:
    lat: float
    lon: float
    precision: str          # "verified", "interpolated", "street", ...
    source: str             # which geocoder produced it
    corrected_street: str | None = None   # set if a typo was fixed


class Geocoder(Protocol):
    def geocode(self, addr: ParsedAddress) -> GeocodeResult | None: ...


# <<grid_tables>>
# Chicago's grid: origin at State St and Madison St, 800 numbers per mile.
# Position of north-south streets, in address numbers east (+) of State St.
NS_STREETS = {
    "STATE": 0, "MICHIGAN": 100, "KING": 400, "COTTAGE GROVE": 800,
    "WOODLAWN": 1200, "STONY ISLAND": 1600, "JEFFERY": 2000,
    "HALSTED": -800, "RACINE": -1200, "ASHLAND": -1600, "DAMEN": -2000,
    "WESTERN": -2400, "CALIFORNIA": -2800, "KEDZIE": -3200,
    "PULASKI": -4000, "CICERO": -4800, "CENTRAL": -5600, "HARLEM": -7200,
}
# Position of east-west streets, in address numbers north (+) of Madison St.
EW_STREETS = {
    "MADISON": 0, "CHICAGO": 800, "DIVISION": 1200, "NORTH": 1600,
    "ARMITAGE": 2000, "FULLERTON": 2400, "DIVERSEY": 2800, "BELMONT": 3200,
    "ROSCOE": 3400, "ADDISON": 3600, "IRVING PARK": 4000, "MONTROSE": 4400,
    "LAWRENCE": 4800, "FOSTER": 5200, "BRYN MAWR": 5600, "DEVON": 6400,
    "TOUHY": 7200, "HOWARD": 7600, "ROOSEVELT": -1200, "CERMAK": -2200,
    "PERSHING": -3900, "GARFIELD": -5500,
}
NUMBERED_STREET = re.compile(r"^(\d+)(ST|ND|RD|TH)$")   # "47TH" -> 4700 S
# South of Madison the first two miles are numbered more densely.  These
# control points (address number, miles) calibrate the interpolation.
SOUTH_CONTROL = [(0, 0.0), (1200, 1.0), (2200, 2.0), (3900, 4.0)]
# <<end_grid_tables>>


class ChicagoGridGeocoder:
    """Offline geocoder for addresses on Chicago's regular street grid."""

    ORIGIN = (41.88207, -87.62792)   # State St & Madison St

    def __init__(self, fuzzy_cutoff: float = 0.8):
        self.fuzzy_cutoff = fuzzy_cutoff

    # <<grid_geocode>>
    def geocode(self, addr: ParsedAddress) -> GeocodeResult | None:
        street, corrected = self._match_street(addr.street)
        if street is None:
            return None                          # not a grid street
        if street in NS_STREETS:                 # number runs north-south
            if addr.direction not in "NS":
                return None
            east = NS_STREETS[street]
            north = addr.number if addr.direction == "N" else -addr.number
        else:                                    # number runs east-west
            if addr.direction not in "EW":
                return None
            north = self._ew_position(street)
            east = addr.number if addr.direction == "E" else -addr.number
        lat, lon = self.to_latlon(self._miles_north(north), east / 800)
        return GeocodeResult(lat, lon, "interpolated", "chicago-grid",
                             street if corrected else None)
    # <<end_grid_geocode>>

    def _match_street(self, name: str) -> tuple[str | None, bool]:
        if name in NS_STREETS or name in EW_STREETS or NUMBERED_STREET.match(name):
            return name, False
        close = difflib.get_close_matches(name, [*NS_STREETS, *EW_STREETS], n=1,
                                          cutoff=self.fuzzy_cutoff)
        return (close[0], True) if close else (None, False)

    @staticmethod
    def _ew_position(street: str) -> int:
        m = NUMBERED_STREET.match(street)
        return -100 * int(m.group(1)) if m else EW_STREETS[street]

    @staticmethod
    def _miles_north(n: int) -> float:
        """Address number north (+) or south (-) of Madison -> miles."""
        if n >= 0:
            return n / 800
        s = -n
        for (a0, m0), (a1, m1) in zip(SOUTH_CONTROL, SOUTH_CONTROL[1:]):
            if s <= a1:
                return -(m0 + (s - a0) * (m1 - m0) / (a1 - a0))
        a_last, m_last = SOUTH_CONTROL[-1]
        return -(m_last + (s - a_last) / 800)

    @classmethod
    def to_latlon(cls, miles_north: float, miles_east: float) -> tuple[float, float]:
        """Local flat-earth projection around the origin (spherical Earth)."""
        lat0, lon0 = cls.ORIGIN
        miles_per_deg = EARTH_RADIUS_M * math.pi / 180 / METERS_PER_MILE
        lat = lat0 + miles_north / miles_per_deg
        lon = lon0 + miles_east / (miles_per_deg * math.cos(math.radians(lat0)))
        return lat, lon


class CachedGeocoder:
    """Addresses resolved before, keyed by ParsedAddress.key (JSON file)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def geocode(self, addr: ParsedAddress) -> GeocodeResult | None:
        hit = self.data.get(addr.key)
        if hit is None:
            return None
        return GeocodeResult(hit["lat"], hit["lon"], hit.get("precision", "cached"),
                             "cache")

    def store(self, addr: ParsedAddress, result: GeocodeResult, note: str = "") -> None:
        self.data[addr.key] = {**asdict(result), "source": note or result.source}
        self.path.write_text(json.dumps(self.data, indent=2))


class NominatimGeocoder:
    """OpenStreetMap Nominatim via geopy (pip install geopy).

    The public server allows at most one request per second and requires
    an identifying user agent. For more than occasional use, run your own
    Nominatim server or use a commercial geocoder.
    """

    def __init__(self, user_agent: str, city: str = "Chicago", state: str = "IL"):
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim
        self._geocode = RateLimiter(Nominatim(user_agent=user_agent).geocode,
                                    min_delay_seconds=1.0)
        self.city, self.state = city, state

    def geocode(self, addr: ParsedAddress) -> GeocodeResult | None:
        loc = self._geocode(f"{addr.key}, {self.city}, {self.state}", addressdetails=True)
        if loc is None:
            return None
        kind = loc.raw.get("addresstype") or loc.raw.get("type", "")
        precision = "rooftop" if kind in ("house", "building") else "street"
        return GeocodeResult(loc.latitude, loc.longitude, precision, "nominatim")


# <<chain>>
class GeocoderChain:
    """Try geocoders in order. The first answer wins."""

    def __init__(self, *geocoders: Geocoder):
        self.geocoders = geocoders

    def geocode(self, addr: ParsedAddress) -> GeocodeResult | None:
        for g in self.geocoders:
            result = g.geocode(addr)
            if result is not None:
                return result
        return None
# <<end_chain>>
