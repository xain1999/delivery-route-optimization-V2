"""Normalizing and parsing US street addresses (Chapter 3).

This is a deliberately small parser for the address forms that appear in
the book's data: "<number> <direction> <street name> <suffix>".  For
production use across many cities, a dedicated library such as
usaddress or libpostal is more robust. The interface below is designed so
that one can be swapped in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DIRECTIONS = {"N": "N", "NORTH": "N", "S": "S", "SOUTH": "S",
              "E": "E", "EAST": "E", "W": "W", "WEST": "W"}
SUFFIXES = {"AVENUE": "AVE", "AVE": "AVE", "AV": "AVE", "STREET": "ST", "ST": "ST",
            "ROAD": "RD", "RD": "RD", "BOULEVARD": "BLVD", "BLVD": "BLVD",
            "DRIVE": "DR", "DR": "DR", "PARKWAY": "PKWY", "PKWY": "PKWY",
            "PLACE": "PL", "PL": "PL", "COURT": "CT", "CT": "CT"}
PO_BOX = re.compile(r"^\s*P\.?\s*O\.?\s*BOX\b", re.IGNORECASE)


class AddressError(ValueError):
    """Raised when an address cannot be used for delivery routing."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# <<parsed_address>>
@dataclass(frozen=True)
class ParsedAddress:
    number: int
    direction: str      # N, S, E or W
    street: str         # e.g. "ASHLAND", "COTTAGE GROVE", "47TH"
    suffix: str | None  # e.g. "AVE"

    @property
    def key(self) -> str:
        """Canonical form, used to merge orders and as a cache key."""
        parts = [str(self.number), self.direction, self.street]
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)


def normalize(text: str) -> str:
    """Upper-case, drop punctuation, collapse whitespace."""
    text = re.sub(r"[.,#]", " ", text.upper())
    return re.sub(r"\s+", " ", text).strip()


def parse(text: str) -> ParsedAddress:
    """Parse '<number> <dir> <street...> [suffix]' or raise AddressError."""
    if PO_BOX.match(text):
        raise AddressError("po_box", "PO boxes cannot receive deliveries")
    tokens = normalize(text).split()
    if not tokens or not tokens[0].isdigit():
        raise AddressError("no_house_number", "address has no house number")
    if len(tokens) < 3 or tokens[1] not in DIRECTIONS:
        raise AddressError("unparseable", f"cannot parse address {text!r}")
    number, direction, rest = int(tokens[0]), DIRECTIONS[tokens[1]], tokens[2:]
    suffix = SUFFIXES.get(rest[-1]) if len(rest) > 1 else None
    if suffix:
        rest = rest[:-1]
    return ParsedAddress(number, direction, " ".join(rest), suffix)
# <<end_parsed_address>>
