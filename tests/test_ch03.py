"""Tests for the Chapter 3 data pipeline.  Run with:  pytest -q"""
from datetime import datetime
from pathlib import Path

import pytest

from vrpbook.addresses import AddressError, parse
from vrpbook.geocode import ChicagoGridGeocoder, CachedGeocoder, GeocoderChain
from vrpbook.orders import Rules, clean_orders, load_raw

DATA = Path(__file__).resolve().parent.parent / "data"


def test_parse_normalizes_words_and_case():
    a = parse("1600 north damen avenue")
    assert a.key == "1600 N DAMEN AVE"


@pytest.mark.parametrize("text,code", [("PO Box 12", "po_box"),
                                       ("W Belmont Ave", "no_house_number"),
                                       ("1500 Chicago Ave", "unparseable")])
def test_parse_rejects(text, code):
    with pytest.raises(AddressError) as e:
        parse(text)
    assert e.value.code == code


def test_grid_origin_and_one_mile():
    g = ChicagoGridGeocoder()
    origin = g.geocode(parse("0 N State St"))
    north = g.geocode(parse("800 N State St"))
    assert origin.lat == pytest.approx(41.88207) and origin.lon == pytest.approx(-87.62792)
    # 800 numbers = 1 mile = about 0.01447 degrees of latitude
    assert north.lat - origin.lat == pytest.approx(1 / 69.09, rel=1e-3)


def test_grid_south_side_control_points():
    g = ChicagoGridGeocoder()
    roosevelt = g.geocode(parse("1200 S State St"))
    s47 = g.geocode(parse("100 W 47th St"))
    origin = g.geocode(parse("0 N State St"))
    assert (origin.lat - roosevelt.lat) * 69.09 == pytest.approx(1.0, rel=1e-3)
    assert (origin.lat - s47.lat) * 69.09 == pytest.approx(5.0, rel=1e-3)


def test_typo_is_corrected():
    r = ChicagoGridGeocoder().geocode(parse("2400 N Ashlnad Ave"))
    assert r is not None and r.corrected_street == "ASHLAND"


def test_diagonal_street_needs_another_geocoder():
    assert ChicagoGridGeocoder().geocode(parse("1500 N Milwaukee Ave")) is None


def test_pipeline_accounts_for_every_order():
    rows = load_raw(DATA / "chicago_tuesday_orders_raw.csv")
    geo = GeocoderChain(CachedGeocoder(DATA / "geocode_cache.json"), ChicagoGridGeocoder())
    res = clean_orders(rows, Rules(cutoff=datetime(2026, 10, 12, 23, 59, 59)), geo)
    planned = {o for s in res.stops for o in s.order_ids}
    explained = {i.order_id for i in res.issues if i.code != "split_delivery"}
    all_orders = {r["order_id"] for r in rows}
    # every order is either planned or has an issue explaining why not
    assert all_orders <= planned | explained
    assert all(s.totes <= 100 for s in res.stops)


# ---- dirty-data boundary: the review's reproductions -----------------------
from vrpbook.geocode import GeocodeResult  # noqa: E402

CUTOFF = datetime(2026, 10, 12, 23, 59, 59)


def order(oid, **kw):
    row = dict(order_id=oid, placed_at="2026-10-12T09:00:00", customer_id="C1",
               address="5600 N Ashland Ave", unit="", city="Chicago", state="IL",
               zip="60660", totes="10", weight_lb="200", status="paid")
    row.update(kw)
    return row


class FixedGeocoder:
    def __init__(self, precision):
        self.precision = precision

    def geocode(self, addr):
        return GeocodeResult(41.98, -87.67, self.precision, "test")


def codes(res, oid):
    return {i.code for i in res.issues if i.order_id == oid}


def test_units_are_kept_and_different_units_are_different_stops():
    rows = [order("A", unit="Apt 3"), order("B", unit="Apt 4"), order("C", unit="apt. 3")]
    res = clean_orders(rows, Rules(cutoff=CUTOFF), ChicagoGridGeocoder())
    assert sorted(s.unit for s in res.stops) == ["APT 3", "APT 4"]
    apt3 = next(s for s in res.stops if s.unit == "APT 3")
    assert apt3.order_ids == ["A", "C"] and apt3.totes == 20     # same unit: merged


@pytest.mark.parametrize("precision,kept", [("street", False), ("zip", False),
                                            ("interpolated", True), ("rooftop", True)])
def test_coarse_geocodes_go_to_review(precision, kept):
    res = clean_orders([order("A")], Rules(cutoff=CUTOFF), FixedGeocoder(precision))
    assert bool(res.stops) == kept
    assert ("coarse_geocode" in codes(res, "A")) != kept
    if kept:
        assert res.stops[0].precision == precision


@pytest.mark.parametrize("field,value,code", [
    ("totes", "abc", "bad_quantity"), ("totes", "-3", "bad_quantity"),
    ("totes", "2.5", "bad_quantity"), ("weight_lb", "-40", "bad_weight"),
    ("weight_lb", "heavy", "bad_weight"), ("placed_at", "yesterday", "bad_timestamp"),
    ("customer_id", "", "missing_field")])
def test_malformed_fields_become_issues_not_crashes(field, value, code):
    res = clean_orders([order("A", **{field: value}), order("B", customer_id="C2")],
                       Rules(cutoff=CUTOFF), ChicagoGridGeocoder())
    assert code in codes(res, "A")
    assert [s.order_ids for s in res.stops] == [["B"]]           # the good order still flows


def test_split_weights_add_up_to_the_order_weight():
    res = clean_orders([order("A", totes="140", weight_lb="101")], Rules(cutoff=CUTOFF),
                       ChicagoGridGeocoder())
    assert [s.totes for s in res.stops] == [70, 70]
    assert sum(s.weight_lb for s in res.stops) == 101
