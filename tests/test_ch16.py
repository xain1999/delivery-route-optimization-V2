"""Tests for Chapter 16: the Hearthline Home pipeline."""
import pytest

pytest.importorskip("pydantic", reason='needs pip install -e ".[service]"')

from case_study.hearthline import (Fleet, by_day, clean, evaluate, load,  # noqa: E402
                                 manual_plan, parse_slot, to_request)


def test_slots_in_every_format():
    assert parse_slot("8-10") == parse_slot("8:00-10:00") == parse_slot("8am-10am") == (8, 10)
    assert parse_slot("12pm-2pm") == (12, 14)
    assert parse_slot("4pm-6pm") == (16, 18)
    with pytest.raises(ValueError):
        parse_slot("noonish")


def test_cleaning_removes_duplicates_and_flags_missing_locations():
    row = {"order_id": "HH-1", "delivery_date": "2026-09-14", "customer_area": "Midtown",
           "lat": "33.78", "lon": "-84.38", "items": "sofa; washer", "slot": "8-10",
           "notes": "3rd floor walk-up"}
    nowhere = dict(row, order_id="HH-2", lat="", lon="")
    orders, issues = clean([row, dict(row), nowhere])
    assert [o.id for o in orders] == ["HH-1"]
    assert issues == {"duplicate row": ["HH-1"], "no coordinates": ["HH-2"]}
    assert orders[0].cuft == 60 + 30 and orders[0].minutes == 10 + 15 + 45 + 25


def test_manual_plan_serves_everyone_once_within_capacity():
    orders, _ = load()
    day = next(iter(by_day(orders).values()))
    req = to_request(day, Fleet())
    routes = manual_plan(req)
    served = sorted(k for r in routes for k in r)
    assert served == list(range(1, len(day) + 1))
    assert evaluate(req, routes)["overloaded"] == 0


@pytest.mark.parametrize("text,slot", [("8-10", (8, 10)), ("8:00-10:00", (8, 10)),
                                       ("10am-12pm", (10, 12)), ("12am-1am", (0, 1)),
                                       ("11pm-12am", (23, 24)), ("4pm-6pm", (16, 18))])
def test_parse_slot_reads_every_form(text, slot):
    assert parse_slot(text) == slot


@pytest.mark.parametrize("text", ["8-10-12", "8", "25-26", "13pm-2pm", "10-8", "8:30-10", ""])
def test_parse_slot_rejects_malformed_slots(text):
    with pytest.raises(ValueError):
        parse_slot(text)
