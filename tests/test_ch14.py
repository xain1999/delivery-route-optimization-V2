"""Tests for Chapter 14: the same-day courier simulator."""
from vrpbook.dynamic import (MIN, Geometry, Order, generate_day, latest_departure,
                             simulate, trip_times)


def test_generated_days_are_reproducible():
    a, b = generate_day(3), generate_day(3)
    assert [(o.release, o.lat) for o in a] == [(o.release, o.lat) for o in b]
    assert all(o.deadline - o.release == 180 * MIN for o in a)
    assert len(generate_day(3, busy=2.0)) > len(a)


def test_latest_departure_is_the_slack_of_the_tightest_order():
    orders = [Order(0, 33.46, -112.00, 0, 10 * 3600), Order(1, 33.47, -112.01, 0, 9 * 3600)]
    G = Geometry(orders)
    when, back = trip_times(8 * 3600, [0, 1], orders, G)
    latest = latest_departure(8 * 3600, [0, 1], orders, G)
    assert latest == 8 * 3600 + min(10 * 3600 - when[0], 9 * 3600 - when[1])
    assert back > when[-1]


def test_a_simulated_day_is_consistent():
    orders = generate_day(7, busy=0.3)
    res = simulate(orders, 2, "accept", seconds=0.1, epoch_min=30)
    assert res["delivered"] + res["rejected"] == res["orders"]
    for o in orders:
        if o.status == "delivered":
            assert o.delivered >= o.release
