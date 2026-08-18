import pandas as pd

from src.data_generation.generate_competitor_events import (
    generate_competitor_events,
)

from src.data_generation.apply_competitor_events import (
    apply_competitor_events,
)


def create_product_master():
    return pd.DataFrame(
        {
            "sku_id": [
                "SKU001",
                "SKU002",
            ],
            "regular_sell_price": [
                100.00,
                200.00,
            ],
        }
    )


def create_pricing_history():
    dates = pd.date_range(
        "2024-07-01",
        "2026-06-30",
        freq="D",
    )

    records = []

    for sku_id, sell_price in [
        ("SKU001", 100.00),
        ("SKU002", 200.00),
    ]:
        for date in dates:
            records.append(
                {
                    "date": date,
                    "sku_id": sku_id,
                    "cost_price": sell_price * 0.60,
                    "baseline_sell_price": sell_price,
                    "regular_sell_price": sell_price,
                }
            )

    return pd.DataFrame(records)


def test_each_sku_has_initial_competitor_price():
    product_master = create_product_master()
    pricing_history = create_pricing_history()

    events = generate_competitor_events(
        product_master,
        pricing_history,
        seed=42,
    )

    initial = events[
        events["event_type"] == "Initial Price"
    ]

    assert len(initial) == len(
        product_master
    )

    assert (
        initial.groupby("sku_id")
        .size()
        .eq(1)
        .all()
    )


def test_competitor_prices_are_positive():
    product_master = create_product_master()
    pricing_history = create_pricing_history()

    events = generate_competitor_events(
        product_master,
        pricing_history,
        seed=42,
    )

    assert (
        events["new_competitor_price"] > 0
    ).all()


def test_daily_history_preserves_row_count():
    product_master = create_product_master()
    pricing_history = create_pricing_history()

    events = generate_competitor_events(
        product_master,
        pricing_history,
        seed=42,
    )

    result = apply_competitor_events(
        pricing_history,
        events,
    )

    assert len(result) == len(
        pricing_history
    )


def test_price_index_calculation():
    product_master = create_product_master()
    pricing_history = create_pricing_history()

    events = generate_competitor_events(
        product_master,
        pricing_history,
        seed=42,
    )

    result = apply_competitor_events(
        pricing_history,
        events,
    )

    expected_index = (
        result["regular_sell_price"]
        / result["competitor_price"]
    )

    assert (
        (
            result["price_index"]
            - expected_index
        ).abs()
        < 1e-10
    ).all()


def test_price_gap_calculation():
    product_master = create_product_master()
    pricing_history = create_pricing_history()

    events = generate_competitor_events(
        product_master,
        pricing_history,
        seed=42,
    )

    result = apply_competitor_events(
        pricing_history,
        events,
    )

    expected_gap = (
        result["regular_sell_price"]
        - result["competitor_price"]
    )

    assert (
        (
            result["price_gap_dollars"]
            - expected_gap
        ).abs()
        < 1e-10
    ).all()