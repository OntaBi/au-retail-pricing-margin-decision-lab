import pandas as pd

from src.data_generation.generate_price_response_events import (
    generate_price_response_events,
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


def create_cost_events():
    return pd.DataFrame(
        {
            "sku_id": [
                "SKU001",
                "SKU002",
            ],
            "event_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-01",
                ]
            ),
            "old_cost_price": [
                60.00,
                120.00,
            ],
            "new_cost_price": [
                66.00,
                114.00,
            ],
            "cost_change_pct": [
                0.10,
                -0.05,
            ],
        }
    )


def test_one_response_per_cost_event():
    product_master = create_product_master()
    cost_events = create_cost_events()

    result = generate_price_response_events(
        cost_events,
        product_master,
        seed=42,
    )

    assert len(result) == len(cost_events)


def test_response_prices_are_positive():
    product_master = create_product_master()
    cost_events = create_cost_events()

    result = generate_price_response_events(
        cost_events,
        product_master,
        seed=42,
    )

    assert (
        result["new_sell_price"] > 0
    ).all()


def test_zero_recovery_has_no_response_date():
    product_master = create_product_master()
    cost_events = create_cost_events()

    result = generate_price_response_events(
        cost_events,
        product_master,
        seed=42,
    )

    zero_recovery = result[
        result["recovery_rate"] == 0
    ]

    assert (
        zero_recovery["response_date"]
        .isna()
        .all()
    )


def test_price_change_matches_recovery_logic():
    product_master = create_product_master()
    cost_events = create_cost_events()

    result = generate_price_response_events(
        cost_events,
        product_master,
        seed=42,
    )

    expected_sell = (
        result["old_sell_price"]
        + (
            result["new_cost_price"]
            - result["old_cost_price"]
        )
        * result["recovery_rate"]
    ).round(2)

    assert (
        result["new_sell_price"]
        == expected_sell
    ).all()