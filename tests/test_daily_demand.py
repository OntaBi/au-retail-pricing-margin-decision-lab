import pandas as pd

from src.data_generation.generate_daily_demand import (
    generate_daily_demand,
)


def create_history():
    dates = pd.date_range(
        "2025-03-01",
        periods=14,
        freq="D",
    )

    records = []

    for sku_id, sell_price, competitor_price in [
        ("SKU001", 100.00, 100.00),
        ("SKU002", 200.00, 190.00),
    ]:
        for date in dates:
            records.append(
                {
                    "date": date,
                    "sku_id": sku_id,
                    "cost_price": sell_price * 0.60,
                    "baseline_sell_price": sell_price,
                    "regular_sell_price": sell_price,
                    "competitor_price": competitor_price,
                    "price_index": (
                        sell_price
                        / competitor_price
                    ),
                    "price_gap_dollars": (
                        sell_price
                        - competitor_price
                    ),
                    "price_gap_pct": (
                        sell_price
                        / competitor_price
                        - 1
                    ),
                }
            )

    return pd.DataFrame(records)


def create_profiles():
    return pd.DataFrame(
        {
            "sku_id": [
                "SKU001",
                "SKU002",
            ],
            "baseline_daily_units": [
                10.0,
                5.0,
            ],
            "true_price_elasticity": [
                -1.5,
                -1.0,
            ],
            "demand_volatility": [
                0.15,
                0.15,
            ],
        }
    )


def test_daily_demand_preserves_row_count():
    history = create_history()
    profiles = create_profiles()

    result = generate_daily_demand(
        history,
        profiles,
        seed=42,
    )

    assert len(result) == len(history)


def test_units_sold_are_non_negative_integers():
    history = create_history()
    profiles = create_profiles()

    result = generate_daily_demand(
        history,
        profiles,
        seed=42,
    )

    assert (
        result["units_sold"] >= 0
    ).all()

    assert (
        result["units_sold"] % 1 == 0
    ).all()


def test_sales_dollars_match_units_times_price():
    history = create_history()
    profiles = create_profiles()

    result = generate_daily_demand(
        history,
        profiles,
        seed=42,
    )

    expected_sales = (
        result["units_sold"]
        * result["regular_sell_price"]
    )

    assert (
        (
            result["sales_dollars"]
            - expected_sales
        ).abs()
        < 1e-10
    ).all()


def test_margin_dollars_match_units_times_unit_margin():
    history = create_history()
    profiles = create_profiles()

    result = generate_daily_demand(
        history,
        profiles,
        seed=42,
    )

    expected_margin = (
        result["units_sold"]
        * (
            result["regular_sell_price"]
            - result["cost_price"]
        )
    )

    assert (
        (
            result["gross_margin_dollars"]
            - expected_margin
        ).abs()
        < 1e-10
    ).all()


def test_higher_own_price_reduces_expected_demand():
    history = create_history().copy()

    profiles = create_profiles()

    sku_mask = (
        history["sku_id"] == "SKU001"
    )

    midpoint = history.loc[
        sku_mask,
        "date",
    ].sort_values().iloc[7]

    higher_price_mask = (
        sku_mask
        & (
            history["date"] >= midpoint
        )
    )

    history.loc[
        higher_price_mask,
        "regular_sell_price",
    ] = 110.00

    history.loc[
        higher_price_mask,
        "price_index",
    ] = 1.10

    history.loc[
        higher_price_mask,
        "price_gap_dollars",
    ] = 10.00

    history.loc[
        higher_price_mask,
        "price_gap_pct",
    ] = 0.10

    result = generate_daily_demand(
        history,
        profiles,
        seed=42,
    )

    sku_result = result[
        result["sku_id"] == "SKU001"
    ]

    before = sku_result[
        sku_result["date"] < midpoint
    ][
        "own_price_demand_factor"
    ].mean()

    after = sku_result[
        sku_result["date"] >= midpoint
    ][
        "own_price_demand_factor"
    ].mean()

    assert after < before