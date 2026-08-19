import pandas as pd

from src.decision_engine.optimise_pricing_scenarios import (
    apply_guardrails,
    optimise_scenarios,
)


def create_scenario_rows(
    sku_id="SKU001",
    confidence="High",
    current_index=0.95,
    margin_pct=0.35,
):
    rows = []

    current_price = 100.0
    competitor_price = (
        current_price / current_index
    )

    current_margin = 1000.0
    current_sales = 3000.0
    current_units = 30.0

    for change in [
        -0.05,
        -0.025,
        0.0,
        0.025,
        0.05,
        0.075,
    ]:
        sell_price = (
            current_price
            * (1 + change)
        )

        scenario_index = (
            sell_price
            / competitor_price
        )

        expected_unit_change = (
            -1.5 * change
        )

        units = (
            current_units
            * (
                1
                + expected_unit_change
            )
        )

        sales = (
            current_sales
            * (
                1
                + change
                + expected_unit_change
            )
        )

        # Synthetic test economics:
        # moderate increases improve margin.
        margin = (
            current_margin
            * (
                1
                + 1.8 * change
            )
        )

        rows.append(
            {
                "sku_id": sku_id,
                "department": "Technology",
                "category": "Computers",
                "product_class": "Laptops",
                "price_band": "Core",
                "lifecycle_stage": "Core",
                "price_change_pct": change,
                "current_sell_price": current_price,
                "scenario_sell_price": sell_price,
                "competitor_price": competitor_price,
                "scenario_price_index": scenario_index,
                "scenario_margin_pct": margin_pct,
                "scenario_unit_margin": (
                    sell_price * margin_pct
                ),
                "scenario_28d_units": units,
                "scenario_28d_sales": sales,
                "scenario_28d_margin": margin,
                "expected_unit_change_pct": (
                    expected_unit_change
                ),
                "calibrated_elasticity": -1.5,
                "decision_confidence": confidence,
                "decision_source": "Product Class",
            }
        )

    return pd.DataFrame(rows)


def test_high_confidence_increase_respects_headroom():

    scenarios = create_scenario_rows(
        confidence="High",
        current_index=0.94,
    )

    guarded = apply_guardrails(
        scenarios
    )

    eligible_increases = guarded[
        (guarded["price_change_pct"] > 0)
        & guarded["scenario_eligible"]
    ]

    assert (
        eligible_increases[
            "price_change_pct"
        ].max()
        <= 0.075
    )


def test_no_increase_when_currently_expensive():

    scenarios = create_scenario_rows(
        current_index=1.07,
    )

    guarded = apply_guardrails(
        scenarios
    )

    increases = guarded[
        guarded["price_change_pct"] > 0
    ]

    assert not (
        increases[
            "scenario_eligible"
        ].any()
    )


def test_reduction_allowed_when_materially_expensive():

    scenarios = create_scenario_rows(
        current_index=1.12,
    )

    guarded = apply_guardrails(
        scenarios
    )

    reductions = guarded[
        guarded["price_change_pct"] < 0
    ]

    assert (
        reductions[
            "scenario_eligible"
        ].any()
    )


def test_low_confidence_limits_price_increase():

    scenarios = create_scenario_rows(
        confidence="Low",
        current_index=0.90,
    )

    guarded = apply_guardrails(
        scenarios
    )

    eligible_increases = guarded[
        (guarded["price_change_pct"] > 0)
        & guarded["scenario_eligible"]
    ]

    if not eligible_increases.empty:
        assert (
            eligible_increases[
                "price_change_pct"
            ].max()
            <= 0.025
        )


def test_optimiser_returns_one_recommendation_per_sku():

    sku1 = create_scenario_rows(
        sku_id="SKU001",
        current_index=0.94,
    )

    sku2 = create_scenario_rows(
        sku_id="SKU002",
        current_index=1.12,
    )

    scenarios = pd.concat(
        [
            sku1,
            sku2,
        ],
        ignore_index=True,
    )

    result = optimise_scenarios(
        scenarios
    )

    assert len(result) == 2

    assert (
        result["sku_id"].nunique()
        == 2
    )


def test_recommendation_actions_are_valid():

    scenarios = create_scenario_rows(
        current_index=0.96,
    )

    result = optimise_scenarios(
        scenarios
    )

    assert result[
        "recommended_action"
    ].isin(
        [
            "Increase Price",
            "Reduce Price",
            "Hold Price",
            "Review",
        ]
    ).all()