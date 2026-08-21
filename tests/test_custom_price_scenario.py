import pytest

from src.app.scenario_math import (
    calculate_custom_price_scenario,
)


# =========================================================
# TEST 1
# CURRENT PRICE SHOULD PRODUCE A NEUTRAL SCENARIO
# =========================================================

def test_current_price_produces_neutral_scenario():

    result = calculate_custom_price_scenario(
        current_price=100.0,
        proposed_price=100.0,
        competitor_price=100.0,
        cost_price=60.0,
        elasticity=-1.5,
        current_28d_units=100.0,
        current_28d_sales=10000.0,
        current_28d_margin=4000.0,
    )

    assert result["price_change_pct"] == pytest.approx(0.0)

    assert result[
        "expected_unit_change_pct"
    ] == pytest.approx(0.0)

    assert result[
        "scenario_28d_units"
    ] == pytest.approx(100.0)

    assert result[
        "scenario_28d_sales"
    ] == pytest.approx(10000.0)

    assert result[
        "scenario_28d_margin"
    ] == pytest.approx(4000.0)

    assert result[
        "incremental_units"
    ] == pytest.approx(0.0)

    assert result[
        "incremental_sales"
    ] == pytest.approx(0.0)

    assert result[
        "incremental_margin"
    ] == pytest.approx(0.0)


# =========================================================
# TEST 2
# PRICE INCREASE SHOULD REDUCE EXPECTED DEMAND
# =========================================================

def test_price_increase_reduces_expected_units():

    result = calculate_custom_price_scenario(
        current_price=100.0,
        proposed_price=110.0,
        competitor_price=105.0,
        cost_price=60.0,
        elasticity=-1.5,
        current_28d_units=100.0,
        current_28d_sales=10000.0,
        current_28d_margin=4000.0,
    )

    # +10% price change
    assert result[
        "price_change_pct"
    ] == pytest.approx(0.10)

    # -1.5 elasticity × +10% price = -15% units
    assert result[
        "expected_unit_change_pct"
    ] == pytest.approx(-0.15)

    assert result[
        "scenario_28d_units"
    ] == pytest.approx(85.0)

    assert result[
        "incremental_units"
    ] < 0

    # Sales = 85 × $110
    assert result[
        "scenario_28d_sales"
    ] == pytest.approx(9350.0)

    # Margin = 85 × ($110 - $60)
    assert result[
        "scenario_28d_margin"
    ] == pytest.approx(4250.0)

    # Margin should improve despite lower units
    assert result[
        "incremental_margin"
    ] == pytest.approx(250.0)

    assert result[
        "incremental_margin"
    ] > 0


# =========================================================
# TEST 3
# PRICE REDUCTION SHOULD INCREASE DEMAND AND IMPROVE
# COMPETITIVE POSITION
# =========================================================

def test_price_reduction_increases_units_and_improves_price_index():

    result = calculate_custom_price_scenario(
        current_price=100.0,
        proposed_price=90.0,
        competitor_price=95.0,
        cost_price=60.0,
        elasticity=-1.5,
        current_28d_units=100.0,
        current_28d_sales=10000.0,
        current_28d_margin=4000.0,
    )

    # -10% price change
    assert result[
        "price_change_pct"
    ] == pytest.approx(-0.10)

    # -1.5 elasticity × -10% price = +15% units
    assert result[
        "expected_unit_change_pct"
    ] == pytest.approx(0.15)

    assert result[
        "scenario_28d_units"
    ] == pytest.approx(115.0)

    assert result[
        "incremental_units"
    ] > 0

    # Proposed price is below competitor
    assert result[
        "scenario_price_index"
    ] == pytest.approx(
        90.0 / 95.0
    )

    assert result[
        "scenario_price_index"
    ] < 1.0

    # Sales = 115 × $90
    assert result[
        "scenario_28d_sales"
    ] == pytest.approx(10350.0)

    # Margin = 115 × ($90 - $60)
    assert result[
        "scenario_28d_margin"
    ] == pytest.approx(3450.0)

    # Demand increases, but margin declines
    assert result[
        "incremental_margin"
    ] == pytest.approx(-550.0)