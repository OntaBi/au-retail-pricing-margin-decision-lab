def calculate_custom_price_scenario(
    *,
    current_price: float,
    proposed_price: float,
    competitor_price: float,
    cost_price: float,
    elasticity: float,
    current_28d_units: float,
    current_28d_sales: float,
    current_28d_margin: float,
) -> dict:
    """
    Calculate commercial impact for a custom sell-price scenario.

    Uses the same linear elasticity convention as the pricing
    scenario foundation:
        expected unit change % = elasticity * price change %
    """

    if current_price <= 0:
        raise ValueError(
            "current_price must be greater than zero."
        )

    if proposed_price <= 0:
        raise ValueError(
            "proposed_price must be greater than zero."
        )

    if competitor_price <= 0:
        raise ValueError(
            "competitor_price must be greater than zero."
        )

    price_change_pct = (
        proposed_price
        / current_price
        - 1
    )

    expected_unit_change_pct = (
        elasticity
        * price_change_pct
    )

    scenario_28d_units = (
        current_28d_units
        * (
            1
            + expected_unit_change_pct
        )
    )

    scenario_28d_units = max(
        scenario_28d_units,
        0,
    )

    scenario_28d_sales = (
        scenario_28d_units
        * proposed_price
    )

    unit_margin = (
        proposed_price
        - cost_price
    )

    scenario_28d_margin = (
        scenario_28d_units
        * unit_margin
    )

    scenario_margin_pct = (
        unit_margin
        / proposed_price
    )

    scenario_price_index = (
        proposed_price
        / competitor_price
    )

    return {
        "price_change_pct":
            price_change_pct,

        "expected_unit_change_pct":
            expected_unit_change_pct,

        "scenario_28d_units":
            scenario_28d_units,

        "scenario_28d_sales":
            scenario_28d_sales,

        "scenario_28d_margin":
            scenario_28d_margin,

        "scenario_margin_pct":
            scenario_margin_pct,

        "scenario_price_index":
            scenario_price_index,

        "incremental_units":
            scenario_28d_units
            - current_28d_units,

        "incremental_sales":
            scenario_28d_sales
            - current_28d_sales,

        "incremental_margin":
            scenario_28d_margin
            - current_28d_margin,
    }