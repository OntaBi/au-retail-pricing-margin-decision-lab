from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_scenarios.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_recommendations.parquet"
)


# =========================================================
# COMMERCIAL GUARDRAILS
# =========================================================

MIN_MARGIN_PCT = 0.10

# Do not recommend operational changes for trivial benefit.
MIN_INCREMENTAL_MARGIN_DOLLARS = 25.0
MIN_INCREMENTAL_MARGIN_PCT = 0.01

# Maximum permitted price decrease.
MAX_PRICE_DECREASE = -0.05

# Maximum price increase based on elasticity confidence.
CONFIDENCE_MAX_INCREASE = {
    "High": 0.075,
    "Medium": 0.050,
    "Low": 0.025,
}

# Maximum expected unit decline from a recommended increase.
CONFIDENCE_MAX_UNIT_DECLINE = {
    "High": -0.12,
    "Medium": -0.08,
    "Low": -0.05,
}

# Competitive position determines available price headroom.
#
# Current index:
# < 0.95     strong headroom
# 0.95-1.00  moderate headroom
# 1.00-1.05  small headroom
# > 1.05     no increase
COMPETITIVE_HEADROOM = [
    (-np.inf, 0.95, 0.075),
    (0.95, 1.00, 0.050),
    (1.00, 1.05, 0.025),
    (1.05, np.inf, 0.000),
]

# Final recommended price should not become materially
# expensive relative to competitor.
MAX_RECOMMENDED_PRICE_INDEX = 1.08

# Reduction logic.
REDUCTION_REVIEW_INDEX = 1.05
STRONG_REDUCTION_INDEX = 1.10

# We allow some gross-margin sacrifice for an overpriced SKU
# if price investment materially improves competitiveness.
MAX_REDUCTION_MARGIN_DECLINE_PCT = -0.05

# Sales sacrifice guardrail for price increases.
#
# Example:
# +$1 margin should not require losing >$5 sales.
MAX_SALES_LOSS_PER_MARGIN_GAIN = 5.0


def get_confidence_max_increase(
    confidence: str,
) -> float:

    return CONFIDENCE_MAX_INCREASE.get(
        confidence,
        0.025,
    )


def get_max_unit_decline(
    confidence: str,
) -> float:

    return CONFIDENCE_MAX_UNIT_DECLINE.get(
        confidence,
        -0.05,
    )


def get_competitive_headroom(
    current_price_index: float,
) -> float:

    for lower, upper, max_increase in COMPETITIVE_HEADROOM:

        if (
            current_price_index >= lower
            and current_price_index < upper
        ):
            return max_increase

    return 0.0


def apply_guardrails(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:

    df = scenarios.copy()

    # -----------------------------------------------------
    # Current SKU position
    # -----------------------------------------------------

    current_lookup = (
        df[
            df["price_change_pct"] == 0
        ][
            [
                "sku_id",
                "scenario_price_index",
                "scenario_28d_margin",
                "scenario_28d_sales",
                "scenario_28d_units",
            ]
        ]
        .rename(
            columns={
                "scenario_price_index":
                    "base_price_index",
                "scenario_28d_margin":
                    "base_28d_margin",
                "scenario_28d_sales":
                    "base_28d_sales",
                "scenario_28d_units":
                    "base_28d_units",
            }
        )
    )

    df = df.merge(
        current_lookup,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # Confidence limits
    # -----------------------------------------------------

    df["confidence_max_increase"] = (
        df["decision_confidence"]
        .map(CONFIDENCE_MAX_INCREASE)
        .fillna(0.025)
    )

    df["max_unit_decline_pct"] = (
        df["decision_confidence"]
        .map(CONFIDENCE_MAX_UNIT_DECLINE)
        .fillna(-0.05)
    )

    # -----------------------------------------------------
    # Competitive price headroom
    # -----------------------------------------------------

    df["competitive_max_increase"] = (
        df["base_price_index"]
        .apply(
            get_competitive_headroom
        )
    )

    df["max_allowed_increase"] = (
        df[
            [
                "confidence_max_increase",
                "competitive_max_increase",
            ]
        ]
        .min(axis=1)
    )

    # -----------------------------------------------------
    # Commercial impact vs current
    # -----------------------------------------------------

    df["scenario_margin_change"] = (
        df["scenario_28d_margin"]
        - df["base_28d_margin"]
    )

    df["scenario_sales_change"] = (
        df["scenario_28d_sales"]
        - df["base_28d_sales"]
    )

    df["scenario_units_change"] = (
        df["scenario_28d_units"]
        - df["base_28d_units"]
    )

    df["scenario_unit_change_pct"] = np.where(
        df["base_28d_units"] > 0,
        (
            df["scenario_units_change"]
            / df["base_28d_units"]
        ),
        0.0,
    )

    df["scenario_margin_change_pct"] = np.where(
        df["base_28d_margin"] != 0,
        (
            df["scenario_margin_change"]
            / abs(
                df["base_28d_margin"]
            )
        ),
        np.nan,
    )

    # -----------------------------------------------------
    # Basic safety guardrails
    # -----------------------------------------------------

    df["passes_margin_guardrail"] = (
        df["scenario_margin_pct"]
        >= MIN_MARGIN_PCT
    )

    df["passes_unit_margin_guardrail"] = (
        df["scenario_unit_margin"] > 0
    )

    df["passes_price_decrease_limit"] = (
        df["price_change_pct"]
        >= MAX_PRICE_DECREASE
    )

    # -----------------------------------------------------
    # Increase guardrails
    # -----------------------------------------------------

    is_increase = (
        df["price_change_pct"] > 0
    )

    df["passes_increase_limit"] = (
        df["price_change_pct"]
        <= (
            df["max_allowed_increase"]
            + 1e-9
        )
    )

    df["passes_final_price_index"] = (
        df["scenario_price_index"]
        <= MAX_RECOMMENDED_PRICE_INDEX
    )

    df["passes_demand_risk"] = (
        df["scenario_unit_change_pct"]
        >= df["max_unit_decline_pct"]
    )

    # -----------------------------------------------------
    # Sales / margin trade-off
    # -----------------------------------------------------

    positive_margin_gain = (
        df["scenario_margin_change"] > 0
    )

    sales_loss = (
        -df["scenario_sales_change"]
    ).clip(lower=0)

    df["sales_loss_per_margin_gain"] = np.where(
        positive_margin_gain,
        (
            sales_loss
            / df["scenario_margin_change"]
        ),
        np.inf,
    )

    df["passes_sales_tradeoff"] = (
        (
            df["scenario_sales_change"] >= 0
        )
        |
        (
            df["sales_loss_per_margin_gain"]
            <= MAX_SALES_LOSS_PER_MARGIN_GAIN
        )
    )

    # -----------------------------------------------------
    # Reduction eligibility
    # -----------------------------------------------------

    is_reduction = (
        df["price_change_pct"] < 0
    )

    df["reduction_candidate"] = (
        df["base_price_index"]
        > REDUCTION_REVIEW_INDEX
    )

    df["passes_reduction_margin_tolerance"] = (
        df["scenario_margin_change_pct"]
        >= MAX_REDUCTION_MARGIN_DECLINE_PCT
    )

    df["improves_competitive_position"] = (
        df["scenario_price_index"]
        < df["base_price_index"]
    )

    # -----------------------------------------------------
    # Hold scenario
    # -----------------------------------------------------

    is_hold = (
        df["price_change_pct"] == 0
    )

    # -----------------------------------------------------
    # Final scenario eligibility
    # -----------------------------------------------------

    increase_eligible = (
        is_increase
        & df["passes_increase_limit"]
        & df["passes_final_price_index"]
        & df["passes_demand_risk"]
        & df["passes_sales_tradeoff"]
        & df["passes_margin_guardrail"]
        & df["passes_unit_margin_guardrail"]
    )

    reduction_eligible = (
        is_reduction
        & df["reduction_candidate"]
        & df["passes_price_decrease_limit"]
        & df["passes_reduction_margin_tolerance"]
        & df["improves_competitive_position"]
        & df["passes_margin_guardrail"]
        & df["passes_unit_margin_guardrail"]
    )

    hold_eligible = is_hold

    df["scenario_eligible"] = (
        increase_eligible
        | reduction_eligible
        | hold_eligible
    )

    return df


def is_material_margin_gain(
    candidate: pd.Series,
    current: pd.Series,
) -> bool:

    margin_gain = (
        candidate["scenario_28d_margin"]
        - current["scenario_28d_margin"]
    )

    current_margin = (
        current["scenario_28d_margin"]
    )

    if margin_gain < MIN_INCREMENTAL_MARGIN_DOLLARS:
        return False

    if current_margin == 0:
        return True

    margin_gain_pct = (
        margin_gain
        / abs(current_margin)
    )

    return (
        margin_gain_pct
        >= MIN_INCREMENTAL_MARGIN_PCT
    )


def choose_reduction_scenario(
    eligible: pd.DataFrame,
    current: pd.Series,
):

    reductions = eligible[
        eligible["price_change_pct"] < 0
    ].copy()

    if reductions.empty:
        return None

    current_index = (
        current["scenario_price_index"]
    )

    # Strongly overpriced SKU:
    # prioritise moving closer to competitive parity while
    # protecting most of the current margin.
    if current_index > STRONG_REDUCTION_INDEX:

        reductions["distance_from_parity"] = (
            reductions[
                "scenario_price_index"
            ]
            - 1.00
        ).abs()

        return (
            reductions.sort_values(
                [
                    "distance_from_parity",
                    "scenario_28d_margin",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .iloc[0]
            .copy()
        )

    # Moderately overpriced:
    # only reduce where margin economics are relatively strong.
    profitable_reductions = reductions[
        reductions["scenario_margin_change"]
        >= 0
    ]

    if not profitable_reductions.empty:

        return (
            profitable_reductions.sort_values(
                "scenario_28d_margin",
                ascending=False,
            )
            .iloc[0]
            .copy()
        )

    return None


def choose_best_scenario(
    sku_scenarios: pd.DataFrame,
) -> pd.Series:

    current_rows = sku_scenarios[
        sku_scenarios["price_change_pct"] == 0
    ]

    if current_rows.empty:
        raise ValueError(
            "Current scenario is missing."
        )

    current = (
        current_rows.iloc[0].copy()
    )

    eligible = sku_scenarios[
        sku_scenarios["scenario_eligible"]
    ].copy()

    if eligible.empty:

        recommended = current.copy()

        return pd.Series(
            build_recommendation_record(
                recommended,
                current,
                (
                    "Hold price - no scenario "
                    "passes commercial guardrails"
                ),
                "Review",
            )
        )

    current_index = (
        current["scenario_price_index"]
    )

    # -----------------------------------------------------
    # 1. Investigate price reduction where currently
    #    materially expensive vs competitor.
    # -----------------------------------------------------

    reduction = choose_reduction_scenario(
        eligible,
        current,
    )

    if reduction is not None:

        return pd.Series(
            build_recommendation_record(
                reduction,
                current,
                (
                    "Reduce price - currently "
                    "expensive versus competitor "
                    "and price investment improves "
                    "competitive position within "
                    "margin tolerance"
                ),
                "Reduce Price",
            )
        )

    # -----------------------------------------------------
    # 2. Evaluate eligible price increases.
    # -----------------------------------------------------

    increases = eligible[
        eligible["price_change_pct"] > 0
    ].copy()

    if not increases.empty:

        best_increase = (
            increases.sort_values(
                [
                    "scenario_28d_margin",
                    "price_change_pct",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
            .copy()
        )

        if is_material_margin_gain(
            best_increase,
            current,
        ):

            reason = (
                build_increase_reason(
                    best_increase,
                    current,
                )
            )

            return pd.Series(
                build_recommendation_record(
                    best_increase,
                    current,
                    reason,
                    "Increase Price",
                )
            )

    # -----------------------------------------------------
    # 3. Hold if no change produces a material,
    #    commercially acceptable outcome.
    # -----------------------------------------------------

    if current_index > STRONG_REDUCTION_INDEX:

        action = "Review"

        reason = (
            "Review - SKU remains materially "
            "more expensive than competitor but "
            "available reductions do not meet "
            "margin guardrails"
        )

    else:

        action = "Hold Price"

        reason = (
            "Hold price - no material "
            "commercially acceptable change "
            "identified"
        )

    return pd.Series(
        build_recommendation_record(
            current,
            current,
            reason,
            action,
        )
    )


def build_increase_reason(
    recommended: pd.Series,
    current: pd.Series,
) -> str:

    current_index = (
        current["scenario_price_index"]
    )

    if current_index < 0.95:

        return (
            "Increase price - strong competitive "
            "headroom with material expected "
            "margin improvement"
        )

    if current_index < 1.00:

        return (
            "Increase price - moderate competitive "
            "headroom with acceptable demand risk"
        )

    return (
        "Increase price - limited competitive "
        "headroom but expected margin improvement "
        "remains material within demand guardrails"
    )


def build_recommendation_record(
    recommended: pd.Series,
    current: pd.Series,
    reason: str,
    action: str,
) -> dict:

    incremental_units = (
        recommended["scenario_28d_units"]
        - current["scenario_28d_units"]
    )

    incremental_sales = (
        recommended["scenario_28d_sales"]
        - current["scenario_28d_sales"]
    )

    incremental_margin = (
        recommended["scenario_28d_margin"]
        - current["scenario_28d_margin"]
    )

    return {
        "sku_id": (
            recommended["sku_id"]
        ),
        "department": (
            recommended["department"]
        ),
        "category": (
            recommended["category"]
        ),
        "product_class": (
            recommended["product_class"]
        ),
        "price_band": (
            recommended["price_band"]
        ),
        "lifecycle_stage": (
            recommended["lifecycle_stage"]
        ),
        "recommended_action": action,
        "recommendation_reason": reason,
        "recommended_price_change_pct": (
            recommended["price_change_pct"]
        ),
        "current_sell_price": (
            current["current_sell_price"]
        ),
        "recommended_sell_price": (
            recommended["scenario_sell_price"]
        ),
        "competitor_price": (
            current["competitor_price"]
        ),
        "current_price_index": (
            current["scenario_price_index"]
        ),
        "recommended_price_index": (
            recommended["scenario_price_index"]
        ),
        "current_margin_pct": (
            current["scenario_margin_pct"]
        ),
        "recommended_margin_pct": (
            recommended["scenario_margin_pct"]
        ),
        "calibrated_elasticity": (
            recommended["calibrated_elasticity"]
        ),
        "decision_confidence": (
            recommended["decision_confidence"]
        ),
        "decision_source": (
            recommended["decision_source"]
        ),
        "current_28d_units": (
            current["scenario_28d_units"]
        ),
        "recommended_28d_units": (
            recommended["scenario_28d_units"]
        ),
        "incremental_units": (
            incremental_units
        ),
        "current_28d_sales": (
            current["scenario_28d_sales"]
        ),
        "recommended_28d_sales": (
            recommended["scenario_28d_sales"]
        ),
        "incremental_sales": (
            incremental_sales
        ),
        "current_28d_margin": (
            current["scenario_28d_margin"]
        ),
        "recommended_28d_margin": (
            recommended["scenario_28d_margin"]
        ),
        "incremental_margin": (
            incremental_margin
        ),
    }


def optimise_scenarios(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:

    guarded = apply_guardrails(
        scenarios
    )

    results = []

    for _, sku_scenarios in guarded.groupby(
        "sku_id",
        sort=True,
    ):

        results.append(
            choose_best_scenario(
                sku_scenarios
            )
        )

    return pd.DataFrame(
        results
    )


def validate_result(
    recommendations: pd.DataFrame,
) -> None:

    assert len(recommendations) == 1500

    assert (
        recommendations[
            "sku_id"
        ].nunique()
        == 1500
    )

    assert recommendations[
        "recommended_action"
    ].isin(
        [
            "Increase Price",
            "Reduce Price",
            "Hold Price",
            "Review",
        ]
    ).all()

    assert (
        recommendations[
            "recommended_sell_price"
        ] > 0
    ).all()

    assert (
        recommendations[
            "decision_confidence"
        ]
        .notna()
        .all()
    )

    changed = recommendations[
        "recommended_action"
    ].isin(
        [
            "Increase Price",
            "Reduce Price",
        ]
    )

    assert (
        recommendations.loc[
            changed,
            "recommended_margin_pct",
        ]
        >= MIN_MARGIN_PCT - 1e-9
    ).all()

    increases = (
        recommendations[
            "recommended_action"
        ]
        == "Increase Price"
    )

    assert (
        recommendations.loc[
            increases,
            "recommended_price_index",
        ]
        <= MAX_RECOMMENDED_PRICE_INDEX + 1e-9
    ).all()

    reductions = (
        recommendations[
            "recommended_action"
        ]
        == "Reduce Price"
    )

    assert (
        recommendations.loc[
            reductions,
            "recommended_price_change_pct",
        ]
        >= MAX_PRICE_DECREASE - 1e-9
    ).all()

    print("\nValidation passed.")


def main() -> None:

    scenarios = pd.read_parquet(
        SCENARIO_PATH
    )

    recommendations = (
        optimise_scenarios(
            scenarios
        )
    )

    validate_result(
        recommendations
    )

    recommendations.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nCOMMERCIAL PRICING "
        "RECOMMENDATION ENGINE"
    )

    print("=" * 75)

    print(
        f"SKUs                 : "
        f"{len(recommendations):,}"
    )

    print("\nRecommended Actions:")

    print(
        recommendations[
            "recommended_action"
        ]
        .value_counts()
    )

    print(
        "\nRecommended Price Changes:"
    )

    print(
        recommendations[
            "recommended_price_change_pct"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nPortfolio Commercial Impact:"
    )

    print(
        f"Incremental units     : "
        f"{recommendations['incremental_units'].sum():,.0f}"
    )

    print(
        f"Incremental sales     : "
        f"${recommendations['incremental_sales'].sum():,.0f}"
    )

    print(
        f"Incremental margin    : "
        f"${recommendations['incremental_margin'].sum():,.0f}"
    )

    print("\nAverage Price Index:")

    print(
        f"Current               : "
        f"{recommendations['current_price_index'].mean():.3f}"
    )

    print(
        f"Recommended           : "
        f"{recommendations['recommended_price_index'].mean():.3f}"
    )

    print("\nAverage Margin %:")

    print(
        f"Current               : "
        f"{recommendations['current_margin_pct'].mean():.1%}"
    )

    print(
        f"Recommended           : "
        f"{recommendations['recommended_margin_pct'].mean():.1%}"
    )

    print("\nActions by Confidence:")

    print(
        pd.crosstab(
            recommendations[
                "decision_confidence"
            ],
            recommendations[
                "recommended_action"
            ],
        )
    )

    print("\nActions by Current Price Position:")

    temp = recommendations.copy()

    temp["current_price_position"] = (
        pd.cut(
            temp["current_price_index"],
            bins=[
                -np.inf,
                0.95,
                1.00,
                1.05,
                1.10,
                np.inf,
            ],
            labels=[
                "<0.95",
                "0.95-1.00",
                "1.00-1.05",
                "1.05-1.10",
                ">1.10",
            ],
        )
    )

    print(
        pd.crosstab(
            temp[
                "current_price_position"
            ],
            temp[
                "recommended_action"
            ],
        )
    )

    print("\nTop Margin Opportunities:")

    top = (
        recommendations[
            recommendations[
                "recommended_action"
            ]
            == "Increase Price"
        ]
        .sort_values(
            "incremental_margin",
            ascending=False,
        )
        [
            [
                "sku_id",
                "department",
                "product_class",
                "recommended_price_change_pct",
                "current_sell_price",
                "recommended_sell_price",
                "current_price_index",
                "recommended_price_index",
                "incremental_sales",
                "incremental_margin",
                "decision_confidence",
            ]
        ]
        .head(15)
    )

    print(
        top.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nTop Price Investment Opportunities:")

    reductions = (
        recommendations[
            recommendations[
                "recommended_action"
            ]
            == "Reduce Price"
        ]
        .sort_values(
            "current_price_index",
            ascending=False,
        )
        [
            [
                "sku_id",
                "department",
                "product_class",
                "recommended_price_change_pct",
                "current_sell_price",
                "recommended_sell_price",
                "current_price_index",
                "recommended_price_index",
                "incremental_units",
                "incremental_sales",
                "incremental_margin",
                "decision_confidence",
            ]
        ]
        .head(15)
    )

    print(
        reductions.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nRecommendation Reasons:")

    print(
        recommendations[
            "recommendation_reason"
        ]
        .value_counts()
    )

    print(
        f"\nOutput               : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()