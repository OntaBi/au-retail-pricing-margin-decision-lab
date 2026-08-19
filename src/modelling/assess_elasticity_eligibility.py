from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_modelling_dataset.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_eligibility.parquet"
)


def assess_eligibility(
    weekly: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        weekly.groupby("sku_id")
        .agg(
            department=(
                "department",
                "first",
            ),
            category=(
                "category",
                "first",
            ),
            product_class=(
                "product_class",
                "first",
            ),
            price_band=(
                "price_band",
                "first",
            ),
            weeks=(
                "week_start",
                "nunique",
            ),
            unique_prices=(
                "avg_sell_price",
                "nunique",
            ),
            min_price=(
                "avg_sell_price",
                "min",
            ),
            max_price=(
                "avg_sell_price",
                "max",
            ),
            price_change_weeks=(
                "price_changed",
                "sum",
            ),
            total_units=(
                "units_sold",
                "sum",
            ),
            avg_weekly_units=(
                "units_sold",
                "mean",
            ),
            zero_unit_weeks=(
                "units_sold",
                lambda x: (x == 0).sum(),
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Price variation
    # ---------------------------------------------------------

    summary["price_range_pct"] = (
        summary["max_price"]
        / summary["min_price"]
        - 1
    )

    # ---------------------------------------------------------
    # Demand coverage
    # ---------------------------------------------------------

    summary["non_zero_weeks"] = (
        summary["weeks"]
        - summary["zero_unit_weeks"]
    )

    summary["non_zero_week_pct"] = (
        summary["non_zero_weeks"]
        / summary["weeks"]
    )

    # ---------------------------------------------------------
    # Evidence classification
    #
    # These are modelling eligibility rules, not estimates.
    # We can tune them after inspecting the distribution.
    # ---------------------------------------------------------

    strong = (
        (summary["unique_prices"] >= 4)
        & (summary["price_range_pct"] >= 0.05)
        & (summary["price_change_weeks"] >= 3)
        & (summary["non_zero_weeks"] >= 80)
    )

    moderate = (
        (summary["unique_prices"] >= 3)
        & (summary["price_range_pct"] >= 0.025)
        & (summary["price_change_weeks"] >= 2)
        & (summary["non_zero_weeks"] >= 60)
    )

    limited = (
        (summary["unique_prices"] >= 2)
        & (summary["price_range_pct"] >= 0.01)
        & (summary["price_change_weeks"] >= 1)
        & (summary["non_zero_weeks"] >= 40)
    )

    summary["evidence_tier"] = "Insufficient"

    summary.loc[
        limited,
        "evidence_tier",
    ] = "Limited"

    summary.loc[
        moderate,
        "evidence_tier",
    ] = "Moderate"

    summary.loc[
        strong,
        "evidence_tier",
    ] = "Strong"

    # ---------------------------------------------------------
    # Recommended modelling level
    # ---------------------------------------------------------

    summary["recommended_level"] = (
        "Product Class"
    )

    summary.loc[
        summary["evidence_tier"].isin(
            [
                "Strong",
                "Moderate",
            ]
        ),
        "recommended_level",
    ] = "SKU"

    return summary


def validate_result(
    result: pd.DataFrame,
) -> None:

    assert len(result) == 1500

    assert (
        result["sku_id"].nunique()
        == 1500
    )

    assert (
        result["weeks"] > 0
    ).all()

    assert (
        result["unique_prices"] >= 1
    ).all()

    assert (
        result["price_range_pct"] >= 0
    ).all()

    assert result[
        "evidence_tier"
    ].isin(
        [
            "Strong",
            "Moderate",
            "Limited",
            "Insufficient",
        ]
    ).all()

    print("\nValidation passed.")


def main() -> None:

    weekly = pd.read_parquet(
        INPUT_PATH
    )

    result = assess_eligibility(
        weekly
    )

    validate_result(
        result
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nELASTICITY MODELLING ELIGIBILITY")
    print("=" * 60)

    print(
        f"SKUs                 : "
        f"{len(result):,}"
    )

    print("\nEvidence Tiers:")

    evidence_summary = (
        result["evidence_tier"]
        .value_counts()
        .reindex(
            [
                "Strong",
                "Moderate",
                "Limited",
                "Insufficient",
            ],
            fill_value=0,
        )
    )

    evidence_pct = (
        evidence_summary
        / len(result)
        * 100
    )

    evidence_table = pd.DataFrame(
        {
            "skus": evidence_summary,
            "pct": evidence_pct,
        }
    )

    print(
        evidence_table.round(1)
    )

    print("\nRecommended Modelling Level:")

    print(
        result[
            "recommended_level"
        ]
        .value_counts()
    )

    print("\nDiagnostics by Evidence Tier:")

    diagnostics = (
        result.groupby(
            "evidence_tier"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            avg_unique_prices=(
                "unique_prices",
                "mean",
            ),
            avg_price_range_pct=(
                "price_range_pct",
                "mean",
            ),
            avg_price_change_weeks=(
                "price_change_weeks",
                "mean",
            ),
            avg_non_zero_weeks=(
                "non_zero_weeks",
                "mean",
            ),
        )
        .round(3)
    )

    print(diagnostics)

    print("\nTop SKU-level Candidates:")

    candidates = (
        result[
            result["recommended_level"]
            == "SKU"
        ]
        .sort_values(
            [
                "price_range_pct",
                "unique_prices",
            ],
            ascending=False,
        )
        [
            [
                "sku_id",
                "department",
                "product_class",
                "evidence_tier",
                "unique_prices",
                "price_change_weeks",
                "price_range_pct",
                "avg_weekly_units",
            ]
        ]
        .head(15)
    )

    print(
        candidates.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        f"\nOutput               : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()