from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DECISION_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "decision_elasticity.parquet"
)

DEMAND_PROFILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "demand_profiles.parquet"
)


def main():

    decision = pd.read_parquet(
        DECISION_PATH
    )

    demand = pd.read_parquet(
        DEMAND_PROFILE_PATH
    )

    truth = demand[
        [
            "sku_id",
            "true_price_elasticity",
        ]
    ].rename(
        columns={
            "true_price_elasticity":
                "true_elasticity"
        }
    )

    df = decision.merge(
        truth,
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Errors
    # ---------------------------------------------------------

    df["decision_error"] = (
        df["decision_elasticity"]
        - df["true_elasticity"]
    )

    df["absolute_error"] = (
        df["decision_error"].abs()
    )

    df["benchmark_error"] = (
        df["benchmark_elasticity"]
        - df["true_elasticity"]
    )

    df["benchmark_absolute_error"] = (
        df["benchmark_error"].abs()
    )

    # ---------------------------------------------------------
    # Core diagnostics
    # ---------------------------------------------------------

    mae = df["absolute_error"].mean()

    median_ae = (
        df["absolute_error"].median()
    )

    bias = (
        df["decision_error"].mean()
    )

    correlation = (
        df[
            [
                "decision_elasticity",
                "true_elasticity",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    benchmark_mae = (
        df[
            "benchmark_absolute_error"
        ].mean()
    )

    # ---------------------------------------------------------
    # SKU evidence impact
    # ---------------------------------------------------------

    sku_used = (
        df["sku_weight"] > 0
    )

    no_sku = ~sku_used

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print(
        "\nDECISION ELASTICITY VALIDATION"
    )
    print("=" * 70)

    print(
        f"SKUs                       : "
        f"{len(df):,}"
    )

    print(
        f"Mean decision elasticity   : "
        f"{df['decision_elasticity'].mean():.3f}"
    )

    print(
        f"Mean true elasticity       : "
        f"{df['true_elasticity'].mean():.3f}"
    )

    print(
        f"Mean bias                  : "
        f"{bias:.3f}"
    )

    print(
        f"MAE                        : "
        f"{mae:.3f}"
    )

    print(
        f"Median AE                  : "
        f"{median_ae:.3f}"
    )

    print(
        f"Truth correlation          : "
        f"{correlation:.3f}"
    )

    print(
        f"Benchmark-only MAE         : "
        f"{benchmark_mae:.3f}"
    )

    print("\nError Distribution:")

    print(
        df["decision_error"]
        .describe()
        .round(3)
    )

    print("\nAccuracy by Decision Source:")

    source_summary = (
        df.groupby(
            "decision_source"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            mean_estimate=(
                "decision_elasticity",
                "mean",
            ),
            mean_truth=(
                "true_elasticity",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            bias=(
                "decision_error",
                "mean",
            ),
        )
        .round(3)
    )

    print(source_summary)

    print("\nAccuracy by Confidence:")

    confidence_summary = (
        df.groupby(
            "decision_confidence"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            mean_estimate=(
                "decision_elasticity",
                "mean",
            ),
            mean_truth=(
                "true_elasticity",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            bias=(
                "decision_error",
                "mean",
            ),
        )
        .round(3)
    )

    print(confidence_summary)

    print("\nSKU Evidence Comparison:")

    comparison = pd.DataFrame(
        {
            "group": [
                "SKU evidence used",
                "Benchmark only",
            ],
            "skus": [
                sku_used.sum(),
                no_sku.sum(),
            ],
            "mae": [
                df.loc[
                    sku_used,
                    "absolute_error",
                ].mean(),
                df.loc[
                    no_sku,
                    "absolute_error",
                ].mean(),
            ],
            "bias": [
                df.loc[
                    sku_used,
                    "decision_error",
                ].mean(),
                df.loc[
                    no_sku,
                    "decision_error",
                ].mean(),
            ],
        }
    )

    print(
        comparison.round(3).to_string(
            index=False
        )
    )

    print("\nAccuracy by Department:")

    department_summary = (
        df.groupby(
            "department"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            mean_estimate=(
                "decision_elasticity",
                "mean",
            ),
            mean_truth=(
                "true_elasticity",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            bias=(
                "decision_error",
                "mean",
            ),
        )
        .round(3)
    )

    print(department_summary)

    print("\nWorst Decision Estimates:")

    worst = (
        df.sort_values(
            "absolute_error",
            ascending=False,
        )
        [
            [
                "sku_id",
                "product_class",
                "decision_source",
                "decision_confidence",
                "decision_elasticity",
                "true_elasticity",
                "absolute_error",
                "sku_weight",
            ]
        ]
        .head(20)
    )

    print(
        worst.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    assert len(df) == 1500

    assert (
        df["true_elasticity"]
        .notna()
        .all()
    )

    assert (
        df["decision_elasticity"]
        .notna()
        .all()
    )

    assert (
        df["decision_elasticity"] < 0
    ).all()

    print("\nValidation passed.")


if __name__ == "__main__":
    main()