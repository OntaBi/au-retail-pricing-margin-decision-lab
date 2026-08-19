from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCT_MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "product_master.parquet"
)

SKU_ESTIMATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "sku_elasticity_estimates.parquet"
)

CLASS_ESTIMATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "class_elasticity_estimates.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "decision_elasticity.parquet"
)


def build_category_benchmarks(
    class_estimates: pd.DataFrame,
) -> pd.DataFrame:

    valid = class_estimates[
        class_estimates["model_status"]
        == "Success"
    ].copy()

    category = (
        valid.groupby(
            [
                "department",
                "category",
            ],
            as_index=False,
        )
        .apply(
            lambda group: pd.Series(
                {
                    "category_elasticity": np.average(
                        group[
                            "estimated_elasticity"
                        ],
                        weights=group[
                            "n_price_moves"
                        ].clip(lower=1),
                    ),
                    "category_classes": (
                        group[
                            "product_class"
                        ].nunique()
                    ),
                    "category_price_moves": (
                        group[
                            "n_price_moves"
                        ].sum()
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )

    return category


def build_department_benchmarks(
    class_estimates: pd.DataFrame,
) -> pd.DataFrame:

    valid = class_estimates[
        class_estimates["model_status"]
        == "Success"
    ].copy()

    department = (
        valid.groupby(
            "department",
            as_index=False,
        )
        .apply(
            lambda group: pd.Series(
                {
                    "department_elasticity": np.average(
                        group[
                            "estimated_elasticity"
                        ],
                        weights=group[
                            "n_price_moves"
                        ].clip(lower=1),
                    ),
                    "department_classes": (
                        group[
                            "product_class"
                        ].nunique()
                    ),
                    "department_price_moves": (
                        group[
                            "n_price_moves"
                        ].sum()
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )

    return department


def calculate_sku_weight(
    evidence_tier: str,
    price_range_pct: float,
    unique_prices: float,
    elasticity_std_error: float,
) -> float:

    if evidence_tier == "Strong":
        base_weight = 0.55

    elif evidence_tier == "Moderate":
        base_weight = 0.30

    else:
        return 0.0

    # ---------------------------------------------------------
    # Price variation adjustment
    # ---------------------------------------------------------

    variation_factor = np.clip(
        price_range_pct / 0.10,
        0.25,
        1.00,
    )

    # ---------------------------------------------------------
    # Number of distinct observed prices
    # ---------------------------------------------------------

    price_count_factor = np.clip(
        unique_prices / 5.0,
        0.40,
        1.00,
    )

    # ---------------------------------------------------------
    # Model uncertainty
    #
    # Lower standard error means more trust in SKU estimate.
    # ---------------------------------------------------------

    if pd.isna(elasticity_std_error):
        uncertainty_factor = 0.40

    else:
        uncertainty_factor = np.clip(
            1.0 / (
                1.0
                + elasticity_std_error
            ),
            0.25,
            1.00,
        )

    weight = (
        base_weight
        * variation_factor
        * price_count_factor
        * uncertainty_factor
    )

    return float(
        np.clip(
            weight,
            0.0,
            0.65,
        )
    )


def build_elasticity_hierarchy(
    product_master: pd.DataFrame,
    sku_estimates: pd.DataFrame,
    class_estimates: pd.DataFrame,
) -> pd.DataFrame:

    product = product_master[
        [
            "sku_id",
            "department",
            "category",
            "product_class",
            "price_band",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Product-class estimate
    # ---------------------------------------------------------

    class_lookup = class_estimates[
        [
            "department",
            "category",
            "product_class",
            "estimated_elasticity",
            "elasticity_std_error",
            "elasticity_p_value",
            "n_skus",
            "n_price_moves",
            "confidence",
            "model_status",
        ]
    ].rename(
        columns={
            "estimated_elasticity":
                "class_elasticity",
            "elasticity_std_error":
                "class_std_error",
            "elasticity_p_value":
                "class_p_value",
            "n_skus":
                "class_skus",
            "n_price_moves":
                "class_price_moves",
            "confidence":
                "class_confidence",
            "model_status":
                "class_model_status",
        }
    )

    result = product.merge(
        class_lookup,
        on=[
            "department",
            "category",
            "product_class",
        ],
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Category benchmark
    # ---------------------------------------------------------

    category = build_category_benchmarks(
        class_estimates
    )

    result = result.merge(
        category,
        on=[
            "department",
            "category",
        ],
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Department benchmark
    # ---------------------------------------------------------

    department = build_department_benchmarks(
        class_estimates
    )

    result = result.merge(
        department,
        on="department",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # SKU estimate
    # ---------------------------------------------------------

    sku_lookup = sku_estimates[
        [
            "sku_id",
            "estimated_elasticity",
            "elasticity_std_error",
            "elasticity_p_value",
            "evidence_tier",
            "unique_prices",
            "price_change_weeks",
            "price_range_pct",
            "avg_weekly_units",
            "model_status",
        ]
    ].rename(
        columns={
            "estimated_elasticity":
                "sku_elasticity",
            "elasticity_std_error":
                "sku_std_error",
            "elasticity_p_value":
                "sku_p_value",
            "model_status":
                "sku_model_status",
        }
    )

    result = result.merge(
        sku_lookup,
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Base benchmark hierarchy
    #
    # Prefer class.
    # If class unavailable, fall back to category.
    # Then department.
    # ---------------------------------------------------------

    result[
        "benchmark_elasticity"
    ] = result[
        "class_elasticity"
    ]

    result[
        "benchmark_source"
    ] = "Product Class"

    missing_class = (
        result[
            "benchmark_elasticity"
        ].isna()
    )

    result.loc[
        missing_class,
        "benchmark_elasticity",
    ] = result.loc[
        missing_class,
        "category_elasticity",
    ]

    result.loc[
        missing_class,
        "benchmark_source",
    ] = "Category"

    missing_benchmark = (
        result[
            "benchmark_elasticity"
        ].isna()
    )

    result.loc[
        missing_benchmark,
        "benchmark_elasticity",
    ] = result.loc[
        missing_benchmark,
        "department_elasticity",
    ]

    result.loc[
        missing_benchmark,
        "benchmark_source",
    ] = "Department"

    # ---------------------------------------------------------
    # SKU shrinkage weight
    # ---------------------------------------------------------

    result["sku_weight"] = result.apply(
        lambda row: calculate_sku_weight(
            evidence_tier=(
                row["evidence_tier"]
                if pd.notna(
                    row["evidence_tier"]
                )
                else ""
            ),
            price_range_pct=(
                row["price_range_pct"]
                if pd.notna(
                    row["price_range_pct"]
                )
                else 0.0
            ),
            unique_prices=(
                row["unique_prices"]
                if pd.notna(
                    row["unique_prices"]
                )
                else 0.0
            ),
            elasticity_std_error=(
                row["sku_std_error"]
            ),
        ),
        axis=1,
    )

    # ---------------------------------------------------------
    # Only use SKU estimate if:
    # - model succeeded
    # - estimate is commercially plausible
    # - evidence tier supports SKU modelling
    #
    # Do NOT hard-clip accepted values.
    # Implausible estimates are simply not used.
    # ---------------------------------------------------------

    valid_sku_estimate = (
        (
            result[
                "sku_model_status"
            ]
            == "Success"
        )
        & result[
            "evidence_tier"
        ].isin(
            [
                "Strong",
                "Moderate",
            ]
        )
        & result[
            "sku_elasticity"
        ].between(
            -5.0,
            -0.05,
        )
    )

    result.loc[
        ~valid_sku_estimate,
        "sku_weight",
    ] = 0.0

    # ---------------------------------------------------------
    # Decision elasticity
    # ---------------------------------------------------------

    result[
        "decision_elasticity"
    ] = (
        result["sku_weight"]
        * result[
            "sku_elasticity"
        ].fillna(0)
        +
        (
            1
            - result["sku_weight"]
        )
        * result[
            "benchmark_elasticity"
        ]
    )

    # ---------------------------------------------------------
    # Decision source
    # ---------------------------------------------------------

    result[
        "decision_source"
    ] = result[
        "benchmark_source"
    ]

    sku_used = (
        result["sku_weight"] > 0
    )

    result.loc[
        sku_used,
        "decision_source",
    ] = (
        "SKU + "
        + result.loc[
            sku_used,
            "benchmark_source",
        ]
    )

    # ---------------------------------------------------------
    # Decision confidence
    # ---------------------------------------------------------

    result[
        "decision_confidence"
    ] = "Low"

    class_high = (
        result[
            "class_confidence"
        ]
        == "High"
    )

    class_medium = (
        result[
            "class_confidence"
        ]
        == "Medium"
    )

    result.loc[
        class_medium,
        "decision_confidence",
    ] = "Medium"

    result.loc[
        class_high,
        "decision_confidence",
    ] = "High"

    strong_sku = (
        sku_used
        & (
            result[
                "evidence_tier"
            ]
            == "Strong"
        )
        & (
            result[
                "sku_weight"
            ]
            >= 0.25
        )
    )

    moderate_sku = (
        sku_used
        & (
            result[
                "evidence_tier"
            ]
            == "Moderate"
        )
    )

    result.loc[
        moderate_sku
        & (
            result[
                "decision_confidence"
            ]
            == "High"
        ),
        "decision_confidence",
    ] = "Medium"

    result.loc[
        strong_sku
        & class_high,
        "decision_confidence",
    ] = "High"

    # ---------------------------------------------------------
    # Decision diagnostics
    # ---------------------------------------------------------

    result[
        "sku_adjustment"
    ] = (
        result[
            "decision_elasticity"
        ]
        - result[
            "benchmark_elasticity"
        ]
    )

    return result


def validate_result(
    result: pd.DataFrame,
) -> None:

    assert len(result) == 1500

    assert (
        result["sku_id"].nunique()
        == 1500
    )

    assert (
        result[
            "decision_elasticity"
        ].notna()
    ).all()

    assert (
        result[
            "decision_elasticity"
        ] < 0
    ).all()

    assert (
        result[
            "sku_weight"
        ].between(
            0,
            0.65,
        )
    ).all()

    assert result[
        "decision_source"
    ].notna().all()

    print("\nValidation passed.")


def main() -> None:

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    sku_estimates = pd.read_parquet(
        SKU_ESTIMATES_PATH
    )

    class_estimates = pd.read_parquet(
        CLASS_ESTIMATES_PATH
    )

    result = build_elasticity_hierarchy(
        product_master,
        sku_estimates,
        class_estimates,
    )

    validate_result(
        result
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nDECISION ELASTICITY HIERARCHY"
    )
    print("=" * 70)

    print(
        f"SKUs                    : "
        f"{len(result):,}"
    )

    print("\nDecision Elasticity:")

    print(
        result[
            "decision_elasticity"
        ]
        .describe()
        .round(3)
    )

    print("\nDecision Source:")

    print(
        result[
            "decision_source"
        ]
        .value_counts()
    )

    print("\nDecision Confidence:")

    print(
        result[
            "decision_confidence"
        ]
        .value_counts()
    )

    print("\nSKU Weight Distribution:")

    print(
        result[
            "sku_weight"
        ]
        .describe()
        .round(3)
    )

    print(
        "\nSKUs using SKU-level evidence:"
    )

    sku_used = result[
        result["sku_weight"] > 0
    ]

    print(
        f"{len(sku_used):,} "
        f"({len(sku_used) / len(result):.1%})"
    )

    print("\nAverage Decision Elasticity by Department:")

    print(
        result.groupby(
            "department"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            avg_elasticity=(
                "decision_elasticity",
                "mean",
            ),
            avg_sku_weight=(
                "sku_weight",
                "mean",
            ),
        )
        .round(3)
    )

    print("\nLargest SKU Adjustments:")

    adjustments = (
        result[
            result["sku_weight"] > 0
        ]
        .assign(
            abs_adjustment=lambda x: (
                x["sku_adjustment"]
                .abs()
            )
        )
        .sort_values(
            "abs_adjustment",
            ascending=False,
        )
        [
            [
                "sku_id",
                "product_class",
                "evidence_tier",
                "sku_elasticity",
                "benchmark_elasticity",
                "sku_weight",
                "decision_elasticity",
                "decision_confidence",
            ]
        ]
        .head(15)
    )

    print(
        adjustments.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nSample Decision Elasticities:")

    sample = (
        result[
            [
                "sku_id",
                "department",
                "product_class",
                "price_band",
                "decision_elasticity",
                "decision_source",
                "decision_confidence",
                "benchmark_elasticity",
                "sku_elasticity",
                "sku_weight",
            ]
        ]
        .head(20)
    )

    print(
        sample.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        f"\nOutput                  : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()