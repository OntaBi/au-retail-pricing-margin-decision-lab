from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_modelling_dataset.parquet"
)

BASELINE_CLASS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "class_elasticity_estimates.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "class_elasticity_within_estimates.parquet"
)


def estimate_single_class(
    class_data: pd.DataFrame,
) -> dict:
    class_data = (
        class_data
        .sort_values(
            [
                "sku_id",
                "week_start",
            ]
        )
        .copy()
    )

    department = (
        class_data["department"].iloc[0]
    )

    category = (
        class_data["category"].iloc[0]
    )

    product_class = (
        class_data["product_class"].iloc[0]
    )

    # ---------------------------------------------------------
    # Log variables
    # ---------------------------------------------------------

    class_data["log_units_model"] = np.log1p(
        class_data["units_sold"]
    )

    class_data["log_price_model"] = np.log(
        class_data["avg_sell_price"]
    )

    class_data[
        "log_competitor_price_model"
    ] = np.log(
        class_data["avg_competitor_price"]
    )

    # ---------------------------------------------------------
    # Within-SKU demeaning
    #
    # Remove each SKU's normal level so the model focuses on
    # deviations around that SKU's own average behaviour.
    # ---------------------------------------------------------

    sku_groups = class_data.groupby(
        "sku_id"
    )

    class_data[
        "within_log_units"
    ] = (
        class_data["log_units_model"]
        - sku_groups[
            "log_units_model"
        ].transform("mean")
    )

    class_data[
        "within_log_price"
    ] = (
        class_data["log_price_model"]
        - sku_groups[
            "log_price_model"
        ].transform("mean")
    )

    class_data[
        "within_log_competitor_price"
    ] = (
        class_data[
            "log_competitor_price_model"
        ]
        - sku_groups[
            "log_competitor_price_model"
        ].transform("mean")
    )

    # ---------------------------------------------------------
    # Month controls
    # ---------------------------------------------------------

    month_dummies = pd.get_dummies(
        class_data["month"],
        prefix="month",
        drop_first=True,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Model matrix
    #
    # No SKU dummies required because the variables have
    # already been demeaned within SKU.
    # ---------------------------------------------------------

    X = pd.concat(
        [
            class_data[
                [
                    "within_log_price",
                    "within_log_competitor_price",
                ]
            ].reset_index(drop=True),
            month_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    y = (
        class_data[
            "within_log_units"
        ]
        .reset_index(drop=True)
    )

    try:
        model = sm.OLS(
            y,
            X,
        ).fit()

        estimated_elasticity = (
            model.params.get(
                "within_log_price",
                np.nan,
            )
        )

        elasticity_std_error = (
            model.bse.get(
                "within_log_price",
                np.nan,
            )
        )

        elasticity_p_value = (
            model.pvalues.get(
                "within_log_price",
                np.nan,
            )
        )

        competitor_coefficient = (
            model.params.get(
                "within_log_competitor_price",
                np.nan,
            )
        )

        r_squared = (
            model.rsquared
        )

        model_status = "Success"

    except Exception as exc:
        estimated_elasticity = np.nan
        elasticity_std_error = np.nan
        elasticity_p_value = np.nan
        competitor_coefficient = np.nan
        r_squared = np.nan

        model_status = (
            f"Failed: "
            f"{type(exc).__name__}"
        )

    # ---------------------------------------------------------
    # Hidden truth for evaluation only
    # ---------------------------------------------------------

    true_elasticity_mean = (
        class_data.groupby("sku_id")[
            "true_price_elasticity"
        ]
        .first()
        .mean()
    )

    true_elasticity_std = (
        class_data.groupby("sku_id")[
            "true_price_elasticity"
        ]
        .first()
        .std()
    )

    n_skus = (
        class_data["sku_id"].nunique()
    )

    n_observations = (
        len(class_data)
    )

    n_price_moves = int(
        class_data["price_changed"].sum()
    )

    if pd.notna(
        estimated_elasticity
    ):
        estimation_error = (
            estimated_elasticity
            - true_elasticity_mean
        )

        absolute_error = abs(
            estimation_error
        )

    else:
        estimation_error = np.nan
        absolute_error = np.nan

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    if (
        model_status == "Success"
        and n_price_moves >= 50
        and elasticity_p_value < 0.05
        and elasticity_std_error <= 0.40
    ):
        confidence = "High"

    elif (
        model_status == "Success"
        and n_price_moves >= 35
        and elasticity_p_value < 0.10
        and elasticity_std_error <= 0.70
    ):
        confidence = "Medium"

    else:
        confidence = "Low"

    return {
        "department": department,
        "category": category,
        "product_class": product_class,
        "estimated_elasticity": (
            estimated_elasticity
        ),
        "true_elasticity_mean": (
            true_elasticity_mean
        ),
        "true_elasticity_std": (
            true_elasticity_std
        ),
        "estimation_error": (
            estimation_error
        ),
        "absolute_error": (
            absolute_error
        ),
        "elasticity_std_error": (
            elasticity_std_error
        ),
        "elasticity_p_value": (
            elasticity_p_value
        ),
        "competitor_coefficient": (
            competitor_coefficient
        ),
        "r_squared": (
            r_squared
        ),
        "n_skus": (
            n_skus
        ),
        "n_observations": (
            n_observations
        ),
        "n_price_moves": (
            n_price_moves
        ),
        "confidence": (
            confidence
        ),
        "model_status": (
            model_status
        ),
    }


def estimate_all_classes(
    data: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    grouped = data.groupby(
        [
            "department",
            "category",
            "product_class",
        ],
        sort=True,
    )

    for _, class_data in grouped:
        results.append(
            estimate_single_class(
                class_data
            )
        )

    return pd.DataFrame(
        results
    )


def validate_result(
    estimates: pd.DataFrame,
) -> None:
    assert len(estimates) == 26

    assert (
        estimates["product_class"].nunique()
        == 26
    )

    successful = estimates[
        estimates["model_status"]
        == "Success"
    ]

    assert len(successful) == 26

    assert successful[
        "estimated_elasticity"
    ].notna().all()

    print("\nValidation passed.")


def calculate_metrics(
    data: pd.DataFrame,
) -> dict:
    clean = data.dropna(
        subset=[
            "estimated_elasticity",
            "true_elasticity_mean",
        ]
    )

    mae = (
        clean["absolute_error"]
        .mean()
    )

    median_ae = (
        clean["absolute_error"]
        .median()
    )

    correlation = (
        clean[
            [
                "estimated_elasticity",
                "true_elasticity_mean",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    negative_pct = (
        (
            clean[
                "estimated_elasticity"
            ] < 0
        )
        .mean()
        * 100
    )

    significant_pct = (
        (
            clean[
                "elasticity_p_value"
            ] < 0.05
        )
        .mean()
        * 100
    )

    return {
        "mae": mae,
        "median_ae": median_ae,
        "correlation": correlation,
        "negative_pct": negative_pct,
        "significant_pct": significant_pct,
    }


def main() -> None:
    data = pd.read_parquet(
        DATA_PATH
    )

    baseline = pd.read_parquet(
        BASELINE_CLASS_PATH
    )

    estimates = estimate_all_classes(
        data
    )

    validate_result(
        estimates
    )

    estimates.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    successful = estimates[
        estimates["model_status"]
        == "Success"
    ].copy()

    baseline_successful = baseline[
        baseline["model_status"]
        == "Success"
    ].copy()

    baseline_metrics = calculate_metrics(
        baseline_successful
    )

    within_metrics = calculate_metrics(
        successful
    )

    print(
        "\nWITHIN-SKU PRODUCT-CLASS "
        "ELASTICITY ESTIMATION"
    )

    print("=" * 75)

    print(
        f"Product classes       : "
        f"{len(successful):,}"
    )

    print("\nModel Comparison:")

    comparison = pd.DataFrame(
        {
            "Class Fixed Effects": [
                baseline_metrics["mae"],
                baseline_metrics[
                    "median_ae"
                ],
                baseline_metrics[
                    "correlation"
                ],
                baseline_metrics[
                    "negative_pct"
                ],
                baseline_metrics[
                    "significant_pct"
                ],
            ],
            "Within-SKU Model": [
                within_metrics["mae"],
                within_metrics[
                    "median_ae"
                ],
                within_metrics[
                    "correlation"
                ],
                within_metrics[
                    "negative_pct"
                ],
                within_metrics[
                    "significant_pct"
                ],
            ],
        },
        index=[
            "MAE",
            "Median AE",
            "Truth Correlation",
            "Negative %",
            "p-value < 0.05 %",
        ],
    )

    print(
        comparison.round(3)
    )

    print("\nEstimated Elasticity:")

    print(
        successful[
            "estimated_elasticity"
        ]
        .describe()
        .round(3)
    )

    print(
        "\nConfidence Distribution:"
    )

    print(
        successful[
            "confidence"
        ]
        .value_counts()
    )

    print("\nClass Estimates:")

    class_output = (
        successful[
            [
                "department",
                "category",
                "product_class",
                "estimated_elasticity",
                "true_elasticity_mean",
                "absolute_error",
                "elasticity_std_error",
                "elasticity_p_value",
                "r_squared",
                "n_skus",
                "n_price_moves",
                "confidence",
            ]
        ]
        .sort_values(
            [
                "department",
                "category",
                "product_class",
            ]
        )
    )

    print(
        class_output.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nBest Within-SKU Estimates:")

    best = (
        successful.sort_values(
            "absolute_error"
        )
        [
            [
                "product_class",
                "estimated_elasticity",
                "true_elasticity_mean",
                "absolute_error",
                "n_price_moves",
                "confidence",
            ]
        ]
        .head(10)
    )

    print(
        best.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nWorst Within-SKU Estimates:")

    worst = (
        successful.sort_values(
            "absolute_error",
            ascending=False,
        )
        [
            [
                "product_class",
                "estimated_elasticity",
                "true_elasticity_mean",
                "absolute_error",
                "n_price_moves",
                "confidence",
            ]
        ]
        .head(10)
    )

    print(
        worst.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        f"\nOutput                : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()