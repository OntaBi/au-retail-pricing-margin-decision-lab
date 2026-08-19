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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "class_elasticity_estimates.parquet"
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
    # Modelling fields
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
    # SKU fixed effects
    #
    # These absorb persistent differences in demand level
    # between SKUs inside the same product class.
    # ---------------------------------------------------------

    sku_dummies = pd.get_dummies(
        class_data["sku_id"],
        prefix="sku",
        drop_first=True,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Month fixed effects
    # ---------------------------------------------------------

    month_dummies = pd.get_dummies(
        class_data["month"],
        prefix="month",
        drop_first=True,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Model matrix
    # ---------------------------------------------------------

    X = pd.concat(
        [
            class_data[
                [
                    "log_price_model",
                    "log_competitor_price_model",
                ]
            ].reset_index(drop=True),
            sku_dummies.reset_index(drop=True),
            month_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    y = (
        class_data["log_units_model"]
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Fit pooled fixed-effects regression
    # ---------------------------------------------------------

    try:
        model = sm.OLS(
            y,
            X,
        ).fit()

        estimated_elasticity = (
            model.params.get(
                "log_price_model",
                np.nan,
            )
        )

        elasticity_std_error = (
            model.bse.get(
                "log_price_model",
                np.nan,
            )
        )

        elasticity_p_value = (
            model.pvalues.get(
                "log_price_model",
                np.nan,
            )
        )

        competitor_coefficient = (
            model.params.get(
                "log_competitor_price_model",
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

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    n_skus = (
        class_data["sku_id"].nunique()
    )

    n_observations = (
        len(class_data)
    )

    n_price_moves = int(
        class_data["price_changed"].sum()
    )

    price_summary = (
        class_data.groupby("sku_id")[
            "avg_sell_price"
        ]
        .agg(
            [
                "min",
                "max",
            ]
        )
    )

    price_summary["range_pct"] = (
        price_summary["max"]
        / price_summary["min"]
        - 1
    )

    avg_price_range_pct = (
        price_summary["range_pct"].mean()
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
    # Confidence rule
    # ---------------------------------------------------------

    if (
        model_status == "Success"
        and n_price_moves >= 50
        and elasticity_p_value < 0.05
        and elasticity_std_error <= 0.50
    ):
        confidence = "High"

    elif (
        model_status == "Success"
        and n_price_moves >= 35
        and elasticity_p_value < 0.10
        and elasticity_std_error <= 0.80
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
        "avg_price_range_pct": (
            avg_price_range_pct
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
        result = estimate_single_class(
            class_data
        )

        results.append(
            result
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

    assert len(successful) > 0

    assert successful[
        "estimated_elasticity"
    ].notna().all()

    print("\nValidation passed.")


def main() -> None:
    data = pd.read_parquet(
        DATA_PATH
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

    print(
        "\nPRODUCT-CLASS ELASTICITY ESTIMATION"
    )
    print("=" * 70)

    print(
        f"Product classes       : "
        f"{len(estimates):,}"
    )

    print(
        f"Successful models     : "
        f"{len(successful):,}"
    )

    print(
        f"Failed models         : "
        f"{len(estimates) - len(successful):,}"
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
        "\nHidden True Class Elasticity:"
    )

    print(
        successful[
            "true_elasticity_mean"
        ]
        .describe()
        .round(3)
    )

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------

    mae = (
        successful[
            "absolute_error"
        ].mean()
    )

    median_ae = (
        successful[
            "absolute_error"
        ].median()
    )

    correlation = (
        successful[
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
            successful[
                "estimated_elasticity"
            ] < 0
        )
        .mean()
        * 100
    )

    significant_pct = (
        (
            successful[
                "elasticity_p_value"
            ] < 0.05
        )
        .mean()
        * 100
    )

    print("\nEstimation Accuracy:")

    print(
        f"Mean absolute error   : "
        f"{mae:.3f}"
    )

    print(
        f"Median absolute error : "
        f"{median_ae:.3f}"
    )

    print(
        f"Truth correlation     : "
        f"{correlation:.3f}"
    )

    print(
        f"Negative estimates    : "
        f"{negative_pct:.1f}%"
    )

    print(
        f"p-value < 0.05        : "
        f"{significant_pct:.1f}%"
    )

    print("\nConfidence Distribution:")

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

    print("\nBest Class Estimates:")

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

    print("\nWorst Class Estimates:")

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