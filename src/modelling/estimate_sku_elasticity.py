from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELLING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_modelling_dataset.parquet"
)

ELIGIBILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_eligibility.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "sku_elasticity_estimates.parquet"
)


def estimate_single_sku(
    sku_data: pd.DataFrame,
) -> dict:

    sku_data = (
        sku_data
        .sort_values("week_start")
        .copy()
    )

    sku_id = sku_data["sku_id"].iloc[0]

    # ---------------------------------------------------------
    # Month controls
    # ---------------------------------------------------------

    month_dummies = pd.get_dummies(
        sku_data["month"],
        prefix="month",
        drop_first=True,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Model matrix
    # ---------------------------------------------------------

    X = pd.concat(
        [
            sku_data[
                [
                    "log_price",
                    "log_competitor_price",
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
        sku_data["log_units"]
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Fit OLS model
    # ---------------------------------------------------------

    try:
        model = sm.OLS(
            y,
            X,
        ).fit()

        estimated_elasticity = (
            model.params.get(
                "log_price",
                np.nan,
            )
        )

        elasticity_std_error = (
            model.bse.get(
                "log_price",
                np.nan,
            )
        )

        elasticity_p_value = (
            model.pvalues.get(
                "log_price",
                np.nan,
            )
        )

        r_squared = model.rsquared

        n_obs = int(model.nobs)

        model_status = "Success"

    except Exception as exc:

        estimated_elasticity = np.nan
        elasticity_std_error = np.nan
        elasticity_p_value = np.nan
        r_squared = np.nan
        n_obs = len(sku_data)

        model_status = (
            f"Failed: {type(exc).__name__}"
        )

    # ---------------------------------------------------------
    # Hidden truth is retained ONLY for model evaluation.
    # It is not used as a predictor.
    # ---------------------------------------------------------

    true_elasticity = (
        sku_data[
            "true_price_elasticity"
        ].iloc[0]
    )

    if pd.notna(
        estimated_elasticity
    ):
        estimation_error = (
            estimated_elasticity
            - true_elasticity
        )

        absolute_error = abs(
            estimation_error
        )

    else:
        estimation_error = np.nan
        absolute_error = np.nan

    return {
        "sku_id": sku_id,
        "department": (
            sku_data[
                "department"
            ].iloc[0]
        ),
        "category": (
            sku_data[
                "category"
            ].iloc[0]
        ),
        "product_class": (
            sku_data[
                "product_class"
            ].iloc[0]
        ),
        "price_band": (
            sku_data[
                "price_band"
            ].iloc[0]
        ),
        "estimated_elasticity": (
            estimated_elasticity
        ),
        "true_elasticity": (
            true_elasticity
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
        "r_squared": (
            r_squared
        ),
        "n_obs": (
            n_obs
        ),
        "model_status": (
            model_status
        ),
    }


def estimate_all_skus(
    weekly: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:

    eligible = eligibility[
        eligibility[
            "evidence_tier"
        ].isin(
            [
                "Strong",
                "Moderate",
            ]
        )
    ].copy()

    eligible_skus = set(
        eligible["sku_id"]
    )

    modelling_data = weekly[
        weekly["sku_id"].isin(
            eligible_skus
        )
    ].copy()

    results = []

    grouped = modelling_data.groupby(
        "sku_id",
        sort=True,
    )

    for sku_id, sku_data in grouped:

        result = estimate_single_sku(
            sku_data
        )

        results.append(result)

    estimates = pd.DataFrame(
        results
    )

    estimates = estimates.merge(
        eligible[
            [
                "sku_id",
                "evidence_tier",
                "unique_prices",
                "price_change_weeks",
                "price_range_pct",
                "avg_weekly_units",
            ]
        ],
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    return estimates


def validate_result(
    estimates: pd.DataFrame,
) -> None:

    assert len(estimates) > 0

    assert (
        estimates["sku_id"].nunique()
        == len(estimates)
    )

    assert estimates[
        "evidence_tier"
    ].isin(
        [
            "Strong",
            "Moderate",
        ]
    ).all()

    successful = estimates[
        estimates["model_status"]
        == "Success"
    ]

    assert len(successful) > 0

    print("\nValidation passed.")


def main() -> None:

    weekly = pd.read_parquet(
        MODELLING_DATA_PATH
    )

    eligibility = pd.read_parquet(
        ELIGIBILITY_PATH
    )

    estimates = estimate_all_skus(
        weekly,
        eligibility,
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

    print("\nSKU ELASTICITY ESTIMATION")
    print("=" * 65)

    print(
        f"Eligible SKUs         : "
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

    print("\nHidden True Elasticity:")

    print(
        successful[
            "true_elasticity"
        ]
        .describe()
        .round(3)
    )

    print("\nEstimation Accuracy:")

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
                "true_elasticity",
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

    print("\nAccuracy by Evidence Tier:")

    tier_summary = (
        successful.groupby(
            "evidence_tier"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            mean_estimate=(
                "estimated_elasticity",
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
            median_r2=(
                "r_squared",
                "median",
            ),
        )
        .round(3)
    )

    print(
        tier_summary
    )

    print("\nSample Estimates:")

    sample = (
        successful[
            [
                "sku_id",
                "product_class",
                "evidence_tier",
                "estimated_elasticity",
                "true_elasticity",
                "absolute_error",
                "elasticity_p_value",
                "r_squared",
                "unique_prices",
                "price_range_pct",
            ]
        ]
        .sort_values(
            "absolute_error"
        )
        .head(15)
    )

    print(
        sample.to_string(
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