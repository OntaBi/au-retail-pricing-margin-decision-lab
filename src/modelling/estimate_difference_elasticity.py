from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIFFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_difference_dataset.parquet"
)

ELIGIBILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_eligibility.parquet"
)

BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "sku_elasticity_estimates.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "sku_difference_elasticity_estimates.parquet"
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
    # Keep observations where either our price or competitor
    # price moved.
    #
    # Stable weeks add no information to the differenced
    # price coefficients.
    # ---------------------------------------------------------

    model_data = sku_data[
        sku_data["own_price_moved"]
        | sku_data["competitor_price_moved"]
    ].copy()

    n_move_obs = len(model_data)

    n_own_moves = int(
        model_data["own_price_moved"].sum()
    )

    n_competitor_moves = int(
        model_data[
            "competitor_price_moved"
        ].sum()
    )

    # ---------------------------------------------------------
    # Need enough observations to fit both coefficients.
    # ---------------------------------------------------------

    if n_move_obs < 3 or n_own_moves < 2:

        return build_failed_result(
            sku_data=sku_data,
            sku_id=sku_id,
            n_move_obs=n_move_obs,
            n_own_moves=n_own_moves,
            n_competitor_moves=n_competitor_moves,
            status="Insufficient move observations",
        )

    # ---------------------------------------------------------
    # Difference regression
    #
    # Δlog(units) =
    #     beta_own * Δlog(our price)
    #   + beta_comp * Δlog(competitor price)
    #
    # We deliberately do NOT add an intercept.
    # If neither price changes, predicted price-driven demand
    # change should be zero.
    # ---------------------------------------------------------

    X = model_data[
        [
            "delta_log_price",
            "delta_log_competitor_price",
        ]
    ].astype(float)

    y = model_data[
        "delta_log_units"
    ].astype(float)

    try:

        model = sm.OLS(
            y,
            X,
        ).fit()

        estimated_elasticity = (
            model.params.get(
                "delta_log_price",
                np.nan,
            )
        )

        elasticity_std_error = (
            model.bse.get(
                "delta_log_price",
                np.nan,
            )
        )

        elasticity_p_value = (
            model.pvalues.get(
                "delta_log_price",
                np.nan,
            )
        )

        competitor_coefficient = (
            model.params.get(
                "delta_log_competitor_price",
                np.nan,
            )
        )

        r_squared = model.rsquared

        model_status = "Success"

    except Exception as exc:

        return build_failed_result(
            sku_data=sku_data,
            sku_id=sku_id,
            n_move_obs=n_move_obs,
            n_own_moves=n_own_moves,
            n_competitor_moves=n_competitor_moves,
            status=(
                f"Failed: "
                f"{type(exc).__name__}"
            ),
        )

    true_elasticity = (
        sku_data[
            "true_price_elasticity"
        ].iloc[0]
    )

    estimation_error = (
        estimated_elasticity
        - true_elasticity
    )

    absolute_error = abs(
        estimation_error
    )

    return {
        "sku_id": sku_id,
        "department": (
            sku_data["department"].iloc[0]
        ),
        "category": (
            sku_data["category"].iloc[0]
        ),
        "product_class": (
            sku_data[
                "product_class"
            ].iloc[0]
        ),
        "price_band": (
            sku_data["price_band"].iloc[0]
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
        "competitor_coefficient": (
            competitor_coefficient
        ),
        "r_squared": (
            r_squared
        ),
        "n_move_obs": (
            n_move_obs
        ),
        "n_own_moves": (
            n_own_moves
        ),
        "n_competitor_moves": (
            n_competitor_moves
        ),
        "model_status": (
            model_status
        ),
    }


def build_failed_result(
    sku_data,
    sku_id,
    n_move_obs,
    n_own_moves,
    n_competitor_moves,
    status,
):

    return {
        "sku_id": sku_id,
        "department": (
            sku_data["department"].iloc[0]
        ),
        "category": (
            sku_data["category"].iloc[0]
        ),
        "product_class": (
            sku_data[
                "product_class"
            ].iloc[0]
        ),
        "price_band": (
            sku_data["price_band"].iloc[0]
        ),
        "estimated_elasticity": np.nan,
        "true_elasticity": (
            sku_data[
                "true_price_elasticity"
            ].iloc[0]
        ),
        "estimation_error": np.nan,
        "absolute_error": np.nan,
        "elasticity_std_error": np.nan,
        "elasticity_p_value": np.nan,
        "competitor_coefficient": np.nan,
        "r_squared": np.nan,
        "n_move_obs": n_move_obs,
        "n_own_moves": n_own_moves,
        "n_competitor_moves": (
            n_competitor_moves
        ),
        "model_status": status,
    }


def estimate_all_skus(
    differences: pd.DataFrame,
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

    results = []

    for _, sku_data in differences.groupby(
        "sku_id",
        sort=True,
    ):

        results.append(
            estimate_single_sku(
                sku_data
            )
        )

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

    assert len(estimates) == 326

    assert (
        estimates["sku_id"].nunique()
        == 326
    )

    successful = estimates[
        estimates["model_status"]
        == "Success"
    ]

    assert len(successful) > 0

    print("\nValidation passed.")


def calculate_metrics(
    data: pd.DataFrame,
) -> dict:

    clean = data.dropna(
        subset=[
            "estimated_elasticity",
            "true_elasticity",
        ]
    )

    if len(clean) == 0:

        return {
            "skus": 0,
            "mae": np.nan,
            "median_ae": np.nan,
            "correlation": np.nan,
            "negative_pct": np.nan,
        }

    mae = clean[
        "absolute_error"
    ].mean()

    median_ae = clean[
        "absolute_error"
    ].median()

    correlation = clean[
        [
            "estimated_elasticity",
            "true_elasticity",
        ]
    ].corr().iloc[0, 1]

    negative_pct = (
        (
            clean[
                "estimated_elasticity"
            ] < 0
        ).mean()
        * 100
    )

    return {
        "skus": len(clean),
        "mae": mae,
        "median_ae": median_ae,
        "correlation": correlation,
        "negative_pct": negative_pct,
    }


def main() -> None:

    differences = pd.read_parquet(
        DIFFERENCE_DATA_PATH
    )

    eligibility = pd.read_parquet(
        ELIGIBILITY_PATH
    )

    baseline = pd.read_parquet(
        BASELINE_PATH
    )

    estimates = estimate_all_skus(
        differences,
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

    print(
        "\nDIFFERENCE ELASTICITY ESTIMATION"
    )

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
        f"Insufficient / failed : "
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

    print("\nModel Comparison:")

    # Compare only SKUs for which challenger succeeded.
    comparison_skus = set(
        successful["sku_id"]
    )

    baseline_comparable = baseline[
        baseline["sku_id"].isin(
            comparison_skus
        )
    ].copy()

    baseline_metrics = (
        calculate_metrics(
            baseline_comparable
        )
    )

    challenger_metrics = (
        calculate_metrics(
            successful
        )
    )

    comparison = pd.DataFrame(
        {
            "Baseline OLS": [
                baseline_metrics["skus"],
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
            ],
            "Difference Model": [
                challenger_metrics["skus"],
                challenger_metrics["mae"],
                challenger_metrics[
                    "median_ae"
                ],
                challenger_metrics[
                    "correlation"
                ],
                challenger_metrics[
                    "negative_pct"
                ],
            ],
        },
        index=[
            "SKUs",
            "MAE",
            "Median AE",
            "Truth Correlation",
            "Negative %",
        ],
    )

    print(
        comparison.round(3)
    )

    print(
        "\nDifference Model by Evidence Tier:"
    )

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
            avg_own_moves=(
                "n_own_moves",
                "mean",
            ),
        )
        .round(3)
    )

    print(tier_summary)

    print("\nBest Challenger Estimates:")

    best = (
        successful.sort_values(
            "absolute_error"
        )
        [
            [
                "sku_id",
                "product_class",
                "evidence_tier",
                "estimated_elasticity",
                "true_elasticity",
                "absolute_error",
                "n_own_moves",
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

    print("\nWorst Challenger Estimates:")

    worst = (
        successful.sort_values(
            "absolute_error",
            ascending=False,
        )
        [
            [
                "sku_id",
                "product_class",
                "evidence_tier",
                "estimated_elasticity",
                "true_elasticity",
                "absolute_error",
                "n_own_moves",
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