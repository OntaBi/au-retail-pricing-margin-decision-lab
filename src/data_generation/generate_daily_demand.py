from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_competitor.parquet"
)

DEMAND_PROFILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "demand_profiles.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_demand_history.parquet"
)


def generate_daily_demand(
    history: pd.DataFrame,
    profiles: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    result = history.merge(
        profiles,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Own-price effect
    # ---------------------------------------------------------

    result["own_price_change_pct"] = (
        result["regular_sell_price"]
        / result["baseline_sell_price"]
        - 1
    )

    result["own_price_demand_factor"] = (
        1
        + (
            result["true_price_elasticity"]
            * result["own_price_change_pct"]
        )
    )

    result["own_price_demand_factor"] = (
        result["own_price_demand_factor"]
        .clip(
            lower=0.20,
            upper=2.00,
        )
    )

    # ---------------------------------------------------------
    # Competitive-position effect
    #
    # If we are more expensive than the competitor,
    # demand receives additional downward pressure.
    # ---------------------------------------------------------

    result["competitive_gap"] = (
        result["price_index"] - 1
    )

    result["competitive_demand_factor"] = np.where(
        result["competitive_gap"] >= 0,
        1 - (
            0.80
            * result["competitive_gap"]
        ),
        1 - (
            0.35
            * result["competitive_gap"]
        ),
    )

    result["competitive_demand_factor"] = (
        result["competitive_demand_factor"]
        .clip(
            lower=0.65,
            upper=1.20,
        )
    )

    # ---------------------------------------------------------
    # Day-of-week effect
    # ---------------------------------------------------------

    day_of_week = result["date"].dt.dayofweek

    weekday_factor = np.select(
        [
            day_of_week == 0,
            day_of_week == 1,
            day_of_week == 2,
            day_of_week == 3,
            day_of_week == 4,
            day_of_week == 5,
            day_of_week == 6,
        ],
        [
            0.94,
            0.98,
            1.00,
            1.02,
            1.08,
            1.12,
            0.86,
        ],
        default=1.00,
    )

    result["weekday_factor"] = (
        weekday_factor
    )

    # ---------------------------------------------------------
    # Seasonal effect
    #
    # Simple AU retail seasonality for now.
    # Later we can introduce explicit promotional events.
    # ---------------------------------------------------------

    month = result["date"].dt.month

    seasonal_factor = np.select(
        [
            month == 1,
            month == 2,
            month == 6,
            month == 7,
            month == 11,
            month == 12,
        ],
        [
            0.94,
            0.96,
            1.06,
            1.04,
            1.12,
            1.18,
        ],
        default=1.00,
    )

    result["seasonal_factor"] = (
        seasonal_factor
    )

    # ---------------------------------------------------------
    # Expected demand before random variation
    # ---------------------------------------------------------

    result["expected_units"] = (
        result["baseline_daily_units"]
        * result["own_price_demand_factor"]
        * result["competitive_demand_factor"]
        * result["weekday_factor"]
        * result["seasonal_factor"]
    )

    # ---------------------------------------------------------
    # Random daily variation
    # ---------------------------------------------------------

    noise = rng.normal(
        loc=0.0,
        scale=result["demand_volatility"].to_numpy(),
    )

    result["demand_noise_factor"] = (
        1 + noise
    )

    result["demand_noise_factor"] = (
        result["demand_noise_factor"]
        .clip(
            lower=0.25,
            upper=1.75,
        )
    )

    noisy_expected_units = (
        result["expected_units"]
        * result["demand_noise_factor"]
    )

    noisy_expected_units = (
        noisy_expected_units
        .clip(lower=0)
    )

    # Poisson sampling gives us integer transaction demand
    # while preserving low-volume zero-sales days.

    result["units_sold"] = rng.poisson(
        noisy_expected_units
    )

    # ---------------------------------------------------------
    # Commercial measures
    # ---------------------------------------------------------

    result["sales_dollars"] = (
        result["units_sold"]
        * result["regular_sell_price"]
    )

    result["gross_margin_dollars"] = (
        result["units_sold"]
        * (
            result["regular_sell_price"]
            - result["cost_price"]
        )
    )

    return result


def validate_result(
    result: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    assert len(result) == len(history)

    assert (
        result["sku_id"].nunique()
        == history["sku_id"].nunique()
    )

    assert (
        result["units_sold"] >= 0
    ).all()

    assert (
        result["units_sold"]
        % 1
        == 0
    ).all()

    assert (
        result["sales_dollars"] >= 0
    ).all()

    assert (
        result["expected_units"] >= 0
    ).all()

    required = [
        "baseline_daily_units",
        "true_price_elasticity",
        "expected_units",
        "units_sold",
    ]

    assert (
        result[required]
        .isna()
        .sum()
        .sum()
        == 0
    )

    print("\nValidation passed.")


def main() -> None:
    history = pd.read_parquet(
        HISTORY_PATH
    )

    profiles = pd.read_parquet(
        DEMAND_PROFILE_PATH
    )

    result = generate_daily_demand(
        history,
        profiles,
    )

    validate_result(
        result,
        history,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDAILY DEMAND FOUNDATION")
    print("=" * 60)

    print(
        f"Rows                  : "
        f"{len(result):,}"
    )

    print(
        f"SKUs                  : "
        f"{result['sku_id'].nunique():,}"
    )

    print(
        f"Dates                 : "
        f"{result['date'].nunique():,}"
    )

    print(
        f"Total units           : "
        f"{result['units_sold'].sum():,.0f}"
    )

    print(
        f"Avg units / SKU / day : "
        f"{result['units_sold'].mean():.2f}"
    )

    print(
        f"Zero-sales days       : "
        f"{(result['units_sold'] == 0).mean():.1%}"
    )

    print(
        f"Total sales           : "
        f"${result['sales_dollars'].sum():,.0f}"
    )

    print(
        f"Total gross margin    : "
        f"${result['gross_margin_dollars'].sum():,.0f}"
    )

    print(
        f"Output                : "
        f"{OUTPUT_PATH}"
    )

    print("\nDemand by Price Position:")

    result["price_position"] = pd.cut(
        result["price_index"],
        bins=[
            -np.inf,
            0.95,
            1.05,
            np.inf,
        ],
        labels=[
            "Cheaper",
            "Near Parity",
            "More Expensive",
        ],
    )

    summary = (
        result.groupby(
            "price_position",
            observed=True,
        )
        .agg(
            rows=("sku_id", "size"),
            avg_units=("units_sold", "mean"),
            avg_expected_units=(
                "expected_units",
                "mean",
            ),
            avg_price_index=(
                "price_index",
                "mean",
            ),
        )
        .round(2)
    )

    print(summary)

    print("\nSample:")

    sample = result[
        [
            "date",
            "sku_id",
            "regular_sell_price",
            "competitor_price",
            "price_index",
            "true_price_elasticity",
            "expected_units",
            "units_sold",
            "sales_dollars",
            "gross_margin_dollars",
        ]
    ].head(15)

    print(
        sample.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


if __name__ == "__main__":
    main()