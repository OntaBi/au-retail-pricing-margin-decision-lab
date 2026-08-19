from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
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
    / "elasticity_difference_dataset.parquet"
)


def prepare_differences(
    weekly: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:

    eligible_skus = eligibility.loc[
        eligibility["evidence_tier"].isin(
            ["Strong", "Moderate"]
        ),
        "sku_id",
    ]

    df = weekly[
        weekly["sku_id"].isin(
            eligible_skus
        )
    ].copy()

    df = df.sort_values(
        ["sku_id", "week_start"]
    )

    # ---------------------------------------------------------
    # Log levels
    # ---------------------------------------------------------

    df["log_units_level"] = np.log1p(
        df["units_sold"]
    )

    df["log_price_level"] = np.log(
        df["avg_sell_price"]
    )

    df["log_competitor_price_level"] = np.log(
        df["avg_competitor_price"]
    )

    # ---------------------------------------------------------
    # Week-over-week differences
    # ---------------------------------------------------------

    grouped = df.groupby("sku_id")

    df["delta_log_units"] = grouped[
        "log_units_level"
    ].diff()

    df["delta_log_price"] = grouped[
        "log_price_level"
    ].diff()

    df["delta_log_competitor_price"] = grouped[
        "log_competitor_price_level"
    ].diff()

    # ---------------------------------------------------------
    # Flags
    # ---------------------------------------------------------

    tolerance = 1e-10

    df["own_price_moved"] = (
        df["delta_log_price"].abs()
        > tolerance
    )

    df["competitor_price_moved"] = (
        df[
            "delta_log_competitor_price"
        ].abs()
        > tolerance
    )

    # Remove first observation per SKU because diff = NaN
    result = df[
        df["delta_log_units"].notna()
    ].copy()

    return result


def validate_result(
    result: pd.DataFrame,
) -> None:

    assert len(result) > 0

    assert (
        result["sku_id"].nunique()
        == 326
    )

    assert result[
        "delta_log_units"
    ].notna().all()

    assert result[
        "delta_log_price"
    ].notna().all()

    assert result[
        "delta_log_competitor_price"
    ].notna().all()

    print("\nValidation passed.")


def main() -> None:

    weekly = pd.read_parquet(
        DATA_PATH
    )

    eligibility = pd.read_parquet(
        ELIGIBILITY_PATH
    )

    result = prepare_differences(
        weekly,
        eligibility,
    )

    validate_result(result)

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nELASTICITY DIFFERENCE DATASET")
    print("=" * 60)

    print(
        f"Rows                 : "
        f"{len(result):,}"
    )

    print(
        f"SKUs                 : "
        f"{result['sku_id'].nunique():,}"
    )

    print(
        f"Own-price move rows  : "
        f"{result['own_price_moved'].sum():,}"
    )

    print(
        f"Competitor move rows : "
        f"{result['competitor_price_moved'].sum():,}"
    )

    both = (
        result["own_price_moved"]
        & result["competitor_price_moved"]
    )

    print(
        f"Both moved           : "
        f"{both.sum():,}"
    )

    print("\nOwn Price Change Distribution:")

    own_moves = result.loc[
        result["own_price_moved"],
        "delta_log_price",
    ]

    print(
        own_moves.describe().round(4)
    )

    print("\nDemand Change During Own Price Moves:")

    demand_moves = result.loc[
        result["own_price_moved"],
        "delta_log_units",
    ]

    print(
        demand_moves.describe().round(4)
    )

    print("\nPrice-Move Rows per SKU:")

    moves_per_sku = (
        result[
            result["own_price_moved"]
        ]
        .groupby("sku_id")
        .size()
    )

    print(
        moves_per_sku
        .describe()
        .round(2)
    )

    print("\nRaw Change Correlation:")

    move_rows = result[
        result["own_price_moved"]
    ]

    correlation = move_rows[
        [
            "delta_log_price",
            "delta_log_units",
        ]
    ].corr().iloc[0, 1]

    print(
        f"Price vs demand change : "
        f"{correlation:.3f}"
    )

    print(
        f"\nOutput               : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()