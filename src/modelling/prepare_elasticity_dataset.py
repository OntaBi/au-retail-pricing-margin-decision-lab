from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_demand_history.parquet"
)

PRODUCT_MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "product_master.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_modelling_dataset.parquet"
)


def prepare_elasticity_dataset(
    history: pd.DataFrame,
    product_master: pd.DataFrame,
) -> pd.DataFrame:

    df = history.copy()

    # ---------------------------------------------------------
    # Add product hierarchy
    # ---------------------------------------------------------

    hierarchy = product_master[
        [
            "sku_id",
            "department",
            "category",
            "product_class",
            "price_band",
        ]
    ].copy()

    df = df.merge(
        hierarchy,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Calendar foundation
    # ---------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["week_start"] = (
        df["date"]
        - pd.to_timedelta(
            df["date"].dt.dayofweek,
            unit="D",
        )
    )

    # ---------------------------------------------------------
    # Aggregate daily observations to SKU-week
    # ---------------------------------------------------------

    weekly = (
        df.groupby(
            [
                "sku_id",
                "week_start",
            ],
            as_index=False,
        )
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
            avg_sell_price=(
                "regular_sell_price",
                "mean",
            ),
            avg_competitor_price=(
                "competitor_price",
                "mean",
            ),
            avg_cost_price=(
                "cost_price",
                "mean",
            ),
            units_sold=(
                "units_sold",
                "sum",
            ),
            sales_dollars=(
                "sales_dollars",
                "sum",
            ),
            gross_margin_dollars=(
                "gross_margin_dollars",
                "sum",
            ),
            baseline_daily_units=(
                "baseline_daily_units",
                "first",
            ),
            true_price_elasticity=(
                "true_price_elasticity",
                "first",
            ),
        )
    )

    # ---------------------------------------------------------
    # Competitive position
    # ---------------------------------------------------------

    weekly["price_index"] = (
        weekly["avg_sell_price"]
        / weekly["avg_competitor_price"]
    )

    weekly["price_gap_pct"] = (
        weekly["price_index"] - 1
    )

    # ---------------------------------------------------------
    # SKU-relative price movement
    #
    # Normalising against each SKU's own median price prevents
    # $5 stationery and $500 furniture from being compared
    # directly on absolute price.
    # ---------------------------------------------------------

    sku_reference_price = (
        weekly.groupby("sku_id")[
            "avg_sell_price"
        ]
        .transform("median")
    )

    weekly["relative_price"] = (
        weekly["avg_sell_price"]
        / sku_reference_price
    )

    # ---------------------------------------------------------
    # Log transforms
    #
    # In a log-log demand model, the coefficient on log_price
    # can be interpreted as price elasticity.
    # ---------------------------------------------------------

    weekly["log_price"] = np.log(
        weekly["relative_price"]
    )

    weekly["log_competitor_price"] = np.log(
        weekly["avg_competitor_price"]
    )

    # log1p lets us retain weeks with zero units.
    weekly["log_units"] = np.log1p(
        weekly["units_sold"]
    )

    # ---------------------------------------------------------
    # Calendar controls
    # ---------------------------------------------------------

    weekly["month"] = (
        weekly["week_start"].dt.month
    )

    weekly["year"] = (
        weekly["week_start"].dt.year
    )

    # ---------------------------------------------------------
    # Price movement diagnostics
    # ---------------------------------------------------------

    weekly["price_changed"] = (
        weekly.groupby("sku_id")[
            "avg_sell_price"
        ]
        .diff()
        .abs()
        .fillna(0)
        > 0.01
    )

    return weekly


def validate_dataset(
    weekly: pd.DataFrame,
) -> None:

    assert len(weekly) > 0

    assert (
        weekly["sku_id"].nunique()
        == 1500
    )

    assert (
        weekly["avg_sell_price"] > 0
    ).all()

    assert (
        weekly["avg_competitor_price"] > 0
    ).all()

    assert (
        weekly["units_sold"] >= 0
    ).all()

    assert (
        weekly[
            [
                "department",
                "category",
                "product_class",
                "price_band",
            ]
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )

    assert (
        weekly[
            [
                "log_price",
                "log_competitor_price",
                "log_units",
            ]
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )

    print("\nValidation passed.")


def main() -> None:

    history = pd.read_parquet(
        INPUT_PATH
    )

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    weekly = prepare_elasticity_dataset(
        history,
        product_master,
    )

    validate_dataset(
        weekly
    )

    weekly.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    skus_with_price_move = (
        weekly.loc[
            weekly["price_changed"],
            "sku_id",
        ]
        .nunique()
    )

    print("\nELASTICITY MODELLING DATASET")
    print("=" * 60)

    print(
        f"Rows                 : "
        f"{len(weekly):,}"
    )

    print(
        f"SKUs                 : "
        f"{weekly['sku_id'].nunique():,}"
    )

    print(
        f"Weeks                : "
        f"{weekly['week_start'].nunique():,}"
    )

    print(
        f"Date range           : "
        f"{weekly['week_start'].min().date()}"
        f" to "
        f"{weekly['week_start'].max().date()}"
    )

    print(
        f"Price-change weeks   : "
        f"{weekly['price_changed'].sum():,}"
    )

    print(
        f"SKUs with price move : "
        f"{skus_with_price_move:,}"
    )

    print("\nWeekly Demand Summary:")

    print(
        weekly["units_sold"]
        .describe()
        .round(2)
    )

    print("\nPrice Variation by SKU:")

    variation = (
        weekly.groupby("sku_id")[
            "avg_sell_price"
        ]
        .agg(
            ["min", "max", "nunique"]
        )
    )

    variation["range_pct"] = (
        variation["max"]
        / variation["min"]
        - 1
    )

    print(
        variation["range_pct"]
        .describe()
        .round(3)
    )

    print("\nSKUs by Number of Unique Weekly Prices:")

    print(
        variation["nunique"]
        .value_counts()
        .sort_index()
        .head(15)
    )

    print("\nSample:")

    sample = weekly[
        [
            "sku_id",
            "week_start",
            "department",
            "product_class",
            "avg_sell_price",
            "avg_competitor_price",
            "price_index",
            "units_sold",
            "log_price",
            "log_units",
            "price_changed",
        ]
    ].head(15)

    print(
        sample.to_string(
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