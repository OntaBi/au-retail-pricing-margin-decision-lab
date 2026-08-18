from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "pricing_history.parquet"
)

START_DATE = "2024-07-01"
END_DATE = "2026-06-30"


def generate_pricing_history(
    product_master: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.DataFrame(
        {
            "date": pd.date_range(
                start=START_DATE,
                end=END_DATE,
                freq="D",
            )
        }
    )

    skus = product_master[
        [
            "sku_id",
            "cost_price",
            "regular_sell_price",
        ]
    ].copy()

    dates["join_key"] = 1
    skus["join_key"] = 1

    pricing_history = dates.merge(
        skus,
        on="join_key",
        how="inner",
    ).drop(
        columns="join_key"
    )

    pricing_history = pricing_history[
        [
            "date",
            "sku_id",
            "cost_price",
            "regular_sell_price",
        ]
    ]

    pricing_history = pricing_history.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    return pricing_history


def validate_pricing_history(
    pricing_history: pd.DataFrame,
    product_master: pd.DataFrame,
) -> None:
    expected_dates = len(
        pd.date_range(
            START_DATE,
            END_DATE,
            freq="D",
        )
    )

    expected_rows = (
        len(product_master)
        * expected_dates
    )

    assert len(pricing_history) == expected_rows
    assert pricing_history["sku_id"].nunique() == len(product_master)
    assert pricing_history["date"].nunique() == expected_dates
    assert pricing_history.isna().sum().sum() == 0

    print("\nValidation passed.")


def main() -> None:
    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    pricing_history = generate_pricing_history(
        product_master
    )

    validate_pricing_history(
        pricing_history,
        product_master,
    )

    pricing_history.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nPRICING HISTORY FOUNDATION")
    print("=" * 50)

    print(
        f"Date range : "
        f"{pricing_history['date'].min().date()} "
        f"to "
        f"{pricing_history['date'].max().date()}"
    )

    print(
        f"SKUs       : "
        f"{pricing_history['sku_id'].nunique():,}"
    )

    print(
        f"Dates      : "
        f"{pricing_history['date'].nunique():,}"
    )

    print(
        f"Rows       : "
        f"{len(pricing_history):,}"
    )

    print(
        f"Output     : "
        f"{OUTPUT_PATH}"
    )

    print("\nSample:")
    print(
        pricing_history.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()