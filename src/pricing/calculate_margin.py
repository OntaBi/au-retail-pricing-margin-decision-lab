from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_costs.parquet"
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
    / "pricing_history_with_margin.parquet"
)


def calculate_margin(
    pricing_history: pd.DataFrame,
    product_master: pd.DataFrame,
) -> pd.DataFrame:
    df = pricing_history.copy()

    baseline = product_master[
        [
            "sku_id",
            "gross_margin_pct",
        ]
    ].rename(
        columns={
            "gross_margin_pct": "baseline_margin_pct"
        }
    )

    df = df.merge(
        baseline,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    df["gross_margin_dollars"] = (
        df["regular_sell_price"]
        - df["cost_price"]
    )

    df["gross_margin_pct"] = (
        df["gross_margin_dollars"]
        / df["regular_sell_price"]
    )

    df["margin_erosion_ppt"] = (
        df["baseline_margin_pct"]
        - df["gross_margin_pct"]
    ) * 100

    df["gross_margin_dollars"] = (
        df["gross_margin_dollars"].round(2)
    )

    df["gross_margin_pct"] = (
        df["gross_margin_pct"].round(4)
    )

    df["baseline_margin_pct"] = (
        df["baseline_margin_pct"].round(4)
    )

    df["margin_erosion_ppt"] = (
        df["margin_erosion_ppt"].round(2)
    )

    return df


def validate_result(
    df: pd.DataFrame,
) -> None:
    assert len(df) == 1_095_000

    assert df.isna().sum().sum() == 0

    assert (
        df["regular_sell_price"] > 0
    ).all()

    print("\nValidation passed.")


def main() -> None:
    pricing_history = pd.read_parquet(
        INPUT_PATH
    )

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    result = calculate_margin(
        pricing_history,
        product_master,
    )

    validate_result(result)

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    latest_date = result["date"].max()

    latest = result[
        result["date"] == latest_date
    ].copy()

    erosion = latest[
        latest["margin_erosion_ppt"] > 0
    ]

    material_erosion = latest[
        latest["margin_erosion_ppt"] >= 2
    ]

    severe_erosion = latest[
        latest["margin_erosion_ppt"] >= 5
    ]

    print("\nMARGIN ANALYSIS")
    print("=" * 50)

    print(
        f"Rows                    : "
        f"{len(result):,}"
    )

    print(
        f"Latest date             : "
        f"{latest_date.date()}"
    )

    print(
        f"SKUs with erosion       : "
        f"{len(erosion):,}"
    )

    print(
        f"SKUs >= 2ppt erosion    : "
        f"{len(material_erosion):,}"
    )

    print(
        f"SKUs >= 5ppt erosion    : "
        f"{len(severe_erosion):,}"
    )

    print(
        f"Max margin erosion      : "
        f"{latest['margin_erosion_ppt'].max():.2f}ppt"
    )

    print(
        f"Output                  : "
        f"{OUTPUT_PATH}"
    )

    print("\nTop 15 Margin Erosion SKUs:")

    print(
        latest[
            [
                "sku_id",
                "cost_price",
                "regular_sell_price",
                "baseline_margin_pct",
                "gross_margin_pct",
                "margin_erosion_ppt",
            ]
        ]
        .sort_values(
            "margin_erosion_ppt",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()