from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "generated" / "product_master.parquet"


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    print("\nPRODUCT MASTER")
    print("=" * 80)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample SKUs:")
    print(df.head(10).to_string(index=False))

    print("\nSKUs by Department:")
    print(
        df.groupby("department")
        .size()
        .sort_values(ascending=False)
    )

    print("\nSKUs by Category:")
    print(
        df.groupby(["department", "category"])
        .size()
        .sort_values(ascending=False)
    )

    print("\nPricing by Department:")
    print(
        df.groupby("department")
        .agg(
            skus=("sku_id", "count"),
            avg_cost=("cost_price", "mean"),
            avg_sell=("regular_sell_price", "mean"),
            avg_margin_pct=("gross_margin_pct", "mean"),
        )
        .round(2)
    )

    print("\nPricing by Price Band:")
    print(
        df.groupby("price_band")
        .agg(
            skus=("sku_id", "count"),
            avg_cost=("cost_price", "mean"),
            avg_sell=("regular_sell_price", "mean"),
            avg_margin_pct=("gross_margin_pct", "mean"),
        )
        .round(2)
    )

    print("\nLifecycle Distribution:")
    print(
        df["lifecycle_stage"]
        .value_counts()
    )

    print("\nMissing Values:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()