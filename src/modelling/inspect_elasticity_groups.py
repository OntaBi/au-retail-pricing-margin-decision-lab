from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_modelling_dataset.parquet"
)


def main() -> None:

    df = pd.read_parquet(
        DATA_PATH
    )

    # ---------------------------------------------------------
    # Product-class diagnostics
    # ---------------------------------------------------------

    class_summary = (
        df.groupby(
            [
                "department",
                "category",
                "product_class",
            ]
        )
        .agg(
            skus=(
                "sku_id",
                "nunique",
            ),
            observations=(
                "sku_id",
                "size",
            ),
            price_move_rows=(
                "price_changed",
                "sum",
            ),
            unique_prices=(
                "avg_sell_price",
                "nunique",
            ),
            avg_weekly_units=(
                "units_sold",
                "mean",
            ),
            true_elasticity_mean=(
                "true_price_elasticity",
                "mean",
            ),
            true_elasticity_std=(
                "true_price_elasticity",
                "std",
            ),
        )
        .reset_index()
    )

    class_summary[
        "moves_per_sku"
    ] = (
        class_summary[
            "price_move_rows"
        ]
        / class_summary["skus"]
    )

    # ---------------------------------------------------------
    # Category diagnostics
    # ---------------------------------------------------------

    category_summary = (
        df.groupby(
            [
                "department",
                "category",
            ]
        )
        .agg(
            skus=(
                "sku_id",
                "nunique",
            ),
            observations=(
                "sku_id",
                "size",
            ),
            price_move_rows=(
                "price_changed",
                "sum",
            ),
            avg_weekly_units=(
                "units_sold",
                "mean",
            ),
            true_elasticity_mean=(
                "true_price_elasticity",
                "mean",
            ),
            true_elasticity_std=(
                "true_price_elasticity",
                "std",
            ),
        )
        .reset_index()
    )

    category_summary[
        "moves_per_sku"
    ] = (
        category_summary[
            "price_move_rows"
        ]
        / category_summary["skus"]
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print(
        "\nELASTICITY GROUP DIAGNOSTICS"
    )
    print("=" * 70)

    print(
        f"Product classes       : "
        f"{len(class_summary):,}"
    )

    print(
        f"Categories            : "
        f"{len(category_summary):,}"
    )

    print("\nProduct-Class SKU Coverage:")

    print(
        class_summary["skus"]
        .describe()
        .round(2)
    )

    print("\nProduct-Class Price Moves:")

    print(
        class_summary[
            "price_move_rows"
        ]
        .describe()
        .round(2)
    )

    print("\nMoves per SKU by Product Class:")

    print(
        class_summary[
            "moves_per_sku"
        ]
        .describe()
        .round(2)
    )

    print(
        "\nSmallest Product Classes:"
    )

    smallest = (
        class_summary.sort_values(
            [
                "skus",
                "price_move_rows",
            ]
        )
        .head(15)
    )

    print(
        smallest[
            [
                "department",
                "category",
                "product_class",
                "skus",
                "observations",
                "price_move_rows",
                "moves_per_sku",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(
        "\nLargest Product Classes:"
    )

    largest = (
        class_summary.sort_values(
            "skus",
            ascending=False,
        )
        .head(15)
    )

    print(
        largest[
            [
                "department",
                "category",
                "product_class",
                "skus",
                "observations",
                "price_move_rows",
                "moves_per_sku",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(
        "\nTrue Elasticity Variation "
        "within Product Classes:"
    )

    print(
        class_summary[
            "true_elasticity_std"
        ]
        .describe()
        .round(3)
    )

    print(
        "\nCategory Coverage:"
    )

    print(
        category_summary[
            [
                "department",
                "category",
                "skus",
                "price_move_rows",
                "moves_per_sku",
            ]
        ]
        .sort_values(
            "skus",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


if __name__ == "__main__":
    main()