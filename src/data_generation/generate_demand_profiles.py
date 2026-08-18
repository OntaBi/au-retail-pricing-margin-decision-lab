from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42

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
    / "demand_profiles.parquet"
)


CLASS_DEMAND = {
    "Laptops": ((0.4, 2.0), (-2.6, -1.4)),
    "Monitors": ((0.8, 3.5), (-2.2, -1.2)),
    "Keyboards & Mice": ((2.0, 10.0), (-1.9, -0.9)),
    "Computer Accessories": ((3.0, 15.0), (-1.8, -0.8)),
    "Printers": ((0.7, 3.0), (-2.0, -1.1)),
    "Ink & Toner": ((2.0, 9.0), (-1.4, -0.6)),
    "Printer Accessories": ((2.0, 8.0), (-1.6, -0.7)),
    "Headphones": ((1.5, 8.0), (-2.2, -1.0)),
    "Speakers": ((0.8, 4.0), (-2.0, -1.0)),
    "Audio Accessories": ((2.0, 10.0), (-1.8, -0.8)),
    "Pens": ((8.0, 35.0), (-1.0, -0.3)),
    "Notebooks": ((5.0, 22.0), (-1.1, -0.4)),
    "Filing": ((3.0, 12.0), (-1.0, -0.4)),
    "Desk Accessories": ((2.0, 9.0), (-1.3, -0.5)),
    "Copy Paper": ((8.0, 30.0), (-0.9, -0.3)),
    "Specialty Paper": ((3.0, 12.0), (-1.2, -0.5)),
    "Labels": ((3.0, 14.0), (-1.1, -0.4)),
    "Mailing": ((4.0, 18.0), (-1.0, -0.4)),
    "Packaging": ((3.0, 14.0), (-1.1, -0.4)),
    "Tape": ((5.0, 22.0), (-0.9, -0.3)),
    "Desks": ((0.3, 1.8), (-1.8, -0.8)),
    "Office Chairs": ((0.4, 2.2), (-2.0, -0.9)),
    "Storage": ((0.4, 2.0), (-1.6, -0.7)),
    "Home Office Desks": ((0.4, 2.0), (-1.8, -0.8)),
    "Home Office Chairs": ((0.4, 2.2), (-1.9, -0.8)),
    "Ergonomic Accessories": ((1.0, 5.0), (-1.5, -0.6)),
}


PRICE_BAND_VOLUME_MULTIPLIER = {
    "Entry": 1.35,
    "Value": 1.15,
    "Core": 1.00,
    "Premium": 0.65,
}


LIFECYCLE_VOLUME_MULTIPLIER = {
    "New": 0.75,
    "Core": 1.00,
    "Mature": 0.85,
    "Exit": 0.45,
}


def generate_demand_profiles(
    product_master: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    records = []

    for row in product_master.itertuples():
        volume_range, elasticity_range = (
            CLASS_DEMAND[row.product_class]
        )

        baseline_units = rng.uniform(
            volume_range[0],
            volume_range[1],
        )

        baseline_units *= (
            PRICE_BAND_VOLUME_MULTIPLIER[
                row.price_band
            ]
        )

        baseline_units *= (
            LIFECYCLE_VOLUME_MULTIPLIER[
                row.lifecycle_stage
            ]
        )

        elasticity = rng.uniform(
            elasticity_range[0],
            elasticity_range[1],
        )

        demand_volatility = rng.uniform(
            0.08,
            0.28,
        )

        records.append(
            {
                "sku_id": row.sku_id,
                "baseline_daily_units": round(
                    baseline_units,
                    3,
                ),
                "true_price_elasticity": round(
                    elasticity,
                    3,
                ),
                "demand_volatility": round(
                    demand_volatility,
                    3,
                ),
            }
        )

    return pd.DataFrame(records)


def validate_profiles(
    profiles: pd.DataFrame,
    product_master: pd.DataFrame,
) -> None:
    assert len(profiles) == len(
        product_master
    )

    assert (
        profiles["sku_id"].nunique()
        == len(product_master)
    )

    assert (
        profiles["baseline_daily_units"] > 0
    ).all()

    assert (
        profiles["true_price_elasticity"] < 0
    ).all()

    assert (
        profiles["demand_volatility"]
        .between(0, 1)
        .all()
    )

    assert (
        profiles.isna().sum().sum()
        == 0
    )

    print("\nValidation passed.")


def main() -> None:
    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    profiles = generate_demand_profiles(
        product_master
    )

    validate_profiles(
        profiles,
        product_master,
    )

    profiles.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    analysis = product_master[
        [
            "sku_id",
            "department",
            "product_class",
            "price_band",
            "lifecycle_stage",
        ]
    ].merge(
        profiles,
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    print("\nDEMAND PROFILE FOUNDATION")
    print("=" * 60)

    print(
        f"SKUs                  : "
        f"{len(profiles):,}"
    )

    print(
        f"Avg daily units       : "
        f"{profiles['baseline_daily_units'].mean():.2f}"
    )

    print(
        f"Median daily units    : "
        f"{profiles['baseline_daily_units'].median():.2f}"
    )

    print(
        f"Avg elasticity        : "
        f"{profiles['true_price_elasticity'].mean():.2f}"
    )

    print(
        f"Min elasticity        : "
        f"{profiles['true_price_elasticity'].min():.2f}"
    )

    print(
        f"Max elasticity        : "
        f"{profiles['true_price_elasticity'].max():.2f}"
    )

    print(
        f"Output                : "
        f"{OUTPUT_PATH}"
    )

    print("\nDemand by Department:")

    print(
        analysis.groupby(
            "department"
        ).agg(
            skus=("sku_id", "count"),
            avg_daily_units=(
                "baseline_daily_units",
                "mean",
            ),
            avg_elasticity=(
                "true_price_elasticity",
                "mean",
            ),
        ).round(2)
    )

    print("\nDemand by Price Band:")

    print(
        analysis.groupby(
            "price_band"
        ).agg(
            skus=("sku_id", "count"),
            avg_daily_units=(
                "baseline_daily_units",
                "mean",
            ),
            avg_elasticity=(
                "true_price_elasticity",
                "mean",
            ),
        ).round(2)
    )

    print("\nSample:")

    print(
        analysis[
            [
                "sku_id",
                "department",
                "product_class",
                "price_band",
                "baseline_daily_units",
                "true_price_elasticity",
                "demand_volatility",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()