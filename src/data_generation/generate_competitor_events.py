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

PRICING_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_prices.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "competitor_price_events.parquet"
)

START_DATE = pd.Timestamp("2024-07-01")
END_DATE = pd.Timestamp("2026-06-30")


def generate_competitor_events(
    product_master: pd.DataFrame,
    pricing_history: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    our_price_lookup = (
        pricing_history
        .set_index(["sku_id", "date"])[
            "regular_sell_price"
        ]
        .to_dict()
    )

    records = []

    for row in product_master.itertuples():
        starting_position = rng.normal(
            loc=-0.015,
            scale=0.045,
        )

        starting_position = np.clip(
            starting_position,
            -0.12,
            0.12,
        )

        current_price = (
            row.regular_sell_price
            * (1 + starting_position)
        )

        current_price = round(
            max(current_price, 0.50),
            2,
        )

        records.append(
            {
                "sku_id": row.sku_id,
                "event_date": START_DATE,
                "event_type": "Initial Price",
                "old_competitor_price": current_price,
                "new_competitor_price": current_price,
                "competitor_change_pct": 0.0,
            }
        )

        event_count = int(
            rng.choice(
                [0, 1, 2, 3, 4],
                p=[
                    0.10,
                    0.25,
                    0.30,
                    0.25,
                    0.10,
                ],
            )
        )

        if event_count == 0:
            continue

        possible_dates = pd.date_range(
            START_DATE + pd.Timedelta(days=30),
            END_DATE - pd.Timedelta(days=14),
            freq="D",
        )

        event_dates = pd.to_datetime(
            rng.choice(
                possible_dates,
                size=event_count,
                replace=False,
            )
        ).sort_values()

        for event_date in event_dates:
            our_price = our_price_lookup[
                (
                    row.sku_id,
                    pd.Timestamp(event_date),
                )
            ]

            current_index = (
                our_price / current_price
            )

            competitive_gap = (
                current_index - 1
            )

            random_change = rng.normal(
                loc=0.003,
                scale=0.040,
            )

            reversion_adjustment = (
                competitive_gap * 0.35
            )

            change_pct = (
                random_change
                + reversion_adjustment
            )

            change_pct = np.clip(
                change_pct,
                -0.15,
                0.15,
            )

            old_price = current_price

            new_price = (
                old_price
                * (1 + change_pct)
            )

            new_price = round(
                max(new_price, 0.50),
                2,
            )

            actual_change_pct = (
                new_price / old_price - 1
            )

            records.append(
                {
                    "sku_id": row.sku_id,
                    "event_date": event_date,
                    "event_type": "Price Change",
                    "old_competitor_price": old_price,
                    "new_competitor_price": new_price,
                    "competitor_change_pct": round(
                        actual_change_pct,
                        4,
                    ),
                }
            )

            current_price = new_price

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "sku_id",
                "event_date",
            ]
        )
        .reset_index(drop=True)
    )


def validate_events(
    events: pd.DataFrame,
    product_master: pd.DataFrame,
) -> None:
    assert (
        events["sku_id"].nunique()
        == len(product_master)
    )

    assert (
        events["new_competitor_price"] > 0
    ).all()

    assert events.isna().sum().sum() == 0

    initial = events[
        events["event_type"] == "Initial Price"
    ]

    assert len(initial) == len(
        product_master
    )

    assert (
        initial.groupby("sku_id")
        .size()
        .eq(1)
        .all()
    )

    print("\nValidation passed.")


def main() -> None:
    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    pricing_history = pd.read_parquet(
        PRICING_HISTORY_PATH
    )

    events = generate_competitor_events(
        product_master,
        pricing_history,
    )

    validate_events(
        events,
        product_master,
    )

    events.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    price_changes = events[
        events["event_type"] == "Price Change"
    ]

    print("\nCOMPETITOR PRICE EVENTS")
    print("=" * 60)

    print(
        f"SKUs                  : "
        f"{events['sku_id'].nunique():,}"
    )

    print(
        f"Initial prices        : "
        f"{(events['event_type'] == 'Initial Price').sum():,}"
    )

    print(
        f"Price change events   : "
        f"{len(price_changes):,}"
    )

    print(
        f"SKUs with changes     : "
        f"{price_changes['sku_id'].nunique():,}"
    )

    print(
        f"Avg competitor change : "
        f"{price_changes['competitor_change_pct'].mean():.2%}"
    )

    print(
        f"Avg absolute change   : "
        f"{price_changes['competitor_change_pct'].abs().mean():.2%}"
    )

    print(
        f"Price decreases       : "
        f"{(price_changes['competitor_change_pct'] < 0).sum():,}"
    )

    print(
        f"Price increases       : "
        f"{(price_changes['competitor_change_pct'] > 0).sum():,}"
    )

    print(
        f"Output                : "
        f"{OUTPUT_PATH}"
    )

    print("\nSample:")
    print(
        events.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()