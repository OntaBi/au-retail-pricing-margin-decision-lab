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
    / "cost_change_events.parquet"
)

START_DATE = pd.Timestamp("2024-09-01")
END_DATE = pd.Timestamp("2026-04-30")


def generate_cost_change_events(
    product_master: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    eligible = product_master[
        product_master["lifecycle_stage"] != "Exit"
    ].copy()

    selected_skus = eligible.sample(
        frac=0.55,
        random_state=seed,
    )

    events = []

    for row in selected_skus.itertuples():
        event_count = rng.choice(
            [1, 2, 3],
            p=[0.65, 0.28, 0.07],
        )

        event_dates = pd.to_datetime(
            rng.choice(
                pd.date_range(
                    START_DATE,
                    END_DATE,
                    freq="D",
                ),
                size=event_count,
                replace=False,
            )
        ).sort_values()

        current_cost = row.cost_price

        for event_date in event_dates:
            change_pct = rng.normal(
                loc=0.045,
                scale=0.035,
            )

            change_pct = np.clip(
                change_pct,
                -0.06,
                0.18,
            )

            old_cost = current_cost
            new_cost = old_cost * (
                1 + change_pct
            )

            old_cost = round(old_cost, 2)
            new_cost = round(new_cost, 2)

            actual_change_pct = (
                new_cost / old_cost
            ) - 1

            events.append(
                {
                    "sku_id": row.sku_id,
                    "event_date": event_date,
                    "old_cost_price": old_cost,
                    "new_cost_price": new_cost,
                    "cost_change_pct": round(
                        actual_change_pct,
                        4,
                    ),
                }
            )

            current_cost = new_cost

    event_df = pd.DataFrame(events)

    event_df = event_df.sort_values(
        [
            "sku_id",
            "event_date",
        ]
    ).reset_index(drop=True)

    return event_df


def validate_events(
    events: pd.DataFrame,
) -> None:
    assert not events.empty

    assert events.isna().sum().sum() == 0

    assert (
        events["new_cost_price"] > 0
    ).all()

    assert (
        events["old_cost_price"] > 0
    ).all()

    assert events["event_date"].between(
        START_DATE,
        END_DATE,
    ).all()

    print("\nValidation passed.")


def main() -> None:
    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    events = generate_cost_change_events(
        product_master
    )

    validate_events(events)

    events.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nCOST CHANGE EVENTS")
    print("=" * 50)

    print(
        f"SKUs affected : "
        f"{events['sku_id'].nunique():,}"
    )

    print(
        f"Events        : "
        f"{len(events):,}"
    )

    print(
        f"Avg change    : "
        f"{events['cost_change_pct'].mean():.2%}"
    )

    print(
        f"Min change    : "
        f"{events['cost_change_pct'].min():.2%}"
    )

    print(
        f"Max change    : "
        f"{events['cost_change_pct'].max():.2%}"
    )

    print(
        f"Output        : "
        f"{OUTPUT_PATH}"
    )

    print("\nSample:")
    print(
        events.head(15).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()