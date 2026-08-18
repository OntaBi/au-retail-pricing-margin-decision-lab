from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COST_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "cost_change_events.parquet"
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
    / "price_response_events.parquet"
)


COST_INCREASE_RESPONSES = {
    "No Response": 0.00,
    "Partial Cost Recovery": 0.60,
    "Full Cost Recovery": 1.00,
    "Over Recovery": 1.15,
}

COST_DECREASE_RESPONSES = {
    "Retain Margin": 0.00,
    "Partial Pass Through": 0.50,
    "Full Pass Through": 1.00,
}


def generate_price_response_events(
    cost_events: pd.DataFrame,
    product_master: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sku_attributes = product_master[
        [
            "sku_id",
            "regular_sell_price",
        ]
    ].rename(
        columns={
            "regular_sell_price": "baseline_sell_price"
        }
    )

    events = cost_events.merge(
        sku_attributes,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    response_records = []

    current_sell_prices = dict(
        zip(
            product_master["sku_id"],
            product_master["regular_sell_price"],
        )
    )

    events = events.sort_values(
        [
            "sku_id",
            "event_date",
        ]
    )

    for row in events.itertuples():
        current_sell = current_sell_prices[
            row.sku_id
        ]

        cost_change = (
            row.new_cost_price
            - row.old_cost_price
        )

        if cost_change >= 0:
            response_type = rng.choice(
                list(
                    COST_INCREASE_RESPONSES.keys()
                ),
                p=[
                    0.20,
                    0.35,
                    0.35,
                    0.10,
                ],
            )

            recovery_rate = (
                COST_INCREASE_RESPONSES[
                    response_type
                ]
            )

        else:
            response_type = rng.choice(
                list(
                    COST_DECREASE_RESPONSES.keys()
                ),
                p=[
                    0.50,
                    0.30,
                    0.20,
                ],
            )

            recovery_rate = (
                COST_DECREASE_RESPONSES[
                    response_type
                ]
            )

        proposed_sell_change = (
            cost_change
            * recovery_rate
        )

        new_sell_price = (
            current_sell
            + proposed_sell_change
        )

        if recovery_rate == 0:
            delay_days = 0
            response_date = pd.NaT
            new_sell_price = current_sell

        else:
            delay_days = int(
                rng.integers(
                    7,
                    46,
                )
            )

            response_date = (
                row.event_date
                + pd.Timedelta(
                    days=delay_days
                )
            )

        new_sell_price = round(
            new_sell_price,
            2,
        )

        sell_change_pct = (
            new_sell_price
            / current_sell
            - 1
        )

        response_records.append(
            {
                "sku_id": row.sku_id,
                "cost_event_date": row.event_date,
                "response_date": response_date,
                "response_type": response_type,
                "response_delay_days": delay_days,
                "recovery_rate": recovery_rate,
                "old_cost_price": row.old_cost_price,
                "new_cost_price": row.new_cost_price,
                "old_sell_price": current_sell,
                "new_sell_price": new_sell_price,
                "sell_change_pct": round(
                    sell_change_pct,
                    4,
                ),
            }
        )

        if recovery_rate > 0:
            current_sell_prices[
                row.sku_id
            ] = new_sell_price

    return (
        pd.DataFrame(response_records)
        .sort_values(
            [
                "sku_id",
                "cost_event_date",
            ]
        )
        .reset_index(drop=True)
    )


def validate_events(
    events: pd.DataFrame,
    cost_events: pd.DataFrame,
) -> None:
    assert len(events) == len(cost_events)

    assert (
        events["new_sell_price"] > 0
    ).all()

    price_changed = (
        events["recovery_rate"] > 0
    )

    assert (
        events.loc[
            price_changed,
            "response_date",
        ].notna()
    ).all()

    assert (
        events.loc[
            ~price_changed,
            "response_date",
        ].isna()
    ).all()

    assert (
        events.loc[
            ~price_changed,
            "response_delay_days",
        ] == 0
    ).all()

    assert (
        events.loc[
            ~price_changed,
            "new_sell_price",
        ]
        == events.loc[
            ~price_changed,
            "old_sell_price",
        ]
    ).all()

    print("\nValidation passed.")


def main() -> None:
    cost_events = pd.read_parquet(
        COST_EVENTS_PATH
    )

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    events = generate_price_response_events(
        cost_events,
        product_master,
    )

    validate_events(
        events,
        cost_events,
    )

    events.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nPRICE RESPONSE EVENTS")
    print("=" * 55)

    print(
        f"Cost events     : "
        f"{len(events):,}"
    )

    print(
        f"SKUs            : "
        f"{events['sku_id'].nunique():,}"
    )

    print("\nResponse Mix:")

    response_mix = (
        events["response_type"]
        .value_counts()
        .to_frame("events")
    )

    response_mix["pct"] = (
        response_mix["events"]
        / len(events)
        * 100
    ).round(1)

    print(response_mix)

    responded = events[
        events["response_type"]
        != "No Response"
    ]

    print(
        f"\nAvg response delay : "
        f"{responded['response_delay_days'].mean():.1f} days"
    )

    print(
        f"Avg sell change    : "
        f"{responded['sell_change_pct'].mean():.2%}"
    )

    print(
        f"Output             : "
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