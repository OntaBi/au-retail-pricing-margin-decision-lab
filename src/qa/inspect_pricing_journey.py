from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_prices.parquet"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "price_response_events.parquet"
)

PRODUCT_MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "product_master.parquet"
)


def calculate_margin(
    sell_price: pd.Series,
    cost_price: pd.Series,
) -> pd.Series:
    return (
        sell_price - cost_price
    ) / sell_price


def find_example_event(
    events: pd.DataFrame,
) -> pd.Series:
    candidates = events[
        (events["response_type"] == "Full Cost Recovery")
        & (events["new_cost_price"] > events["old_cost_price"])
        & (events["response_delay_days"] >= 14)
        & (events["response_delay_days"] <= 40)
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No suitable pricing journey found."
        )

    candidates["cost_change_pct"] = (
        candidates["new_cost_price"]
        / candidates["old_cost_price"]
        - 1
    )

    candidates = candidates[
        candidates["cost_change_pct"] >= 0.05
    ]

    if candidates.empty:
        raise ValueError(
            "No suitable >=5% cost increase found."
        )

    return candidates.sort_values(
        "cost_change_pct",
        ascending=False,
    ).iloc[0]


def main() -> None:
    history = pd.read_parquet(
        HISTORY_PATH
    )

    events = pd.read_parquet(
        EVENTS_PATH
    )

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    event = find_example_event(
        events
    )

    sku_id = event["sku_id"]

    sku_master = product_master[
        product_master["sku_id"] == sku_id
    ].iloc[0]

    sku_history = history[
        history["sku_id"] == sku_id
    ].copy()

    sku_history["gross_margin_pct"] = (
        calculate_margin(
            sku_history["regular_sell_price"],
            sku_history["cost_price"],
        )
    )

    cost_event_date = pd.Timestamp(
        event["cost_event_date"]
    )

    response_date = pd.Timestamp(
        event["response_date"]
    )

    dates_to_show = [
        cost_event_date - pd.Timedelta(days=1),
        cost_event_date,
        cost_event_date + pd.Timedelta(days=1),
        response_date - pd.Timedelta(days=1),
        response_date,
        response_date + pd.Timedelta(days=1),
    ]

    journey = sku_history[
        sku_history["date"].isin(
            dates_to_show
        )
    ][
        [
            "date",
            "cost_price",
            "regular_sell_price",
            "gross_margin_pct",
        ]
    ].copy()

    journey["gross_margin_pct"] = (
        journey["gross_margin_pct"] * 100
    ).round(2)

    print("\nPRICING JOURNEY")
    print("=" * 65)

    print(
        f"SKU           : {sku_id}"
    )

    print(
        f"Department    : "
        f"{sku_master['department']}"
    )

    print(
        f"Category      : "
        f"{sku_master['category']}"
    )

    print(
        f"Product Class : "
        f"{sku_master['product_class']}"
    )

    print(
        f"Price Band    : "
        f"{sku_master['price_band']}"
    )

    print(
        f"\nResponse      : "
        f"{event['response_type']}"
    )

    print(
        f"Response Delay: "
        f"{int(event['response_delay_days'])} days"
    )

    print("\nJourney:")
    print(
        journey.to_string(
            index=False
        )
    )

    before_margin = (
        (
            event["old_sell_price"]
            - event["old_cost_price"]
        )
        / event["old_sell_price"]
        * 100
    )

    eroded_margin = (
        (
            event["old_sell_price"]
            - event["new_cost_price"]
        )
        / event["old_sell_price"]
        * 100
    )

    recovered_margin = (
        (
            event["new_sell_price"]
            - event["new_cost_price"]
        )
        / event["new_sell_price"]
        * 100
    )

    print("\nCommercial Impact:")
    print(
        f"Margin before cost change : "
        f"{before_margin:.2f}%"
    )

    print(
        f"Margin after cost change  : "
        f"{eroded_margin:.2f}%"
    )

    print(
        f"Margin after price change : "
        f"{recovered_margin:.2f}%"
    )

    print(
        f"Immediate erosion         : "
        f"{before_margin - eroded_margin:.2f}ppt"
    )

    print(
        f"Residual erosion          : "
        f"{before_margin - recovered_margin:.2f}ppt"
    )


if __name__ == "__main__":
    main()