from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_competitor.parquet"
)

OUR_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "price_response_events.parquet"
)

COMPETITOR_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "competitor_price_events.parquet"
)

PRODUCT_MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "product_master.parquet"
)


def main() -> None:
    history = pd.read_parquet(
        HISTORY_PATH
    )

    our_events = pd.read_parquet(
        OUR_EVENTS_PATH
    )

    competitor_events = pd.read_parquet(
        COMPETITOR_EVENTS_PATH
    )

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    our_changes = our_events[
        our_events["response_date"].notna()
    ][
        [
            "sku_id",
            "response_date",
        ]
    ].copy()

    competitor_changes = competitor_events[
        competitor_events["event_type"]
        == "Price Change"
    ][
        [
            "sku_id",
            "event_date",
        ]
    ].copy()

    candidate_skus = set(
        our_changes["sku_id"]
    ).intersection(
        competitor_changes["sku_id"]
    )

    if not candidate_skus:
        raise ValueError(
            "No SKU found with both own and competitor price changes."
        )

    candidates = history[
        history["sku_id"].isin(
            candidate_skus
        )
    ].copy()

    latest_date = candidates["date"].max()

    latest = candidates[
        candidates["date"] == latest_date
    ].copy()

    latest["distance_from_parity"] = (
        latest["price_index"] - 1
    ).abs()

    sku_id = (
        latest.sort_values(
            "distance_from_parity",
            ascending=False,
        )
        .iloc[0]["sku_id"]
    )

    sku_master = product_master[
        product_master["sku_id"] == sku_id
    ].iloc[0]

    sku_history = history[
        history["sku_id"] == sku_id
    ].copy()

    sku_our_events = our_changes[
        our_changes["sku_id"] == sku_id
    ]

    sku_comp_events = competitor_changes[
        competitor_changes["sku_id"] == sku_id
    ]

    event_dates = set(
        sku_our_events["response_date"]
    ).union(
        set(
            sku_comp_events["event_date"]
        )
    )

    dates_to_show = set()

    for event_date in event_dates:
        event_date = pd.Timestamp(
            event_date
        )

        dates_to_show.add(
            event_date
            - pd.Timedelta(days=1)
        )

        dates_to_show.add(
            event_date
        )

        dates_to_show.add(
            event_date
            + pd.Timedelta(days=1)
        )

    journey = sku_history[
        sku_history["date"].isin(
            dates_to_show
        )
    ][
        [
            "date",
            "regular_sell_price",
            "competitor_price",
            "price_gap_dollars",
            "price_gap_pct",
            "price_index",
        ]
    ].copy()

    journey["price_gap_pct"] = (
        journey["price_gap_pct"] * 100
    ).round(2)

    journey["price_index"] = (
        journey["price_index"]
        .round(3)
    )

    print("\nCOMPETITIVE PRICING JOURNEY")
    print("=" * 75)

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
        f"\nOur price events       : "
        f"{len(sku_our_events)}"
    )

    print(
        f"Competitor price events: "
        f"{len(sku_comp_events)}"
    )

    print("\nJourney:")

    print(
        journey.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()