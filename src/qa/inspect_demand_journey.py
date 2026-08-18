from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_demand_history.parquet"
)

PRODUCT_MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "product_master.parquet"
)

PRICE_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "price_response_events.parquet"
)


def main() -> None:
    history = pd.read_parquet(
        HISTORY_PATH
    )

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    price_events = pd.read_parquet(
        PRICE_EVENTS_PATH
    )

    valid_events = price_events[
        price_events["response_date"].notna()
    ].copy()

    event_counts = (
        valid_events.groupby("sku_id")
        .size()
        .sort_values(ascending=False)
    )

    candidate_skus = event_counts[
        event_counts >= 2
    ].index

    candidates = history[
        history["sku_id"].isin(
            candidate_skus
        )
    ].copy()

    # Prefer a SKU with meaningful elasticity
    # and enough daily demand to observe the response.
    sku_summary = (
        candidates.groupby("sku_id")
        .agg(
            avg_units=("units_sold", "mean"),
            elasticity=(
                "true_price_elasticity",
                "first",
            ),
        )
    )

    sku_summary = sku_summary[
        (sku_summary["avg_units"] >= 3)
        & (sku_summary["elasticity"] <= -1.0)
    ]

    if sku_summary.empty:
        raise ValueError(
            "No suitable SKU found."
        )

    sku_id = (
        sku_summary
        .sort_values(
            "avg_units",
            ascending=False,
        )
        .index[0]
    )

    master = product_master[
        product_master["sku_id"] == sku_id
    ].iloc[0]

    sku_history = history[
        history["sku_id"] == sku_id
    ].copy()

    sku_events = valid_events[
        valid_events["sku_id"] == sku_id
    ].sort_values("response_date")

    sku_event_dates = set(
        pd.to_datetime(
            sku_events["response_date"]
        )
    )

    print("\nDEMAND & PRICING JOURNEY")
    print("=" * 75)

    print(
        f"SKU             : {sku_id}"
    )

    print(
        f"Department      : "
        f"{master['department']}"
    )

    print(
        f"Category        : "
        f"{master['category']}"
    )

    print(
        f"Product Class   : "
        f"{master['product_class']}"
    )

    print(
        f"Price Band      : "
        f"{master['price_band']}"
    )

    elasticity = (
        sku_history[
            "true_price_elasticity"
        ].iloc[0]
    )

    print(
        f"True Elasticity : "
        f"{elasticity:.2f}"
    )

    print(
        f"Price Events    : "
        f"{len(sku_events)}"
    )

    for event in sku_events.itertuples():
        event_date = pd.Timestamp(
            event.response_date
        )

        before_start = (
            event_date
            - pd.Timedelta(days=28)
        )

        before_end = (
            event_date
            - pd.Timedelta(days=1)
        )

        after_start = event_date

        after_end = (
            event_date
            + pd.Timedelta(days=27)
        )

        overlapping_events = [
            other_date
            for other_date in sku_event_dates
            if (
                other_date != event_date
                and before_start
                <= other_date
                <= after_end
            )
        ]

        before = sku_history[
            sku_history["date"].between(
                before_start,
                before_end,
            )
        ]

        after = sku_history[
            sku_history["date"].between(
                after_start,
                after_end,
            )
        ]

        if before.empty or after.empty:
            continue

        before_price = (
            before["regular_sell_price"]
            .mean()
        )

        after_price = (
            after["regular_sell_price"]
            .mean()
        )

        before_units = (
            before["units_sold"].mean()
        )

        after_units = (
            after["units_sold"].mean()
        )

        before_adjusted_units = (
            before["units_sold"]
            / (
                before["weekday_factor"]
                * before["seasonal_factor"]
            )
        ).mean()

        after_adjusted_units = (
            after["units_sold"]
            / (
                after["weekday_factor"]
                * after["seasonal_factor"]
            )
        ).mean()

        before_index = (
            before["price_index"].mean()
        )

        after_index = (
            after["price_index"].mean()
        )

        before_sales = (
            before["sales_dollars"].sum()
        )

        after_sales = (
            after["sales_dollars"].sum()
        )

        before_margin = (
            before[
                "gross_margin_dollars"
            ].sum()
        )

        after_margin = (
            after[
                "gross_margin_dollars"
            ].sum()
        )

        price_change = (
            after_price
            / before_price
            - 1
        )

        unit_change = (
            after_units
            / before_units
            - 1
        )

        adjusted_unit_change = (
            after_adjusted_units
            / before_adjusted_units
            - 1
        )

        sales_change = (
            after_sales
            / before_sales
            - 1
        )

        if before_margin != 0:
            margin_change = (
                after_margin
                / before_margin
                - 1
            )
        else:
            margin_change = float("nan")

        print(
            f"\nPRICE EVENT: "
            f"{event_date.date()}"
        )

        print("-" * 75)

        if overlapping_events:
            overlap_text = ", ".join(
                str(other_date.date())
                for other_date in sorted(
                    overlapping_events
                )
            )

            print(
                "WARNING: overlapping "
                f"price event(s): "
                f"{overlap_text}"
            )

        print(
            f"Avg sell price   "
            f"${before_price:>8.2f}"
            f"  -> "
            f"${after_price:>8.2f}"
            f"   "
            f"{price_change:+.1%}"
        )

        print(
            f"Avg price index  "
            f"{before_index:>9.3f}"
            f"  -> "
            f"{after_index:>9.3f}"
        )

        print(
            f"Avg daily units  "
            f"{before_units:>9.2f}"
            f"  -> "
            f"{after_units:>9.2f}"
            f"   "
            f"{unit_change:+.1%}"
        )

        print(
            f"Adj daily units  "
            f"{before_adjusted_units:>9.2f}"
            f"  -> "
            f"{after_adjusted_units:>9.2f}"
            f"   "
            f"{adjusted_unit_change:+.1%}"
        )

        print(
            f"28-day sales     "
            f"${before_sales:>8,.0f}"
            f"  -> "
            f"${after_sales:>8,.0f}"
            f"   "
            f"{sales_change:+.1%}"
        )

        print(
            f"28-day margin    "
            f"${before_margin:>8,.0f}"
            f"  -> "
            f"${after_margin:>8,.0f}"
            f"   "
            f"{margin_change:+.1%}"
        )


if __name__ == "__main__":
    main()