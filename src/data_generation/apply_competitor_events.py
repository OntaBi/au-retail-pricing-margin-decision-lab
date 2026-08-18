from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICING_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_prices.parquet"
)

COMPETITOR_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "competitor_price_events.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_competitor.parquet"
)


def apply_competitor_events(
    pricing_history: pd.DataFrame,
    competitor_events: pd.DataFrame,
) -> pd.DataFrame:
    history = pricing_history.copy()

    events = competitor_events[
        [
            "sku_id",
            "event_date",
            "new_competitor_price",
        ]
    ].copy()

    history = history.sort_values(
        ["date", "sku_id"]
    ).reset_index(drop=True)

    events = events.sort_values(
        ["event_date", "sku_id"]
    ).reset_index(drop=True)

    result = pd.merge_asof(
        history,
        events,
        left_on="date",
        right_on="event_date",
        by="sku_id",
        direction="backward",
    )

    result = result.rename(
        columns={
            "new_competitor_price":
                "competitor_price"
        }
    )

    result["price_index"] = (
        result["regular_sell_price"]
        / result["competitor_price"]
    )

    result["price_gap_dollars"] = (
        result["regular_sell_price"]
        - result["competitor_price"]
    )

    result["price_gap_pct"] = (
        result["price_index"] - 1
    )

    result = result.drop(
        columns=["event_date"]
    )

    result = result.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    return result


def validate_result(
    original: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    assert len(result) == len(original)

    assert (
        result["sku_id"].nunique()
        == original["sku_id"].nunique()
    )

    assert (
        result["date"].nunique()
        == original["date"].nunique()
    )

    assert (
        result["competitor_price"] > 0
    ).all()

    assert (
        result["price_index"] > 0
    ).all()

    assert result.isna().sum().sum() == 0

    print("\nValidation passed.")


def main() -> None:
    pricing_history = pd.read_parquet(
        PRICING_HISTORY_PATH
    )

    competitor_events = pd.read_parquet(
        COMPETITOR_EVENTS_PATH
    )

    result = apply_competitor_events(
        pricing_history,
        competitor_events,
    )

    validate_result(
        pricing_history,
        result,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    latest_date = result["date"].max()

    latest = result[
        result["date"] == latest_date
    ].copy()

    print("\nCOMPETITOR EVENTS APPLIED")
    print("=" * 60)

    print(
        f"Rows                : "
        f"{len(result):,}"
    )

    print(
        f"SKUs                : "
        f"{result['sku_id'].nunique():,}"
    )

    print(
        f"Dates               : "
        f"{result['date'].nunique():,}"
    )

    print(
        f"Latest date         : "
        f"{latest_date.date()}"
    )

    print(
        f"Avg price index     : "
        f"{latest['price_index'].mean():.3f}"
    )

    print(
        f"Median price index  : "
        f"{latest['price_index'].median():.3f}"
    )

    print(
        f"Index < 0.95        : "
        f"{(latest['price_index'] < 0.95).sum():,}"
    )

    print(
        f"Index 0.95-1.05     : "
        f"{latest['price_index'].between(0.95, 1.05).sum():,}"
    )

    print(
        f"Index > 1.05        : "
        f"{(latest['price_index'] > 1.05).sum():,}"
    )

    print(
        f"Index > 1.10        : "
        f"{(latest['price_index'] > 1.10).sum():,}"
    )

    print(
        f"Output              : "
        f"{OUTPUT_PATH}"
    )

    print("\nMost Expensive vs Competitor:")

    print(
        latest[
            [
                "sku_id",
                "regular_sell_price",
                "competitor_price",
                "price_gap_dollars",
                "price_gap_pct",
                "price_index",
            ]
        ]
        .nlargest(
            10,
            "price_index",
        )
        .round(
            {
                "price_gap_pct": 3,
                "price_index": 3,
            }
        )
        .to_string(index=False)
    )

    print("\nCheapest vs Competitor:")

    print(
        latest[
            [
                "sku_id",
                "regular_sell_price",
                "competitor_price",
                "price_gap_dollars",
                "price_gap_pct",
                "price_index",
            ]
        ]
        .nsmallest(
            10,
            "price_index",
        )
        .round(
            {
                "price_gap_pct": 3,
                "price_index": 3,
            }
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()