from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICING_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_costs.parquet"
)

PRICE_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "price_response_events.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_prices.parquet"
)


def apply_price_response_events(
    pricing_history: pd.DataFrame,
    price_events: pd.DataFrame,
) -> pd.DataFrame:
    history = pricing_history.copy()

    events = price_events[
        price_events["response_date"].notna()
    ][
        [
            "sku_id",
            "response_date",
            "new_sell_price",
        ]
    ].copy()

    history = history.rename(
        columns={
            "regular_sell_price": "baseline_sell_price"
        }
    )

    history = history.sort_values(
        ["date", "sku_id"]
    ).reset_index(drop=True)

    events = events.sort_values(
        ["response_date", "sku_id"]
    ).reset_index(drop=True)

    result = pd.merge_asof(
        history,
        events,
        left_on="date",
        right_on="response_date",
        by="sku_id",
        direction="backward",
    )

    result["regular_sell_price"] = (
        result["new_sell_price"]
        .fillna(result["baseline_sell_price"])
    )

    result = result.drop(
        columns=[
            "response_date",
            "new_sell_price",
        ]
    )

    result = result[
        [
            "date",
            "sku_id",
            "cost_price",
            "baseline_sell_price",
            "regular_sell_price",
        ]
    ]

    result = result.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    return result


def validate_result(
    original: pd.DataFrame,
    result: pd.DataFrame,
    events: pd.DataFrame,
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

    assert result.isna().sum().sum() == 0

    assert (
        result["regular_sell_price"] > 0
    ).all()

    actual_price_events = events[
        events["response_date"].notna()
    ]

    changed_skus = set(
        result.loc[
            result["regular_sell_price"]
            != result["baseline_sell_price"],
            "sku_id",
        ].unique()
    )

    event_skus = set(
        actual_price_events["sku_id"].unique()
    )

    assert changed_skus.issubset(
        event_skus
    )

    print("\nValidation passed.")


def main() -> None:
    pricing_history = pd.read_parquet(
        PRICING_HISTORY_PATH
    )

    price_events = pd.read_parquet(
        PRICE_EVENTS_PATH
    )

    result = apply_price_response_events(
        pricing_history,
        price_events,
    )

    validate_result(
        pricing_history,
        result,
        price_events,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    actual_events = price_events[
        price_events["response_date"].notna()
    ]

    changed_rows = (
        result["regular_sell_price"]
        != result["baseline_sell_price"]
    ).sum()

    print("\nPRICE RESPONSE EVENTS APPLIED")
    print("=" * 55)

    print(
        f"Rows               : "
        f"{len(result):,}"
    )

    print(
        f"Actual price events: "
        f"{len(actual_events):,}"
    )

    print(
        f"SKUs price changed : "
        f"{actual_events['sku_id'].nunique():,}"
    )

    print(
        f"Changed daily rows : "
        f"{changed_rows:,}"
    )

    print(
        f"Output             : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()