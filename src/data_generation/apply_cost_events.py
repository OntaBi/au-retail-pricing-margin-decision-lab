from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICING_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history.parquet"
)

COST_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "cost_change_events.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_history_with_costs.parquet"
)


def apply_cost_events(
    pricing_history: pd.DataFrame,
    cost_events: pd.DataFrame,
) -> pd.DataFrame:
    pricing_history = pricing_history.copy()

    cost_events = cost_events[
        [
            "sku_id",
            "event_date",
            "new_cost_price",
        ]
    ].copy()

    pricing_history = pricing_history.sort_values(
        ["date", "sku_id"]
    ).reset_index(drop=True)

    cost_events = cost_events.sort_values(
        ["event_date", "sku_id"]
    ).reset_index(drop=True)

    pricing_history = pd.merge_asof(
        pricing_history,
        cost_events,
        left_on="date",
        right_on="event_date",
        by="sku_id",
        direction="backward",
    )

    pricing_history["cost_price"] = (
        pricing_history["new_cost_price"]
        .fillna(pricing_history["cost_price"])
    )

    pricing_history = pricing_history.drop(
        columns=[
            "event_date",
            "new_cost_price",
        ]
    )

    pricing_history = pricing_history.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    return pricing_history


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

    affected_skus = set(
        events["sku_id"].unique()
    )

    changed_skus = set(
        result.loc[
            result["cost_price"]
            != original["cost_price"],
            "sku_id",
        ].unique()
    )

    assert changed_skus.issubset(
        affected_skus
    )

    print("\nValidation passed.")


def main() -> None:
    pricing_history = pd.read_parquet(
        PRICING_HISTORY_PATH
    )

    cost_events = pd.read_parquet(
        COST_EVENTS_PATH
    )

    result = apply_cost_events(
        pricing_history,
        cost_events,
    )

    validate_result(
        pricing_history,
        result,
        cost_events,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    changed_rows = (
        result["cost_price"]
        != pricing_history["cost_price"]
    ).sum()

    print("\nCOST EVENTS APPLIED")
    print("=" * 50)

    print(
        f"Rows              : "
        f"{len(result):,}"
    )

    print(
        f"SKUs with events  : "
        f"{cost_events['sku_id'].nunique():,}"
    )

    print(
        f"Cost events       : "
        f"{len(cost_events):,}"
    )

    print(
        f"Changed daily rows: "
        f"{changed_rows:,}"
    )

    print(
        f"Output            : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()