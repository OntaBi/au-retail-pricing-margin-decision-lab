from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_modelling_dataset.parquet"
)

ELIGIBILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_eligibility.parquet"
)


def main() -> None:

    weekly = pd.read_parquet(DATA_PATH)

    eligibility = pd.read_parquet(
        ELIGIBILITY_PATH
    )

    eligible_skus = eligibility.loc[
        eligibility["evidence_tier"].isin(
            ["Strong", "Moderate"]
        ),
        "sku_id",
    ]

    weekly = weekly[
        weekly["sku_id"].isin(
            eligible_skus
        )
    ].copy()

    # ---------------------------------------------------------
    # Identify weeks containing our own price changes
    # ---------------------------------------------------------

    events = weekly[
        weekly["price_changed"]
    ][
        [
            "sku_id",
            "week_start",
            "avg_sell_price",
        ]
    ].copy()

    events = events.sort_values(
        ["sku_id", "week_start"]
    )

    # ---------------------------------------------------------
    # Previous / next event
    # ---------------------------------------------------------

    events["previous_event"] = (
        events.groupby("sku_id")[
            "week_start"
        ].shift(1)
    )

    events["next_event"] = (
        events.groupby("sku_id")[
            "week_start"
        ].shift(-1)
    )

    events["weeks_since_previous"] = (
        (
            events["week_start"]
            - events["previous_event"]
        ).dt.days
        / 7
    )

    events["weeks_until_next"] = (
        (
            events["next_event"]
            - events["week_start"]
        ).dt.days
        / 7
    )

    # ---------------------------------------------------------
    # Determine clean event windows
    #
    # A clean 4-week event needs:
    # 4 weeks before + event week + 4 weeks after
    # without another price event contaminating the window.
    # ---------------------------------------------------------

    events["clean_2_week"] = (
        (
            events["weeks_since_previous"].isna()
            | (
                events["weeks_since_previous"]
                > 2
            )
        )
        &
        (
            events["weeks_until_next"].isna()
            | (
                events["weeks_until_next"]
                > 2
            )
        )
    )

    events["clean_4_week"] = (
        (
            events["weeks_since_previous"].isna()
            | (
                events["weeks_since_previous"]
                > 4
            )
        )
        &
        (
            events["weeks_until_next"].isna()
            | (
                events["weeks_until_next"]
                > 4
            )
        )
    )

    events["clean_8_week"] = (
        (
            events["weeks_since_previous"].isna()
            | (
                events["weeks_since_previous"]
                > 8
            )
        )
        &
        (
            events["weeks_until_next"].isna()
            | (
                events["weeks_until_next"]
                > 8
            )
        )
    )

    print("\nPRICE EVENT SPACING")
    print("=" * 60)

    print(
        f"Eligible SKUs         : "
        f"{weekly['sku_id'].nunique():,}"
    )

    print(
        f"Price events          : "
        f"{len(events):,}"
    )

    print(
        f"SKUs with events      : "
        f"{events['sku_id'].nunique():,}"
    )

    print("\nEvent Spacing:")

    spacing = pd.concat(
        [
            events["weeks_since_previous"],
            events["weeks_until_next"],
        ]
    ).dropna()

    print(
        spacing.describe().round(2)
    )

    print("\nClean Event Windows:")

    for window in [2, 4, 8]:

        column = f"clean_{window}_week"

        count = int(
            events[column].sum()
        )

        pct = (
            events[column].mean()
            * 100
        )

        skus = events.loc[
            events[column],
            "sku_id",
        ].nunique()

        print(
            f"{window}-week window : "
            f"{count:,} events "
            f"({pct:.1f}%) | "
            f"{skus:,} SKUs"
        )

    print("\nEvents per Eligible SKU:")

    events_per_sku = (
        events.groupby("sku_id")
        .size()
    )

    print(
        events_per_sku
        .describe()
        .round(2)
    )

    print("\nClean 4-week Events per SKU:")

    clean_events = (
        events[
            events["clean_4_week"]
        ]
        .groupby("sku_id")
        .size()
    )

    print(
        clean_events
        .describe()
        .round(2)
    )


if __name__ == "__main__":
    main()