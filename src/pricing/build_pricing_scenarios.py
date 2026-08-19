from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_demand_history.parquet"
)

ELASTICITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "calibrated_decision_elasticity.parquet"
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
    / "pricing_scenarios.parquet"
)


SCENARIO_PRICE_CHANGES = [
    -0.10,
    -0.075,
    -0.05,
    -0.025,
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
]


def build_current_state(
    history: pd.DataFrame,
    elasticity: pd.DataFrame,
    product_master: pd.DataFrame,
) -> pd.DataFrame:

    latest_date = history["date"].max()

    latest = history[
        history["date"] == latest_date
    ].copy()

    # ---------------------------------------------------------
    # Recent demand baseline
    #
    # Use trailing 28 days rather than only latest-day demand.
    # ---------------------------------------------------------

    recent_start = (
        latest_date
        - pd.Timedelta(days=27)
    )

    recent = history[
        history["date"].between(
            recent_start,
            latest_date,
        )
    ].copy()

    demand_baseline = (
        recent.groupby(
            "sku_id",
            as_index=False,
        )
        .agg(
            baseline_daily_units=(
                "units_sold",
                "mean",
            ),
            recent_28d_units=(
                "units_sold",
                "sum",
            ),
            recent_28d_sales=(
                "sales_dollars",
                "sum",
            ),
            recent_28d_margin=(
                "gross_margin_dollars",
                "sum",
            ),
        )
    )

    current = latest[
        [
            "sku_id",
            "cost_price",
            "regular_sell_price",
            "competitor_price",
            "price_index",
        ]
    ].rename(
        columns={
            "regular_sell_price":
                "current_sell_price",
            "price_index":
                "current_price_index",
        }
    )

    current = current.merge(
        demand_baseline,
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    elasticity_lookup = elasticity[
        [
            "sku_id",
            "calibrated_elasticity",
            "decision_confidence",
            "decision_source",
        ]
    ]

    current = current.merge(
        elasticity_lookup,
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    hierarchy = product_master[
        [
            "sku_id",
            "department",
            "category",
            "product_class",
            "price_band",
            "lifecycle_stage",
        ]
    ]

    current = current.merge(
        hierarchy,
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    current["current_unit_margin"] = (
        current["current_sell_price"]
        - current["cost_price"]
    )

    current["current_margin_pct"] = (
        current["current_unit_margin"]
        / current["current_sell_price"]
    )

    return current


def build_scenarios(
    current: pd.DataFrame,
) -> pd.DataFrame:

    scenario_frames = []

    for price_change_pct in SCENARIO_PRICE_CHANGES:

        scenario = current.copy()

        scenario["price_change_pct"] = (
            price_change_pct
        )

        scenario["scenario_sell_price"] = (
            scenario["current_sell_price"]
            * (
                1
                + scenario["price_change_pct"]
            )
        )

        # -----------------------------------------------------
        # Demand response
        #
        # elasticity =
        # % demand change / % price change
        # -----------------------------------------------------

        scenario[
            "expected_unit_change_pct"
        ] = (
            scenario[
                "calibrated_elasticity"
            ]
            * scenario[
                "price_change_pct"
            ]
        )

        scenario[
            "expected_daily_units"
        ] = (
            scenario[
                "baseline_daily_units"
            ]
            * (
                1
                + scenario[
                    "expected_unit_change_pct"
                ]
            )
        )

        scenario[
            "expected_daily_units"
        ] = (
            scenario[
                "expected_daily_units"
            ]
            .clip(lower=0)
        )

        # -----------------------------------------------------
        # Competitive position
        # -----------------------------------------------------

        scenario[
            "scenario_price_index"
        ] = (
            scenario[
                "scenario_sell_price"
            ]
            / scenario[
                "competitor_price"
            ]
        )

        scenario[
            "scenario_price_gap_pct"
        ] = (
            scenario[
                "scenario_price_index"
            ]
            - 1
        )

        # -----------------------------------------------------
        # Unit economics
        # -----------------------------------------------------

        scenario[
            "scenario_unit_margin"
        ] = (
            scenario[
                "scenario_sell_price"
            ]
            - scenario[
                "cost_price"
            ]
        )

        scenario[
            "scenario_margin_pct"
        ] = (
            scenario[
                "scenario_unit_margin"
            ]
            / scenario[
                "scenario_sell_price"
            ]
        )

        # -----------------------------------------------------
        # 28-day commercial outcome
        # -----------------------------------------------------

        scenario[
            "scenario_28d_units"
        ] = (
            scenario[
                "expected_daily_units"
            ]
            * 28
        )

        scenario[
            "scenario_28d_sales"
        ] = (
            scenario[
                "scenario_28d_units"
            ]
            * scenario[
                "scenario_sell_price"
            ]
        )

        scenario[
            "scenario_28d_margin"
        ] = (
            scenario[
                "scenario_28d_units"
            ]
            * scenario[
                "scenario_unit_margin"
            ]
        )

        # -----------------------------------------------------
        # Incremental impact vs current recent baseline
        # -----------------------------------------------------

        scenario[
            "incremental_units"
        ] = (
            scenario[
                "scenario_28d_units"
            ]
            - scenario[
                "recent_28d_units"
            ]
        )

        scenario[
            "incremental_sales"
        ] = (
            scenario[
                "scenario_28d_sales"
            ]
            - scenario[
                "recent_28d_sales"
            ]
        )

        scenario[
            "incremental_margin"
        ] = (
            scenario[
                "scenario_28d_margin"
            ]
            - scenario[
                "recent_28d_margin"
            ]
        )

        scenario_frames.append(
            scenario
        )

    result = pd.concat(
        scenario_frames,
        ignore_index=True,
    )

    result["scenario_name"] = (
        result["price_change_pct"]
        .map(
            lambda x:
                "Current"
                if x == 0
                else f"{x:+.1%}"
        )
    )

    return result


def validate_result(
    scenarios: pd.DataFrame,
) -> None:

    expected_rows = (
        1500
        * len(
            SCENARIO_PRICE_CHANGES
        )
    )

    assert len(scenarios) == expected_rows

    assert (
        scenarios["sku_id"].nunique()
        == 1500
    )

    assert (
        scenarios[
            "scenario_sell_price"
        ] > 0
    ).all()

    assert (
        scenarios[
            "expected_daily_units"
        ] >= 0
    ).all()

    assert (
        scenarios[
            "scenario_price_index"
        ] > 0
    ).all()

    assert (
        scenarios[
            "calibrated_elasticity"
        ] < 0
    ).all()

    assert (
        scenarios.isna()
        .sum()
        .sum()
        == 0
    )

    print(
        "\nValidation passed."
    )


def main() -> None:

    history = pd.read_parquet(
        HISTORY_PATH
    )

    elasticity = pd.read_parquet(
        ELASTICITY_PATH
    )

    product_master = pd.read_parquet(
        PRODUCT_MASTER_PATH
    )

    current = build_current_state(
        history,
        elasticity,
        product_master,
    )

    scenarios = build_scenarios(
        current
    )

    validate_result(
        scenarios
    )

    scenarios.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nPRICING SCENARIO FOUNDATION"
    )
    print("=" * 70)

    print(
        f"SKUs                   : "
        f"{scenarios['sku_id'].nunique():,}"
    )

    print(
        f"Scenarios per SKU      : "
        f"{len(SCENARIO_PRICE_CHANGES)}"
    )

    print(
        f"Rows                   : "
        f"{len(scenarios):,}"
    )

    print(
        "\nPortfolio Scenario Summary:"
    )

    portfolio = (
        scenarios.groupby(
            "scenario_name",
            as_index=False,
        )
        .agg(
            price_change_pct=(
                "price_change_pct",
                "first",
            ),
            expected_units=(
                "scenario_28d_units",
                "sum",
            ),
            sales=(
                "scenario_28d_sales",
                "sum",
            ),
            margin=(
                "scenario_28d_margin",
                "sum",
            ),
            incremental_sales=(
                "incremental_sales",
                "sum",
            ),
            incremental_margin=(
                "incremental_margin",
                "sum",
            ),
            avg_price_index=(
                "scenario_price_index",
                "mean",
            ),
        )
        .sort_values(
            "price_change_pct"
        )
    )

    print(
        portfolio.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print(
        "\nExample SKU:"
    )

    example_sku = (
        scenarios[
            "sku_id"
        ].iloc[0]
    )

    example = scenarios[
        scenarios["sku_id"]
        == example_sku
    ][
        [
            "sku_id",
            "scenario_name",
            "price_change_pct",
            "current_sell_price",
            "scenario_sell_price",
            "calibrated_elasticity",
            "expected_unit_change_pct",
            "scenario_price_index",
            "scenario_margin_pct",
            "scenario_28d_units",
            "scenario_28d_sales",
            "scenario_28d_margin",
            "incremental_margin",
        ]
    ].sort_values(
        "price_change_pct"
    )

    print(
        example.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        f"\nOutput                 : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()