from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from src.app.filters import render_pricing_filters


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RECOMMENDATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_recommendations.parquet"
)


# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_recommendations() -> pd.DataFrame:

    if not RECOMMENDATIONS_PATH.exists():
        raise FileNotFoundError(
            "pricing_recommendations.parquet was not found. "
            "Run the pricing recommendation pipeline first."
        )

    return pd.read_parquet(
        RECOMMENDATIONS_PATH
    )


recommendations = load_recommendations()


# =========================================================
# FILTERS
# =========================================================

filtered, filters = render_pricing_filters(
    recommendations
)


# =========================================================
# PAGE HEADER
# =========================================================

st.title(
    "AU Retail Pricing & Margin Decision Lab"
)

st.caption(
    "Synthetic Australian retail pricing scenario | "
    "1,500 SKUs | Cost → Price → Elasticity → Margin → Decision"
)

st.header(
    "Executive Overview"
)

st.caption(
    "Where are the strongest pricing and margin opportunities "
    "across the portfolio?"
)


# =========================================================
# EMPTY FILTER CHECK
# =========================================================

if filtered.empty:

    st.warning(
        "No SKUs match the current filters."
    )

    st.stop()


# =========================================================
# PORTFOLIO OPPORTUNITY
# =========================================================

st.subheader(
    "Portfolio pricing opportunity"
)

sku_count = len(filtered)

incremental_margin = (
    filtered[
        "incremental_margin"
    ].sum()
)

incremental_sales = (
    filtered[
        "incremental_sales"
    ].sum()
)

incremental_units = (
    filtered[
        "incremental_units"
    ].sum()
)

current_margin = (
    filtered[
        "current_28d_margin"
    ].sum()
)

recommended_margin = (
    filtered[
        "recommended_28d_margin"
    ].sum()
)

margin_uplift_pct = (
    recommended_margin
    / current_margin
    - 1
    if current_margin != 0
    else 0
)


kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "SKUs",
    f"{sku_count:,}",
)

kpi2.metric(
    "28-Day Margin Opportunity",
    f"${incremental_margin:,.0f}",
)

kpi3.metric(
    "Margin Uplift",
    f"{margin_uplift_pct:.1%}",
)

kpi4.metric(
    "Sales Impact",
    f"${incremental_sales:,.0f}",
)

kpi5.metric(
    "Unit Impact",
    f"{incremental_units:,.0f}",
)


# =========================================================
# CURRENT VS RECOMMENDED POSITION
# =========================================================

st.subheader(
    "Current vs recommended position"
)

current_price_index = (
    filtered[
        "current_price_index"
    ].mean()
)

recommended_price_index = (
    filtered[
        "recommended_price_index"
    ].mean()
)

current_margin_pct = (
    filtered[
        "current_margin_pct"
    ].mean()
)

recommended_margin_pct = (
    filtered[
        "recommended_margin_pct"
    ].mean()
)


pos1, pos2, pos3, pos4 = st.columns(4)

pos1.metric(
    "Current Price Index",
    f"{current_price_index:.3f}",
)

pos2.metric(
    "Recommended Price Index",
    f"{recommended_price_index:.3f}",
    delta=(
        f"{recommended_price_index - current_price_index:+.3f}"
    ),
)

pos3.metric(
    "Current Margin %",
    f"{current_margin_pct:.1%}",
)

pos4.metric(
    "Recommended Margin %",
    f"{recommended_margin_pct:.1%}",
    delta=(
        f"{recommended_margin_pct - current_margin_pct:+.1%}"
    ),
)


# =========================================================
# ACTION MIX + COMPETITIVE POSITION
# =========================================================

st.subheader(
    "Portfolio decision mix"
)

action_col, position_col = st.columns(
    [1, 1]
)


# ---------------------------------------------------------
# Recommended action mix
# ---------------------------------------------------------

with action_col:

    st.markdown(
        "**Recommended action mix**"
    )

    action_order = [
        "Increase Price",
        "Hold Price",
        "Reduce Price",
        "Review",
    ]

    action_summary = (
        filtered[
            "recommended_action"
        ]
        .value_counts()
        .reindex(
            action_order,
            fill_value=0,
        )
        .rename_axis(
            "Recommended Action"
        )
        .reset_index(
            name="SKUs"
        )
    )

    action_chart = (
        alt.Chart(action_summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "Recommended Action:N",
                title=None,
                sort=[
                    "Increase Price",
                    "Hold Price",
                    "Reduce Price",
                    "Review",
                ],
                axis=alt.Axis(
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                "SKUs:Q",
                title=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "Recommended Action:N",
                    title="Action",
                ),
                alt.Tooltip(
                    "SKUs:Q",
                    title="SKUs",
                    format=",",
                ),
            ],
        )
        .properties(
            height=300
        )
    )

    st.altair_chart(
        action_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Competitive price position
# ---------------------------------------------------------

with position_col:

    st.markdown(
        "**Current competitive price position**"
    )

    price_position = pd.cut(
        filtered[
            "current_price_index"
        ],
        bins=[
            float("-inf"),
            0.95,
            1.00,
            1.05,
            1.10,
            float("inf"),
        ],
        labels=[
            "<0.95",
            "0.95–1.00",
            "1.00–1.05",
            "1.05–1.10",
            ">1.10",
        ],
    )

    position_summary = (
        price_position
        .value_counts()
        .sort_index()
        .rename_axis(
            "Price Position"
        )
        .reset_index(
            name="SKUs"
        )
    )

    position_chart = (
        alt.Chart(position_summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "Price Position:N",
                title=None,
                sort=[
                    "<0.95",
                    "0.95–1.00",
                    "1.00–1.05",
                    "1.05–1.10",
                    ">1.10",
                ],
                axis=alt.Axis(
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                "SKUs:Q",
                title=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "Price Position:N",
                    title="Price Position",
                ),
                alt.Tooltip(
                    "SKUs:Q",
                    title="SKUs",
                    format=",",
                ),
            ],
        )
        .properties(
            height=300
        )
    )

    st.altair_chart(
        position_chart,
        use_container_width=True,
    )


# =========================================================
# ACTION SUMMARY TABLE
# =========================================================

st.subheader(
    "Recommended actions"
)

action_table = (
    filtered.groupby(
        "recommended_action",
        as_index=False,
    )
    .agg(
        skus=(
            "sku_id",
            "count",
        ),
        incremental_sales=(
            "incremental_sales",
            "sum",
        ),
        incremental_margin=(
            "incremental_margin",
            "sum",
        ),
        avg_current_price_index=(
            "current_price_index",
            "mean",
        ),
        avg_recommended_price_index=(
            "recommended_price_index",
            "mean",
        ),
    )
)

action_table["share"] = (
    action_table["skus"]
    / action_table["skus"].sum()
)

action_table["action_order"] = (
    action_table[
        "recommended_action"
    ].map(
        {
            "Increase Price": 1,
            "Hold Price": 2,
            "Reduce Price": 3,
            "Review": 4,
        }
    )
)

action_table = (
    action_table
    .sort_values(
        "action_order"
    )
    .drop(
        columns=[
            "action_order",
        ]
    )
)

display_action_table = (
    action_table.copy()
)

display_action_table[
    "share"
] = (
    display_action_table[
        "share"
    ]
    .map(
        lambda x: f"{x:.1%}"
    )
)

display_action_table[
    "incremental_sales"
] = (
    display_action_table[
        "incremental_sales"
    ]
    .map(
        lambda x: f"${x:,.0f}"
    )
)

display_action_table[
    "incremental_margin"
] = (
    display_action_table[
        "incremental_margin"
    ]
    .map(
        lambda x: f"${x:,.0f}"
    )
)

display_action_table[
    "avg_current_price_index"
] = (
    display_action_table[
        "avg_current_price_index"
    ]
    .map(
        lambda x: f"{x:.3f}"
    )
)

display_action_table[
    "avg_recommended_price_index"
] = (
    display_action_table[
        "avg_recommended_price_index"
    ]
    .map(
        lambda x: f"{x:.3f}"
    )
)

display_action_table = (
    display_action_table.rename(
        columns={
            "recommended_action":
                "Recommended Action",
            "skus":
                "SKUs",
            "share":
                "Share",
            "incremental_sales":
                "Incremental Sales",
            "incremental_margin":
                "Incremental Margin",
            "avg_current_price_index":
                "Current Price Index",
            "avg_recommended_price_index":
                "Recommended Price Index",
        }
    )
)

st.dataframe(
    display_action_table,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# TOP MARGIN OPPORTUNITIES
# =========================================================

st.subheader(
    "Top margin opportunities"
)

opportunities = (
    filtered[
        filtered[
            "recommended_action"
        ]
        == "Increase Price"
    ]
    .sort_values(
        "incremental_margin",
        ascending=False,
    )
    .head(15)
    [
        [
            "sku_id",
            "department",
            "product_class",
            "recommended_price_change_pct",
            "current_sell_price",
            "recommended_sell_price",
            "current_price_index",
            "recommended_price_index",
            "incremental_sales",
            "incremental_margin",
            "decision_confidence",
        ]
    ]
    .copy()
)


if opportunities.empty:

    st.info(
        "No price-increase opportunities "
        "match the current filters."
    )

else:

    opportunities[
        "recommended_price_change_pct"
    ] = (
        opportunities[
            "recommended_price_change_pct"
        ]
        .map(
            lambda x: f"{x:+.1%}"
        )
    )

    opportunities[
        "current_sell_price"
    ] = (
        opportunities[
            "current_sell_price"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )

    opportunities[
        "recommended_sell_price"
    ] = (
        opportunities[
            "recommended_sell_price"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )

    opportunities[
        "current_price_index"
    ] = (
        opportunities[
            "current_price_index"
        ]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

    opportunities[
        "recommended_price_index"
    ] = (
        opportunities[
            "recommended_price_index"
        ]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

    opportunities[
        "incremental_sales"
    ] = (
        opportunities[
            "incremental_sales"
        ]
        .map(
            lambda x: f"${x:,.0f}"
        )
    )

    opportunities[
        "incremental_margin"
    ] = (
        opportunities[
            "incremental_margin"
        ]
        .map(
            lambda x: f"${x:,.0f}"
        )
    )

    opportunities = (
        opportunities.rename(
            columns={
                "sku_id":
                    "SKU",
                "department":
                    "Department",
                "product_class":
                    "Product Class",
                "recommended_price_change_pct":
                    "Recommended Change",
                "current_sell_price":
                    "Current Price",
                "recommended_sell_price":
                    "Recommended Price",
                "current_price_index":
                    "Current Index",
                "recommended_price_index":
                    "Recommended Index",
                "incremental_sales":
                    "Sales Impact",
                "incremental_margin":
                    "Margin Opportunity",
                "decision_confidence":
                    "Confidence",
            }
        )
    )

    st.dataframe(
        opportunities,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# COMPETITIVE EXPOSURE
# =========================================================

st.subheader(
    "Competitive exposure"
)

exposure = (
    filtered[
        filtered[
            "recommended_action"
        ].isin(
            [
                "Reduce Price",
                "Review",
            ]
        )
    ]
    .sort_values(
        "current_price_index",
        ascending=False,
    )
    .head(15)
    [
        [
            "sku_id",
            "department",
            "product_class",
            "recommended_action",
            "current_sell_price",
            "competitor_price",
            "current_price_index",
            "recommended_sell_price",
            "recommended_price_index",
            "incremental_units",
            "incremental_margin",
            "decision_confidence",
        ]
    ]
    .copy()
)


if exposure.empty:

    st.info(
        "No competitive exposure items "
        "match the current filters."
    )

else:

    exposure[
        "current_sell_price"
    ] = (
        exposure[
            "current_sell_price"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )

    exposure[
        "competitor_price"
    ] = (
        exposure[
            "competitor_price"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )

    exposure[
        "current_price_index"
    ] = (
        exposure[
            "current_price_index"
        ]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

    exposure[
        "recommended_sell_price"
    ] = (
        exposure[
            "recommended_sell_price"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )

    exposure[
        "recommended_price_index"
    ] = (
        exposure[
            "recommended_price_index"
        ]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

    exposure[
        "incremental_units"
    ] = (
        exposure[
            "incremental_units"
        ]
        .map(
            lambda x: f"{x:+,.0f}"
        )
    )

    exposure[
        "incremental_margin"
    ] = (
        exposure[
            "incremental_margin"
        ]
        .map(
            lambda x: f"${x:+,.0f}"
        )
    )

    exposure = (
        exposure.rename(
            columns={
                "sku_id":
                    "SKU",
                "department":
                    "Department",
                "product_class":
                    "Product Class",
                "recommended_action":
                    "Action",
                "current_sell_price":
                    "Current Price",
                "competitor_price":
                    "Competitor Price",
                "current_price_index":
                    "Current Index",
                "recommended_sell_price":
                    "Recommended Price",
                "recommended_price_index":
                    "Recommended Index",
                "incremental_units":
                    "Unit Impact",
                "incremental_margin":
                    "Margin Impact",
                "decision_confidence":
                    "Confidence",
            }
        )
    )

    st.dataframe(
        exposure,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Recommendations combine current cost and sell price, "
    "competitive position, calibrated price elasticity, "
    "expected demand response and commercial pricing guardrails. "
    "Commercial impacts represent a 28-day scenario horizon."
)