from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

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
    "Recommendation Queue"
)

st.caption(
    "Which pricing actions should be prioritised based on "
    "margin opportunity, competitive position and decision confidence?"
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
# DECISION SUMMARY
# =========================================================

st.subheader(
    "Decision summary"
)

queue_count = len(filtered)

increase_count = (
    filtered[
        "recommended_action"
    ]
    .eq("Increase Price")
    .sum()
)

reduce_count = (
    filtered[
        "recommended_action"
    ]
    .eq("Reduce Price")
    .sum()
)

hold_count = (
    filtered[
        "recommended_action"
    ]
    .eq("Hold Price")
    .sum()
)

review_count = (
    filtered[
        "recommended_action"
    ]
    .eq("Review")
    .sum()
)

incremental_margin = (
    filtered[
        "incremental_margin"
    ].sum()
)


kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

kpi1.metric(
    "SKUs in Queue",
    f"{queue_count:,}",
)

kpi2.metric(
    "Increase Price",
    f"{increase_count:,}",
)

kpi3.metric(
    "Reduce Price",
    f"{reduce_count:,}",
)

kpi4.metric(
    "Hold Price",
    f"{hold_count:,}",
)

kpi5.metric(
    "Review",
    f"{review_count:,}",
)

kpi6.metric(
    "28-Day Margin Impact",
    f"${incremental_margin:,.0f}",
)


# =========================================================
# PRIORITY OPPORTUNITIES
# =========================================================

st.subheader(
    "Priority opportunities"
)

margin_col, investment_col = st.columns(
    [1, 1]
)


# ---------------------------------------------------------
# Top margin opportunities
# ---------------------------------------------------------

with margin_col:

    st.markdown(
        "**Top margin opportunities**"
    )

    margin_opportunities = (
        filtered[
            filtered[
                "recommended_action"
            ]
            == "Increase Price"
        ]
        .nlargest(
            10,
            "incremental_margin",
        )
        [
            [
                "sku_id",
                "incremental_margin",
            ]
        ]
        .copy()
    )

    if margin_opportunities.empty:

        st.info(
            "No price-increase opportunities "
            "match the current filters."
        )

    else:

        margin_chart = (
            alt.Chart(
                margin_opportunities
            )
            .mark_bar()
            .encode(
                y=alt.Y(
                    "sku_id:N",
                    title=None,
                    sort="-x",
                    axis=alt.Axis(
                        labelLimit=120,
                    ),
                ),
                x=alt.X(
                    "incremental_margin:Q",
                    title="28-Day Incremental Margin",
                    axis=alt.Axis(
                        format="$,.0f",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "sku_id:N",
                        title="SKU",
                    ),
                    alt.Tooltip(
                        "incremental_margin:Q",
                        title="Margin Opportunity",
                        format="$,.0f",
                    ),
                ],
            )
            .properties(
                height=320
            )
        )

        st.altair_chart(
            margin_chart,
            use_container_width=True,
        )


# ---------------------------------------------------------
# Top price-investment opportunities
# ---------------------------------------------------------

with investment_col:

    st.markdown(
        "**Top price-investment opportunities**"
    )

    price_investment = (
        filtered[
            filtered[
                "recommended_action"
            ]
            == "Reduce Price"
        ]
        .nlargest(
            10,
            "current_price_index",
        )
        [
            [
                "sku_id",
                "current_price_index",
                "recommended_price_index",
            ]
        ]
        .copy()
    )

    if price_investment.empty:

        st.info(
            "No price-reduction opportunities "
            "match the current filters."
        )

    else:

        investment_long = (
            price_investment.melt(
                id_vars=[
                    "sku_id",
                ],
                value_vars=[
                    "current_price_index",
                    "recommended_price_index",
                ],
                var_name="position",
                value_name="price_index",
            )
        )

        investment_long[
            "position"
        ] = (
            investment_long[
                "position"
            ].map(
                {
                    "current_price_index":
                        "Current",
                    "recommended_price_index":
                        "Recommended",
                }
            )
        )

        investment_chart = (
            alt.Chart(
                investment_long
            )
            .mark_bar()
            .encode(
                y=alt.Y(
                    "sku_id:N",
                    title=None,
                    sort=alt.SortField(
                        field="price_index",
                        order="descending",
                    ),
                    axis=alt.Axis(
                        labelLimit=120,
                    ),
                ),
                x=alt.X(
                    "price_index:Q",
                    title="Price Index",
                    scale=alt.Scale(
                        zero=False,
                    ),
                ),
                color=alt.Color(
                    "position:N",
                    title=None,
                ),
                xOffset="position:N",
                tooltip=[
                    alt.Tooltip(
                        "sku_id:N",
                        title="SKU",
                    ),
                    alt.Tooltip(
                        "position:N",
                        title="Position",
                    ),
                    alt.Tooltip(
                        "price_index:Q",
                        title="Price Index",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=320
            )
        )

        st.altair_chart(
            investment_chart,
            use_container_width=True,
        )


# =========================================================
# COMMERCIAL IMPACT BY ACTION
# =========================================================

st.subheader(
    "Commercial impact by action"
)

impact_summary = (
    filtered.groupby(
        "recommended_action",
        as_index=False,
    )
    .agg(
        skus=(
            "sku_id",
            "count",
        ),
        incremental_units=(
            "incremental_units",
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
    )
)

action_order = [
    "Increase Price",
    "Hold Price",
    "Reduce Price",
    "Review",
]

impact_summary[
    "recommended_action"
] = pd.Categorical(
    impact_summary[
        "recommended_action"
    ],
    categories=action_order,
    ordered=True,
)

impact_summary = (
    impact_summary.sort_values(
        "recommended_action"
    )
)


display_impact = impact_summary.copy()

display_impact[
    "incremental_units"
] = (
    display_impact[
        "incremental_units"
    ]
    .map(
        lambda x: f"{x:+,.0f}"
    )
)

display_impact[
    "incremental_sales"
] = (
    display_impact[
        "incremental_sales"
    ]
    .map(
        lambda x: f"${x:+,.0f}"
    )
)

display_impact[
    "incremental_margin"
] = (
    display_impact[
        "incremental_margin"
    ]
    .map(
        lambda x: f"${x:+,.0f}"
    )
)

display_impact = (
    display_impact.rename(
        columns={
            "recommended_action":
                "Recommended Action",
            "skus":
                "SKUs",
            "incremental_units":
                "Unit Impact",
            "incremental_sales":
                "Sales Impact",
            "incremental_margin":
                "Margin Impact",
        }
    )
)

st.dataframe(
    display_impact,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# RECOMMENDATION QUEUE
# =========================================================

st.subheader(
    "Prioritised recommendation queue"
)

st.caption(
    "Sorted by expected 28-day margin impact. "
    "Use the left-hand filters to narrow the queue."
)


queue = filtered[
    [
        "sku_id",
        "department",
        "category",
        "product_class",
        "recommended_action",
        "recommendation_reason",
        "current_sell_price",
        "recommended_sell_price",
        "recommended_price_change_pct",
        "competitor_price",
        "current_price_index",
        "recommended_price_index",
        "current_margin_pct",
        "recommended_margin_pct",
        "incremental_units",
        "incremental_sales",
        "incremental_margin",
        "calibrated_elasticity",
        "decision_confidence",
        "decision_source",
    ]
].copy()


# ---------------------------------------------------------
# Priority score
#
# Used only for queue ordering.
# It does not feed back into the recommendation engine.
# ---------------------------------------------------------

queue[
    "priority_score"
] = (
    queue[
        "incremental_margin"
    ].clip(lower=0)
    +
    (
        (
            queue[
                "current_price_index"
            ]
            - 1.0
        )
        .clip(lower=0)
        * 1000
    )
)


queue = (
    queue.sort_values(
        [
            "priority_score",
            "incremental_margin",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .drop(
        columns=[
            "priority_score",
        ]
    )
)


# =========================================================
# DISPLAY FORMATTING
# =========================================================

queue[
    "current_sell_price"
] = (
    queue[
        "current_sell_price"
    ]
    .map(
        lambda x: f"${x:,.2f}"
    )
)

queue[
    "recommended_sell_price"
] = (
    queue[
        "recommended_sell_price"
    ]
    .map(
        lambda x: f"${x:,.2f}"
    )
)

queue[
    "recommended_price_change_pct"
] = (
    queue[
        "recommended_price_change_pct"
    ]
    .map(
        lambda x: f"{x:+.1%}"
    )
)

queue[
    "competitor_price"
] = (
    queue[
        "competitor_price"
    ]
    .map(
        lambda x: f"${x:,.2f}"
    )
)

queue[
    "current_price_index"
] = (
    queue[
        "current_price_index"
    ]
    .map(
        lambda x: f"{x:.3f}"
    )
)

queue[
    "recommended_price_index"
] = (
    queue[
        "recommended_price_index"
    ]
    .map(
        lambda x: f"{x:.3f}"
    )
)

queue[
    "current_margin_pct"
] = (
    queue[
        "current_margin_pct"
    ]
    .map(
        lambda x: f"{x:.1%}"
    )
)

queue[
    "recommended_margin_pct"
] = (
    queue[
        "recommended_margin_pct"
    ]
    .map(
        lambda x: f"{x:.1%}"
    )
)

queue[
    "incremental_units"
] = (
    queue[
        "incremental_units"
    ]
    .map(
        lambda x: f"{x:+,.0f}"
    )
)

queue[
    "incremental_sales"
] = (
    queue[
        "incremental_sales"
    ]
    .map(
        lambda x: f"${x:+,.0f}"
    )
)

queue[
    "incremental_margin"
] = (
    queue[
        "incremental_margin"
    ]
    .map(
        lambda x: f"${x:+,.0f}"
    )
)

queue[
    "calibrated_elasticity"
] = (
    queue[
        "calibrated_elasticity"
    ]
    .map(
        lambda x: f"{x:.2f}"
    )
)


queue = queue.rename(
    columns={
        "sku_id":
            "SKU",
        "department":
            "Department",
        "category":
            "Category",
        "product_class":
            "Product Class",
        "recommended_action":
            "Action",
        "recommendation_reason":
            "Recommendation Reason",
        "current_sell_price":
            "Current Price",
        "recommended_sell_price":
            "Recommended Price",
        "recommended_price_change_pct":
            "Price Change",
        "competitor_price":
            "Competitor Price",
        "current_price_index":
            "Current Index",
        "recommended_price_index":
            "Recommended Index",
        "current_margin_pct":
            "Current Margin",
        "recommended_margin_pct":
            "Recommended Margin",
        "incremental_units":
            "Unit Impact",
        "incremental_sales":
            "Sales Impact",
        "incremental_margin":
            "Margin Impact",
        "calibrated_elasticity":
            "Elasticity",
        "decision_confidence":
            "Confidence",
        "decision_source":
            "Evidence Source",
    }
)


# =========================================================
# TABLE
# =========================================================

st.dataframe(
    queue,
    hide_index=True,
    use_container_width=True,
    height=620,
)


# =========================================================
# EXPORT
# =========================================================

export_data = filtered.copy()

csv_data = (
    export_data
    .to_csv(
        index=False
    )
    .encode("utf-8")
)

st.download_button(
    label="Download filtered recommendation queue",
    data=csv_data,
    file_name=(
        "pricing_recommendation_queue.csv"
    ),
    mime="text/csv",
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Recommendations are prioritised using expected commercial "
    "impact, competitive position, calibrated elasticity and "
    "decision confidence. Scenario impacts represent a 28-day horizon."
)