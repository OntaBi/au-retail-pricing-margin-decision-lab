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

SCENARIOS_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_scenarios.parquet"
)


# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_recommendations() -> pd.DataFrame:

    if not RECOMMENDATIONS_PATH.exists():
        raise FileNotFoundError(
            "pricing_recommendations.parquet was not found."
        )

    return pd.read_parquet(
        RECOMMENDATIONS_PATH
    )


@st.cache_data
def load_scenarios() -> pd.DataFrame:

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            "pricing_scenarios.parquet was not found."
        )

    return pd.read_parquet(
        SCENARIOS_PATH
    )


recommendations = load_recommendations()
scenarios = load_scenarios()


# =========================================================
# FILTERS
# =========================================================

filtered_recommendations, filters = (
    render_pricing_filters(
        recommendations
    )
)


if filtered_recommendations.empty:

    st.warning(
        "No SKUs match the current filters."
    )

    st.stop()


# =========================================================
# SKU SELECTOR
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Scenario selection"
)

sku_options = (
    filtered_recommendations[
        "sku_id"
    ]
    .dropna()
    .sort_values()
    .tolist()
)

selected_sku = st.sidebar.selectbox(
    "SKU",
    options=sku_options,
    index=0,
    key="scenario_sku",
)


# =========================================================
# SELECTED SKU DATA
# =========================================================

recommendation = (
    filtered_recommendations[
        filtered_recommendations[
            "sku_id"
        ]
        == selected_sku
    ]
    .iloc[0]
)

sku_scenarios = (
    scenarios[
        scenarios["sku_id"]
        == selected_sku
    ]
    .copy()
    .sort_values(
        "price_change_pct"
    )
)


if sku_scenarios.empty:

    st.warning(
        "No scenario data is available for the selected SKU."
    )

    st.stop()


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
    "Scenario Explorer"
)

st.caption(
    "How do alternative price points change expected demand, "
    "sales, margin and competitive position?"
)


# =========================================================
# SKU CONTEXT
# =========================================================

st.subheader(
    "Selected SKU"
)

st.write(
    f"**{selected_sku}**  |  "
    f"{recommendation['department']}  |  "
    f"{recommendation['category']}  |  "
    f"{recommendation['product_class']}"
)


# =========================================================
# CURRENT PRODUCT POSITION
# =========================================================

st.subheader(
    "Current product position"
)

current_price = (
    recommendation[
        "current_sell_price"
    ]
)

competitor_price = (
    recommendation[
        "competitor_price"
    ]
)

current_price_index = (
    recommendation[
        "current_price_index"
    ]
)

current_margin_pct = (
    recommendation[
        "current_margin_pct"
    ]
)

elasticity = (
    recommendation[
        "calibrated_elasticity"
    ]
)

confidence = (
    recommendation[
        "decision_confidence"
    ]
)

current_scenario = (
    sku_scenarios[
        sku_scenarios[
            "price_change_pct"
        ]
        == 0
    ]
    .iloc[0]
)

cost_price = (
    current_scenario[
        "cost_price"
    ]
)


kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = (
    st.columns(6)
)

kpi1.metric(
    "Current Price",
    f"${current_price:,.2f}",
)

kpi2.metric(
    "Unit Cost",
    f"${cost_price:,.2f}",
)

kpi3.metric(
    "Competitor Price",
    f"${competitor_price:,.2f}",
)

kpi4.metric(
    "Price Index",
    f"{current_price_index:.3f}",
)

kpi5.metric(
    "Margin %",
    f"{current_margin_pct:.1%}",
)

kpi6.metric(
    "Elasticity",
    f"{elasticity:.2f}",
    help=(
        f"Decision confidence: {confidence}"
    ),
)


# =========================================================
# RECOMMENDED ACTION
# =========================================================

st.subheader(
    "Recommended action"
)

action_col, reason_col = st.columns(
    [1, 3]
)

with action_col:

    st.metric(
        "Action",
        recommendation[
            "recommended_action"
        ],
    )

    st.metric(
        "Recommended Price",
        f"${recommendation['recommended_sell_price']:,.2f}",
        delta=(
            f"{recommendation['recommended_price_change_pct']:+.1%}"
        ),
    )

with reason_col:

    st.markdown(
        "**Recommendation rationale**"
    )

    st.write(
        recommendation[
            "recommendation_reason"
        ]
    )

    st.caption(
        f"Decision confidence: "
        f"{recommendation['decision_confidence']} | "
        f"Evidence source: "
        f"{recommendation['decision_source']}"
    )


# =========================================================
# SCENARIO CURVES
# =========================================================

st.subheader(
    "Price-response scenarios"
)

response_col, margin_col = st.columns(
    [1, 1]
)


# ---------------------------------------------------------
# PRICE RESPONSE / ELASTICITY CURVE
# ---------------------------------------------------------

with response_col:

    st.markdown(
        "**Expected demand response**"
    )

    response_data = (
        sku_scenarios[
            [
                "price_change_pct",
                "expected_unit_change_pct",
            ]
        ]
        .copy()
    )

    response_data[
        "price_change_display"
    ] = (
        response_data[
            "price_change_pct"
        ]
        * 100
    )

    response_data[
        "unit_change_display"
    ] = (
        response_data[
            "expected_unit_change_pct"
        ]
        * 100
    )

    response_chart = (
        alt.Chart(
            response_data
        )
        .mark_line(
            point=True
        )
        .encode(
            x=alt.X(
                "price_change_display:Q",
                title="Price Change",
                axis=alt.Axis(
                    format="+.0f",
                    labelExpr="datum.value + '%'",
                ),
            ),
            y=alt.Y(
                "unit_change_display:Q",
                title="Expected Unit Change",
                axis=alt.Axis(
                    labelExpr="datum.value + '%'",
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "price_change_display:Q",
                    title="Price Change %",
                    format="+.1f",
                ),
                alt.Tooltip(
                    "unit_change_display:Q",
                    title="Expected Unit Change %",
                    format="+.1f",
                ),
            ],
        )
        .properties(
            height=340
        )
    )

    current_rule = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [0]
                }
            )
        )
        .mark_rule(
            strokeDash=[4, 4]
        )
        .encode(
            x="x:Q"
        )
    )

    st.altair_chart(
        response_chart
        + current_rule,
        use_container_width=True,
    )

    st.caption(
        f"Calibrated elasticity: {elasticity:.2f} | "
        f"Confidence: {confidence}"
    )


# ---------------------------------------------------------
# MARGIN CURVE
# ---------------------------------------------------------

with margin_col:

    st.markdown(
        "**28-day gross margin by price scenario**"
    )

    margin_data = (
        sku_scenarios[
            [
                "price_change_pct",
                "scenario_28d_margin",
            ]
        ]
        .copy()
    )

    margin_data[
        "price_change_display"
    ] = (
        margin_data[
            "price_change_pct"
        ]
        * 100
    )

    margin_chart = (
        alt.Chart(
            margin_data
        )
        .mark_line(
            point=True
        )
        .encode(
            x=alt.X(
                "price_change_display:Q",
                title="Price Change",
                axis=alt.Axis(
                    format="+.0f",
                    labelExpr="datum.value + '%'",
                ),
            ),
            y=alt.Y(
                "scenario_28d_margin:Q",
                title="28-Day Gross Margin",
                axis=alt.Axis(
                    format="$,.0f",
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "price_change_display:Q",
                    title="Price Change %",
                    format="+.1f",
                ),
                alt.Tooltip(
                    "scenario_28d_margin:Q",
                    title="28-Day Margin",
                    format="$,.0f",
                ),
            ],
        )
        .properties(
            height=340
        )
    )

    current_margin_rule = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [0]
                }
            )
        )
        .mark_rule(
            strokeDash=[4, 4]
        )
        .encode(
            x="x:Q"
        )
    )

    recommended_change = (
        recommendation[
            "recommended_price_change_pct"
        ]
        * 100
    )

    recommended_rule = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [
                        recommended_change
                    ]
                }
            )
        )
        .mark_rule()
        .encode(
            x="x:Q"
        )
    )

    st.altair_chart(
        margin_chart
        + current_margin_rule
        + recommended_rule,
        use_container_width=True,
    )

    st.caption(
        "Dashed reference = current price | "
        "Solid reference = recommended price"
    )


# =========================================================
# SCENARIO SUMMARY KPIs
# =========================================================

st.subheader(
    "Recommended scenario impact"
)

impact1, impact2, impact3, impact4 = (
    st.columns(4)
)

impact1.metric(
    "Recommended Price",
    f"${recommendation['recommended_sell_price']:,.2f}",
    delta=(
        f"{recommendation['recommended_price_change_pct']:+.1%}"
    ),
)

impact2.metric(
    "28-Day Unit Impact",
    f"{recommendation['incremental_units']:+,.0f}",
)

def format_signed_currency(value: float) -> str:
    if value > 0:
        return f"+${value:,.0f}"
    if value < 0:
        return f"-${abs(value):,.0f}"
    return "$0"


impact3.metric(
    "28-Day Sales Impact",
    format_signed_currency(
        recommendation["incremental_sales"]
    ),
)

impact4.metric(
    "28-Day Margin Impact",
    format_signed_currency(
        recommendation["incremental_margin"]
    ),
)


# =========================================================
# SCENARIO TABLE
# =========================================================

st.subheader(
    "Scenario comparison"
)

scenario_table = (
    sku_scenarios[
        [
            "price_change_pct",
            "scenario_sell_price",
            "expected_unit_change_pct",
            "scenario_28d_units",
            "scenario_28d_sales",
            "scenario_28d_margin",
            "scenario_margin_pct",
            "scenario_price_index",
        ]
    ]
    .copy()
)


def scenario_label(
    price_change: float,
) -> str:

    recommended_change = (
        recommendation[
            "recommended_price_change_pct"
        ]
    )

    if abs(
        price_change
        - recommended_change
    ) < 1e-9:

        if abs(
            price_change
        ) < 1e-9:
            return "Current / Recommended"

        return "Recommended"

    if abs(
        price_change
    ) < 1e-9:
        return "Current"

    return f"{price_change:+.1%}"


scenario_table[
    "Scenario"
] = (
    scenario_table[
        "price_change_pct"
    ]
    .apply(
        scenario_label
    )
)

scenario_table[
    "Price Change"
] = (
    scenario_table[
        "price_change_pct"
    ]
    .map(
        lambda x: f"{x:+.1%}"
    )
)

scenario_table[
    "Sell Price"
] = (
    scenario_table[
        "scenario_sell_price"
    ]
    .map(
        lambda x: f"${x:,.2f}"
    )
)

scenario_table[
    "Expected Unit Change"
] = (
    scenario_table[
        "expected_unit_change_pct"
    ]
    .map(
        lambda x: f"{x:+.1%}"
    )
)

scenario_table[
    "28-Day Units"
] = (
    scenario_table[
        "scenario_28d_units"
    ]
    .map(
        lambda x: f"{x:,.0f}"
    )
)

scenario_table[
    "28-Day Sales"
] = (
    scenario_table[
        "scenario_28d_sales"
    ]
    .map(
        lambda x: f"${x:,.0f}"
    )
)

scenario_table[
    "28-Day Margin"
] = (
    scenario_table[
        "scenario_28d_margin"
    ]
    .map(
        lambda x: f"${x:,.0f}"
    )
)

scenario_table[
    "Margin %"
] = (
    scenario_table[
        "scenario_margin_pct"
    ]
    .map(
        lambda x: f"{x:.1%}"
    )
)

scenario_table[
    "Price Index"
] = (
    scenario_table[
        "scenario_price_index"
    ]
    .map(
        lambda x: f"{x:.3f}"
    )
)


scenario_table = scenario_table[
    [
        "Scenario",
        "Price Change",
        "Sell Price",
        "Expected Unit Change",
        "28-Day Units",
        "28-Day Sales",
        "28-Day Margin",
        "Margin %",
        "Price Index",
    ]
]


st.dataframe(
    scenario_table,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# COMMERCIAL INTERPRETATION
# =========================================================

st.subheader(
    "Commercial interpretation"
)

if (
    recommendation[
        "recommended_action"
    ]
    == "Increase Price"
):

    st.write(
        "The recommended increase is expected to improve "
        "gross margin despite lower unit demand, while remaining "
        "within the configured competitive and demand-risk guardrails."
    )

elif (
    recommendation[
        "recommended_action"
    ]
    == "Reduce Price"
):

    st.write(
        "The SKU is currently priced materially above the competitor. "
        "The recommended price investment improves competitive position "
        "while keeping the expected margin trade-off within tolerance."
    )

elif (
    recommendation[
        "recommended_action"
    ]
    == "Review"
):

    st.write(
        "The SKU remains competitively exposed, but available price "
        "reductions do not currently meet the commercial margin guardrails. "
        "Further review is recommended."
    )

else:

    st.write(
        "No alternative scenario provides a sufficiently material "
        "commercial improvement within the current pricing guardrails."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Scenario outputs use calibrated price elasticity to estimate "
    "demand response across alternative sell prices. Commercial "
    "impacts represent a 28-day scenario horizon."
)