from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.app.filters import render_pricing_filters

from src.app.scenario_math import (
    calculate_custom_price_scenario,
)


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
# HELPERS
# =========================================================

def format_signed_currency(
    value: float,
) -> str:

    if value > 0:
        return f"+${value:,.0f}"

    if value < 0:
        return f"-${abs(value):,.0f}"

    return "$0"


def format_currency(
    value: float,
) -> str:

    return f"${value:,.0f}"


def format_price(
    value: float,
) -> str:

    return f"${value:,.2f}"


def format_signed_pct(
    value: float,
    decimals: int = 1,
) -> str:

    if abs(value) < 0.0000001:
        value = 0.0

    return f"{value:+.{decimals}%}"


def format_pct(
    value: float,
    decimals: int = 1,
) -> str:

    if abs(value) < 0.0000001:
        value = 0.0

    return f"{value:.{decimals}%}"


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


current_scenario = (
    sku_scenarios[
        sku_scenarios[
            "price_change_pct"
        ]
        == 0
    ]
    .iloc[0]
)


# =========================================================
# CORE SKU VALUES
# =========================================================

current_price = (
    recommendation[
        "current_sell_price"
    ]
)

recommended_price = (
    recommendation[
        "recommended_sell_price"
    ]
)

recommended_price_change = (
    recommendation[
        "recommended_price_change_pct"
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

cost_price = (
    current_scenario[
        "cost_price"
    ]
)

current_28d_units = (
    current_scenario[
        "scenario_28d_units"
    ]
)

current_28d_sales = (
    current_scenario[
        "scenario_28d_sales"
    ]
)

current_28d_margin = (
    current_scenario[
        "scenario_28d_margin"
    ]
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
        f"${recommended_price:,.2f}",
        delta=(
            f"{recommended_price_change:+.1%}"
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
# CUSTOM PRICE WHAT-IF
# =========================================================

st.subheader(
    "Custom Price What-If"
)

st.caption(
    "Test a commercial price different from the model recommendation "
    "and quantify the expected demand, competitive and margin trade-off."
)


whatif_input_col, whatif_context_col = (
    st.columns(
        [1, 2]
    )
)


with whatif_input_col:

    proposed_price = st.number_input(
        "Proposed Sell Price ($)",
        min_value=0.01,
        value=float(
            round(
                recommended_price,
                2,
            )
        ),
        step=0.01,
        format="%.2f",
        key=f"whatif_price_{selected_sku}",
    )


with whatif_context_col:

    st.markdown(
        "**Reference prices**"
    )

    ref1, ref2, ref3 = st.columns(3)

    ref1.metric(
        "Current",
        f"${current_price:,.2f}",
    )

    ref2.metric(
        "Model Recommendation",
        f"${recommended_price:,.2f}",
    )

    ref3.metric(
        "Competitor",
        f"${competitor_price:,.2f}",
    )


# =========================================================
# CUSTOM SCENARIO CALCULATION
# =========================================================

custom_scenario = (
    calculate_custom_price_scenario(
        current_price=current_price,
        proposed_price=proposed_price,
        competitor_price=competitor_price,
        cost_price=cost_price,
        elasticity=elasticity,
        current_28d_units=current_28d_units,
        current_28d_sales=current_28d_sales,
        current_28d_margin=current_28d_margin,
    )
)


custom_price_change_pct = (
    custom_scenario[
        "price_change_pct"
    ]
)

custom_expected_unit_change_pct = (
    custom_scenario[
        "expected_unit_change_pct"
    ]
)

custom_28d_units = (
    custom_scenario[
        "scenario_28d_units"
    ]
)

custom_28d_sales = (
    custom_scenario[
        "scenario_28d_sales"
    ]
)

custom_28d_margin = (
    custom_scenario[
        "scenario_28d_margin"
    ]
)

custom_margin_pct = (
    custom_scenario[
        "scenario_margin_pct"
    ]
)

custom_price_index = (
    custom_scenario[
        "scenario_price_index"
    ]
)

custom_price_gap_pct = (
    custom_price_index
    - 1
)

custom_incremental_units = (
    custom_scenario[
        "incremental_units"
    ]
)

custom_incremental_sales = (
    custom_scenario[
        "incremental_sales"
    ]
)

custom_incremental_margin = (
    custom_scenario[
        "incremental_margin"
    ]
)


# =========================================================
# MODEL RANGE / GUARDRAILS
# =========================================================

min_model_change = (
    sku_scenarios[
        "price_change_pct"
    ].min()
)

max_model_change = (
    sku_scenarios[
        "price_change_pct"
    ].max()
)


if (
    custom_price_change_pct
    < min_model_change
    or custom_price_change_pct
    > max_model_change
):

    st.warning(
        "Proposed price is outside the modelled scenario range "
        f"({min_model_change:+.0%} to {max_model_change:+.0%}). "
        "The demand response therefore represents extrapolation "
        "and should be interpreted with additional caution."
    )


if proposed_price <= cost_price:

    st.warning(
        "Proposed sell price is at or below unit cost and produces "
        "zero or negative unit margin."
    )


# =========================================================
# CUSTOM SCENARIO KPIs
# =========================================================

st.markdown(
    "**Your scenario**"
)

custom1, custom2, custom3, custom4, custom5 = (
    st.columns(5)
)

custom1.metric(
    "Price Change",
    format_signed_pct(
        custom_price_change_pct
    ),
)

custom2.metric(
    "vs Competitor",
    format_signed_pct(
        custom_price_gap_pct
    ),
    help=(
        "Positive values indicate the proposed price "
        "is above the competitor."
    ),
)

custom3.metric(
    "Expected Unit Impact",
    format_signed_pct(
        custom_expected_unit_change_pct
    ),
)

custom4.metric(
    "Margin %",
    format_pct(
        custom_margin_pct
    ),
)

custom5.metric(
    "28-Day Margin Impact",
    format_signed_currency(
        custom_incremental_margin
    ),
)


# =========================================================
# CURRENT VS RECOMMENDED VS CUSTOM
# =========================================================

st.markdown(
    "**Current vs model recommendation vs your scenario**"
)


recommended_28d_units = (
    recommendation[
        "recommended_28d_units"
    ]
)

recommended_28d_sales = (
    recommendation[
        "recommended_28d_sales"
    ]
)

recommended_28d_margin = (
    recommendation[
        "recommended_28d_margin"
    ]
)

recommended_margin_pct = (
    recommendation[
        "recommended_margin_pct"
    ]
)

recommended_price_index = (
    recommendation[
        "recommended_price_index"
    ]
)

recommended_price_gap_pct = (
    recommended_price_index
    - 1
)


comparison_table = pd.DataFrame(
    {
        "Metric": [
            "Sell Price",
            "vs Competitor",
            "28-Day Units",
            "28-Day Sales",
            "Margin %",
            "28-Day Margin",
            "Margin vs Current",
        ],
        "Current": [
            format_price(
                current_price
            ),
            format_signed_pct(
                current_price_index
                - 1
            ),
            f"{current_28d_units:,.0f}",
            format_currency(
                current_28d_sales
            ),
            format_pct(
                current_margin_pct
            ),
            format_currency(
                current_28d_margin
            ),
            "—",
        ],
        "Model Recommendation": [
            format_price(
                recommended_price
            ),
            format_signed_pct(
                recommended_price_gap_pct
            ),
            f"{recommended_28d_units:,.0f}",
            format_currency(
                recommended_28d_sales
            ),
            format_pct(
                recommended_margin_pct
            ),
            format_currency(
                recommended_28d_margin
            ),
            format_signed_currency(
                recommendation[
                    "incremental_margin"
                ]
            ),
        ],
        f"Your Scenario ({proposed_price:,.2f})": [
            format_price(
                proposed_price
            ),
            format_signed_pct(
                custom_price_gap_pct
            ),
            f"{custom_28d_units:,.0f}",
            format_currency(
                custom_28d_sales
            ),
            format_pct(
                custom_margin_pct
            ),
            format_currency(
                custom_28d_margin
            ),
            format_signed_currency(
                custom_incremental_margin
            ),
        ],
    }
)


st.dataframe(
    comparison_table,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# CUSTOM COMMERCIAL INTERPRETATION
# =========================================================

if custom_price_index > 1.10:

    st.info(
        f"Your proposed price remains "
        f"{custom_price_gap_pct:.1%} above the competitor. "
        "Competitive exposure is therefore relatively high."
    )

elif custom_price_index > 1.05:

    st.info(
        f"Your proposed price is "
        f"{custom_price_gap_pct:.1%} above the competitor. "
        "The scenario improves or protects economics, but "
        "competitive positioning should remain under review."
    )

elif custom_price_index >= 0.95:

    st.success(
        "Your proposed price remains broadly within the "
        "competitive parity range."
    )

else:

    st.info(
        f"Your proposed price is "
        f"{abs(custom_price_gap_pct):.1%} below the competitor, "
        "indicating potential price headroom if margin economics allow."
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
            strokeDash=[
                4,
                4,
            ]
        )
        .encode(
            x="x:Q"
        )
    )

    custom_response_rule = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [
                        custom_price_change_pct
                        * 100
                    ]
                }
            )
        )
        .mark_rule(
            strokeDash=[
                2,
                2,
            ]
        )
        .encode(
            x="x:Q"
        )
    )

    st.altair_chart(
        response_chart
        + current_rule
        + custom_response_rule,
        use_container_width=True,
    )

    st.caption(
        f"Calibrated elasticity: {elasticity:.2f} | "
        f"Confidence: {confidence} | "
        "Dashed = current | Dotted = your scenario"
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
            strokeDash=[
                4,
                4,
            ]
        )
        .encode(
            x="x:Q"
        )
    )

    recommended_change = (
        recommended_price_change
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

    custom_margin_rule = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [
                        custom_price_change_pct
                        * 100
                    ]
                }
            )
        )
        .mark_rule(
            strokeDash=[
                2,
                2,
            ]
        )
        .encode(
            x="x:Q"
        )
    )

    st.altair_chart(
        margin_chart
        + current_margin_rule
        + recommended_rule
        + custom_margin_rule,
        use_container_width=True,
    )

    st.caption(
        "Dashed = current | Solid = model recommendation | "
        "Dotted = your scenario"
    )


# =========================================================
# RECOMMENDED SCENARIO IMPACT
# =========================================================

st.subheader(
    "Recommended scenario impact"
)

impact1, impact2, impact3, impact4 = (
    st.columns(4)
)

impact1.metric(
    "Recommended Price",
    f"${recommended_price:,.2f}",
    delta=(
        f"{recommended_price_change:+.1%}"
    ),
)

impact2.metric(
    "28-Day Unit Impact",
    f"{recommendation['incremental_units']:+,.0f}",
)

impact3.metric(
    "28-Day Sales Impact",
    format_signed_currency(
        recommendation[
            "incremental_sales"
        ]
    ),
)

impact4.metric(
    "28-Day Margin Impact",
    format_signed_currency(
        recommendation[
            "incremental_margin"
        ]
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

    if abs(
        price_change
        - recommended_price_change
    ) < 1e-9:

        if abs(
            price_change
        ) < 1e-9:

            return (
                "Current / Recommended"
            )

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
        format_signed_pct
    )
)

scenario_table[
    "Sell Price"
] = (
    scenario_table[
        "scenario_sell_price"
    ]
    .map(
        format_price
    )
)

scenario_table[
    "Expected Unit Change"
] = (
    scenario_table[
        "expected_unit_change_pct"
    ]
    .map(
        format_signed_pct
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
        format_currency
    )
)

scenario_table[
    "28-Day Margin"
] = (
    scenario_table[
        "scenario_28d_margin"
    ]
    .map(
        format_currency
    )
)

scenario_table[
    "Margin %"
] = (
    scenario_table[
        "scenario_margin_pct"
    ]
    .map(
        format_pct
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
        "The model recommendation increases price because the "
        "expected gross-margin improvement remains material after "
        "allowing for lower unit demand, while competitive and "
        "demand-risk guardrails remain acceptable."
    )

elif (
    recommendation[
        "recommended_action"
    ]
    == "Reduce Price"
):

    st.write(
        "The SKU is currently priced materially above the competitor. "
        "The model recommendation invests in price to improve competitive "
        "position while keeping the expected margin trade-off within "
        "configured commercial tolerance."
    )

elif (
    recommendation[
        "recommended_action"
    ]
    == "Review"
):

    st.write(
        "The SKU remains competitively exposed, but the modelled price "
        "reductions do not currently meet the commercial margin guardrails. "
        "Further pricing review is recommended."
    )

else:

    st.write(
        "No alternative modelled scenario provides a sufficiently "
        "material commercial improvement within the current pricing "
        "guardrails."
    )

st.caption(
    "The Custom Price What-If allows a commercial user to challenge "
    "the model recommendation and quantify the expected trade-off "
    "without changing the underlying recommendation engine."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Scenario outputs use calibrated price elasticity to estimate "
    "demand response across alternative sell prices. Commercial "
    "impacts represent a 28-day scenario horizon. Custom prices "
    "outside the modelled scenario range represent extrapolation "
    "and should be interpreted cautiously."
)