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

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "pricing_demand_history.parquet"
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
def load_history() -> pd.DataFrame:

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            "pricing_demand_history.parquet was not found."
        )

    columns = [
        "date",
        "sku_id",
        "cost_price",
        "regular_sell_price",
        "competitor_price",
        "price_index",
        "units_sold",
        "sales_dollars",
        "gross_margin_dollars",
    ]

    data = pd.read_parquet(
        HISTORY_PATH,
        columns=columns,
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    return data


recommendations = load_recommendations()
history = load_history()


# =========================================================
# FILTERS
# =========================================================

filtered, filters = render_pricing_filters(
    recommendations
)


if filtered.empty:

    st.warning(
        "No SKUs match the current filters."
    )

    st.stop()


# =========================================================
# SKU SELECTOR
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "History selection"
)

sku_options = (
    filtered["sku_id"]
    .dropna()
    .sort_values()
    .tolist()
)

selected_sku = st.sidebar.selectbox(
    "SKU",
    options=sku_options,
    index=0,
    key="margin_competition_sku",
)


# =========================================================
# SELECTED SKU DATA
# =========================================================

selected_rec = (
    filtered[
        filtered["sku_id"]
        == selected_sku
    ]
    .iloc[0]
)

sku_history = (
    history[
        history["sku_id"]
        == selected_sku
    ]
    .copy()
    .sort_values(
        "date"
    )
)


if sku_history.empty:

    st.warning(
        "No historical data is available for the selected SKU."
    )

    st.stop()


# =========================================================
# WEEKLY HISTORY
# =========================================================

sku_history[
    "week_start"
] = (
    sku_history["date"]
    - pd.to_timedelta(
        sku_history[
            "date"
        ].dt.weekday,
        unit="D",
    )
)


weekly = (
    sku_history.groupby(
        "week_start",
        as_index=False,
    )
    .agg(
        cost_price=(
            "cost_price",
            "mean",
        ),
        sell_price=(
            "regular_sell_price",
            "mean",
        ),
        competitor_price=(
            "competitor_price",
            "mean",
        ),
        price_index=(
            "price_index",
            "mean",
        ),
        units_sold=(
            "units_sold",
            "sum",
        ),
        sales_dollars=(
            "sales_dollars",
            "sum",
        ),
        gross_margin_dollars=(
            "gross_margin_dollars",
            "sum",
        ),
    )
)


weekly[
    "gross_margin_pct"
] = (
    weekly[
        "gross_margin_dollars"
    ]
    / weekly[
        "sales_dollars"
    ].replace(
        0,
        pd.NA,
    )
)


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


def build_watch_reason(
    row: pd.Series,
) -> str:

    reasons = []

    if (
        row[
            "current_margin_pct"
        ] < 0.20
    ):
        reasons.append(
            "Low Margin"
        )

    if (
        row[
            "current_price_index"
        ] > 1.10
    ):
        reasons.append(
            "Materially Above Competitor"
        )

    elif (
        row[
            "current_price_index"
        ] > 1.05
    ):
        reasons.append(
            "Above Competitor"
        )

    if (
        row[
            "recommended_action"
        ]
        == "Reduce Price"
    ):
        reasons.append(
            "Price Investment"
        )

    if (
        row[
            "recommended_action"
        ]
        == "Review"
    ):
        reasons.append(
            "Manual Review"
        )

    if not reasons:
        return "Watch"

    return " | ".join(
        reasons
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
    "Margin & Competition"
)

st.caption(
    "Where are cost movements, margin pressure and competitive "
    "price gaps creating pricing risk or opportunity?"
)


# =========================================================
# PORTFOLIO POSITION
# =========================================================

st.subheader(
    "Portfolio position"
)

avg_current_price = (
    filtered[
        "current_sell_price"
    ].mean()
)

avg_competitor_price = (
    filtered[
        "competitor_price"
    ].mean()
)

avg_price_index = (
    filtered[
        "current_price_index"
    ].mean()
)

avg_margin_pct = (
    filtered[
        "current_margin_pct"
    ].mean()
)

priced_above_competitor = (
    filtered[
        "current_price_index"
    ]
    .gt(1.0)
    .mean()
)

margin_opportunity = (
    filtered[
        "incremental_margin"
    ].sum()
)


kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = (
    st.columns(6)
)

kpi1.metric(
    "Avg Current Price",
    f"${avg_current_price:,.2f}",
)

kpi2.metric(
    "Avg Competitor Price",
    f"${avg_competitor_price:,.2f}",
)

kpi3.metric(
    "Avg Price Index",
    f"{avg_price_index:.3f}",
)

kpi4.metric(
    "Avg Margin %",
    f"{avg_margin_pct:.1%}",
)

kpi5.metric(
    "Priced Above Competitor",
    f"{priced_above_competitor:.1%}",
)

kpi6.metric(
    "28-Day Margin Opportunity",
    f"${margin_opportunity:,.0f}",
)

st.caption(
    "Portfolio-level views below reflect all SKUs matching the "
    "current left-hand filters, rather than only the selected SKU."
)


# =========================================================
# SELECTED SKU ANALYSIS
# =========================================================

st.subheader(
    "Selected SKU analysis"
)

st.caption(
    "The following views relate only to the SKU selected "
    "in the left-hand History selection."
)


# =========================================================
# SELECTED SKU
# =========================================================

st.subheader(
    "Selected SKU"
)

st.write(
    f"**{selected_sku}**  |  "
    f"{selected_rec['department']}  |  "
    f"{selected_rec['category']}  |  "
    f"{selected_rec['product_class']}"
)


context1, context2, context3, context4 = (
    st.columns(4)
)

context1.metric(
    "Current Price",
    f"${selected_rec['current_sell_price']:,.2f}",
)

context2.metric(
    "Competitor Price",
    f"${selected_rec['competitor_price']:,.2f}",
)

context3.metric(
    "Current Price Index",
    f"{selected_rec['current_price_index']:.3f}",
)

context4.metric(
    "Current Margin %",
    f"{selected_rec['current_margin_pct']:.1%}",
)


# =========================================================
# COST PASS-THROUGH
# =========================================================

st.subheader(
    "Cost pass-through"
)

latest_date = (
    sku_history[
        "date"
    ].max()
)

lookback_date = (
    latest_date
    - pd.DateOffset(
        years=1
    )
)

latest_row = (
    sku_history.iloc[-1]
)

lookback_candidates = (
    sku_history[
        sku_history[
            "date"
        ]
        <= lookback_date
    ]
)


if not lookback_candidates.empty:

    base_row = (
        lookback_candidates
        .iloc[-1]
    )

else:

    base_row = (
        sku_history.iloc[0]
    )


if (
    base_row[
        "cost_price"
    ] != 0
):

    cost_change_pct = (
        latest_row[
            "cost_price"
        ]
        / base_row[
            "cost_price"
        ]
        - 1
    )

else:

    cost_change_pct = 0


if (
    base_row[
        "regular_sell_price"
    ] != 0
):

    sell_change_pct = (
        latest_row[
            "regular_sell_price"
        ]
        / base_row[
            "regular_sell_price"
        ]
        - 1
    )

else:

    sell_change_pct = 0


pass_through_gap = (
    sell_change_pct
    - cost_change_pct
)


if (
    base_row[
        "regular_sell_price"
    ] != 0
):

    base_margin_pct = (
        (
            base_row[
                "regular_sell_price"
            ]
            - base_row[
                "cost_price"
            ]
        )
        / base_row[
            "regular_sell_price"
        ]
    )

else:

    base_margin_pct = 0


if (
    latest_row[
        "regular_sell_price"
    ] != 0
):

    latest_margin_pct = (
        (
            latest_row[
                "regular_sell_price"
            ]
            - latest_row[
                "cost_price"
            ]
        )
        / latest_row[
            "regular_sell_price"
        ]
    )

else:

    latest_margin_pct = 0


margin_change_ppt = (
    latest_margin_pct
    - base_margin_pct
)


pass1, pass2, pass3, pass4 = (
    st.columns(4)
)

pass1.metric(
    "12M Cost Change",
    f"{cost_change_pct:+.1%}",
)

pass2.metric(
    "12M Sell Price Change",
    f"{sell_change_pct:+.1%}",
)

pass3.metric(
    "Pass-Through Gap",
    f"{pass_through_gap:+.1%}",
    help=(
        "Sell price change minus cost change. "
        "Negative values indicate cost movement "
        "has not been fully passed through."
    ),
)

pass4.metric(
    "Margin Change",
    f"{margin_change_ppt * 100:+.1f} ppt",
)


# =========================================================
# COST / SELL / COMPETITOR HISTORY
# =========================================================

st.subheader(
    "Cost, sell price & competitor history"
)

price_history = (
    weekly[
        [
            "week_start",
            "cost_price",
            "sell_price",
            "competitor_price",
        ]
    ]
    .melt(
        id_vars=[
            "week_start",
        ],
        value_vars=[
            "cost_price",
            "sell_price",
            "competitor_price",
        ],
        var_name="series",
        value_name="price",
    )
)


price_history[
    "series"
] = (
    price_history[
        "series"
    ].map(
        {
            "cost_price":
                "Unit Cost",
            "sell_price":
                "Sell Price",
            "competitor_price":
                "Competitor Price",
        }
    )
)


price_chart = (
    alt.Chart(
        price_history
    )
    .mark_line()
    .encode(
        x=alt.X(
            "week_start:T",
            title=None,
        ),
        y=alt.Y(
            "price:Q",
            title="Price",
            axis=alt.Axis(
                format="$,.0f",
            ),
            scale=alt.Scale(
                zero=False,
            ),
        ),
        color=alt.Color(
            "series:N",
            title=None,
        ),
        tooltip=[
            alt.Tooltip(
                "week_start:T",
                title="Week",
            ),
            alt.Tooltip(
                "series:N",
                title="Series",
            ),
            alt.Tooltip(
                "price:Q",
                title="Price",
                format="$,.2f",
            ),
        ],
    )
    .properties(
        height=360
    )
)


st.altair_chart(
    price_chart,
    use_container_width=True,
)

st.caption(
    "Weekly average unit cost, sell price and competitor price "
    "for the selected SKU."
)


# =========================================================
# MARGIN + PRICE INDEX TREND
# =========================================================

st.subheader(
    "Margin and competitive position over time"
)

margin_col, index_col = (
    st.columns(
        [1, 1]
    )
)


with margin_col:

    st.markdown(
        "**Gross margin % trend**"
    )

    margin_chart = (
        alt.Chart(
            weekly
        )
        .mark_line()
        .encode(
            x=alt.X(
                "week_start:T",
                title=None,
            ),
            y=alt.Y(
                "gross_margin_pct:Q",
                title="Gross Margin %",
                axis=alt.Axis(
                    format=".1%",
                ),
                scale=alt.Scale(
                    zero=False,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "week_start:T",
                    title="Week",
                ),
                alt.Tooltip(
                    "gross_margin_pct:Q",
                    title="Gross Margin %",
                    format=".2%",
                ),
            ],
        )
        .properties(
            height=300
        )
    )

    st.altair_chart(
        margin_chart,
        use_container_width=True,
    )


with index_col:

    st.markdown(
        "**Price index trend**"
    )

    index_chart = (
        alt.Chart(
            weekly
        )
        .mark_line()
        .encode(
            x=alt.X(
                "week_start:T",
                title=None,
            ),
            y=alt.Y(
                "price_index:Q",
                title="Price Index",
                scale=alt.Scale(
                    zero=False,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "week_start:T",
                    title="Week",
                ),
                alt.Tooltip(
                    "price_index:Q",
                    title="Price Index",
                    format=".3f",
                ),
            ],
        )
        .properties(
            height=300
        )
    )

    parity_rule = (
        alt.Chart(
            pd.DataFrame(
                {
                    "parity": [
                        1.0
                    ]
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
            y="parity:Q"
        )
    )

    st.altair_chart(
        index_chart
        + parity_rule,
        use_container_width=True,
    )


# =========================================================
# PORTFOLIO ANALYSIS
# =========================================================

st.divider()

st.subheader(
    "Portfolio analysis"
)

st.caption(
    "Portfolio-level views below reflect all SKUs matching the "
    "current left-hand filters, rather than only the selected SKU."
)


# =========================================================
# PRICE INDEX x MARGIN SCATTER
# =========================================================

st.subheader(
    "Portfolio price position vs margin"
)

scatter_data = (
    filtered[
        [
            "sku_id",
            "department",
            "product_class",
            "current_price_index",
            "current_margin_pct",
            "incremental_margin",
            "recommended_action",
            "decision_confidence",
        ]
    ]
    .copy()
)


scatter_data[
    "margin_opportunity_size"
] = (
    scatter_data[
        "incremental_margin"
    ]
    .clip(
        lower=0
    )
    + 1
)


scatter = (
    alt.Chart(
        scatter_data
    )
    .mark_circle(
        opacity=0.65
    )
    .encode(
        x=alt.X(
            "current_price_index:Q",
            title="Current Price Index",
            scale=alt.Scale(
                zero=False,
            ),
        ),
        y=alt.Y(
            "current_margin_pct:Q",
            title="Current Margin %",
            axis=alt.Axis(
                format=".0%",
            ),
            scale=alt.Scale(
                zero=False,
            ),
        ),
        size=alt.Size(
            "margin_opportunity_size:Q",
            title="Margin Opportunity",
            legend=None,
        ),
        color=alt.Color(
            "recommended_action:N",
            title="Recommended Action",
        ),
        tooltip=[
            alt.Tooltip(
                "sku_id:N",
                title="SKU",
            ),
            alt.Tooltip(
                "department:N",
                title="Department",
            ),
            alt.Tooltip(
                "product_class:N",
                title="Product Class",
            ),
            alt.Tooltip(
                "current_price_index:Q",
                title="Price Index",
                format=".3f",
            ),
            alt.Tooltip(
                "current_margin_pct:Q",
                title="Margin %",
                format=".1%",
            ),
            alt.Tooltip(
                "incremental_margin:Q",
                title="28-Day Margin Opportunity",
                format="$,.0f",
            ),
            alt.Tooltip(
                "recommended_action:N",
                title="Action",
            ),
        ],
    )
    .properties(
        height=420
    )
)


parity_vertical = (
    alt.Chart(
        pd.DataFrame(
            {
                "x": [
                    1.0
                ]
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


st.altair_chart(
    scatter
    + parity_vertical,
    use_container_width=True,
)

st.caption(
    "Left of parity indicates pricing below the competitor; "
    "right of parity indicates pricing above the competitor."
)


# =========================================================
# MOST PRICE-SENSITIVE PRODUCT CLASSES
# =========================================================

st.subheader(
    "Most price-sensitive product classes"
)

elasticity_summary = (
    filtered.groupby(
        [
            "department",
            "product_class",
        ],
        as_index=False,
    )
    .agg(
        avg_elasticity=(
            "calibrated_elasticity",
            "mean",
        ),
        skus=(
            "sku_id",
            "count",
        ),
    )
    .sort_values(
        "avg_elasticity",
        ascending=True,
    )
    .head(10)
)


elasticity_chart = (
    alt.Chart(
        elasticity_summary
    )
    .mark_bar()
    .encode(
        y=alt.Y(
            "product_class:N",
            title=None,
            sort=alt.SortField(
                field="avg_elasticity",
                order="ascending",
            ),
            axis=alt.Axis(
                labelLimit=200,
            ),
        ),
        x=alt.X(
            "avg_elasticity:Q",
            title="Average Calibrated Elasticity",
        ),
        tooltip=[
            alt.Tooltip(
                "product_class:N",
                title="Product Class",
            ),
            alt.Tooltip(
                "department:N",
                title="Department",
            ),
            alt.Tooltip(
                "avg_elasticity:Q",
                title="Elasticity",
                format=".2f",
            ),
            alt.Tooltip(
                "skus:Q",
                title="SKUs",
                format=",",
            ),
        ],
    )
    .properties(
        height=340
    )
)


st.altair_chart(
    elasticity_chart,
    use_container_width=True,
)

st.caption(
    "More negative elasticity indicates stronger expected "
    "demand response to changes in sell price."
)


# =========================================================
# COMMERCIAL WATCHLIST
# =========================================================

st.subheader(
    "Commercial watchlist"
)

st.caption(
    "SKUs with margin pressure, competitive exposure or "
    "recommended price investment requiring attention."
)


watchlist = (
    filtered[
        (
            filtered[
                "current_price_index"
            ] > 1.05
        )
        |
        (
            filtered[
                "current_margin_pct"
            ] < 0.20
        )
        |
        (
            filtered[
                "recommended_action"
            ].isin(
                [
                    "Reduce Price",
                    "Review",
                ]
            )
        )
    ]
    .copy()
)


if not watchlist.empty:

    watchlist[
        "watch_reason"
    ] = (
        watchlist.apply(
            build_watch_reason,
            axis=1,
        )
    )

    watchlist[
        "watch_priority"
    ] = (
        (
            watchlist[
                "current_price_index"
            ]
            - 1
        )
        .clip(
            lower=0
        )
        * 100
        +
        (
            0.20
            - watchlist[
                "current_margin_pct"
            ]
        )
        .clip(
            lower=0
        )
        * 100
    )

    watchlist = (
        watchlist.sort_values(
            [
                "watch_priority",
                "current_price_index",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(30)
    )


if watchlist.empty:

    st.info(
        "No commercial watchlist items "
        "match the current filters."
    )

else:

    display_watchlist = (
        watchlist[
            [
                "sku_id",
                "department",
                "product_class",
                "recommended_action",
                "watch_reason",
                "current_sell_price",
                "competitor_price",
                "current_price_index",
                "current_margin_pct",
                "calibrated_elasticity",
                "incremental_margin",
                "decision_confidence",
            ]
        ]
        .copy()
    )


    display_watchlist[
        "current_sell_price"
    ] = (
        display_watchlist[
            "current_sell_price"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )


    display_watchlist[
        "competitor_price"
    ] = (
        display_watchlist[
            "competitor_price"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )


    display_watchlist[
        "current_price_index"
    ] = (
        display_watchlist[
            "current_price_index"
        ]
        .map(
            lambda x: f"{x:.3f}"
        )
    )


    display_watchlist[
        "current_margin_pct"
    ] = (
        display_watchlist[
            "current_margin_pct"
        ]
        .map(
            lambda x: f"{x:.1%}"
        )
    )


    display_watchlist[
        "calibrated_elasticity"
    ] = (
        display_watchlist[
            "calibrated_elasticity"
        ]
        .map(
            lambda x: f"{x:.2f}"
        )
    )


    display_watchlist[
        "incremental_margin"
    ] = (
        display_watchlist[
            "incremental_margin"
        ]
        .map(
            format_signed_currency
        )
    )


    display_watchlist = (
        display_watchlist.rename(
            columns={
                "sku_id":
                    "SKU",
                "department":
                    "Department",
                "product_class":
                    "Product Class",
                "recommended_action":
                    "Action",
                "watch_reason":
                    "Watch Reason",
                "current_sell_price":
                    "Current Price",
                "competitor_price":
                    "Competitor Price",
                "current_price_index":
                    "Price Index",
                "current_margin_pct":
                    "Margin %",
                "calibrated_elasticity":
                    "Elasticity",
                "incremental_margin":
                    "Margin Opportunity",
                "decision_confidence":
                    "Confidence",
            }
        )
    )


    st.dataframe(
        display_watchlist,
        hide_index=True,
        use_container_width=True,
        height=560,
    )


# =========================================================
# WATCHLIST DOWNLOAD
# =========================================================

watchlist_csv = watchlist.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download commercial watchlist",
    data=watchlist_csv,
    file_name="commercial_watchlist.csv",
    mime="text/csv",
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Historical charts use weekly aggregation of the synthetic "
    "two-year daily pricing history. Portfolio views use current "
    "pricing recommendations and calibrated elasticity estimates."
)