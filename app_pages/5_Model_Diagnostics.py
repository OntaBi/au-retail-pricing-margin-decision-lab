from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

ELIGIBILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "elasticity_eligibility.parquet"
)

SKU_ESTIMATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "sku_elasticity_estimates.parquet"
)

CLASS_ESTIMATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "class_elasticity_estimates.parquet"
)

DECISION_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "decision_elasticity.parquet"
)

CALIBRATED_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "calibrated_decision_elasticity.parquet"
)


# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():

    eligibility = pd.read_parquet(
        ELIGIBILITY_PATH
    )

    sku_estimates = pd.read_parquet(
        SKU_ESTIMATES_PATH
    )

    class_estimates = pd.read_parquet(
        CLASS_ESTIMATES_PATH
    )

    decision = pd.read_parquet(
        DECISION_PATH
    )

    calibrated = pd.read_parquet(
        CALIBRATED_PATH
    )

    return (
        eligibility,
        sku_estimates,
        class_estimates,
        decision,
        calibrated,
    )


(
    eligibility,
    sku_estimates,
    class_estimates,
    decision,
    calibrated,
) = load_data()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header(
    "Filters"
)


department_options = [
    "All",
    *sorted(
        calibrated[
            "department"
        ]
        .dropna()
        .unique()
        .tolist()
    ),
]

selected_department = (
    st.sidebar.selectbox(
        "Department",
        options=department_options,
        index=0,
        key="diagnostics_department",
    )
)


if (
    selected_department
    == "All"
):

    diagnostics_base = (
        calibrated.copy()
    )

else:

    diagnostics_base = (
        calibrated[
            calibrated[
                "department"
            ]
            == selected_department
        ]
        .copy()
    )


product_class_options = [
    "All",
    *sorted(
        diagnostics_base[
            "product_class"
        ]
        .dropna()
        .unique()
        .tolist()
    ),
]

selected_product_class = (
    st.sidebar.selectbox(
        "Product class",
        options=product_class_options,
        index=0,
        key="diagnostics_product_class",
    )
)


if (
    selected_product_class
    != "All"
):

    diagnostics_base = (
        diagnostics_base[
            diagnostics_base[
                "product_class"
            ]
            == selected_product_class
        ]
        .copy()
    )


confidence_order = [
    "High",
    "Medium",
    "Low",
]

confidence_available = (
    diagnostics_base[
        "decision_confidence"
    ]
    .dropna()
    .unique()
    .tolist()
)

confidence_options = [
    "All",
    *[
        x
        for x
        in confidence_order
        if x in confidence_available
    ],
]

selected_confidence = (
    st.sidebar.selectbox(
        "Decision confidence",
        options=confidence_options,
        index=0,
        key="diagnostics_confidence",
    )
)


if (
    selected_confidence
    != "All"
):

    diagnostics_base = (
        diagnostics_base[
            diagnostics_base[
                "decision_confidence"
            ]
            == selected_confidence
        ]
        .copy()
    )


source_options = [
    "All",
    *sorted(
        diagnostics_base[
            "decision_source"
        ]
        .dropna()
        .unique()
        .tolist()
    ),
]

selected_source = (
    st.sidebar.selectbox(
        "Decision source",
        options=source_options,
        index=0,
        key="diagnostics_source",
    )
)


if (
    selected_source
    != "All"
):

    diagnostics_base = (
        diagnostics_base[
            diagnostics_base[
                "decision_source"
            ]
            == selected_source
        ]
        .copy()
    )


if diagnostics_base.empty:

    st.warning(
        "No model diagnostics match the current filters."
    )

    st.stop()


# =========================================================
# FILTER RELATED DATASETS
# =========================================================

selected_skus = set(
    diagnostics_base[
        "sku_id"
    ]
    .tolist()
)


eligibility_filtered = (
    eligibility[
        eligibility[
            "sku_id"
        ].isin(
            selected_skus
        )
    ]
    .copy()
)


sku_estimates_filtered = (
    sku_estimates[
        sku_estimates[
            "sku_id"
        ].isin(
            selected_skus
        )
    ]
    .copy()
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
    "Model Diagnostics"
)

st.caption(
    "How reliable is the pricing signal, and what evidence "
    "supports each recommendation?"
)


# =========================================================
# DECISION MODEL COVERAGE
# =========================================================

st.subheader(
    "Decision model coverage"
)

sku_count = (
    diagnostics_base[
        "sku_id"
    ].nunique()
)

sku_level_count = (
    diagnostics_base[
        "decision_source"
    ]
    .eq(
        "SKU + Product Class"
    )
    .sum()
)

sku_level_pct = (
    sku_level_count
    / sku_count
    if sku_count > 0
    else 0
)


fallback_pct = (
    1
    - sku_level_pct
)

high_confidence_pct = (
    diagnostics_base[
        "decision_confidence"
    ]
    .eq(
        "High"
    )
    .mean()
)

median_elasticity = (
    diagnostics_base[
        "calibrated_elasticity"
    ]
    .median()
)

median_model_weight = (
    diagnostics_base[
        "model_weight"
    ]
    .median()
)


kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = (
    st.columns(6)
)

kpi1.metric(
    "SKUs Evaluated",
    f"{sku_count:,}",
)

kpi2.metric(
    "SKU-Level Evidence",
    f"{sku_level_pct:.1%}",
)

kpi3.metric(
    "Benchmark / Fallback",
    f"{fallback_pct:.1%}",
)

kpi4.metric(
    "High Confidence",
    f"{high_confidence_pct:.1%}",
)

kpi5.metric(
    "Median Elasticity",
    f"{median_elasticity:.2f}",
)

kpi6.metric(
    "Median Model Weight",
    f"{median_model_weight:.2f}",
    help=(
        "Higher model weight places more emphasis on "
        "the statistical estimate relative to the commercial prior."
    ),
)


# =========================================================
# EVIDENCE & CONFIDENCE
# =========================================================

st.subheader(
    "Evidence & confidence"
)

source_col, confidence_col, evidence_col = (
    st.columns(
        [1, 1, 1]
    )
)


# ---------------------------------------------------------
# DECISION SOURCE
# ---------------------------------------------------------

with source_col:

    st.markdown(
        "**Decision source mix**"
    )

    source_summary = (
        diagnostics_base[
            "decision_source"
        ]
        .value_counts()
        .rename_axis(
            "Decision Source"
        )
        .reset_index(
            name="SKUs"
        )
    )

    source_chart = (
        alt.Chart(
            source_summary
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "Decision Source:N",
                title=None,
                axis=alt.Axis(
                    labelAngle=0,
                    labelLimit=140,
                ),
            ),
            y=alt.Y(
                "SKUs:Q",
                title="SKUs",
            ),
            tooltip=[
                alt.Tooltip(
                    "Decision Source:N",
                    title="Source",
                ),
                alt.Tooltip(
                    "SKUs:Q",
                    title="SKUs",
                    format=",",
                ),
            ],
        )
        .properties(
            height=280
        )
    )

    st.altair_chart(
        source_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# CONFIDENCE
# ---------------------------------------------------------

with confidence_col:

    st.markdown(
        "**Decision confidence mix**"
    )

    confidence_summary = (
        diagnostics_base[
            "decision_confidence"
        ]
        .value_counts()
        .reindex(
            [
                "High",
                "Medium",
                "Low",
            ],
            fill_value=0,
        )
        .rename_axis(
            "Confidence"
        )
        .reset_index(
            name="SKUs"
        )
    )

    confidence_chart = (
        alt.Chart(
            confidence_summary
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "Confidence:N",
                title=None,
                sort=[
                    "High",
                    "Medium",
                    "Low",
                ],
                axis=alt.Axis(
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                "SKUs:Q",
                title="SKUs",
            ),
            tooltip=[
                alt.Tooltip(
                    "Confidence:N",
                    title="Confidence",
                ),
                alt.Tooltip(
                    "SKUs:Q",
                    title="SKUs",
                    format=",",
                ),
            ],
        )
        .properties(
            height=280
        )
    )

    st.altair_chart(
        confidence_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# EVIDENCE TIER
# ---------------------------------------------------------

with evidence_col:

    st.markdown(
        "**Elasticity evidence tier**"
    )

    evidence_summary = (
        eligibility_filtered[
            "evidence_tier"
        ]
        .value_counts()
        .reindex(
            [
                "Strong",
                "Moderate",
                "Limited",
                "Insufficient",
            ],
            fill_value=0,
        )
        .rename_axis(
            "Evidence Tier"
        )
        .reset_index(
            name="SKUs"
        )
    )

    evidence_chart = (
        alt.Chart(
            evidence_summary
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "Evidence Tier:N",
                title=None,
                sort=[
                    "Strong",
                    "Moderate",
                    "Limited",
                    "Insufficient",
                ],
                axis=alt.Axis(
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                "SKUs:Q",
                title="SKUs",
            ),
            tooltip=[
                alt.Tooltip(
                    "Evidence Tier:N",
                    title="Evidence",
                ),
                alt.Tooltip(
                    "SKUs:Q",
                    title="SKUs",
                    format=",",
                ),
            ],
        )
        .properties(
            height=280
        )
    )

    st.altair_chart(
        evidence_chart,
        use_container_width=True,
    )


# =========================================================
# ELASTICITY CALIBRATION
# =========================================================

st.subheader(
    "Elasticity calibration"
)

st.caption(
    "Raw decision elasticity is blended with commercial priors "
    "to reduce over-reaction to noisy statistical estimates."
)


raw_mean = (
    diagnostics_base[
        "decision_elasticity"
    ].mean()
)

calibrated_mean = (
    diagnostics_base[
        "calibrated_elasticity"
    ].mean()
)

avg_adjustment = (
    diagnostics_base[
        "calibration_adjustment"
    ].mean()
)

median_abs_adjustment = (
    diagnostics_base[
        "calibration_adjustment"
    ]
    .abs()
    .median()
)


cal1, cal2, cal3, cal4 = (
    st.columns(4)
)

cal1.metric(
    "Mean Raw Elasticity",
    f"{raw_mean:.2f}",
)

cal2.metric(
    "Mean Calibrated Elasticity",
    f"{calibrated_mean:.2f}",
)

cal3.metric(
    "Average Adjustment",
    f"{avg_adjustment:+.2f}",
)

cal4.metric(
    "Median Absolute Adjustment",
    f"{median_abs_adjustment:.2f}",
)


# =========================================================
# RAW VS CALIBRATED
# =========================================================

raw_vs_calibrated = (
    diagnostics_base[
        [
            "sku_id",
            "department",
            "product_class",
            "decision_elasticity",
            "calibrated_elasticity",
            "decision_confidence",
            "model_weight",
        ]
    ]
    .copy()
)


raw_cal_chart = (
    alt.Chart(
        raw_vs_calibrated
    )
    .mark_circle(
        opacity=0.55
    )
    .encode(
        x=alt.X(
            "decision_elasticity:Q",
            title="Raw Decision Elasticity",
            scale=alt.Scale(
                zero=False,
            ),
        ),
        y=alt.Y(
            "calibrated_elasticity:Q",
            title="Calibrated Elasticity",
            scale=alt.Scale(
                zero=False,
            ),
        ),
        size=alt.Size(
            "model_weight:Q",
            title="Model Weight",
        ),
        color=alt.Color(
            "decision_confidence:N",
            title="Confidence",
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
                "decision_elasticity:Q",
                title="Raw Elasticity",
                format=".2f",
            ),
            alt.Tooltip(
                "calibrated_elasticity:Q",
                title="Calibrated Elasticity",
                format=".2f",
            ),
            alt.Tooltip(
                "model_weight:Q",
                title="Model Weight",
                format=".2f",
            ),
            alt.Tooltip(
                "decision_confidence:N",
                title="Confidence",
            ),
        ],
    )
    .properties(
        height=400
    )
)


parity = (
    alt.Chart(
        pd.DataFrame(
            {
                "x": [
                    -4.0,
                    0.0,
                ],
                "y": [
                    -4.0,
                    0.0,
                ],
            }
        )
    )
    .mark_line(
        strokeDash=[
            4,
            4,
        ]
    )
    .encode(
        x="x:Q",
        y="y:Q",
    )
)


st.altair_chart(
    raw_cal_chart
    + parity,
    use_container_width=True,
)

st.caption(
    "Points below or above the diagonal indicate calibration "
    "has changed the magnitude of the raw elasticity signal."
)


# =========================================================
# ELASTICITY DISTRIBUTION
# =========================================================

st.subheader(
    "Calibrated elasticity distribution"
)


elasticity_hist = (
    alt.Chart(
        diagnostics_base
    )
    .mark_bar()
    .encode(
        x=alt.X(
            "calibrated_elasticity:Q",
            bin=alt.Bin(
                maxbins=30
            ),
            title="Calibrated Elasticity",
        ),
        y=alt.Y(
            "count():Q",
            title="SKUs",
        ),
        tooltip=[
            alt.Tooltip(
                "count():Q",
                title="SKUs",
            ),
        ],
    )
    .properties(
        height=300
    )
)


st.altair_chart(
    elasticity_hist,
    use_container_width=True,
)


# =========================================================
# SKU ESTIMATION QUALITY
# =========================================================

st.subheader(
    "SKU estimation quality"
)

st.caption(
    "Synthetic data includes the hidden true elasticity, allowing "
    "direct validation of the SKU-level model where sufficient "
    "price variation exists."
)


if (
    sku_estimates_filtered.empty
):

    st.info(
        "No SKU-level elasticity estimates match "
        "the current filters."
    )

else:

    successful_models = (
        sku_estimates_filtered[
            "model_status"
        ]
        .eq(
            "Success"
        )
        .mean()
    )

    median_r2 = (
        sku_estimates_filtered[
            "r_squared"
        ]
        .median()
    )

    median_std_error = (
        sku_estimates_filtered[
            "elasticity_std_error"
        ]
        .median()
    )

    median_abs_error = (
        sku_estimates_filtered[
            "absolute_error"
        ]
        .median()
    )

    statistically_significant = (
        sku_estimates_filtered[
            "elasticity_p_value"
        ]
        .lt(
            0.05
        )
        .mean()
    )


    q1, q2, q3, q4, q5 = (
        st.columns(5)
    )

    q1.metric(
        "Model Success Rate",
        f"{successful_models:.1%}",
    )

    q2.metric(
        "Median R²",
        f"{median_r2:.3f}",
    )

    q3.metric(
        "Median Std Error",
        f"{median_std_error:.3f}",
    )

    q4.metric(
        "Median Absolute Error",
        f"{median_abs_error:.2f}",
    )

    q5.metric(
        "p-value < 0.05",
        f"{statistically_significant:.1%}",
    )


    # -----------------------------------------------------
    # ESTIMATED VS TRUE ELASTICITY
    # -----------------------------------------------------

    st.markdown(
        "**Estimated vs hidden true elasticity**"
    )

    estimate_scatter = (
        alt.Chart(
            sku_estimates_filtered
        )
        .mark_circle(
            opacity=0.65
        )
        .encode(
            x=alt.X(
                "true_elasticity:Q",
                title="Hidden True Elasticity",
                scale=alt.Scale(
                    zero=False,
                ),
            ),
            y=alt.Y(
                "estimated_elasticity:Q",
                title="Estimated Elasticity",
                scale=alt.Scale(
                    zero=False,
                ),
            ),
            color=alt.Color(
                "evidence_tier:N",
                title="Evidence Tier",
            ),
            size=alt.Size(
                "price_range_pct:Q",
                title="Price Range",
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "sku_id:N",
                    title="SKU",
                ),
                alt.Tooltip(
                    "product_class:N",
                    title="Product Class",
                ),
                alt.Tooltip(
                    "true_elasticity:Q",
                    title="True Elasticity",
                    format=".2f",
                ),
                alt.Tooltip(
                    "estimated_elasticity:Q",
                    title="Estimated Elasticity",
                    format=".2f",
                ),
                alt.Tooltip(
                    "absolute_error:Q",
                    title="Absolute Error",
                    format=".2f",
                ),
                alt.Tooltip(
                    "r_squared:Q",
                    title="R²",
                    format=".3f",
                ),
                alt.Tooltip(
                    "evidence_tier:N",
                    title="Evidence Tier",
                ),
            ],
        )
        .properties(
            height=400
        )
    )


    estimate_parity = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [
                        -4.0,
                        0.0,
                    ],
                    "y": [
                        -4.0,
                        0.0,
                    ],
                }
            )
        )
        .mark_line(
            strokeDash=[
                4,
                4,
            ]
        )
        .encode(
            x="x:Q",
            y="y:Q",
        )
    )


    st.altair_chart(
        estimate_scatter
        + estimate_parity,
        use_container_width=True,
    )


    st.caption(
        "SKU-level elasticity estimates can be unstable where historical "
        "price variation is limited. The decision hierarchy therefore "
        "uses evidence strength, product-class benchmarks and calibration "
        "rather than relying on raw SKU estimates alone."
    )


# =========================================================
# PRODUCT CLASS DIAGNOSTICS
# =========================================================

st.subheader(
    "Product-class model diagnostics"
)


class_filtered = (
    class_estimates.copy()
)


if (
    selected_department
    != "All"
):

    class_filtered = (
        class_filtered[
            class_filtered[
                "department"
            ]
            == selected_department
        ]
        .copy()
    )


if (
    selected_product_class
    != "All"
):

    class_filtered = (
        class_filtered[
            class_filtered[
                "product_class"
            ]
            == selected_product_class
        ]
        .copy()
    )


class_table = (
    class_filtered[
        [
            "department",
            "category",
            "product_class",
            "estimated_elasticity",
            "true_elasticity_mean",
            "absolute_error",
            "r_squared",
            "n_skus",
            "n_price_moves",
            "confidence",
            "model_status",
        ]
    ]
    .sort_values(
        [
            "confidence",
            "absolute_error",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .copy()
)


class_table[
    "estimated_elasticity"
] = (
    class_table[
        "estimated_elasticity"
    ]
    .map(
        lambda x: f"{x:.2f}"
    )
)

class_table[
    "true_elasticity_mean"
] = (
    class_table[
        "true_elasticity_mean"
    ]
    .map(
        lambda x: f"{x:.2f}"
    )
)

class_table[
    "absolute_error"
] = (
    class_table[
        "absolute_error"
    ]
    .map(
        lambda x: f"{x:.2f}"
    )
)

class_table[
    "r_squared"
] = (
    class_table[
        "r_squared"
    ]
    .map(
        lambda x: f"{x:.3f}"
    )
)


class_table = (
    class_table.rename(
        columns={
            "department":
                "Department",
            "category":
                "Category",
            "product_class":
                "Product Class",
            "estimated_elasticity":
                "Estimated Elasticity",
            "true_elasticity_mean":
                "True Elasticity",
            "absolute_error":
                "Absolute Error",
            "r_squared":
                "R²",
            "n_skus":
                "SKUs",
            "n_price_moves":
                "Price Moves",
            "confidence":
                "Confidence",
            "model_status":
                "Model Status",
        }
    )
)


st.dataframe(
    class_table,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# MODEL REVIEW QUEUE
# =========================================================

st.subheader(
    "Model review queue"
)

st.caption(
    "SKUs with weaker evidence, larger estimation error or "
    "larger calibration adjustments requiring additional scrutiny."
)


review_queue = (
    diagnostics_base[
        [
            "sku_id",
            "department",
            "product_class",
            "decision_source",
            "decision_confidence",
            "evidence_tier",
            "decision_elasticity",
            "calibrated_elasticity",
            "calibration_adjustment",
            "model_weight",
        ]
    ]
    .copy()
)


review_queue = (
    review_queue.merge(
        sku_estimates[
            [
                "sku_id",
                "estimated_elasticity",
                "absolute_error",
                "r_squared",
                "elasticity_p_value",
                "price_range_pct",
            ]
        ],
        on="sku_id",
        how="left",
    )
)


review_queue[
    "review_score"
] = (
    review_queue[
        "calibration_adjustment"
    ]
    .abs()
    +
    review_queue[
        "absolute_error"
    ]
    .fillna(
        0
    )
    +
    (
        1
        - review_queue[
            "model_weight"
        ]
    )
)


review_queue = (
    review_queue.sort_values(
        "review_score",
        ascending=False,
    )
    .head(30)
)


display_review = (
    review_queue[
        [
            "sku_id",
            "department",
            "product_class",
            "decision_source",
            "decision_confidence",
            "evidence_tier",
            "decision_elasticity",
            "calibrated_elasticity",
            "calibration_adjustment",
            "model_weight",
            "absolute_error",
            "r_squared",
            "elasticity_p_value",
        ]
    ]
    .copy()
)


for column in [
    "decision_elasticity",
    "calibrated_elasticity",
    "calibration_adjustment",
    "model_weight",
    "absolute_error",
    "r_squared",
    "elasticity_p_value",
]:

    display_review[
        column
    ] = (
        display_review[
            column
        ]
        .map(
            lambda x: (
                f"{x:.3f}"
                if pd.notna(x)
                else ""
            )
        )
    )


display_review = (
    display_review.rename(
        columns={
            "sku_id":
                "SKU",
            "department":
                "Department",
            "product_class":
                "Product Class",
            "decision_source":
                "Decision Source",
            "decision_confidence":
                "Confidence",
            "evidence_tier":
                "Evidence Tier",
            "decision_elasticity":
                "Raw Elasticity",
            "calibrated_elasticity":
                "Calibrated Elasticity",
            "calibration_adjustment":
                "Calibration Adjustment",
            "model_weight":
                "Model Weight",
            "absolute_error":
                "SKU Absolute Error",
            "r_squared":
                "SKU R²",
            "elasticity_p_value":
                "SKU p-value",
        }
    )
)


st.dataframe(
    display_review,
    hide_index=True,
    use_container_width=True,
    height=560,
)


review_queue_csv = (
    review_queue
    .drop(
        columns=["review_score"],
        errors="ignore",
    )
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download model review queue",
    data=review_queue_csv,
    file_name="model_review_queue.csv",
    mime="text/csv",
)


# =========================================================
# METHODOLOGY
# =========================================================

st.subheader(
    "Methodology"
)


with st.expander(
    "How the elasticity decision hierarchy works"
):

    st.markdown(
        """
The pricing engine uses a hierarchical elasticity approach rather
than relying on a single SKU-level regression estimate.

**1. SKU evidence**

SKUs with sufficient historical price variation are eligible for
SKU-level elasticity estimation.

**2. Product-class benchmark**

Product-class estimates provide a broader benchmark where individual
SKU evidence is limited or noisy.

**3. Decision hierarchy**

SKU and product-class signals are blended according to evidence strength.
Where SKU-level evidence is insufficient, the hierarchy falls back to
the broader benchmark.

**4. Commercial calibration**

The resulting decision elasticity is blended with a commercial prior.
The model weight is determined by confidence, limiting the impact of
extreme or unstable statistical estimates.

**5. Decision engine**

Calibrated elasticity feeds the pricing scenario engine together with
cost, competitor price, margin and commercial guardrails to generate
the final pricing recommendation.
"""
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Model diagnostics use synthetic hidden true elasticity for "
    "validation. In a real retail environment, model quality would "
    "instead be assessed through holdout performance, experiment "
    "results, stability monitoring and realised pricing outcomes."
)