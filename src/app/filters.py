import pandas as pd
import streamlit as st


def _single_select(
    label: str,
    options: list[str],
    key: str,
) -> str:

    display_options = [
        "All",
        *options,
    ]

    return st.sidebar.selectbox(
        label,
        options=display_options,
        index=0,
        key=key,
    )


def render_pricing_filters(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    st.sidebar.header("Filters")

    filtered = data.copy()

    # =====================================================
    # DEPARTMENT
    # =====================================================

    department_options = sorted(
        filtered["department"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_department = _single_select(
        label="Department",
        options=department_options,
        key="filter_department",
    )

    if selected_department != "All":

        filtered = filtered[
            filtered["department"]
            == selected_department
        ].copy()

    # =====================================================
    # CATEGORY
    # =====================================================

    category_options = sorted(
        filtered["category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_category = _single_select(
        label="Category",
        options=category_options,
        key="filter_category",
    )

    if selected_category != "All":

        filtered = filtered[
            filtered["category"]
            == selected_category
        ].copy()

    # =====================================================
    # PRODUCT CLASS
    # =====================================================

    product_class_options = sorted(
        filtered["product_class"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_product_class = _single_select(
        label="Product class",
        options=product_class_options,
        key="filter_product_class",
    )

    if selected_product_class != "All":

        filtered = filtered[
            filtered["product_class"]
            == selected_product_class
        ].copy()

    # =====================================================
    # DECISION CONFIDENCE
    # =====================================================

    confidence_order = [
        "High",
        "Medium",
        "Low",
    ]

    confidence_available = set(
        filtered[
            "decision_confidence"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    confidence_options = [
        value
        for value in confidence_order
        if value in confidence_available
    ]

    selected_confidence = _single_select(
        label="Decision confidence",
        options=confidence_options,
        key="filter_confidence",
    )

    if selected_confidence != "All":

        filtered = filtered[
            filtered[
                "decision_confidence"
            ]
            == selected_confidence
        ].copy()

    # =====================================================
    # RECOMMENDED ACTION
    # =====================================================

    action_order = [
        "Increase Price",
        "Hold Price",
        "Reduce Price",
        "Review",
    ]

    action_available = set(
        filtered[
            "recommended_action"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    action_options = [
        value
        for value in action_order
        if value in action_available
    ]

    selected_action = _single_select(
        label="Recommended action",
        options=action_options,
        key="filter_action",
    )

    if selected_action != "All":

        filtered = filtered[
            filtered[
                "recommended_action"
            ]
            == selected_action
        ].copy()

    # =====================================================
    # ACTIVE FILTER SUMMARY
    # =====================================================

    filters = {
        "department": selected_department,
        "category": selected_category,
        "product_class": (
            selected_product_class
        ),
        "decision_confidence": (
            selected_confidence
        ),
        "recommended_action": (
            selected_action
        ),
    }

    return filtered, filters