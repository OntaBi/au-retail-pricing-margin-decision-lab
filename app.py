from pathlib import Path

import streamlit as st


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="AU Retail Pricing & Margin Decision Lab",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# PATHS
# =========================================================

APP_ROOT = Path(__file__).resolve().parent


# =========================================================
# PAGE NAVIGATION
# =========================================================

pages = [
    st.Page(
        APP_ROOT
        / "app_pages"
        / "1_Executive_Overview.py",
        title="Executive Overview",
        default=True,
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "2_Recommendation_Queue.py",
        title="Recommendation Queue",
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "3_Scenario_Explorer.py",
        title="Scenario Explorer",
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "4_Margin_Competition.py",
        title="Margin & Competition",
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "5_Model_Diagnostics.py",
        title="Model Diagnostics",
    ),
]


# =========================================================
# TOP NAVIGATION
# =========================================================

navigation = st.navigation(
    pages,
    position="top",
)

navigation.run()